#!/usr/bin/env python3
"""Local Flask app for Web of Science tagged-text exports.

For records with <= 30 authors, AU, AF and C1 are retained in full.
For records with > 30 authors, retains first/last authors plus all KTH authors,
and retains only C1 affiliation entries associated with retained authors.
KTH authors are marked with $$$ in AU and AF. C1 affiliation text itself is
preserved verbatim, except that entries no longer associated with any retained
author are omitted in shortened records.
"""
from __future__ import annotations

import io
import re
import secrets
import time
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

from flask import Flask, Response, abort, render_template_string, request, send_file

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

MAX_AUTHORS = 30
BATCH_SIZE = 25
RESULT_TTL_SECONDS = 2 * 60 * 60
KTH_PATTERNS = (
    "royal institute of technology",
    "royal inst technol",
    "kungliga tekniska hgsk",
)
RESULTS: dict[str, dict[str, object]] = {}

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WoS / KTH file processor</title>
<style>
body { font-family: system-ui, sans-serif; line-height: 1.45; max-width: 800px; margin: 3rem auto; padding: 0 1rem; }
main { border: 1px solid #ccc; border-radius: .5rem; padding: 1.5rem; }
label { display:block; margin: 1rem 0 .35rem; font-weight: 600; }
button { background:#0869b2; color:#fff; border:0; border-radius:.3rem; padding:.7rem 1rem; font-size:1rem; cursor:pointer; }
small, .note { color:#444; }.error { border-left:4px solid #b00020; padding:.7rem; background:#fff1f2; }
.success { border-left:4px solid #16803b; padding:.7rem; background:#effcf3; } code { background:#f4f4f4; padding:.1rem .25rem; }
ul { padding-left:1.25rem; } li { margin:.55rem 0; } a.download { color:#0645ad; font-weight:600; }
</style></head><body><main>
<h1>WoS / KTH file processor</h1>
<p>Upload a Web of Science tagged-text <code>.txt</code> export. KTH authors are marked with <code>$$$</code>. Each output WoS file contains at most 25 records.</p>
{% if error %}<p class="error">{{ error }}</p>{% endif %}
{% if results %}<div class="success"><p><strong>Processing complete.</strong> Right-click a link below and choose <em>Save link as…</em>, or click it to download.</p><ul>
{% for result in results %}<li><a class="download" href="{{ result.url }}" download="{{ result.name }}">{{ result.name }}</a> <small>({{ result.size }} bytes)</small></li>{% endfor %}
</ul><p><small>For privacy, links expire after two hours or when this application is restarted.</small></p></div>{% endif %}
<form method="post" enctype="multipart/form-data"><label for="file">WoS text export</label><input id="file" name="file" type="file" accept=".txt,text/plain" required>
<p class="note"><small>KTH markers are always enabled. No files are emailed.</small></p><button type="submit">Process file</button></form>
</main></body></html>"""


@dataclass
class Record:
    lines: list[str]
    authors_au: list[str] = field(default_factory=list)
    authors_af: list[str] = field(default_factory=list)
    kth_au: set[int] = field(default_factory=set)
    kth_af: set[int] = field(default_factory=set)


def clean_expired_results() -> None:
    now = time.time()
    for token in list(RESULTS):
        if float(RESULTS[token]["expires"]) <= now:
            del RESULTS[token]


def one_month_from_today(today: date | None = None) -> date:
    today = today or date.today()
    year = today.year + (today.month == 12)
    month = 1 if today.month == 12 else today.month + 1
    return date(year, month, min(today.day, monthrange(year, month)[1]))


def eol(line: str) -> str:
    return "\r\n" if line.endswith("\r\n") else "\n"


def tag(line: str) -> str:
    return line[:2] if len(line) >= 2 else ""


def clean_abstract_line(line: str) -> str:
    if line.startswith("AB"):
        match = re.search(r"(?:\(c\)|©)", line, flags=re.IGNORECASE)
        if match:
            return line[:match.start()].rstrip() + eol(line)
    return line


def split_records(lines: Iterable[str]) -> tuple[list[str], list[Record]]:
    header: list[str] = []
    records: list[Record] = []
    current: list[str] | None = None
    for raw in lines:
        line = clean_abstract_line(raw)
        if line.startswith("PT "):
            if current is not None:
                records.append(Record(current))
            current = [line]
        elif current is None:
            if not line.startswith("EF"):
                header.append(line)
        else:
            current.append(line)
            if line.startswith("ER"):
                records.append(Record(current))
                current = None
    if current is not None:
        records.append(Record(current))
    return header, records


def field_values(lines: list[str], wanted: str) -> list[str]:
    values: list[str] = []
    collecting = False
    for line in lines:
        current = tag(line)
        if current == wanted:
            values.append(line[3:].rstrip("\r\n"))
            collecting = True
        elif collecting and current == "  ":
            values.append(line[3:].rstrip("\r\n"))
        else:
            collecting = False
    return values


def normalise_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.replace("$$$", "")).strip().casefold()


def c1_blocks(lines: list[str]) -> list[list[str]]:
    """Return raw C1 blocks, including their original continuation lines."""
    blocks: list[list[str]] = []
    position = 0
    while position < len(lines):
        if tag(lines[position]) != "C1":
            position += 1
            continue
        block = [lines[position]]
        position += 1
        while position < len(lines) and tag(lines[position]) == "  ":
            block.append(lines[position])
            position += 1
        blocks.append(block)
    return blocks


def c1_block_text(block: list[str]) -> str:
    return " ".join(line[3:].rstrip("\r\n").strip() for line in block)


def names_in_c1_block(block: list[str]) -> set[str]:
    match = re.search(r"\[([^\]]+)\]", c1_block_text(block))
    if not match:
        return set()
    return {normalise_name(name) for name in match.group(1).split(";") if normalise_name(name)}


def is_kth_c1_block(block: list[str]) -> bool:
    text = c1_block_text(block).casefold()
    return (
        any(pattern in text for pattern in KTH_PATTERNS)
        or ("kth" in text and "sweden" in text)
        or ("inst technol" in text and "sweden" in text)
        or ("royal inst" in text and "sweden" in text)
    )


def author_matches_c1(author: str, c1_name: str) -> bool:
    """Match AU/AF forms conservatively, allowing full-name vs initials forms."""
    author = normalise_name(author)
    c1_name = normalise_name(c1_name)
    if author == c1_name:
        return True
    author_surname = author.split(",", 1)[0]
    c1_surname = c1_name.split(",", 1)[0]
    return bool(author_surname and author_surname == c1_surname)


def mark_kth(record: Record) -> None:
    record.authors_au = field_values(record.lines, "AU")
    record.authors_af = field_values(record.lines, "AF")
    kth_names: set[str] = set()
    for block in c1_blocks(record.lines):
        if is_kth_c1_block(block):
            kth_names.update(names_in_c1_block(block))

    for index, name in enumerate(record.authors_au):
        if any(author_matches_c1(name, kth_name) for kth_name in kth_names):
            record.kth_au.add(index)
    for index, name in enumerate(record.authors_af):
        if any(author_matches_c1(name, kth_name) for kth_name in kth_names):
            record.kth_af.add(index)

    for index in record.kth_af:
        if index < len(record.authors_au):
            record.kth_au.add(index)
    for index in record.kth_au:
        if index < len(record.authors_af):
            record.kth_af.add(index)


def selected_indices(count: int, kth: set[int]) -> list[int]:
    if count <= MAX_AUTHORS:
        return list(range(count))
    return sorted({0, count - 1, *kth})


def retained_name_forms(record: Record, au_keep: list[int], af_keep: list[int]) -> set[str]:
    names: set[str] = set()
    for index in au_keep:
        if index < len(record.authors_au):
            names.add(normalise_name(record.authors_au[index]))
    for index in af_keep:
        if index < len(record.authors_af):
            names.add(normalise_name(record.authors_af[index]))
    return names


def keep_c1_block(block: list[str], retained_names: set[str]) -> bool:
    names = names_in_c1_block(block)
    if not names:
        # An unlinked address cannot safely be attributed after author truncation.
        return False
    return any(
        author_matches_c1(retained_name, c1_name)
        for retained_name in retained_names
        for c1_name in names
    )


def render_field(field: str, values: list[str], keep: list[int], kth: set[int], newline: str) -> list[str]:
    output: list[str] = []
    for out_index, source_index in enumerate(keep):
        prefix = f"{field} " if out_index == 0 else "   "
        value = values[source_index]
        if source_index in kth:
            value = "$$$" + value
        output.append(prefix + value + newline)
    if len(values) > MAX_AUTHORS:
        output.append("   et al." + newline)
    return output


def transform_record(record: Record) -> tuple[str, int, str]:
    mark_kth(record)
    author_count = len(record.authors_af)
    au_keep = selected_indices(len(record.authors_au), record.kth_au)
    af_keep = selected_indices(author_count, record.kth_af)
    short_record = author_count > MAX_AUTHORS
    retained_names = retained_name_forms(record, au_keep, af_keep)
    newline = "\r\n" if any(line.endswith("\r\n") for line in record.lines) else "\n"

    output: list[str] = []
    position = 0
    while position < len(record.lines):
        current = tag(record.lines[position])
        if current in {"AU", "AF"}:
            field = current
            while position < len(record.lines) and tag(record.lines[position]) in {field, "  "}:
                position += 1
            if field == "AU":
                output.extend(render_field("AU", record.authors_au, au_keep, record.kth_au, newline))
            else:
                output.extend(render_field("AF", record.authors_af, af_keep, record.kth_af, newline))
            continue

        if current == "C1":
            block = [record.lines[position]]
            position += 1
            while position < len(record.lines) and tag(record.lines[position]) == "  ":
                block.append(record.lines[position])
                position += 1
            if not short_record or keep_c1_block(block, retained_names):
                output.extend(block)
            continue

        output.append(record.lines[position])
        position += 1

    title = " ".join(field_values(record.lines, "TI")).strip()
    return "".join(output), author_count, title


def process_wos(data: bytes) -> dict[str, bytes]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    header, records = split_records(text.splitlines(keepends=True))
    if not records:
        raise ValueError("No WoS records were found. Expected records beginning with 'PT ' and ending with 'ER'.")

    processed: list[str] = []
    report = ["Record\tAuthor count (AF)\tTitle"]
    for number, record in enumerate(records, start=1):
        rendered, author_count, title = transform_record(record)
        processed.append(rendered)
        if author_count > MAX_AUTHORS:
            report.append(f"{number}\t{author_count}\t{title}")

    output_date = one_month_from_today().isoformat()
    base = f"Tas efter {output_date}"
    files: dict[str, bytes] = {}
    for batch_number, start in enumerate(range(0, len(processed), BATCH_SIZE), start=1):
        filename = f"{base}_UT_{batch_number:03d}.txt"
        body = "".join(header) + "".join(processed[start:start + BATCH_SIZE]) + "EF\n"
        files[filename] = body.encode("utf-8")
    files[f"{base}_ANTAL_FF.txt"] = ("\n".join(report) + "\n").encode("utf-8")
    return files


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    clean_expired_results()
    if request.method == "GET":
        return render_template_string(PAGE, error=None, results=None)
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return render_template_string(PAGE, error="Choose a .txt WoS export first.", results=None), 400
    if not uploaded.filename.lower().endswith(".txt"):
        return render_template_string(PAGE, error="Only .txt files are accepted.", results=None), 400
    try:
        files = process_wos(uploaded.read())
    except ValueError as exc:
        return render_template_string(PAGE, error=str(exc), results=None), 400

    token = secrets.token_urlsafe(24)
    RESULTS[token] = {"expires": time.time() + RESULT_TTL_SECONDS, "files": files}
    result_links = [{"name": name, "size": len(content), "url": f"/download/{token}/{name}"} for name, content in files.items()]
    return render_template_string(PAGE, error=None, results=result_links)


@app.route("/download/<token>/<path:filename>")
def download(token: str, filename: str) -> Response:
    clean_expired_results()
    collection = RESULTS.get(token)
    if collection is None:
        abort(404, "This result collection has expired. Process the input file again.")
    files = collection["files"]
    assert isinstance(files, dict)
    content = files.get(filename)
    if not isinstance(content, bytes):
        abort(404)
    return send_file(io.BytesIO(content), mimetype="text/plain; charset=utf-8", as_attachment=True, download_name=filename)


@app.errorhandler(413)
def too_large(_: object) -> tuple[str, int]:
    return render_template_string(PAGE, error="The file is larger than 10 MB.", results=None), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
