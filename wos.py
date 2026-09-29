#!/usr/bin/env python3
"""Local Flask processor for Web of Science tagged-text exports.

DiVA import behaviour
----------------------
* KTH authors are identified ONLY when an AF full name exactly matches a name
  in a bracketed C1 entry whose affiliation is KTH.
* AU remains unchanged.
* Only those KTH authors receive $$$ in AF and in the same literal C1 names.
* For records with > 30 authors, retain first author, last author and all
  detected KTH authors, plus complete C1 fields linked to retained authors.
* C1 brackets, semicolons, affiliations and line wrapping are preserved; only
  exact KTH names are prefixed with $$$.
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
    "kth royal inst technol",
    "royal institute of technology",
    "royal inst technol",
    "royal inst technol kth",
    "kungliga tekniska högskolan",
    "kungliga tekniska hogskolan",
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
<p>Upload a Web of Science tagged-text <code>.txt</code> export. Only authors explicitly linked to a KTH <code>C1</code> affiliation are marked with <code>$$$</code> in <code>AF</code> and <code>C1</code>; <code>AU</code> is unchanged.</p>
{% if error %}<p class="error">{{ error }}</p>{% endif %}
{% if results %}<div class="success"><p><strong>Processing complete.</strong> Click a link to download its file.</p><ul>
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


@dataclass
class FieldBlock:
    tag: str
    lines: list[str]


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


def parse_blocks(lines: list[str]) -> list[FieldBlock]:
    blocks: list[FieldBlock] = []
    position = 0
    while position < len(lines):
        first = lines[position]
        block = [first]
        field_tag = tag(first)
        position += 1
        while position < len(lines) and tag(lines[position]) == "  ":
            block.append(lines[position])
            position += 1
        blocks.append(FieldBlock(field_tag, block))
    return blocks


def values_from_blocks(blocks: list[FieldBlock], wanted: str) -> list[str]:
    values: list[str] = []
    for block in blocks:
        if block.tag == wanted:
            values.extend(line[3:].rstrip("\r\n") for line in block.lines)
    return values


def field_values(lines: list[str], wanted: str) -> list[str]:
    return values_from_blocks(parse_blocks(lines), wanted)


def normalise_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.replace("$$$", "")).strip().casefold()


def name_key(name: str) -> tuple[str, str]:
    """Used only for C1 filtering after a >30-author reduction."""
    cleaned = normalise_name(name)
    if "," not in cleaned:
        return cleaned, ""
    family, given = cleaned.split(",", 1)
    word = re.search(r"[\wÀ-ÖØ-öø-ÿ]+", given)
    return family.strip(), word.group(0)[0] if word else ""


def c1_text(block: FieldBlock) -> str:
    """Unfold C1 for recognition only; never use it as output."""
    return " ".join(line[3:].rstrip("\r\n").strip() for line in block.lines)


def names_in_c1_block(block: FieldBlock) -> set[str]:
    names: set[str] = set()
    for group in re.findall(r"\[([^\]]+)\]", c1_text(block)):
        for name in group.split(";"):
            normalized = normalise_name(name)
            if normalized:
                names.add(normalized)
    return names


def is_kth_c1_block(block: FieldBlock) -> bool:
    text = c1_text(block).casefold()
    return any(pattern in text for pattern in KTH_PATTERNS)


def mark_kth(record: Record, blocks: list[FieldBlock]) -> None:
    """Mark only exact AF full names explicitly linked to a KTH C1 block."""
    record.authors_au = values_from_blocks(blocks, "AU")
    record.authors_af = values_from_blocks(blocks, "AF")

    kth_c1_names: set[str] = set()
    for block in blocks:
        if block.tag == "C1" and is_kth_c1_block(block):
            kth_c1_names.update(names_in_c1_block(block))

    for index, af_name in enumerate(record.authors_af):
        if normalise_name(af_name) in kth_c1_names:
            record.kth_af.add(index)
            if index < len(record.authors_au):
                record.kth_au.add(index)


def selected_indices(count: int, kth: set[int]) -> list[int]:
    if count <= MAX_AUTHORS:
        return list(range(count))
    return sorted({0, count - 1, *kth})


def retained_name_keys(record: Record, au_keep: list[int], af_keep: list[int]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for index in au_keep:
        if index < len(record.authors_au):
            keys.add(name_key(record.authors_au[index]))
    for index in af_keep:
        if index < len(record.authors_af):
            keys.add(name_key(record.authors_af[index]))
    keys.discard(("", ""))
    return keys


def c1_has_retained_author(block: FieldBlock, retained_keys: set[tuple[str, str]]) -> bool:
    return any(name_key(name) in retained_keys for name in names_in_c1_block(block))


def render_au(values: list[str], keep: list[int], newline: str) -> list[str]:
    output: list[str] = []
    for out_index, source_index in enumerate(keep):
        prefix = "AU " if out_index == 0 else "   "
        output.append(prefix + values[source_index] + newline)
    if len(values) > MAX_AUTHORS:
        output.append("   et al." + newline)
    return output


def render_af(values: list[str], keep: list[int], kth: set[int], newline: str) -> list[str]:
    output: list[str] = []
    for out_index, source_index in enumerate(keep):
        prefix = "AF " if out_index == 0 else "   "
        marker = "$$$" if source_index in kth else ""
        output.append(prefix + marker + values[source_index] + newline)
    if len(values) > MAX_AUTHORS:
        output.append("   et al." + newline)
    return output


def mark_c1_block_literal(block: FieldBlock, kth_full_names: set[str]) -> list[str]:
    """Mark exact KTH AF names in C1 while preserving all C1 punctuation."""
    marked: list[str] = []
    for line in block.lines:
        changed = line
        for name in sorted(kth_full_names, key=len, reverse=True):
            changed = changed.replace(name, "$$$" + name)
        marked.append(changed)
    return marked


def transform_record(record: Record) -> tuple[str, int, str, list[str]]:
    blocks = parse_blocks(record.lines)
    mark_kth(record, blocks)

    author_count = len(record.authors_af) or len(record.authors_au)
    short_record = author_count > MAX_AUTHORS
    au_keep = selected_indices(len(record.authors_au), record.kth_au)
    af_keep = selected_indices(len(record.authors_af), record.kth_af)
    retained_keys = retained_name_keys(record, au_keep, af_keep)
    kth_full_names = {
        record.authors_af[index]
        for index in record.kth_af
        if index < len(record.authors_af)
    }
    newline = "\r\n" if any(line.endswith("\r\n") for line in record.lines) else "\n"

    output: list[str] = []
    au_rendered = False
    af_rendered = False
    for block in blocks:
        if block.tag == "AU":
            if not au_rendered:
                output.extend(render_au(record.authors_au, au_keep, newline))
                au_rendered = True
            continue

        if block.tag == "AF":
            if not af_rendered:
                output.extend(render_af(record.authors_af, af_keep, record.kth_af, newline))
                af_rendered = True
            continue

        if block.tag == "C1":
            if not short_record or c1_has_retained_author(block, retained_keys):
                output.extend(mark_c1_block_literal(block, kth_full_names))
            continue

        output.extend(block.lines)

    title = " ".join(field_values(record.lines, "TI")).strip()
    kth_names = [record.authors_af[index] for index in sorted(record.kth_af) if index < len(record.authors_af)]
    return "".join(output), author_count, title, kth_names


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
    kth_report = ["Record\tWoS UT\tTitle\tKTH authors"]
    for number, record in enumerate(records, start=1):
        rendered, author_count, title, kth_names = transform_record(record)
        processed.append(rendered)
        if author_count > MAX_AUTHORS:
            report.append(f"{number}\t{author_count}\t{title}")
        ut_values = field_values(record.lines, "UT")
        ut = ut_values[0] if ut_values else ""
        kth_report.append(f"{number}\t{ut}\t{title}\t{'; '.join(kth_names)}")

    output_date = one_month_from_today().isoformat()
    base = f"Tas efter {output_date}"
    files: dict[str, bytes] = {}
    for batch_number, start in enumerate(range(0, len(processed), BATCH_SIZE), start=1):
        filename = f"{base}_UT_{batch_number:03d}.txt"
        body = "".join(header) + "".join(processed[start:start + BATCH_SIZE]) + "EF\n"
        files[filename] = body.encode("utf-8")

    files[f"{base}_ANTAL_FF.txt"] = ("\n".join(report) + "\n").encode("utf-8")
    files[f"{base}_KTH_FORFATTARE.txt"] = ("\n".join(kth_report) + "\n").encode("utf-8")
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
    result_links = [
        {"name": name, "size": len(content), "url": f"/download/{token}/{name}"}
        for name, content in files.items()
    ]
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
    return send_file(
        io.BytesIO(content),
        mimetype="text/plain; charset=utf-8",
        as_attachment=True,
        download_name=filename,
    )


@app.errorhandler(413)
def too_large(_: object) -> tuple[str, int]:
    return render_template_string(PAGE, error="The file is larger than 10 MB.", results=None), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
