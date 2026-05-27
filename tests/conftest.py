"""Test configuration and shared fixtures."""

from __future__ import annotations

import csv
import gzip
import io
import json
from datetime import datetime
from pathlib import Path

import pytest
import zstandard as zstd


# ---------------------------------------------------------------------------
# Synthetic JSONL fixtures
#
# The thin slice exercises:
#   - simple top-level fields (text, category)
#   - dotted-path extraction (attributes.body)
#   - recursive directory walks across multiple files
#   - compressed files (.gz, .zst)
#   - plain text files (line and whole-file modes)
#   - multi-source records for routed text extraction
#
# Fixtures are generated per test session into a tmp dir, so we don't commit
# binary-ish corpus files and tests stay deterministic.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fixtures_root(tmp_path_factory) -> Path:
    """Root of generated synthetic fixtures, shared across the session."""
    root = tmp_path_factory.mktemp("ignite_fixtures")

    # Plain JSONL fixtures.
    _write_simple_jsonl(root / "simple.jsonl", count=50)
    _write_nested_jsonl(root / "nested" / "file_a.jsonl", count=30, source="github")
    _write_nested_jsonl(root / "nested" / "file_b.jsonl", count=30, source="reddit")
    _write_simple_jsonl(root / "noisy.jsonl", count=20, with_garbage=True)

    # Compressed JSONL: same content as simple.jsonl, just gzipped / zstd'd.
    _write_simple_jsonl_gz(root / "compressed" / "simple.jsonl.gz", count=15)
    _write_simple_jsonl_zst(root / "compressed" / "simple.jsonl.zst", count=15)

    # Plain text fixtures.
    _write_text_file(
        root / "text" / "doc1.txt",
        "first paragraph of the document\nsecond line\nthird line\n",
    )
    _write_text_file(
        root / "text" / "doc2.md",
        "# A markdown doc\nwith two content lines\n\n",
    )

    # Multi-source JSONL for routed text extraction.
    _write_routed_jsonl(root / "routed" / "events.jsonl")

    # Tabular fixtures.
    _write_simple_csv(root / "tabular" / "products.csv", count=12)
    _write_simple_tsv(root / "tabular" / "products.tsv", count=12)
    _write_simple_csv_gz(root / "tabular" / "products.csv.gz", count=12)
    _write_simple_csv_zst(root / "tabular" / "products.csv.zst", count=12)
    _write_routed_csv(root / "tabular" / "events_routed.csv")
    _write_quirky_csv(root / "tabular" / "quirky.csv")

    # Pipeline fixture: timestamped JSONL with multiple labels and short
    # records, used to exercise normalization, time/label filters, and
    # all sampling modes from the same corpus.
    _write_timeline_jsonl(root / "pipeline" / "timeline.jsonl")

    return root


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def _simple_records(count: int) -> list[dict]:
    return [
        {
            "id": f"rec-{i:04d}",
            "text": f"sample message number {i} about topic-{i % 5}",
            "category": f"topic-{i % 5}",
        }
        for i in range(count)
    ]


def _write_simple_jsonl(path: Path, count: int, with_garbage: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in _simple_records(count):
            f.write(json.dumps(record) + "\n")
        if with_garbage:
            f.write("not-a-json-line\n")
            f.write("\n")
            f.write(json.dumps({"id": "no-text", "category": "topic-1"}) + "\n")
            f.write(json.dumps({"id": "empty-text", "text": "   ", "category": "x"}) + "\n")


def _write_nested_jsonl(path: Path, count: int, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(count):
            record = {
                "id": f"{source}-{i:04d}",
                "attributes": {
                    "source": source,
                    "body": f"{source} body content {i}",
                    "title": f"{source} title {i}",
                },
            }
            f.write(json.dumps(record) + "\n")


def _write_simple_jsonl_gz(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(r) for r in _simple_records(count)).encode("utf-8") + b"\n"
    with gzip.open(path, "wb") as f:
        f.write(payload)


def _write_simple_jsonl_zst(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(r) for r in _simple_records(count)).encode("utf-8") + b"\n"
    cctx = zstd.ZstdCompressor()
    with path.open("wb") as f:
        f.write(cctx.compress(payload))


def _write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_routed_jsonl(path: Path) -> None:
    """Multi-source records exercising routed text extraction.

    Sources differ in which field carries the primary content:
      - github: attributes.body
      - reddit: attributes.body
      - hackernews: attributes.text  (NOT body)
      - producthunt: attributes.title (NOT body)
      - unknown_source: only top-level `body` (exercises _default fallback)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(5):
        records.append({
            "id": f"gh-{i}",
            "attributes": {"source": "github", "body": f"gh body {i}", "title": f"gh title {i}"},
        })
    for i in range(5):
        records.append({
            "id": f"rd-{i}",
            "attributes": {"source": "reddit", "body": f"rd body {i}", "title": f"rd title {i}"},
        })
    for i in range(5):
        records.append({
            "id": f"hn-{i}",
            "attributes": {"source": "hackernews", "text": f"hn text {i}", "title": f"hn title {i}"},
        })
    for i in range(5):
        records.append({
            "id": f"ph-{i}",
            "attributes": {"source": "producthunt", "title": f"ph title {i}"},
        })
    # Fallback case: source unknown to the routing table.
    for i in range(3):
        records.append({
            "id": f"un-{i}",
            "attributes": {"source": "unknown_source"},
            "body": f"un body {i}",
        })
    # Edge case: no source field at all (router_field missing entirely).
    records.append({"id": "no-src-1", "body": "fallback body"})

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Tabular generators (CSV / TSV)
# ---------------------------------------------------------------------------


def _simple_tabular_rows(count: int) -> list[list[str]]:
    rows = [["id", "description", "name", "category"]]
    for i in range(count):
        rows.append([
            f"prod-{i:03d}",
            f"product description number {i}",
            f"name-{i}",
            f"cat-{i % 3}",
        ])
    return rows


def _format_csv(rows: list[list[str]], delimiter: str = ",") -> bytes:
    out = io.StringIO()
    writer = csv.writer(out, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
    for row in rows:
        writer.writerow(row)
    return out.getvalue().encode("utf-8")


def _write_simple_csv(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_format_csv(_simple_tabular_rows(count)))


def _write_simple_tsv(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_format_csv(_simple_tabular_rows(count), delimiter="\t"))


def _write_simple_csv_gz(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _format_csv(_simple_tabular_rows(count))
    with gzip.open(path, "wb") as f:
        f.write(payload)


def _write_simple_csv_zst(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _format_csv(_simple_tabular_rows(count))
    cctx = zstd.ZstdCompressor()
    path.write_bytes(cctx.compress(payload))


def _write_routed_csv(path: Path) -> None:
    """CSV with per-source field divergence, mirroring the routed JSONL fixture.

    Columns: id, source, body, title, summary
      - github (5):  body, title set
      - reddit (5):  body, title set
      - hackernews (5): no body, summary set
      - unknown (3): only summary set (exercises _default fallback to summary)
    """
    rows = [["id", "source", "body", "title", "summary"]]
    for i in range(5):
        rows.append([f"gh-{i}", "github", f"gh body {i}", f"gh title {i}", ""])
    for i in range(5):
        rows.append([f"rd-{i}", "reddit", f"rd body {i}", f"rd title {i}", ""])
    for i in range(5):
        rows.append([f"hn-{i}", "hackernews", "", f"hn title {i}", f"hn summary {i}"])
    for i in range(3):
        rows.append([f"un-{i}", "unknown_source", "", "", f"un summary {i}"])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_format_csv(rows))


def _write_quirky_csv(path: Path) -> None:
    """CSV exercising quoted multi-line cells and embedded delimiters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        'id,description,category\n'
        '1,"a description, with a comma",cat-a\n'
        '2,"multi\nline\ndescription",cat-b\n'
        '3,plain text,cat-a\n'
    )
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Pipeline fixture (normalize + filter + sample)
# ---------------------------------------------------------------------------


def _write_timeline_jsonl(path: Path) -> None:
    """Timestamped multi-label fixture for pipeline-integration tests.

    24 records spanning three months and three labels (`a` / `b` / `c`).
    Some records intentionally have URLs / SHAs / mixed case / extra
    whitespace so normalization tests have something to remove.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    base = datetime(2026, 1, 1, 0, 0, 0)
    labels = ["a", "b", "c"]
    quirks = [
        "Visit  https://example.com  for details",
        "Commit deadbeefcafebabe1234567890abcdef12345678 added",
        "Plain message about TOPIC-{i}",
        "  whitespace surrounded message  ",
        "MIXED Case Letters Are Fine",
    ]

    records = []
    for i in range(24):
        # 3 records per month for 8 months would be 24 — instead spread over
        # 3 months with 8 records each so time filter tests have headroom.
        month = 1 + (i // 8)
        day = 1 + (i % 8)
        ts = base.replace(month=month, day=day, hour=12)
        records.append({
            "id": f"tl-{i:03d}",
            "ts": ts.isoformat() + "Z",
            "label": labels[i % len(labels)],
            "text": quirks[i % len(quirks)].format(i=i),
        })

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
