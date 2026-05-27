"""Tests for ``ignite_tools.core.format`` (read_corpus)."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from ignite_tools.core.config import FormatConfig
from ignite_tools.core.format import Item, read_corpus


def _cfg(path: Path, **overrides) -> FormatConfig:
    data = {
        "storage": {"type": "local", "path": str(path), "recursive": True},
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
    }
    for key, value in overrides.items():
        data[key] = value
    return FormatConfig.from_dict(data)


# ---------------------------------------------------------------------------
# JSONL: simple shape
# ---------------------------------------------------------------------------


def test_read_corpus_returns_generator(fixtures_root: Path):
    cfg = _cfg(fixtures_root / "simple.jsonl")
    result = read_corpus(cfg)
    assert inspect.isgenerator(result)


def test_read_corpus_simple_file(fixtures_root: Path):
    cfg = _cfg(fixtures_root / "simple.jsonl")
    items = list(read_corpus(cfg))
    assert len(items) == 50
    assert all(isinstance(i, Item) for i in items)
    assert items[0].text.startswith("sample message number 0")
    assert items[0].source_file == "simple.jsonl"
    assert items[0].id == "simple.jsonl:1"


def test_read_corpus_uses_id_field_when_configured(fixtures_root: Path):
    cfg = _cfg(fixtures_root / "simple.jsonl", id={"field": "id"})
    items = list(read_corpus(cfg))
    assert items[0].id == "rec-0000"
    assert items[-1].id == "rec-0049"


def test_read_corpus_dotted_path_extraction(fixtures_root: Path):
    data = {
        "storage": {
            "type": "local",
            "path": str(fixtures_root / "nested"),
            "recursive": True,
        },
        "format": {"type": "jsonl"},
        "text": {"fields": ["attributes.body", "attributes.title"]},
        "id": {"field": "id"},
    }
    cfg = FormatConfig.from_dict(data)
    items = list(read_corpus(cfg))

    assert len(items) == 60
    assert items[0].text.startswith("github body content")
    assert items[0].id.startswith("github-")
    assert items[0].source_file == "file_a.jsonl"
    assert items[-1].text.startswith("reddit body content")
    assert items[-1].source_file == "file_b.jsonl"


def test_read_corpus_first_field_with_value_wins(tmp_path: Path):
    p = tmp_path / "fallback.jsonl"
    p.write_text(
        '{"a": "", "b": "from-b"}\n'
        '{"a": "from-a", "b": "from-b"}\n'
        '{"a": null, "b": "from-b"}\n'
    )
    cfg = FormatConfig.from_dict({
        "storage": {"type": "local", "path": str(p)},
        "format": {"type": "jsonl"},
        "text": {"fields": ["a", "b"]},
    })
    items = list(read_corpus(cfg))
    assert [i.text for i in items] == ["from-b", "from-a", "from-b"]


def test_read_corpus_skips_malformed_and_empty_records(fixtures_root: Path):
    cfg = _cfg(fixtures_root / "noisy.jsonl")
    items = list(read_corpus(cfg))
    # 20 valid + 1 garbage + 1 blank + 1 missing-text + 1 whitespace-text
    # → 20 valid items emitted; the rest silently skipped (lenient default).
    assert len(items) == 20


def test_read_corpus_empty_directory(tmp_path: Path):
    cfg = _cfg(tmp_path)
    items = list(read_corpus(cfg))
    assert items == []


# ---------------------------------------------------------------------------
# JSONL: compressed (.gz, .zst)
# ---------------------------------------------------------------------------


def test_read_corpus_jsonl_gz(fixtures_root: Path):
    cfg = _cfg(fixtures_root / "compressed" / "simple.jsonl.gz", id={"field": "id"})
    items = list(read_corpus(cfg))
    assert len(items) == 15
    assert items[0].id == "rec-0000"
    assert items[0].text.startswith("sample message number 0")
    assert items[0].source_file == "simple.jsonl.gz"


def test_read_corpus_jsonl_zst(fixtures_root: Path):
    cfg = _cfg(fixtures_root / "compressed" / "simple.jsonl.zst", id={"field": "id"})
    items = list(read_corpus(cfg))
    assert len(items) == 15
    assert items[0].id == "rec-0000"
    assert items[0].text.startswith("sample message number 0")


# ---------------------------------------------------------------------------
# Recursive aggregation across the full fixture set
# ---------------------------------------------------------------------------


def test_read_corpus_recursive_aggregates_files(fixtures_root: Path):
    """Walks the entire fixture root, picks up every JSONL variant.

    Field order: try ``text`` (simple/noisy/compressed/timeline), then
    ``attributes.body`` (nested/file_a, nested/file_b, routed gh/rd).

    Routed/events.jsonl distribution under this fields list:
      - 5 github (attributes.body)            → matched
      - 5 reddit (attributes.body)            → matched
      - 5 hackernews (attributes.text only)   → not matched
      - 5 producthunt (attributes.title only) → not matched
      - 3 unknown_source (top-level body)     → not matched
      - 1 no-source-field (top-level body)    → not matched
      = 10 matches from the routed file under the simple-fields config.

    Plain text fixtures (doc1.txt, doc2.md) are filtered out by the JSONL
    default include globs.
    """
    cfg = FormatConfig.from_dict({
        "storage": {"type": "local", "path": str(fixtures_root), "recursive": True},
        "format": {"type": "jsonl"},
        "text": {"fields": ["text", "attributes.body"]},
    })
    items = list(read_corpus(cfg))

    by_file: dict[str, int] = {}
    for it in items:
        by_file[it.source_file] = by_file.get(it.source_file, 0) + 1

    assert by_file == {
        "simple.jsonl": 50,
        "noisy.jsonl": 20,
        "nested/file_a.jsonl": 30,
        "nested/file_b.jsonl": 30,
        "compressed/simple.jsonl.gz": 15,
        "compressed/simple.jsonl.zst": 15,
        "routed/events.jsonl": 10,
        "pipeline/timeline.jsonl": 24,
    }
    assert len(items) == sum(by_file.values())


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------


def test_read_corpus_text_unit_line(fixtures_root: Path):
    cfg = FormatConfig.from_dict({
        "storage": {"type": "local", "path": str(fixtures_root / "text")},
        "format": {"type": "text", "unit": "line"},
    })
    items = list(read_corpus(cfg))

    # doc1.txt: 3 non-empty lines; doc2.md: 2 non-empty lines (the blank
    # trailing line is skipped). Sorted file order: doc1.txt before doc2.md.
    assert [i.source_file for i in items] == [
        "doc1.txt", "doc1.txt", "doc1.txt",
        "doc2.md", "doc2.md",
    ]
    assert items[0].text == "first paragraph of the document"
    assert items[0].id == "doc1.txt:1"
    assert items[3].text == "# A markdown doc"
    assert items[3].id == "doc2.md:1"


def test_read_corpus_text_unit_file(fixtures_root: Path):
    cfg = FormatConfig.from_dict({
        "storage": {"type": "local", "path": str(fixtures_root / "text")},
        "format": {"type": "text", "unit": "file"},
    })
    items = list(read_corpus(cfg))

    assert len(items) == 2
    assert items[0].id == "doc1.txt"
    assert items[0].source_file == "doc1.txt"
    assert "first paragraph" in items[0].text
    assert "third line" in items[0].text
    assert items[1].id == "doc2.md"
    assert "markdown doc" in items[1].text


def test_read_corpus_text_unit_file_skips_empty_files(tmp_path: Path):
    (tmp_path / "empty.txt").write_text("")
    (tmp_path / "spaces.txt").write_text("   \n\n  \n")
    (tmp_path / "real.txt").write_text("real content here\n")

    cfg = FormatConfig.from_dict({
        "storage": {"type": "local", "path": str(tmp_path)},
        "format": {"type": "text", "unit": "file"},
    })
    items = list(read_corpus(cfg))
    assert len(items) == 1
    assert items[0].source_file == "real.txt"


# ---------------------------------------------------------------------------
# Routed text extraction
# ---------------------------------------------------------------------------


def _routed_storage(path: Path) -> dict:
    return {"type": "local", "path": str(path)}


def test_read_corpus_routed_extraction(fixtures_root: Path):
    """Each source picks its declared fields. _default not configured here,
    so unmatched router values drop silently."""
    cfg = FormatConfig.from_dict({
        "storage": _routed_storage(fixtures_root / "routed" / "events.jsonl"),
        "format": {"type": "jsonl"},
        "text": {
            "router_field": "attributes.source",
            "routes": {
                "github":     {"fields": ["attributes.body", "attributes.title"]},
                "reddit":     {"fields": ["attributes.body", "attributes.title"]},
                "hackernews": {"fields": ["attributes.text", "attributes.title"]},
                "producthunt": {"fields": ["attributes.title"]},
            },
        },
        "id": {"field": "id"},
    })
    items = list(read_corpus(cfg))
    by_prefix: dict[str, int] = {}
    for it in items:
        prefix = it.id.split("-")[0]
        by_prefix[prefix] = by_prefix.get(prefix, 0) + 1

    # 5 each of gh/rd/hn/ph routed; un-* and no-src dropped (no _default).
    assert by_prefix == {"gh": 5, "rd": 5, "hn": 5, "ph": 5}
    assert len(items) == 20

    # Spot-check that the right field was used per source.
    text_by_id = {it.id: it.text for it in items}
    assert text_by_id["hn-0"] == "hn text 0"
    assert text_by_id["ph-0"] == "ph title 0"


def test_read_corpus_routed_default_fallback(fixtures_root: Path):
    """``_default`` catches records with unknown router values."""
    cfg = FormatConfig.from_dict({
        "storage": _routed_storage(fixtures_root / "routed" / "events.jsonl"),
        "format": {"type": "jsonl"},
        "text": {
            "router_field": "attributes.source",
            "routes": {
                "github":   {"fields": ["attributes.body"]},
                "_default": {"fields": ["body"]},
            },
        },
        "id": {"field": "id"},
    })
    items = list(read_corpus(cfg))
    by_prefix: dict[str, int] = {}
    for it in items:
        prefix = it.id.split("-")[0]
        by_prefix[prefix] = by_prefix.get(prefix, 0) + 1

    # github → 5 (matched), reddit → 5 (no body in route → wait, reddit has
    # attributes.body but the route is github only; reddit goes to _default
    # which uses top-level `body` which reddit DOESN'T have → 0 reddit).
    # hn (no body anywhere) → 0; ph (no body) → 0; un (top-level body) → 3.
    # Plus the no-source-field record routed via _default → 1 (top-level body)
    assert by_prefix["gh"] == 5
    assert by_prefix.get("un", 0) == 3
    assert by_prefix.get("no", 0) == 1
    # reddit / hn / ph have nothing matching the _default field set
    assert by_prefix.get("rd", 0) == 0
    assert by_prefix.get("hn", 0) == 0
    assert by_prefix.get("ph", 0) == 0


def test_read_corpus_routed_no_default_drops_unknown(fixtures_root: Path):
    """Without ``_default``, records whose router value isn't in routes are dropped."""
    cfg = FormatConfig.from_dict({
        "storage": _routed_storage(fixtures_root / "routed" / "events.jsonl"),
        "format": {"type": "jsonl"},
        "text": {
            "router_field": "attributes.source",
            "routes": {"github": {"fields": ["attributes.body"]}},
        },
        "id": {"field": "id"},
    })
    items = list(read_corpus(cfg))
    assert len(items) == 5
    assert all(it.id.startswith("gh-") for it in items)


# ---------------------------------------------------------------------------
# CSV / TSV
# ---------------------------------------------------------------------------


def _csv_storage(path: Path) -> dict:
    return {"type": "local", "path": str(path)}


def test_read_corpus_csv_simple(fixtures_root: Path):
    cfg = FormatConfig.from_dict({
        "storage": _csv_storage(fixtures_root / "tabular" / "products.csv"),
        "format": {"type": "csv"},
        "text": {"fields": ["description"]},
        "id": {"field": "id"},
    })
    items = list(read_corpus(cfg))

    assert len(items) == 12
    assert items[0].id == "prod-000"
    assert items[0].text == "product description number 0"
    assert items[0].source_file == "products.csv"
    assert items[-1].id == "prod-011"


def test_read_corpus_csv_field_fallback(fixtures_root: Path):
    """First-non-empty wins, like the JSONL extractor."""
    cfg = FormatConfig.from_dict({
        "storage": _csv_storage(fixtures_root / "tabular" / "events_routed.csv"),
        "format": {"type": "csv"},
        "text": {"fields": ["body", "summary"]},
        "id": {"field": "id"},
    })
    items = list(read_corpus(cfg))
    by_id = {it.id: it.text for it in items}

    # github + reddit have body; hackernews + unknown only have summary.
    assert by_id["gh-0"] == "gh body 0"
    assert by_id["rd-0"] == "rd body 0"
    assert by_id["hn-0"] == "hn summary 0"
    assert by_id["un-0"] == "un summary 0"
    assert len(items) == 18  # 5 gh + 5 rd + 5 hn + 3 un


def test_read_corpus_csv_auto_id_uses_row_number(fixtures_root: Path):
    cfg = FormatConfig.from_dict({
        "storage": _csv_storage(fixtures_root / "tabular" / "products.csv"),
        "format": {"type": "csv"},
        "text": {"fields": ["description"]},
        # no id block
    })
    items = list(read_corpus(cfg))
    # Auto IDs: <rel>:<row_no>, 1-based, header excluded.
    assert items[0].id == "products.csv:1"
    assert items[-1].id == "products.csv:12"


def test_read_corpus_tsv(fixtures_root: Path):
    cfg = FormatConfig.from_dict({
        "storage": _csv_storage(fixtures_root / "tabular" / "products.tsv"),
        "format": {"type": "tsv"},
        "text": {"fields": ["description"]},
        "id": {"field": "id"},
    })
    items = list(read_corpus(cfg))
    assert len(items) == 12
    assert items[0].text == "product description number 0"
    assert items[0].source_file == "products.tsv"


def test_read_corpus_csv_gz(fixtures_root: Path):
    """polars handles .gz transparently."""
    cfg = FormatConfig.from_dict({
        "storage": _csv_storage(fixtures_root / "tabular" / "products.csv.gz"),
        "format": {"type": "csv"},
        "text": {"fields": ["description"]},
        "id": {"field": "id"},
    })
    items = list(read_corpus(cfg))
    assert len(items) == 12
    assert items[0].text == "product description number 0"
    assert items[0].source_file == "products.csv.gz"


def test_read_corpus_csv_zst(fixtures_root: Path):
    """polars handles .zst transparently."""
    cfg = FormatConfig.from_dict({
        "storage": _csv_storage(fixtures_root / "tabular" / "products.csv.zst"),
        "format": {"type": "csv"},
        "text": {"fields": ["description"]},
        "id": {"field": "id"},
    })
    items = list(read_corpus(cfg))
    assert len(items) == 12
    assert items[0].text == "product description number 0"


def test_read_corpus_csv_routed_extraction(fixtures_root: Path):
    """Routed extraction works on CSV the same way as JSONL."""
    cfg = FormatConfig.from_dict({
        "storage": _csv_storage(fixtures_root / "tabular" / "events_routed.csv"),
        "format": {"type": "csv"},
        "text": {
            "router_field": "source",
            "routes": {
                "github":     {"fields": ["body", "title"]},
                "reddit":     {"fields": ["body", "title"]},
                "hackernews": {"fields": ["summary", "title"]},
                "_default":   {"fields": ["summary"]},
            },
        },
        "id": {"field": "id"},
    })
    items = list(read_corpus(cfg))
    by_id = {it.id: it.text for it in items}
    by_prefix: dict[str, int] = {}
    for it in items:
        by_prefix[it.id.split("-")[0]] = by_prefix.get(it.id.split("-")[0], 0) + 1

    assert by_prefix == {"gh": 5, "rd": 5, "hn": 5, "un": 3}
    assert by_id["gh-0"] == "gh body 0"
    assert by_id["hn-0"] == "hn summary 0"
    assert by_id["un-0"] == "un summary 0"  # via _default


def test_read_corpus_csv_handles_quoted_multiline_cells(fixtures_root: Path):
    """polars resolves quoted multi-line cells and embedded delimiters."""
    cfg = FormatConfig.from_dict({
        "storage": _csv_storage(fixtures_root / "tabular" / "quirky.csv"),
        "format": {"type": "csv"},
        "text": {"fields": ["description"]},
        "id": {"field": "id"},
    })
    items = list(read_corpus(cfg))
    assert len(items) == 3
    assert items[0].text == "a description, with a comma"
    assert items[1].text == "multi\nline\ndescription"
    assert items[2].text == "plain text"


def test_read_corpus_csv_recursive_directory(fixtures_root: Path):
    """A directory of mixed .csv / .csv.gz / .csv.zst all read together."""
    cfg = FormatConfig.from_dict({
        "storage": {
            "type": "local",
            "path": str(fixtures_root / "tabular"),
            "recursive": True,
            "include": ["products.csv*"],  # only the products variants
        },
        "format": {"type": "csv"},
        "text": {"fields": ["description"]},
        "id": {"field": "id"},
    })
    items = list(read_corpus(cfg))
    by_file: dict[str, int] = {}
    for it in items:
        by_file[it.source_file] = by_file.get(it.source_file, 0) + 1

    assert by_file == {
        "products.csv": 12,
        "products.csv.gz": 12,
        "products.csv.zst": 12,
    }
    assert len(items) == 36


def test_read_corpus_tsv_with_explicit_delimiter_override(tmp_path: Path):
    """User can declare type:csv but pass a tab delimiter for a .tsv file."""
    p = tmp_path / "tabbed.csv"  # extension intentionally wrong
    p.write_text("id\ttext\n1\thello\n2\tworld\n", encoding="utf-8")
    cfg = FormatConfig.from_dict({
        "storage": _csv_storage(p),
        "format": {"type": "csv", "delimiter": "\t"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
    })
    items = list(read_corpus(cfg))
    assert [it.text for it in items] == ["hello", "world"]
    assert items[0].id == "1"


# ---------------------------------------------------------------------------
# Pipeline integration: labels, normalization, filtering, sampling
# ---------------------------------------------------------------------------


def _timeline_storage(fixtures_root: Path) -> dict:
    return {"type": "local", "path": str(fixtures_root / "pipeline" / "timeline.jsonl")}


def test_pipeline_labels_populated(fixtures_root: Path):
    cfg = FormatConfig.from_dict({
        "storage": _timeline_storage(fixtures_root),
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
        "labels": {"field": "label"},
    })
    items = list(read_corpus(cfg))
    assert len(items) == 24
    assert all(it.label in {"a", "b", "c"} for it in items)
    # Distribution is round-robin: 8 per label.
    by_label: dict[str, int] = {}
    for it in items:
        by_label[it.label] = by_label.get(it.label, 0) + 1
    assert by_label == {"a": 8, "b": 8, "c": 8}


def test_pipeline_normalize_lowercase_and_masks(fixtures_root: Path):
    cfg = FormatConfig.from_dict({
        "storage": _timeline_storage(fixtures_root),
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
        "normalize": {
            "lowercase": True,
            "strip": True,
            "masks": [
                {"pattern": r"https?://\S+", "replacement": "<url>"},
                {"pattern": r"\b[0-9a-f]{40}\b", "replacement": "<sha>"},
            ],
        },
    })
    items = list(read_corpus(cfg))
    by_id = {it.id: it.text for it in items}

    # tl-000 had "Visit  https://example.com  for details"
    assert "<url>" in by_id["tl-000"]
    assert "https://" not in by_id["tl-000"]
    assert by_id["tl-000"] == by_id["tl-000"].lower()

    # tl-001 had a 40-hex-char SHA.
    assert "<sha>" in by_id["tl-001"]
    assert "deadbeef" not in by_id["tl-001"]

    # tl-003 had leading+trailing whitespace.
    assert not by_id["tl-003"].startswith(" ")
    assert not by_id["tl-003"].endswith(" ")


def test_pipeline_normalize_collapse_whitespace(fixtures_root: Path):
    cfg = FormatConfig.from_dict({
        "storage": _timeline_storage(fixtures_root),
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
        "normalize": {"collapse_whitespace": True, "strip": True},
    })
    items = list(read_corpus(cfg))
    # tl-000 had two spaces between "Visit" and the URL — should become one.
    by_id = {it.id: it.text for it in items}
    assert "  " not in by_id["tl-000"]


def test_pipeline_normalize_trim_drops_short_records(tmp_path: Path):
    p = tmp_path / "tiny.jsonl"
    p.write_text(
        '{"id":"a","text":"abc"}\n'        # 3 chars, drops
        '{"id":"b","text":"hello world"}\n'  # 11 chars, keeps
        '{"id":"c","text":""}\n'            # extraction drops (empty)
        '{"id":"d","text":"longer text content"}\n'
    )
    cfg = FormatConfig.from_dict({
        "storage": {"type": "local", "path": str(p)},
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
        "normalize": {"trim": {"min_chars": 5}},
    })
    items = list(read_corpus(cfg))
    assert {it.id for it in items} == {"b", "d"}


def test_pipeline_normalize_trim_truncates_long_records(tmp_path: Path):
    p = tmp_path / "long.jsonl"
    p.write_text(
        '{"id":"x","text":"this is a long enough message body"}\n'
    )
    cfg = FormatConfig.from_dict({
        "storage": {"type": "local", "path": str(p)},
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
        "normalize": {"trim": {"max_chars": 10}},
    })
    items = list(read_corpus(cfg))
    assert items[0].text == "this is a "
    assert len(items[0].text) == 10


def test_pipeline_filter_labels_include(fixtures_root: Path):
    cfg = FormatConfig.from_dict({
        "storage": _timeline_storage(fixtures_root),
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
        "labels": {"field": "label"},
        "filters": {"labels_include": ["a", "b"]},
    })
    items = list(read_corpus(cfg))
    assert all(it.label in {"a", "b"} for it in items)
    assert len(items) == 16  # 8 + 8


def test_pipeline_filter_labels_exclude(fixtures_root: Path):
    cfg = FormatConfig.from_dict({
        "storage": _timeline_storage(fixtures_root),
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
        "labels": {"field": "label"},
        "filters": {"labels_exclude": ["c"]},
    })
    items = list(read_corpus(cfg))
    assert all(it.label != "c" for it in items)
    assert len(items) == 16


def test_pipeline_filter_time_window(fixtures_root: Path):
    """Records span 2026-01-01 through 2026-03-08; filter to Feb only."""
    cfg = FormatConfig.from_dict({
        "storage": _timeline_storage(fixtures_root),
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
        "filters": {
            "time_field": "ts",
            "time_from": "2026-02-01T00:00:00",
            "time_to": "2026-02-28T23:59:59",
        },
    })
    items = list(read_corpus(cfg))
    # tl-008..tl-015 are in February (8 records).
    assert {it.id for it in items} == {f"tl-{i:03d}" for i in range(8, 16)}


def test_pipeline_filter_drops_records_without_time(tmp_path: Path):
    """Records missing the time field are dropped when a time filter is on."""
    p = tmp_path / "mixed.jsonl"
    p.write_text(
        '{"id":"with-time","ts":"2026-01-15","text":"has timestamp"}\n'
        '{"id":"no-time","text":"no timestamp here"}\n'
    )
    cfg = FormatConfig.from_dict({
        "storage": {"type": "local", "path": str(p)},
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
        "filters": {
            "time_field": "ts",
            "time_from": "2026-01-01",
            "time_to": "2026-12-31",
        },
    })
    items = list(read_corpus(cfg))
    assert [it.id for it in items] == ["with-time"]


def test_pipeline_sampling_head(fixtures_root: Path):
    cfg = FormatConfig.from_dict({
        "storage": _timeline_storage(fixtures_root),
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
        "sampling": {"mode": "head", "total": 5},
    })
    items = list(read_corpus(cfg))
    assert [it.id for it in items] == [f"tl-{i:03d}" for i in range(5)]


def test_pipeline_sampling_random_deterministic(fixtures_root: Path):
    cfg = FormatConfig.from_dict({
        "storage": _timeline_storage(fixtures_root),
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
        "sampling": {"mode": "random", "total": 10, "seed": 7},
    })
    a = list(read_corpus(cfg))
    b = list(read_corpus(cfg))
    assert [it.id for it in a] == [it.id for it in b]
    assert len(a) == 10


def test_pipeline_sampling_stratified(fixtures_root: Path):
    cfg = FormatConfig.from_dict({
        "storage": _timeline_storage(fixtures_root),
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
        "labels": {"field": "label"},
        "sampling": {
            "mode": "stratified",
            "group_field": "label",
            "per_group": 3,
            "seed": 42,
        },
    })
    items = list(read_corpus(cfg))
    by_label: dict[str, int] = {}
    for it in items:
        by_label[it.label] = by_label.get(it.label, 0) + 1
    assert by_label == {"a": 3, "b": 3, "c": 3}


def test_pipeline_sampling_weighted(fixtures_root: Path):
    cfg = FormatConfig.from_dict({
        "storage": _timeline_storage(fixtures_root),
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
        "labels": {"field": "label"},
        "sampling": {
            "mode": "weighted",
            "group_field": "label",
            "total": 12,
            "weights": {"a": 0.5, "b": 0.25, "c": 0.25},
            "seed": 42,
        },
    })
    items = list(read_corpus(cfg))
    by_label: dict[str, int] = {}
    for it in items:
        by_label[it.label] = by_label.get(it.label, 0) + 1
    assert by_label == {"a": 6, "b": 3, "c": 3}


def test_pipeline_sampling_per_file_cap(fixtures_root: Path):
    """Cap applies per source_file across multiple files in the corpus."""
    cfg = FormatConfig.from_dict({
        "storage": {
            "type": "local",
            "path": str(fixtures_root / "nested"),
            "recursive": True,
        },
        "format": {"type": "jsonl"},
        "text": {"fields": ["attributes.body"]},
        "id": {"field": "id"},
        "sampling": {"per_file_cap": 5},
    })
    items = list(read_corpus(cfg))
    by_file: dict[str, int] = {}
    for it in items:
        by_file[it.source_file] = by_file.get(it.source_file, 0) + 1
    assert by_file == {"file_a.jsonl": 5, "file_b.jsonl": 5}


def test_pipeline_filter_then_sample_ordering(fixtures_root: Path):
    """Filtering happens before sampling — sample target counts only kept items."""
    cfg = FormatConfig.from_dict({
        "storage": _timeline_storage(fixtures_root),
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
        "labels": {"field": "label"},
        "filters": {"labels_include": ["a"]},  # restricts to 8 records
        "sampling": {"mode": "head", "total": 5},
    })
    items = list(read_corpus(cfg))
    assert all(it.label == "a" for it in items)
    assert len(items) == 5


def test_pipeline_normalize_then_filter_then_sample(fixtures_root: Path):
    """All three layers compose: normalize -> filter -> sample."""
    cfg = FormatConfig.from_dict({
        "storage": _timeline_storage(fixtures_root),
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
        "labels": {"field": "label"},
        "normalize": {
            "lowercase": True,
            "strip": True,
            "masks": [{"pattern": r"https?://\S+", "replacement": "<url>"}],
        },
        "filters": {"labels_include": ["a", "b"]},
        "sampling": {
            "mode": "stratified",
            "group_field": "label",
            "per_group": 2,
            "seed": 1,
        },
    })
    items = list(read_corpus(cfg))
    by_label: dict[str, int] = {}
    for it in items:
        by_label[it.label] = by_label.get(it.label, 0) + 1
    assert by_label == {"a": 2, "b": 2}
    # Normalization survived: every text is lowercase.
    assert all(it.text == it.text.lower() for it in items)


def test_pipeline_csv_with_label_filter_and_sample(fixtures_root: Path):
    """Same pipeline machinery applies to CSV inputs."""
    cfg = FormatConfig.from_dict({
        "storage": {"type": "local", "path": str(fixtures_root / "tabular" / "products.csv")},
        "format": {"type": "csv"},
        "text": {"fields": ["description"]},
        "id": {"field": "id"},
        "labels": {"field": "category"},
        "filters": {"labels_include": ["cat-0", "cat-1"]},
        "sampling": {"mode": "head", "total": 5},
    })
    items = list(read_corpus(cfg))
    assert all(it.label in {"cat-0", "cat-1"} for it in items)
    assert len(items) == 5
