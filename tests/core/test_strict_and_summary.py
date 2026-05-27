"""Tests for strict mode and ReadSummary."""

from __future__ import annotations

from pathlib import Path

import pytest

from ignite_tools.core.config import FormatConfig
from ignite_tools.core.format import (
    CorpusReadError,
    ReadSummary,
    read_corpus,
)


# ---------------------------------------------------------------------------
# CorpusReadError shape
# ---------------------------------------------------------------------------


def test_corpus_read_error_renders_path_and_line():
    err = CorpusReadError("malformed JSON", source_file="x.jsonl", line_no=12)
    assert "[x.jsonl:12]" in str(err)
    assert "malformed JSON" in str(err)


def test_corpus_read_error_without_line():
    err = CorpusReadError("file is empty", source_file="x.txt")
    assert "[x.txt]" in str(err)
    assert ":" not in str(err).split("]")[0]  # no line_no in prefix


def test_corpus_read_error_without_source():
    err = CorpusReadError("generic")
    assert str(err) == "generic"


# ---------------------------------------------------------------------------
# Strict: JSONL
# ---------------------------------------------------------------------------


def _jsonl(tmp_path: Path, body: str, name: str = "in.jsonl") -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def _jsonl_cfg(path: Path) -> FormatConfig:
    return FormatConfig.from_dict({
        "storage": {"type": "local", "path": str(path)},
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
    })


def test_strict_raises_on_malformed_jsonl(tmp_path: Path):
    p = _jsonl(tmp_path, '{"id":"a","text":"ok"}\nnot-json\n')
    cfg = _jsonl_cfg(p)
    with pytest.raises(CorpusReadError) as exc:
        list(read_corpus(cfg, strict=True))
    assert exc.value.line_no == 2
    assert exc.value.source_file == "in.jsonl"


def test_strict_raises_when_record_is_not_object(tmp_path: Path):
    p = _jsonl(tmp_path, '"just a string"\n')
    cfg = _jsonl_cfg(p)
    with pytest.raises(CorpusReadError, match="not an object"):
        list(read_corpus(cfg, strict=True))


def test_strict_raises_when_text_missing(tmp_path: Path):
    p = _jsonl(tmp_path, '{"id":"a","other":"x"}\n')
    cfg = _jsonl_cfg(p)
    with pytest.raises(CorpusReadError, match="no text field"):
        list(read_corpus(cfg, strict=True))


def test_strict_raises_for_unrouted_routed_extraction(tmp_path: Path):
    p = _jsonl(tmp_path, '{"id":"a","src":"unknown","body":"x"}\n')
    cfg = FormatConfig.from_dict({
        "storage": {"type": "local", "path": str(p)},
        "format": {"type": "jsonl"},
        "text": {
            "router_field": "src",
            "routes": {"github": {"fields": ["body"]}},  # no _default
        },
        "id": {"field": "id"},
    })
    with pytest.raises(CorpusReadError, match="did not match any route"):
        list(read_corpus(cfg, strict=True))


def test_lenient_continues_on_malformed_jsonl(tmp_path: Path):
    p = _jsonl(
        tmp_path,
        '{"id":"a","text":"ok"}\n'
        'not-json\n'
        '{"id":"b","text":"also ok"}\n'
    )
    cfg = _jsonl_cfg(p)
    items = list(read_corpus(cfg, strict=False))
    assert [it.id for it in items] == ["a", "b"]


# ---------------------------------------------------------------------------
# Strict: text format
# ---------------------------------------------------------------------------


def test_strict_raises_on_empty_text_file(tmp_path: Path):
    (tmp_path / "empty.txt").write_text("   \n  \n")
    cfg = FormatConfig.from_dict({
        "storage": {"type": "local", "path": str(tmp_path)},
        "format": {"type": "text", "unit": "file"},
    })
    with pytest.raises(CorpusReadError, match="empty"):
        list(read_corpus(cfg, strict=True))


# ---------------------------------------------------------------------------
# Strict: normalization (trim.min_chars)
# ---------------------------------------------------------------------------


def test_strict_raises_when_normalization_drops_record(tmp_path: Path):
    p = _jsonl(tmp_path, '{"id":"a","text":"hi"}\n')  # 2 chars, < 5
    cfg = FormatConfig.from_dict({
        "storage": {"type": "local", "path": str(p)},
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
        "normalize": {"trim": {"min_chars": 5}},
    })
    with pytest.raises(CorpusReadError, match="below normalize.trim.min_chars"):
        list(read_corpus(cfg, strict=True))


def test_strict_raises_on_invalid_configured_timestamp(tmp_path: Path):
    p = tmp_path / "bad-time.jsonl"
    p.write_text('{"id":"1","text":"hello","ts":"not-a-date"}\n', encoding="utf-8")
    cfg = FormatConfig.from_dict({
        "storage": {"type": "local", "path": str(p)},
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "filters": {"time_field": "ts", "time_from": "2026-01-01T00:00:00"},
    })

    with pytest.raises(CorpusReadError, match="invalid timestamp"):
        list(read_corpus(cfg, strict=True))


# ---------------------------------------------------------------------------
# ReadSummary counters
# ---------------------------------------------------------------------------


def test_summary_records_emitted(fixtures_root: Path):
    cfg = FormatConfig.from_dict({
        "storage": {"type": "local", "path": str(fixtures_root / "simple.jsonl")},
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
    })
    summary = ReadSummary()
    items = list(read_corpus(cfg, summary=summary))
    assert summary.records_emitted == len(items) == 50
    assert summary.files_scanned == 1
    assert summary.total_skipped == 0


def test_summary_counts_skips(fixtures_root: Path):
    """``noisy.jsonl`` has 1 garbage line + 1 missing-text + 1 whitespace-text."""
    cfg = FormatConfig.from_dict({
        "storage": {"type": "local", "path": str(fixtures_root / "noisy.jsonl")},
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
    })
    summary = ReadSummary()
    list(read_corpus(cfg, summary=summary))

    # 20 valid emitted; the rest accounted as skips.
    assert summary.records_emitted == 20
    assert summary.skipped_malformed == 1     # garbage line
    # Missing-text + whitespace-text both count as skipped_extraction.
    assert summary.skipped_extraction == 2


def test_summary_counts_filter_skips(fixtures_root: Path):
    cfg = FormatConfig.from_dict({
        "storage": {"type": "local", "path": str(fixtures_root / "pipeline" / "timeline.jsonl")},
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
        "labels": {"field": "label"},
        "filters": {"labels_include": ["a"]},  # 8 of 24 records
    })
    summary = ReadSummary()
    list(read_corpus(cfg, summary=summary))
    assert summary.records_emitted == 8
    assert summary.skipped_filter == 16


def test_summary_counts_per_file_cap_skips(fixtures_root: Path):
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
    summary = ReadSummary()
    list(read_corpus(cfg, summary=summary))
    # Two files of 30 records each; cap to 5 -> 25 dropped per file.
    assert summary.records_emitted == 10
    assert summary.skipped_per_file_cap == 50
    assert summary.files_scanned == 2


def test_summary_counts_normalization_skips(tmp_path: Path):
    p = _jsonl(
        tmp_path,
        '{"id":"a","text":"hi"}\n'             # 2 chars - dropped
        '{"id":"b","text":"hello world"}\n'    # 11 chars - kept
        '{"id":"c","text":"abc"}\n'            # 3 chars - dropped
    )
    cfg = FormatConfig.from_dict({
        "storage": {"type": "local", "path": str(p)},
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
        "normalize": {"trim": {"min_chars": 5}},
    })
    summary = ReadSummary()
    list(read_corpus(cfg, summary=summary))
    assert summary.records_emitted == 1
    assert summary.skipped_normalization == 2


def test_summary_counts_unrouted_separately(tmp_path: Path):
    p = _jsonl(
        tmp_path,
        '{"id":"a","src":"github","body":"x"}\n'
        '{"id":"b","src":"unknown","body":"y"}\n'
    )
    cfg = FormatConfig.from_dict({
        "storage": {"type": "local", "path": str(p)},
        "format": {"type": "jsonl"},
        "text": {
            "router_field": "src",
            "routes": {"github": {"fields": ["body"]}},  # no _default
        },
        "id": {"field": "id"},
    })
    summary = ReadSummary()
    list(read_corpus(cfg, summary=summary))
    assert summary.records_emitted == 1
    assert summary.skipped_unrouted == 1
    assert summary.skipped_extraction == 0


def test_summary_format_text_only_lists_nonzero(fixtures_root: Path):
    summary = ReadSummary(files_scanned=3, records_emitted=10)
    text = summary.format_text()
    assert "Files scanned:          3" in text
    assert "Records emitted:        10" in text
    assert "Skipped" not in text  # all skip categories are zero

    summary.skipped_malformed = 4
    text = summary.format_text()
    assert "Skipped (malformed):    4" in text


def test_summary_total_skipped_property():
    summary = ReadSummary(
        skipped_malformed=1,
        skipped_extraction=2,
        skipped_unrouted=3,
        skipped_normalization=4,
        skipped_filter=5,
        skipped_per_file_cap=6,
    )
    assert summary.total_skipped == 21
