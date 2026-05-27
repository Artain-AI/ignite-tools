"""Tests for ``ignite_tools.core.sniffer``."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from ignite_tools.core.config import FormatConfig
from ignite_tools.core.sniffer import (
    SniffResult,
    build_proposal,
    format_human_summary,
    proposal_to_config,
    sniff_path,
)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def test_sniff_jsonl_single_file(tmp_path: Path):
    p = tmp_path / "in.jsonl"
    p.write_text('{"id":"a","text":"hello"}\n{"id":"b","text":"world"}\n')

    result = sniff_path(p)
    assert result.format_type == "jsonl"
    assert result.file_count == 1
    assert result.compression is None
    assert result.sample_records_inspected == 2


def test_sniff_jsonl_gzip(tmp_path: Path):
    p = tmp_path / "in.jsonl.gz"
    payload = b'{"id":"a","text":"hello"}\n'
    with gzip.open(p, "wb") as f:
        f.write(payload)
    result = sniff_path(p)
    assert result.format_type == "jsonl"
    assert result.compression == "gzip"


def test_sniff_csv_single_file(tmp_path: Path):
    p = tmp_path / "data.csv"
    p.write_text("id,description,category\n1,hello world,a\n2,foo bar,b\n")
    result = sniff_path(p)
    assert result.format_type == "csv"
    assert "description" in result.csv_columns


def test_sniff_tsv(tmp_path: Path):
    p = tmp_path / "data.tsv"
    p.write_text("id\tdescription\n1\thello there\n2\tworld\n")
    result = sniff_path(p)
    assert result.format_type == "tsv"


def test_sniff_text(tmp_path: Path):
    (tmp_path / "doc1.txt").write_text("first doc\nsecond line\n")
    (tmp_path / "doc2.md").write_text("# heading\nbody\n")
    result = sniff_path(tmp_path)
    assert result.format_type == "text"
    assert result.file_count == 2


def test_sniff_directory_picks_dominant_format(tmp_path: Path):
    """Mixed extensions: JSONL wins over CSV/TSV/text by tie-break."""
    (tmp_path / "a.jsonl").write_text('{"x":1}\n')
    (tmp_path / "b.jsonl").write_text('{"x":2}\n')
    (tmp_path / "c.txt").write_text("plain")
    result = sniff_path(tmp_path)
    assert result.format_type == "jsonl"


def test_sniff_missing_path_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        sniff_path(tmp_path / "no-such-dir")


def test_sniff_empty_directory_returns_zero(tmp_path: Path):
    result = sniff_path(tmp_path)
    assert result.file_count == 0


# ---------------------------------------------------------------------------
# JSONL: key coverage and field guesses
# ---------------------------------------------------------------------------


def test_sniff_jsonl_key_coverage_dotted(tmp_path: Path):
    p = tmp_path / "nested.jsonl"
    records = [
        {"id": "a", "attributes": {"source": "github", "body": "x"}},
        {"id": "b", "attributes": {"source": "reddit", "body": "y"}},
        {"id": "c", "attributes": {"source": "github", "body": "z"}},
    ]
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    result = sniff_path(p)
    assert result.jsonl_key_coverage["id"] == 1.0
    assert result.jsonl_key_coverage["attributes.source"] == 1.0
    assert result.jsonl_key_coverage["attributes.body"] == 1.0


def test_sniff_jsonl_router_field_guess(tmp_path: Path):
    """Low-cardinality categorical field is identified as a router candidate."""
    p = tmp_path / "routed.jsonl"
    records = []
    for i in range(10):
        records.append({
            "id": f"r{i:03d}",
            "attributes": {
                "source": ["github", "reddit", "hackernews"][i % 3],
                "body": f"content about topic {i}",
            },
        })
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    result = sniff_path(p)
    assert result.router_field_guess == "attributes.source"
    assert set(result.router_values) == {"github", "reddit", "hackernews"}


def test_sniff_jsonl_text_field_well_known_name_wins(tmp_path: Path):
    """When a well-known name like 'text' is present, prefer it."""
    p = tmp_path / "x.jsonl"
    records = [
        {"id": "a", "title": "short", "text": "much longer body content here"},
        {"id": "b", "title": "tiny", "text": "another long text content body"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    result = sniff_path(p)
    assert result.text_field_guess == "text"


def test_sniff_jsonl_id_field_detected(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    records = [{"id": f"r-{i}", "text": f"body {i}"} for i in range(20)]
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    result = sniff_path(p)
    assert result.id_field_guess == "id"


def test_sniff_jsonl_avoids_categorical_as_text(tmp_path: Path):
    """A short, categorical field shouldn't be picked as the text field."""
    p = tmp_path / "x.jsonl"
    records = [
        {"id": f"r{i}", "category": ["a", "b"][i % 2], "body": f"longer content {i}"}
        for i in range(8)
    ]
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    result = sniff_path(p)
    assert result.text_field_guess == "body"  # 'category' is too short / categorical


# ---------------------------------------------------------------------------
# CSV / TSV: column guessing
# ---------------------------------------------------------------------------


def test_sniff_csv_text_column_is_longest(tmp_path: Path):
    p = tmp_path / "p.csv"
    p.write_text(
        "id,name,description,category\n"
        "1,short,this is a much longer description for embedding,a\n"
        "2,abc,another long description goes here,b\n"
        "3,def,short and sweet description,a\n"
    )
    result = sniff_path(p)
    assert result.csv_text_guess == "description"


def test_sniff_csv_label_is_low_cardinality(tmp_path: Path):
    p = tmp_path / "p.csv"
    p.write_text(
        "id,description,category\n"
        "1,longer description here,cat-a\n"
        "2,another long description,cat-b\n"
        "3,yet another description,cat-a\n"
    )
    result = sniff_path(p)
    assert result.csv_label_guess == "category"


def test_sniff_csv_router_guess_for_categorical(tmp_path: Path):
    """Low-cardinality categorical column is also flagged as router candidate."""
    p = tmp_path / "p.csv"
    p.write_text(
        "id,source,description\n"
        "1,github,something descriptive here\n"
        "2,reddit,some other description\n"
        "3,github,more description\n"
    )
    result = sniff_path(p)
    assert result.csv_router_guess == "source"
    assert sorted(result.csv_router_values) == ["github", "reddit"]


# ---------------------------------------------------------------------------
# Proposal builder
# ---------------------------------------------------------------------------


def test_build_proposal_simple_jsonl(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    records = [{"id": f"r{i}", "text": f"some text content {i}"} for i in range(8)]
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    result = sniff_path(p)
    proposal = build_proposal(result)

    assert proposal["format"]["type"] == "jsonl"
    assert proposal["text"] == {"fields": ["text"]}
    assert proposal["id"] == {"field": "id"}
    # Validates as a real config.
    cfg = proposal_to_config(proposal)
    assert cfg.parser.type == "jsonl"


def test_build_proposal_routed_jsonl(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    records = []
    for i in range(12):
        records.append({
            "id": f"r{i}",
            "attributes": {
                "source": ["github", "reddit", "hackernews"][i % 3],
                "body": f"some content {i}",
            },
        })
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    result = sniff_path(p)
    proposal = build_proposal(result)

    assert proposal["text"]["router_field"] == "attributes.source"
    assert set(proposal["text"]["routes"].keys()) == {"github", "reddit", "hackernews"}
    cfg = proposal_to_config(proposal)
    assert cfg.text.router_field == "attributes.source"


def test_build_proposal_csv(tmp_path: Path):
    p = tmp_path / "p.csv"
    p.write_text(
        "id,description,category\n"
        "1,longer description for embedding,a\n"
        "2,another long description here,b\n"
        "3,more description content here,a\n"
    )
    result = sniff_path(p)
    proposal = build_proposal(result)

    assert proposal["format"]["type"] == "csv"
    assert proposal["text"] == {"fields": ["description"]}
    assert proposal["labels"] == {"field": "category"}
    cfg = proposal_to_config(proposal)
    assert cfg.parser.type == "csv"


def test_build_proposal_text(tmp_path: Path):
    (tmp_path / "doc.txt").write_text("hello world\n")
    result = sniff_path(tmp_path)
    proposal = build_proposal(result)
    assert proposal["format"]["type"] == "text"
    assert "text" not in proposal  # plain text doesn't need a text block


# ---------------------------------------------------------------------------
# Human summary
# ---------------------------------------------------------------------------


def test_format_human_summary_includes_format_and_counts(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"id":"a","text":"hello"}\n{"id":"b","text":"world"}\n')
    result = sniff_path(p)
    summary = format_human_summary(result)
    assert "Detected:" in summary
    assert "jsonl" in summary
    assert "Field coverage" in summary


def test_format_human_summary_handles_empty_result():
    result = SniffResult(file_count=0, total_bytes=0)
    summary = format_human_summary(result)
    # Should not crash; just shows the header lines.
    assert "Detected:" in summary
