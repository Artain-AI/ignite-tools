"""Tests for ``ignite_tools.core.config``."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ignite_tools.core.config import FormatConfig


def _minimal_jsonl_config(path: str = "./data") -> dict:
    return {
        "storage": {"type": "local", "path": path},
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
    }


def _minimal_text_config(path: str = "./data") -> dict:
    return {
        "storage": {"type": "local", "path": path},
        "format": {"type": "text"},
    }


# ---------------------------------------------------------------------------
# Smoke + roundtrip
# ---------------------------------------------------------------------------


def test_minimal_config_validates():
    cfg = FormatConfig.from_dict(_minimal_jsonl_config())
    assert cfg.storage.type == "local"
    assert cfg.storage.path == "./data"
    assert cfg.storage.recursive is True
    assert cfg.parser.type == "jsonl"
    assert cfg.parser.encoding == "utf-8"
    assert cfg.parser.unit == "line"  # default; meaningless for jsonl but harmless
    assert cfg.text.fields == ["text"]
    assert cfg.id_block is None


def test_id_block_via_alias():
    data = _minimal_jsonl_config()
    data["id"] = {"field": "attributes.id"}
    cfg = FormatConfig.from_dict(data)
    assert cfg.id_block is not None
    assert cfg.id_block.field == "attributes.id"


def test_yaml_roundtrip(tmp_path: Path):
    data = _minimal_jsonl_config()
    data["id"] = {"field": "attributes.id"}
    data["storage"]["recursive"] = False

    src_path = tmp_path / "config.yaml"
    with src_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    cfg = FormatConfig.from_yaml(src_path)
    out_path = tmp_path / "out.yaml"
    cfg.to_yaml(out_path)

    reloaded = FormatConfig.from_yaml(out_path)
    assert reloaded == cfg

    with out_path.open("r", encoding="utf-8") as f:
        roundtripped = yaml.safe_load(f)
    assert "format" in roundtripped
    assert "id" in roundtripped
    assert "parser" not in roundtripped
    assert "id_block" not in roundtripped


# ---------------------------------------------------------------------------
# Text format + unit validation
# ---------------------------------------------------------------------------


def test_text_format_default_unit_is_line():
    cfg = FormatConfig.from_dict(_minimal_text_config())
    assert cfg.parser.type == "text"
    assert cfg.parser.unit == "line"
    # text block is optional for plain text
    assert cfg.text is None


def test_text_format_unit_file_accepted():
    data = _minimal_text_config()
    data["format"]["unit"] = "file"
    cfg = FormatConfig.from_dict(data)
    assert cfg.parser.unit == "file"


def test_text_format_unit_invalid_value_rejected():
    data = _minimal_text_config()
    data["format"]["unit"] = "paragraph"
    with pytest.raises(ValidationError):
        FormatConfig.from_dict(data)


def test_unit_rejected_when_format_is_jsonl():
    data = _minimal_jsonl_config()
    data["format"]["unit"] = "line"
    with pytest.raises(ValidationError, match="unit is only valid"):
        FormatConfig.from_dict(data)


def test_jsonl_requires_text_block():
    data = {
        "storage": {"type": "local", "path": "./data"},
        "format": {"type": "jsonl"},
        # no text block
    }
    with pytest.raises(ValidationError, match="requires a 'text' block"):
        FormatConfig.from_dict(data)


# ---------------------------------------------------------------------------
# Text block: simple form vs routed form
# ---------------------------------------------------------------------------


def test_text_xor_rule_both_forms_rejected():
    data = _minimal_jsonl_config()
    data["text"] = {
        "fields": ["a"],
        "router_field": "src",
        "routes": {"x": {"fields": ["a"]}},
    }
    with pytest.raises(ValidationError) as exc:
        FormatConfig.from_dict(data)
    assert "either 'fields'" in str(exc.value)


def test_text_xor_rule_neither_form_rejected():
    data = _minimal_jsonl_config()
    data["text"] = {}
    with pytest.raises(ValidationError) as exc:
        FormatConfig.from_dict(data)
    assert "must specify" in str(exc.value)


def test_routed_form_accepted():
    data = _minimal_jsonl_config()
    data["text"] = {
        "router_field": "attributes.source",
        "routes": {
            "github": {"fields": ["attributes.body", "attributes.title"]},
            "_default": {"fields": ["body", "text"]},
        },
    }
    cfg = FormatConfig.from_dict(data)
    assert cfg.text.router_field == "attributes.source"
    assert "github" in cfg.text.routes
    assert cfg.text.routes["_default"].fields == ["body", "text"]


def test_routed_router_without_routes_rejected():
    data = _minimal_jsonl_config()
    data["text"] = {"router_field": "src"}
    with pytest.raises(ValidationError, match="requires 'routes'"):
        FormatConfig.from_dict(data)


def test_routed_routes_without_router_rejected():
    data = _minimal_jsonl_config()
    data["text"] = {"routes": {"x": {"fields": ["a"]}}}
    with pytest.raises(ValidationError, match="requires 'router_field'"):
        FormatConfig.from_dict(data)


def test_routed_empty_routes_rejected():
    data = _minimal_jsonl_config()
    data["text"] = {"router_field": "src", "routes": {}}
    with pytest.raises(ValidationError, match="at least one route"):
        FormatConfig.from_dict(data)


def test_route_block_rejects_unknown_keys():
    data = _minimal_jsonl_config()
    data["text"] = {
        "router_field": "src",
        "routes": {"x": {"fields": ["a"], "what": "unknown"}},
    }
    with pytest.raises(ValidationError):
        FormatConfig.from_dict(data)


# ---------------------------------------------------------------------------
# Schema strictness
# ---------------------------------------------------------------------------


def test_unknown_top_level_field_rejected():
    data = _minimal_jsonl_config()
    data["unknown_block"] = {"foo": "bar"}
    with pytest.raises(ValidationError):
        FormatConfig.from_dict(data)


def test_unknown_storage_type_rejected():
    data = _minimal_jsonl_config()
    data["storage"]["type"] = "ftp"
    with pytest.raises(ValidationError):
        FormatConfig.from_dict(data)


def test_unsupported_format_rejected():
    data = _minimal_jsonl_config()
    data["format"]["type"] = "parquet"
    with pytest.raises(ValidationError):
        FormatConfig.from_dict(data)


# ---------------------------------------------------------------------------
# CSV / TSV format
# ---------------------------------------------------------------------------


def _minimal_csv_config(path: str = "./products.csv") -> dict:
    return {
        "storage": {"type": "local", "path": path},
        "format": {"type": "csv"},
        "text": {"fields": ["description"]},
    }


def test_csv_format_validates_with_defaults():
    cfg = FormatConfig.from_dict(_minimal_csv_config())
    assert cfg.parser.type == "csv"
    assert cfg.parser.delimiter is None  # resolved at read time
    assert cfg.parser.effective_delimiter() == ","
    assert cfg.parser.has_header is True
    assert cfg.parser.quote == '"'


def test_tsv_default_delimiter_is_tab():
    data = _minimal_csv_config()
    data["format"]["type"] = "tsv"
    cfg = FormatConfig.from_dict(data)
    assert cfg.parser.effective_delimiter() == "\t"


def test_csv_explicit_delimiter_wins():
    data = _minimal_csv_config()
    data["format"]["delimiter"] = ";"
    cfg = FormatConfig.from_dict(data)
    assert cfg.parser.effective_delimiter() == ";"


def test_csv_custom_quote_char():
    data = _minimal_csv_config()
    data["format"]["quote"] = "'"
    cfg = FormatConfig.from_dict(data)
    assert cfg.parser.quote == "'"


def test_csv_no_header():
    data = _minimal_csv_config()
    data["format"]["has_header"] = False
    cfg = FormatConfig.from_dict(data)
    assert cfg.parser.has_header is False


def test_csv_requires_text_block():
    data = {
        "storage": {"type": "local", "path": "./products.csv"},
        "format": {"type": "csv"},
    }
    with pytest.raises(ValidationError, match="requires a 'text' block"):
        FormatConfig.from_dict(data)


def test_tsv_requires_text_block():
    data = {
        "storage": {"type": "local", "path": "./products.tsv"},
        "format": {"type": "tsv"},
    }
    with pytest.raises(ValidationError, match="requires a 'text' block"):
        FormatConfig.from_dict(data)


def test_delimiter_rejected_when_format_is_jsonl():
    data = _minimal_jsonl_config()
    data["format"]["delimiter"] = ","
    with pytest.raises(ValidationError, match="only valid when format.type is 'csv' or 'tsv'"):
        FormatConfig.from_dict(data)


def test_has_header_rejected_when_format_is_jsonl():
    data = _minimal_jsonl_config()
    data["format"]["has_header"] = True
    with pytest.raises(ValidationError, match="only valid when format.type is 'csv' or 'tsv'"):
        FormatConfig.from_dict(data)


def test_quote_rejected_when_format_is_text():
    data = _minimal_text_config()
    data["format"]["quote"] = "'"
    with pytest.raises(ValidationError, match="only valid when format.type is 'csv' or 'tsv'"):
        FormatConfig.from_dict(data)


def test_unit_rejected_when_format_is_csv():
    data = _minimal_csv_config()
    data["format"]["unit"] = "line"
    with pytest.raises(ValidationError, match="unit is only valid"):
        FormatConfig.from_dict(data)


def test_csv_yaml_roundtrip(tmp_path: Path):
    data = _minimal_csv_config()
    data["format"]["delimiter"] = ";"
    data["format"]["has_header"] = False
    data["format"]["quote"] = "'"

    src = tmp_path / "config.yaml"
    with src.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    cfg = FormatConfig.from_yaml(src)
    out = tmp_path / "out.yaml"
    cfg.to_yaml(out)

    reloaded = FormatConfig.from_yaml(out)
    assert reloaded == cfg

    # Round-tripped YAML keeps tabular fields, drops text-only ones.
    with out.open("r", encoding="utf-8") as f:
        round_data = yaml.safe_load(f)
    assert round_data["format"]["delimiter"] == ";"
    assert round_data["format"]["has_header"] is False
    assert round_data["format"]["quote"] == "'"
    assert "unit" not in round_data["format"]


def test_jsonl_yaml_roundtrip_strips_tabular_defaults(tmp_path: Path):
    cfg = FormatConfig.from_dict(_minimal_jsonl_config())
    out = tmp_path / "out.yaml"
    cfg.to_yaml(out)

    with out.open("r", encoding="utf-8") as f:
        round_data = yaml.safe_load(f)
    # The model has has_header=True, quote='"' as defaults; for jsonl they're
    # stripped on dump so reload doesn't trip the "only valid for csv" rule.
    assert "has_header" not in round_data["format"]
    assert "quote" not in round_data["format"]
    assert "delimiter" not in round_data["format"]
    assert "unit" not in round_data["format"]


# ---------------------------------------------------------------------------
# Labels block
# ---------------------------------------------------------------------------


def test_labels_block_validates():
    data = _minimal_jsonl_config()
    data["labels"] = {"field": "category"}
    cfg = FormatConfig.from_dict(data)
    assert cfg.labels is not None
    assert cfg.labels.field == "category"


def test_labels_block_rejects_unknown_keys():
    data = _minimal_jsonl_config()
    data["labels"] = {"field": "category", "other": "x"}
    with pytest.raises(ValidationError):
        FormatConfig.from_dict(data)


def test_labels_block_requires_field():
    data = _minimal_jsonl_config()
    data["labels"] = {}
    with pytest.raises(ValidationError):
        FormatConfig.from_dict(data)


# ---------------------------------------------------------------------------
# Normalize block
# ---------------------------------------------------------------------------


def test_normalize_minimal():
    data = _minimal_jsonl_config()
    data["normalize"] = {"lowercase": True}
    cfg = FormatConfig.from_dict(data)
    assert cfg.normalize is not None
    assert cfg.normalize.lowercase is True
    assert cfg.normalize.collapse_whitespace is False
    assert cfg.normalize.masks is None
    assert cfg.normalize.trim is None


def test_normalize_full_block():
    data = _minimal_jsonl_config()
    data["normalize"] = {
        "lowercase": True,
        "collapse_whitespace": True,
        "strip": True,
        "masks": [
            {"pattern": r"https?://\S+", "replacement": "<url>"},
            {"pattern": r"\b[0-9a-f]{40}\b", "replacement": "<sha>"},
        ],
        "trim": {"max_chars": 256, "min_chars": 10},
    }
    cfg = FormatConfig.from_dict(data)
    assert len(cfg.normalize.masks) == 2
    assert cfg.normalize.trim.max_chars == 256
    assert cfg.normalize.trim.min_chars == 10


def test_normalize_invalid_regex_rejected():
    data = _minimal_jsonl_config()
    data["normalize"] = {"masks": [{"pattern": "(unclosed", "replacement": "<x>"}]}
    with pytest.raises(ValidationError, match="invalid regex"):
        FormatConfig.from_dict(data)


def test_trim_max_smaller_than_min_rejected():
    data = _minimal_jsonl_config()
    data["normalize"] = {"trim": {"max_chars": 5, "min_chars": 10}}
    with pytest.raises(ValidationError, match=">= normalize.trim.min_chars"):
        FormatConfig.from_dict(data)


def test_trim_negative_values_rejected():
    data = _minimal_jsonl_config()
    data["normalize"] = {"trim": {"min_chars": -1}}
    with pytest.raises(ValidationError, match=">= 0"):
        FormatConfig.from_dict(data)


def test_normalize_block_rejects_unknown_keys():
    data = _minimal_jsonl_config()
    data["normalize"] = {"loweRcase": True}  # typo'd key
    with pytest.raises(ValidationError):
        FormatConfig.from_dict(data)


def test_mask_rule_rejects_unknown_keys():
    data = _minimal_jsonl_config()
    data["normalize"] = {"masks": [{"pattern": "x", "replacement": "y", "z": 1}]}
    with pytest.raises(ValidationError):
        FormatConfig.from_dict(data)


# ---------------------------------------------------------------------------
# Filters block
# ---------------------------------------------------------------------------


def test_filters_minimal():
    data = _minimal_jsonl_config()
    data["filters"] = {"labels_include": ["a", "b"]}
    cfg = FormatConfig.from_dict(data)
    assert cfg.filters is not None
    assert cfg.filters.labels_include == ["a", "b"]


def test_filters_time_window():
    data = _minimal_jsonl_config()
    data["filters"] = {
        "time_field": "ts",
        "time_from": "2024-01-01",
        "time_to": "2024-12-31",
    }
    cfg = FormatConfig.from_dict(data)
    assert cfg.filters.time_field == "ts"
    assert cfg.filters.time_from is not None
    assert cfg.filters.time_to is not None


def test_filters_time_bounds_require_time_field():
    data = _minimal_jsonl_config()
    data["filters"] = {"time_from": "2024-01-01"}
    with pytest.raises(ValidationError, match="require 'time_field'"):
        FormatConfig.from_dict(data)


def test_filters_time_from_after_time_to_rejected():
    data = _minimal_jsonl_config()
    data["filters"] = {
        "time_field": "ts",
        "time_from": "2024-12-01",
        "time_to": "2024-01-01",
    }
    with pytest.raises(ValidationError, match="must be <= 'time_to'"):
        FormatConfig.from_dict(data)


def test_filters_block_rejects_unknown_keys():
    data = _minimal_jsonl_config()
    data["filters"] = {"unknown_filter": True}
    with pytest.raises(ValidationError):
        FormatConfig.from_dict(data)


# ---------------------------------------------------------------------------
# Sampling block
# ---------------------------------------------------------------------------


def test_sampling_default_mode_is_full():
    data = _minimal_jsonl_config()
    data["sampling"] = {}
    cfg = FormatConfig.from_dict(data)
    assert cfg.sampling.mode == "full"
    assert cfg.sampling.seed == 42


def test_sampling_head_requires_total():
    data = _minimal_jsonl_config()
    data["sampling"] = {"mode": "head"}
    with pytest.raises(ValidationError, match="'total'"):
        FormatConfig.from_dict(data)


def test_sampling_random_requires_total():
    data = _minimal_jsonl_config()
    data["sampling"] = {"mode": "random"}
    with pytest.raises(ValidationError, match="'total'"):
        FormatConfig.from_dict(data)


def test_sampling_stride_requires_total():
    data = _minimal_jsonl_config()
    data["sampling"] = {"mode": "stride"}
    with pytest.raises(ValidationError, match="'total'"):
        FormatConfig.from_dict(data)


def test_sampling_stratified_requires_group_field():
    data = _minimal_jsonl_config()
    data["sampling"] = {"mode": "stratified", "per_group": 100}
    with pytest.raises(ValidationError, match="'group_field'"):
        FormatConfig.from_dict(data)


def test_sampling_stratified_requires_per_group():
    data = _minimal_jsonl_config()
    data["sampling"] = {"mode": "stratified", "group_field": "label"}
    with pytest.raises(ValidationError, match="'per_group'"):
        FormatConfig.from_dict(data)


def test_sampling_stratified_full_config():
    data = _minimal_jsonl_config()
    data["sampling"] = {
        "mode": "stratified",
        "group_field": "label",
        "per_group": 100,
        "stratified_unknown": "keep",
    }
    cfg = FormatConfig.from_dict(data)
    assert cfg.sampling.mode == "stratified"
    assert cfg.sampling.per_group == 100
    assert cfg.sampling.stratified_unknown == "keep"


def test_sampling_weighted_requires_weights_and_group():
    data = _minimal_jsonl_config()
    data["sampling"] = {"mode": "weighted", "total": 100}
    with pytest.raises(ValidationError, match="'group_field'"):
        FormatConfig.from_dict(data)

    data["sampling"]["group_field"] = "label"
    with pytest.raises(ValidationError, match="'weights'"):
        FormatConfig.from_dict(data)


def test_sampling_weighted_full_config():
    data = _minimal_jsonl_config()
    data["sampling"] = {
        "mode": "weighted",
        "total": 100,
        "group_field": "label",
        "weights": {"a": 0.5, "b": 0.3, "c": 0.2},
    }
    cfg = FormatConfig.from_dict(data)
    assert cfg.sampling.mode == "weighted"
    assert sum(cfg.sampling.weights.values()) == pytest.approx(1.0)


def test_sampling_weighted_rejects_zero_weights():
    data = _minimal_jsonl_config()
    data["sampling"] = {
        "mode": "weighted",
        "total": 100,
        "group_field": "label",
        "weights": {"a": 0.5, "b": 0.0},
    }
    with pytest.raises(ValidationError, match="must all be > 0"):
        FormatConfig.from_dict(data)


def test_sampling_total_must_be_positive():
    data = _minimal_jsonl_config()
    data["sampling"] = {"mode": "head", "total": 0}
    with pytest.raises(ValidationError, match="must be > 0"):
        FormatConfig.from_dict(data)


def test_sampling_per_file_cap_must_be_positive():
    data = _minimal_jsonl_config()
    data["sampling"] = {"per_file_cap": 0}
    with pytest.raises(ValidationError, match="must be > 0"):
        FormatConfig.from_dict(data)


def test_sampling_per_file_cap_with_full_mode_accepted():
    data = _minimal_jsonl_config()
    data["sampling"] = {"per_file_cap": 100}  # mode defaults to 'full'
    cfg = FormatConfig.from_dict(data)
    assert cfg.sampling.mode == "full"
    assert cfg.sampling.per_file_cap == 100


def test_sampling_block_rejects_unknown_keys():
    data = _minimal_jsonl_config()
    data["sampling"] = {"mode": "full", "unknown_key": 1}
    with pytest.raises(ValidationError):
        FormatConfig.from_dict(data)


def test_full_pipeline_yaml_roundtrip(tmp_path: Path):
    """All blocks combined survive a load -> dump -> load round-trip."""
    data = {
        "storage": {"type": "local", "path": "./data", "recursive": True},
        "format": {"type": "jsonl"},
        "text": {"fields": ["body"]},
        "id": {"field": "id"},
        "labels": {"field": "label"},
        "normalize": {
            "lowercase": True,
            "strip": True,
            "masks": [{"pattern": r"https?://\S+", "replacement": "<url>"}],
            "trim": {"max_chars": 256, "min_chars": 10},
        },
        "filters": {
            "time_field": "ts",
            "time_from": "2024-01-01T00:00:00",
            "labels_include": ["a", "b"],
        },
        "sampling": {
            "mode": "stratified",
            "group_field": "label",
            "per_group": 100,
        },
    }
    src = tmp_path / "in.yaml"
    with src.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    cfg = FormatConfig.from_yaml(src)
    out = tmp_path / "out.yaml"
    cfg.to_yaml(out)

    reloaded = FormatConfig.from_yaml(out)
    assert reloaded == cfg


def test_from_yaml_rejects_non_mapping(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="mapping"):
        FormatConfig.from_yaml(bad)
