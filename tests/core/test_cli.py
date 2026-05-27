"""Tests for ``ignite_tools.core.cli``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from ignite_tools.core.cli import (
    DEFAULT_CONFIG_FILENAME,
    LoadedConfig,
    add_read_flags,
    apply_overrides,
    load_config_from_args,
)
from ignite_tools.core.config import FormatConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_read_flags(parser)
    return parser


def _write_jsonl(path: Path, n: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"id": f"r{i}", "text": f"hello {i}"}) + "\n")


def _write_existing_config(path: Path, data_path: Path) -> None:
    cfg = {
        "storage": {"type": "local", "path": str(data_path)},
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def test_add_read_flags_registers_all_documented_flags():
    parser = _make_parser()
    parsed = parser.parse_args(["./data"])
    # All shared flags exist on the namespace.
    for attr in (
        "path",
        "config",
        "recursive",
        "sample",
        "sample_mode",
        "seed",
        "cache_dir",
        "no_cache",
        "yes",
        "save_config",
        "strict",
        "progress",
    ):
        assert hasattr(parsed, attr)


def test_recursive_flag_is_tri_state():
    parser = _make_parser()
    assert parser.parse_args(["./data"]).recursive is None
    assert parser.parse_args(["./data", "--recursive"]).recursive is True
    assert parser.parse_args(["./data", "--no-recursive"]).recursive is False


# ---------------------------------------------------------------------------
# load_config_from_args: each branch
# ---------------------------------------------------------------------------


def test_load_config_via_explicit_config(tmp_path: Path):
    data_path = tmp_path / "data.jsonl"
    _write_jsonl(data_path)
    cfg_path = tmp_path / "cfg.yaml"
    _write_existing_config(cfg_path, data_path)

    parser = _make_parser()
    args = parser.parse_args([str(data_path), "--config", str(cfg_path)])
    loaded = load_config_from_args(args)

    assert isinstance(loaded.config, FormatConfig)
    assert loaded.saved_to is None
    # --path positional overrode storage.path (same value here, but the
    # override path is exercised).
    assert loaded.config.storage.path == str(data_path)


def test_load_config_via_default_filename(tmp_path: Path, monkeypatch):
    """When ./ignite-format.yaml exists in CWD, it's used implicitly."""
    data_path = tmp_path / "data.jsonl"
    _write_jsonl(data_path)
    monkeypatch.chdir(tmp_path)
    _write_existing_config(tmp_path / DEFAULT_CONFIG_FILENAME, data_path)

    parser = _make_parser()
    args = parser.parse_args([])
    loaded = load_config_from_args(args)
    assert loaded.config is not None
    assert loaded.config.parser.type == "jsonl"


def test_load_config_save_config_writes_proposal_and_exits(tmp_path: Path):
    data_path = tmp_path / "data.jsonl"
    _write_jsonl(data_path)
    target = tmp_path / "out.yaml"

    parser = _make_parser()
    args = parser.parse_args([str(data_path), "--save-config", str(target)])
    loaded = load_config_from_args(args)

    assert loaded.config is None
    assert loaded.saved_to == target
    assert target.exists()

    # Written file is a valid FormatConfig.
    cfg = FormatConfig.from_yaml(target)
    assert cfg.parser.type == "jsonl"


def test_load_config_save_config_requires_path(tmp_path: Path):
    parser = _make_parser()
    args = parser.parse_args(["--save-config", str(tmp_path / "out.yaml")])
    with pytest.raises(SystemExit):
        load_config_from_args(args)


def test_load_config_no_args_no_default_raises(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parser = _make_parser()
    args = parser.parse_args([])
    with pytest.raises(SystemExit, match="No config found"):
        load_config_from_args(args)


# ---------------------------------------------------------------------------
# Sniff + interactive prompt
# ---------------------------------------------------------------------------


def test_load_config_sniffer_yes_flag_skips_prompt(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "data.jsonl"
    _write_jsonl(data_path)

    parser = _make_parser()
    args = parser.parse_args([str(data_path), "--yes"])

    def fail_prompt(_msg):
        raise AssertionError("prompt should not be called when --yes is set")

    loaded = load_config_from_args(args, interactive_prompt=fail_prompt)
    assert loaded.config is not None
    assert loaded.config.parser.type == "jsonl"


def test_load_config_sniffer_prompt_y_accepts(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "data.jsonl"
    _write_jsonl(data_path)

    parser = _make_parser()
    args = parser.parse_args([str(data_path)])

    # Force the "interactive" branch by pretending stdin is a TTY.
    with patch("ignite_tools.core.cli._stdin_is_tty", return_value=True):
        loaded = load_config_from_args(args, interactive_prompt=lambda _msg: "y")
    assert loaded.config is not None


def test_load_config_sniffer_prompt_save_writes_default_filename(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "data.jsonl"
    _write_jsonl(data_path)

    parser = _make_parser()
    args = parser.parse_args([str(data_path)])

    with patch("ignite_tools.core.cli._stdin_is_tty", return_value=True):
        loaded = load_config_from_args(
            args, interactive_prompt=lambda _msg: "save"
        )
    assert loaded.config is None
    assert loaded.saved_to == tmp_path / DEFAULT_CONFIG_FILENAME
    assert loaded.saved_to.exists()


def test_load_config_sniffer_prompt_n_aborts(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "data.jsonl"
    _write_jsonl(data_path)

    parser = _make_parser()
    args = parser.parse_args([str(data_path)])

    with patch("ignite_tools.core.cli._stdin_is_tty", return_value=True):
        with pytest.raises(SystemExit, match="Aborted"):
            load_config_from_args(args, interactive_prompt=lambda _msg: "n")


def test_load_config_sniffer_non_tty_auto_accepts(tmp_path: Path, monkeypatch):
    """When stdin is not a TTY (e.g. pipe / CI), auto-accept silently."""
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "data.jsonl"
    _write_jsonl(data_path)

    parser = _make_parser()
    args = parser.parse_args([str(data_path)])

    def fail_prompt(_msg):
        raise AssertionError("prompt should not be called when stdin isn't a TTY")

    with patch("ignite_tools.core.cli._stdin_is_tty", return_value=False):
        loaded = load_config_from_args(args, interactive_prompt=fail_prompt)
    assert loaded.config is not None


# ---------------------------------------------------------------------------
# apply_overrides
# ---------------------------------------------------------------------------


def _base_cfg(path: str = "./data") -> FormatConfig:
    return FormatConfig.from_dict({
        "storage": {"type": "local", "path": path},
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
    })


def test_apply_overrides_sets_path():
    parser = _make_parser()
    args = parser.parse_args(["./other"])
    cfg = apply_overrides(_base_cfg(), args)
    assert cfg.storage.path == "./other"


def test_apply_overrides_sets_recursive():
    parser = _make_parser()
    args = parser.parse_args(["./data", "--no-recursive"])
    cfg = apply_overrides(_base_cfg(), args)
    assert cfg.storage.recursive is False


def test_apply_overrides_sample_implies_random_mode():
    parser = _make_parser()
    args = parser.parse_args(["./data", "--sample", "100"])
    cfg = apply_overrides(_base_cfg(), args)
    assert cfg.sampling.total == 100
    assert cfg.sampling.mode == "random"


def test_apply_overrides_explicit_mode_takes_precedence():
    parser = _make_parser()
    args = parser.parse_args(
        ["./data", "--sample", "100", "--sample-mode", "head"]
    )
    cfg = apply_overrides(_base_cfg(), args)
    assert cfg.sampling.total == 100
    assert cfg.sampling.mode == "head"


def test_apply_overrides_seed():
    parser = _make_parser()
    args = parser.parse_args(["./data", "--sample", "10", "--seed", "99"])
    cfg = apply_overrides(_base_cfg(), args)
    assert cfg.sampling.seed == 99


def test_apply_overrides_cache_dir():
    parser = _make_parser()
    args = parser.parse_args(["./data", "--cache-dir", "/tmp/c"])
    cfg = apply_overrides(_base_cfg(), args)
    assert cfg.storage.cache_dir == "/tmp/c"


def test_apply_overrides_preserves_existing_sampling():
    """Overrides patch the existing sampling block, don't overwrite it."""
    base = FormatConfig.from_dict({
        "storage": {"type": "local", "path": "./data"},
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "sampling": {"mode": "stratified", "group_field": "label", "per_group": 50},
    })
    parser = _make_parser()
    args = parser.parse_args(["./data", "--seed", "7"])
    cfg = apply_overrides(base, args)
    assert cfg.sampling.mode == "stratified"
    assert cfg.sampling.per_group == 50
    assert cfg.sampling.seed == 7


def test_loaded_config_carries_strict_and_progress(tmp_path: Path):
    data_path = tmp_path / "data.jsonl"
    _write_jsonl(data_path)
    cfg_path = tmp_path / "cfg.yaml"
    _write_existing_config(cfg_path, data_path)

    parser = _make_parser()
    args = parser.parse_args([
        str(data_path),
        "--config",
        str(cfg_path),
        "--strict",
        "--progress",
        "--no-cache",
    ])
    loaded = load_config_from_args(args)
    assert loaded.strict is True
    assert loaded.progress is True
    assert loaded.no_cache is True
