"""
Shared CLI helpers for ``ignite-*`` tools.

Config model: ONE file, ONE ``--config`` flag. The file contains:
- Shared data-reading blocks (storage, format, text, labels, normalize,
  filters, sampling) - validated by :class:`FormatConfig`.
- Tool-specific blocks (``report:``, ``eval:``, ``explore:``, ``index:``) -
  popped before FormatConfig validation, returned separately for the tool.

Public surface:
- :func:`add_read_flags` - register the standard flags on an argparse parser.
- :func:`load_config_from_args` - turn parsed args into FormatConfig + tool block.
- :func:`apply_overrides` - merge CLI flags onto a loaded config.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from ignite_tools.core.config import FormatConfig, NoAliasDumper
from ignite_tools.core.sniffer import (
    build_proposal,
    format_human_summary,
    sniff_path,
)

DEFAULT_CONFIG_FILENAME = "ignite-format.yaml"
_USER_CONFIG_DIR = Path.home() / ".config" / "ignite-tools"

# Known tool-specific block names. Popped before FormatConfig validation.
_TOOL_BLOCKS = {"ignite-read", "ignite-eval", "ignite-explore", "ignite-index"}


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def add_read_flags(parser: argparse.ArgumentParser) -> None:
    """Register the standard ignite-tools flags on ``parser``."""
    g = parser.add_argument_group("input (shared flags)")
    g.add_argument(
        "path", nargs="?", type=str, default=None,
        help="Path or URI to the corpus. Overrides storage.path in config.",
    )
    g.add_argument(
        "--config", type=Path, default=None, metavar="PATH",
        help=f"Path to config file (default: ./{DEFAULT_CONFIG_FILENAME}).",
    )
    g.add_argument("--recursive", dest="recursive", action="store_true", default=None)
    g.add_argument("--no-recursive", dest="recursive", action="store_false")
    g.add_argument("--sample", type=int, default=None, metavar="N",
                   help="Override sampling.total (implies mode=random if unset).")
    g.add_argument("--sample-mode", type=str, default=None,
                   choices=["full", "head", "random", "stride", "stratified", "weighted"])
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--cache-dir", type=str, default=None, metavar="PATH")
    g.add_argument("--no-cache", action="store_true", default=False)
    g.add_argument("--yes", action="store_true", default=False,
                   help="Accept sniffer proposal non-interactively.")
    g.add_argument("--save-config", type=Path, default=None, metavar="PATH",
                   help="Sniff, write proposed config, and exit.")
    g.add_argument("--strict", action="store_true", default=False)
    g.add_argument("--progress", action="store_true", default=False)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


@dataclass
class LoadedConfig:
    """Result of :func:`load_config_from_args`.

    ``tool_block`` carries the raw dict from the tool's named block
    (e.g. contents of ``report:``) if present in the config file.
    """
    config: Optional[FormatConfig] = None
    tool_block: Optional[dict] = None
    saved_to: Optional[Path] = None
    strict: bool = False
    no_cache: bool = False
    progress: bool = False


def load_config_from_args(
    args: argparse.Namespace,
    *,
    tool_name: Optional[str] = None,
    interactive_prompt: Optional[callable] = None,
) -> LoadedConfig:
    """Resolve FormatConfig + tool block from CLI arguments.

    Discovery order:
      1. --save-config -> sniff, write, exit.
      2. --config PATH -> load directly.
      3. ./ignite-format.yaml in CWD.
      4. <path>/ignite-format.yaml next to data.
      5. ~/.config/ignite-tools/ignite-format.yaml global.
      6. Sniff and propose.

    ``tool_name`` (e.g. "report") selects which tool block to extract.
    """
    if interactive_prompt is None:
        interactive_prompt = prompt_user

    # Case 1: --save-config
    if args.save_config is not None:
        if args.path is None:
            raise SystemExit("--save-config requires a positional path to sniff.")
        result = sniff_path(args.path)
        proposal = build_proposal(result)
        FormatConfig.from_dict(proposal)
        _write_yaml(proposal, args.save_config)
        return LoadedConfig(saved_to=args.save_config)

    # Case 2: explicit --config
    config_path = getattr(args, "config", None)
    if config_path is not None:
        cfg, tb = _load_and_split(config_path, tool_name)
        cfg = apply_overrides(cfg, args)
        return LoadedConfig(config=cfg, tool_block=tb, strict=args.strict,
                            no_cache=args.no_cache, progress=args.progress)

    # Case 3: CWD
    cwd_cfg = Path.cwd() / DEFAULT_CONFIG_FILENAME
    if cwd_cfg.exists():
        cfg, tb = _load_and_split(cwd_cfg, tool_name)
        cfg = apply_overrides(cfg, args)
        return LoadedConfig(config=cfg, tool_block=tb, strict=args.strict,
                            no_cache=args.no_cache, progress=args.progress)

    # Case 4: next to data
    if args.path is not None:
        data_path = Path(args.path).expanduser()
        data_dir = data_path if data_path.is_dir() else data_path.parent
        adjacent = data_dir / DEFAULT_CONFIG_FILENAME
        if adjacent.exists():
            cfg, tb = _load_and_split(adjacent, tool_name)
            cfg = apply_overrides(cfg, args)
            return LoadedConfig(config=cfg, tool_block=tb, strict=args.strict,
                                no_cache=args.no_cache, progress=args.progress)

    # Case 5: user-level global
    user_cfg = _USER_CONFIG_DIR / DEFAULT_CONFIG_FILENAME
    if user_cfg.exists():
        cfg, tb = _load_and_split(user_cfg, tool_name)
        cfg = apply_overrides(cfg, args)
        return LoadedConfig(config=cfg, tool_block=tb, strict=args.strict,
                            no_cache=args.no_cache, progress=args.progress)

    # Case 6: sniff and propose
    if args.path is None:
        raise SystemExit(
            "No config found and no path given. Provide a path or "
            f"use --config to point at an existing {DEFAULT_CONFIG_FILENAME}."
        )

    result = sniff_path(args.path)
    proposal = build_proposal(result)
    print(format_human_summary(result), file=sys.stderr)
    print("\nProposed configuration:", file=sys.stderr)
    print(_render_yaml(proposal), file=sys.stderr)

    if args.yes or not _stdin_is_tty():
        cfg = FormatConfig.from_dict(proposal)
        cfg = apply_overrides(cfg, args)
        return LoadedConfig(config=cfg, strict=args.strict,
                            no_cache=args.no_cache, progress=args.progress)

    response = interactive_prompt("Proceed? [y/n/save/edit] ").strip().lower()
    if response in {"y", "yes", ""}:
        cfg = FormatConfig.from_dict(proposal)
        cfg = apply_overrides(cfg, args)
        return LoadedConfig(config=cfg, strict=args.strict,
                            no_cache=args.no_cache, progress=args.progress)
    if response == "save":
        target = Path.cwd() / DEFAULT_CONFIG_FILENAME
        _write_yaml(proposal, target)
        return LoadedConfig(saved_to=target)
    if response == "edit":
        target = Path.cwd() / DEFAULT_CONFIG_FILENAME
        _write_yaml(proposal, target)
        editor = os.environ.get("EDITOR", "vi")
        subprocess.run([editor, str(target)])
        cfg, tb = _load_and_split(target, tool_name)
        cfg = apply_overrides(cfg, args)
        return LoadedConfig(config=cfg, tool_block=tb, strict=args.strict,
                            no_cache=args.no_cache, progress=args.progress)
    raise SystemExit("Aborted.")


# ---------------------------------------------------------------------------
# Config loading with tool-block extraction
# ---------------------------------------------------------------------------


def _load_and_split(
    path: Path, tool_name: Optional[str]
) -> tuple[FormatConfig, Optional[dict]]:
    """Load YAML, pop tool blocks, validate FormatConfig from remainder.

    Tool block resolution:
    - If the block value is a dict → use inline.
    - If the block value is a string → treat as a path (relative to the
      config file's directory), load that file as the tool block dict.
    - If absent → None (tool uses defaults).
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping")

    tool_block: Optional[dict] = None
    config_dir = path.parent

    for block_name in _TOOL_BLOCKS:
        popped = data.pop(block_name, None)
        if block_name == tool_name and popped is not None:
            if isinstance(popped, str):
                # It's a path reference - resolve relative to config file.
                ref_path = (config_dir / popped).resolve()
                # Reject references that escape the config directory.
                try:
                    ref_path.relative_to(config_dir.resolve())
                except ValueError:
                    raise ValueError(
                        f"Tool config reference '{popped}' in {path} escapes "
                        f"the config directory. Keep references inside the same "
                        f"directory as the config file."
                    )
                if not ref_path.exists():
                    raise FileNotFoundError(
                        f"Tool config reference '{popped}' in {path} "
                        f"resolves to {ref_path} which does not exist."
                    )
                with ref_path.open("r", encoding="utf-8") as f2:
                    tool_block = yaml.safe_load(f2)
                if not isinstance(tool_block, dict):
                    raise ValueError(
                        f"Tool config at {ref_path} must be a YAML mapping"
                    )
            else:
                tool_block = popped

    cfg = FormatConfig.from_dict(data)
    return cfg, tool_block


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


def apply_overrides(cfg: FormatConfig, args: argparse.Namespace) -> FormatConfig:
    """Merge CLI flags onto a loaded FormatConfig."""
    data = cfg.dump_for_yaml()

    storage = data.get("storage", {})
    if args.path is not None:
        storage["path"] = args.path
    if args.recursive is not None:
        storage["recursive"] = args.recursive
    if getattr(args, "cache_dir", None) is not None:
        storage["cache_dir"] = args.cache_dir
    data["storage"] = storage

    sampling = data.get("sampling") or {}
    if args.sample is not None:
        sampling["total"] = args.sample
        if args.sample_mode is None and not sampling.get("mode"):
            sampling["mode"] = "random"
    if args.sample_mode is not None:
        sampling["mode"] = args.sample_mode
    if args.seed is not None:
        sampling["seed"] = args.seed
    if sampling:
        data["sampling"] = sampling

    return FormatConfig.from_dict(data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def prompt_user(message: str) -> str:
    return input(message)


def _stdin_is_tty() -> bool:
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _render_yaml(data: dict) -> str:
    return yaml.dump(data, Dumper=NoAliasDumper, default_flow_style=False, sort_keys=False).rstrip("\n")


def _write_yaml(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, Dumper=NoAliasDumper, default_flow_style=False, sort_keys=False)
    print(f"Wrote config to {path}", file=sys.stderr)
