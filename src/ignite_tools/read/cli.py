"""
CLI entry point for ``ignite-read``.

Usage:
    ignite-read examples/data/ --config examples/configs/full.yaml
    ignite-read examples/data/ --yes --show 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ignite_tools.core.cli import add_read_flags, load_config_from_args
from ignite_tools.read.config import ReportConfig
from ignite_tools.read.core import format_report, run_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ignite-read",
        description="Read and inspect a corpus. Shows what the read layer sees "
        "before embedding.",
    )
    add_read_flags(parser)

    g = parser.add_argument_group("report options")
    g.add_argument(
        "--show", type=int, default=None, metavar="N",
        help="Override sample text count (default 10).",
    )
    g.add_argument(
        "--quiet", action="store_true", default=False,
        help="Suppress sample texts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Load config - tool_name="ignite-read" extracts the ignite-read: block.
    loaded = load_config_from_args(args, tool_name="ignite-read")

    if loaded.saved_to is not None:
        return 0

    if loaded.config is None:
        print("ERROR: no config resolved.", file=sys.stderr)
        return 1

    # Build ReportConfig from the tool block (or defaults).
    if loaded.tool_block:
        report_cfg = ReportConfig.from_dict(loaded.tool_block)
    else:
        report_cfg = ReportConfig.defaults()

    # CLI overrides.
    if args.show is not None:
        report_cfg.sample.count = args.show
    if args.quiet:
        report_cfg.sample.count = 0

    report = run_report(
        loaded.config,
        report_config=report_cfg,
        strict=loaded.strict,
        config_path=str(args.config) if args.config else None,
    )
    print(format_report(report, report_cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
