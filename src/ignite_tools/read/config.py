"""
Report configuration for ``ignite-read``.

Loaded from ``ignite-read.yaml`` (tool-specific config, separate from the
shared ``ignite-format.yaml``). Controls which analysis sections appear,
their parameters, and display settings.

Discovery follows the same priority chain as ``ignite-format.yaml``:
  1. ``--report-config PATH`` (explicit)
  2. ``./ignite-read.yaml`` in CWD
  3. ``<data-path>/ignite-read.yaml`` next to the data
  4. ``~/.config/ignite-tools/ignite-read.yaml`` user-level global
  5. Built-in defaults (everything on, standard parameters)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------


class TopWordsConfig(BaseModel):
    """Configuration for the top-words analysis section."""

    model_config = ConfigDict(extra="forbid")

    count: int = 25
    min_length: int = 3
    tokenize: str = r"[a-z][a-z0-9_-]*[a-z0-9]|[a-z]"
    stop_words_extend: list[str] = Field(default_factory=list)
    stop_words_replace: Optional[list[str]] = None  # if set, replaces the built-in list entirely

    @field_validator("tokenize")
    @classmethod
    def _tokenize_must_compile(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"top_words.tokenize: invalid regex: {exc}") from exc
        return value


class TimeConfig(BaseModel):
    """Configuration for time-distribution analysis."""

    model_config = ConfigDict(extra="forbid")

    granularity: Literal["month", "week", "day"] = "month"


class PatternConfig(BaseModel):
    """One user-defined regex pattern to count across the corpus."""

    model_config = ConfigDict(extra="forbid")

    name: str
    pattern: str

    @field_validator("pattern")
    @classmethod
    def _pattern_must_compile(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"patterns[].pattern: invalid regex: {exc}") from exc
        return value


class SampleConfig(BaseModel):
    """Configuration for the sample-texts output section."""

    model_config = ConfigDict(extra="forbid")

    count: int = 10
    truncate: int = 72


# All configurable analysis sections.
ANALYSIS_SECTIONS = [
    "corpus_stats",
    "per_source",
    "labels",
    "top_words",
    "patterns",
]


# ---------------------------------------------------------------------------
# Top-level report config
# ---------------------------------------------------------------------------


class ReportConfig(BaseModel):
    """Top-level ``ignite-read.yaml`` model.

    All fields are optional - the tool works with zero config using built-in
    defaults. The config is for power users who want to tune the report.
    """

    model_config = ConfigDict(extra="forbid")

    # Which analysis sections to show, and in what order.
    # Default: all sections in the order listed in ANALYSIS_SECTIONS.
    sections: list[str] = Field(default_factory=lambda: list(ANALYSIS_SECTIONS))

    top_words: TopWordsConfig = Field(default_factory=TopWordsConfig)
    time: TimeConfig = Field(default_factory=TimeConfig)
    patterns: list[PatternConfig] = Field(default_factory=list)
    sample: SampleConfig = Field(default_factory=SampleConfig)

    @field_validator("sections")
    @classmethod
    def _validate_sections(cls, value: list[str]) -> list[str]:
        for s in value:
            if s not in ANALYSIS_SECTIONS:
                raise ValueError(
                    f"Unknown section {s!r}. Available: {ANALYSIS_SECTIONS}"
                )
        return value

    # ------------------------------------------------------------------ I/O

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ReportConfig":
        """Load and validate an ``ignite-read.yaml`` file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError(
                f"Report config at {path} must be a YAML mapping at the top level"
            )
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict) -> "ReportConfig":
        return cls.model_validate(data)

    @classmethod
    def defaults(cls) -> "ReportConfig":
        """Return the built-in default config (no file needed)."""
        return cls()
