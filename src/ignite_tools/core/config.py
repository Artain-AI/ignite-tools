"""
Format config: pydantic models, YAML load/save, validation.

Current scope (after read-layer expansion 3):
- Storage: local filesystem only
- Format: JSONL, plain text, CSV, TSV - with .gz/.zst auto-detected at read time
- Text: simple ``fields`` form OR routed (``router_field`` + ``routes``)
- Id: optional dotted-path field
- Labels: optional dotted-path field, populates ``Item.label``
- Normalize: lowercase / masks / collapse_whitespace / strip / trim
- Filters: time window, labels include/exclude
- Sampling: full / head / random / stride / stratified / weighted

The full schema is documented in docs/format-config.md and the underlying
design in docs/read-layer.md. New blocks land here when their slice is built.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class NoAliasDumper(yaml.SafeDumper):
    """YAML dumper that never emits anchors (&id001) or aliases (*id001).

    PyYAML uses these when the same Python object appears in multiple places
    in the data structure. They're correct YAML but unreadable to most users.
    """

    def ignore_aliases(self, data):  # noqa: ARG002
        return True

# Field names allowed only for specific format types. Centralized here so the
# ``before`` validator and ``to_yaml`` stripper agree on the rule.
_TEXT_ONLY_FIELDS = {"unit"}
_TABULAR_ONLY_FIELDS = {"delimiter", "has_header", "quote"}
_TABULAR_TYPES = {"csv", "tsv"}

# ---------------------------------------------------------------------------
# Block models
# ---------------------------------------------------------------------------


class StorageBlock(BaseModel):
    """`storage:` section. Local filesystem, S3, or Azure Blob."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["local", "s3", "azure"] = "local"
    path: str
    recursive: bool = True
    include: Optional[list[str]] = None
    exclude: Optional[list[str]] = None
    cache_dir: Optional[str] = None
    region: Optional[str] = None

    @field_validator("path")
    @classmethod
    def _path_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("storage.path must not be empty")
        return v

    @model_validator(mode="after")
    def _validate_uri_matches_type(self) -> "StorageBlock":
        path = self.path
        if self.type == "s3" and not path.startswith("s3://"):
            raise ValueError(
                "storage.type 's3' requires path starting with 's3://'"
            )
        if self.type == "azure" and not path.startswith("azure://"):
            raise ValueError(
                "storage.type 'azure' requires path starting with 'azure://'"
            )
        if self.type == "local" and (
            path.startswith("s3://") or path.startswith("azure://")
        ):
            raise ValueError(
                "storage.type 'local' but path looks like a cloud URI; "
                "set storage.type to 's3' or 'azure'"
            )
        if self.region is not None and self.type != "s3":
            raise ValueError(
                "storage.region is only valid when storage.type is 's3'"
            )
        return self


class FormatBlock(BaseModel):
    """`format:` section. Supports JSONL, plain text, and CSV/TSV.

    Type-specific fields are accepted only when their owning ``type`` is set:
    - ``unit`` requires ``type: text``
    - ``delimiter`` / ``has_header`` / ``quote`` require ``type: csv`` or ``tsv``
    Misuse is rejected at config load time so mistakes surface immediately.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["jsonl", "text", "csv", "tsv"] = "jsonl"
    encoding: str = "utf-8"
    # text-only:
    unit: Literal["line", "file"] = "line"
    # csv/tsv-only. ``delimiter`` is left unset by default; effective value is
    # resolved at read time per ``type`` (``,`` for csv, ``\t`` for tsv).
    delimiter: Optional[str] = None
    has_header: bool = True
    quote: str = '"'

    @model_validator(mode="before")
    @classmethod
    def _validate_format_specific_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        type_ = data.get("type", "jsonl")
        present = set(data.keys())

        if type_ != "text":
            misplaced = sorted(present & _TEXT_ONLY_FIELDS)
            if misplaced:
                raise ValueError(
                    f"format.{misplaced[0]} is only valid when format.type is 'text'"
                )

        if type_ not in _TABULAR_TYPES:
            misplaced = sorted(present & _TABULAR_ONLY_FIELDS)
            if misplaced:
                raise ValueError(
                    f"format.{misplaced[0]} is only valid when "
                    "format.type is 'csv' or 'tsv'"
                )
        return data

    # ------------------------------------------------------------- helpers

    def effective_delimiter(self) -> str:
        """Resolved delimiter for tabular reads."""
        if self.delimiter is not None:
            return self.delimiter
        if self.type == "tsv":
            return "\t"
        return ","

    @field_validator("delimiter")
    @classmethod
    def _validate_delimiter(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) != 1:
            raise ValueError("format.delimiter must be a single character")
        return v

    @field_validator("quote")
    @classmethod
    def _validate_quote(cls, v: str) -> str:
        if len(v) != 1:
            raise ValueError("format.quote must be a single character")
        return v


class RouteBlock(BaseModel):
    """One route in the routed text-extraction form."""

    model_config = ConfigDict(extra="forbid")

    fields: list[str]

    @field_validator("fields")
    @classmethod
    def _non_empty_fields(cls, v: list[str]) -> list[str]:
        for i, f in enumerate(v):
            if not f.strip():
                raise ValueError(f"routes.fields[{i}] must not be empty")
        return [f.strip() for f in v]


class TextBlock(BaseModel):
    """`text:` section. Simple OR routed form, never both."""

    model_config = ConfigDict(extra="forbid")

    fields: Optional[list[str]] = None
    router_field: Optional[str] = None
    routes: Optional[dict[str, RouteBlock]] = None

    @field_validator("fields")
    @classmethod
    def _non_empty_fields(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is not None:
            for i, f in enumerate(v):
                if not f.strip():
                    raise ValueError(f"text.fields[{i}] must not be empty")
            return [f.strip() for f in v]
        return v

    @field_validator("router_field")
    @classmethod
    def _non_empty_router(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("text.router_field must not be empty")
        return v.strip() if v is not None else v

    @model_validator(mode="after")
    def _validate_form(self) -> "TextBlock":
        has_simple = bool(self.fields)
        has_router = bool(self.router_field)
        has_routes = self.routes is not None
        has_routed = has_router or has_routes

        if has_simple and has_routed:
            raise ValueError(
                "text: provide either 'fields' or ('router_field' + 'routes'), not both"
            )
        if not has_simple and not has_routed:
            raise ValueError(
                "text: must specify 'fields' or ('router_field' + 'routes')"
            )
        if has_router and not has_routes:
            raise ValueError("text: 'router_field' requires 'routes' to be defined")
        if has_routes and not has_router:
            raise ValueError("text: 'routes' requires 'router_field' to be defined")
        if has_routes and len(self.routes) == 0:
            raise ValueError("text: 'routes' must define at least one route")
        return self


class IdBlock(BaseModel):
    """`id:` section."""

    model_config = ConfigDict(extra="forbid")

    field: str

    @field_validator("field")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("id.field must not be empty")
        return v.strip()


class LabelsBlock(BaseModel):
    """`labels:` section."""

    model_config = ConfigDict(extra="forbid")

    field: str

    @field_validator("field")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("labels.field must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Normalize block (and sub-blocks)
# ---------------------------------------------------------------------------


class MaskRule(BaseModel):
    """One regex-substitution rule under ``normalize.masks``."""

    model_config = ConfigDict(extra="forbid")

    pattern: str
    replacement: str

    @field_validator("pattern")
    @classmethod
    def _pattern_must_compile(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"normalize.masks: invalid regex {value!r}: {exc}") from exc
        return value


class TrimRule(BaseModel):
    """``normalize.trim`` - character-length bounds applied last."""

    model_config = ConfigDict(extra="forbid")

    max_chars: Optional[int] = None
    min_chars: Optional[int] = None

    @model_validator(mode="after")
    def _validate(self) -> "TrimRule":
        if self.max_chars is not None and self.max_chars < 0:
            raise ValueError("normalize.trim.max_chars must be >= 0")
        if self.min_chars is not None and self.min_chars < 0:
            raise ValueError("normalize.trim.min_chars must be >= 0")
        if (
            self.max_chars is not None
            and self.min_chars is not None
            and self.max_chars < self.min_chars
        ):
            raise ValueError(
                "normalize.trim.max_chars must be >= normalize.trim.min_chars"
            )
        return self


class NormalizeBlock(BaseModel):
    """`normalize:` section. Pipeline applied in fixed order:

    1. lowercase
    2. masks (left-to-right)
    3. collapse_whitespace
    4. strip
    5. trim (max_chars, then min_chars)

    Each section is independently optional. Records dropped by trim.min_chars
    after normalization are silently skipped (lenient default).
    """

    model_config = ConfigDict(extra="forbid")

    lowercase: bool = False
    collapse_whitespace: bool = False
    strip: bool = False
    masks: Optional[list[MaskRule]] = None
    trim: Optional[TrimRule] = None


# ---------------------------------------------------------------------------
# Filters block
# ---------------------------------------------------------------------------


class FiltersBlock(BaseModel):
    """`filters:` section. Composable; applied during read, before sampling.

    Time bounds compare naive datetimes - any timezone info on either side
    is stripped before comparison so mixed-aware/naive sources don't raise.
    """

    model_config = ConfigDict(extra="forbid")

    time_field: Optional[str] = None
    time_from: Optional[datetime] = None
    time_to: Optional[datetime] = None

    @field_validator("time_field")
    @classmethod
    def _non_empty_time_field(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("filters.time_field must not be empty")
        return v.strip() if v is not None else v
    labels_include: Optional[list[str]] = None
    labels_exclude: Optional[list[str]] = None

    @model_validator(mode="after")
    def _validate(self) -> "FiltersBlock":
        if (self.time_from is not None or self.time_to is not None) and not self.time_field:
            raise ValueError(
                "filters: 'time_from' / 'time_to' require 'time_field' to be set"
            )
        if self.time_from is not None and self.time_to is not None:
            if strip_tz(self.time_from) > strip_tz(self.time_to):
                raise ValueError("filters: 'time_from' must be <= 'time_to'")
        return self


# ---------------------------------------------------------------------------
# Sampling block
# ---------------------------------------------------------------------------


class SamplingBlock(BaseModel):
    """`sampling:` section. One of six modes; filters compose with all of them.

    Per-mode required fields:
    - ``head``       -> ``total``
    - ``random``     -> ``total``
    - ``stride``     -> ``total``
    - ``stratified`` -> ``per_group`` + ``group_field``
    - ``weighted``   -> ``total`` + ``group_field`` + ``weights``
    - ``full``       -> none required

    ``per_file_cap`` is orthogonal to mode; applies after filtering.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["full", "head", "random", "stride", "stratified", "weighted"] = "full"
    total: Optional[int] = None
    per_group: Optional[int] = None
    group_field: Optional[str] = None
    per_file_cap: Optional[int] = None
    seed: int = 42

    @field_validator("group_field")
    @classmethod
    def _non_empty_group(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("sampling.group_field must not be empty")
        return v.strip() if v is not None else v
    stratified_unknown: Literal["drop", "keep"] = "drop"
    weights: Optional[dict[str, float]] = None

    @model_validator(mode="after")
    def _validate(self) -> "SamplingBlock":
        m = self.mode

        if self.total is not None and self.total <= 0:
            raise ValueError("sampling.total must be > 0")
        if self.per_group is not None and self.per_group <= 0:
            raise ValueError("sampling.per_group must be > 0")
        if self.per_file_cap is not None and self.per_file_cap <= 0:
            raise ValueError("sampling.per_file_cap must be > 0")

        if m in {"head", "random", "stride"} and self.total is None:
            raise ValueError(f"sampling.mode {m!r} requires 'total'")

        if m == "stratified":
            if not self.group_field:
                raise ValueError("sampling.mode 'stratified' requires 'group_field'")
            if self.per_group is None:
                raise ValueError("sampling.mode 'stratified' requires 'per_group'")

        if m == "weighted":
            if not self.group_field:
                raise ValueError("sampling.mode 'weighted' requires 'group_field'")
            if not self.weights:
                raise ValueError("sampling.mode 'weighted' requires 'weights'")
            if self.total is None:
                raise ValueError("sampling.mode 'weighted' requires 'total'")
            if any(v <= 0 for v in self.weights.values()):
                raise ValueError("sampling.weights values must all be > 0")

        return self


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class FormatConfig(BaseModel):
    """Top-level ``ignite-format.yaml`` model.

    Attribute names diverge from YAML keys for two reserved-name fields:
    - YAML ``format:`` -> attribute ``parser`` (avoid shadowing ``format``)
    - YAML ``id:``     -> attribute ``id_block`` (avoid shadowing ``id``)
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    storage: StorageBlock
    parser: FormatBlock = Field(alias="format")
    text: Optional[TextBlock] = None
    id_block: Optional[IdBlock] = Field(default=None, alias="id")
    labels: Optional[LabelsBlock] = None
    normalize: Optional[NormalizeBlock] = None
    filters: Optional[FiltersBlock] = None
    sampling: Optional[SamplingBlock] = None

    @model_validator(mode="after")
    def _validate_text_required_for_record_formats(self) -> "FormatConfig":
        # Record-shaped formats (jsonl, csv, tsv) need a ``text`` block to
        # know which field/column to embed. Plain text doesn't - the file
        # itself is the content.
        if self.parser.type in {"jsonl", "csv", "tsv"} and self.text is None:
            raise ValueError(
                f"format.type {self.parser.type!r} requires a 'text' block "
                "(either 'fields' or 'router_field' + 'routes')"
            )
        return self

    # ------------------------------------------------------------------ I/O

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "FormatConfig":
        """Load and validate an ``ignite-format.yaml`` file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"Format config at {path} must be a YAML mapping at the top level"
            )
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict) -> "FormatConfig":
        """Build and validate a config from an in-memory dict."""
        return cls.model_validate(data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Write this config back to YAML using YAML keys (aliases)."""
        path = Path(path)
        data = self.dump_for_yaml()
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(data, f, Dumper=NoAliasDumper, default_flow_style=False, sort_keys=False)

    def dump_for_yaml(self) -> dict:
        """Return a YAML-ready dict (aliases applied, irrelevant defaults stripped).

        Strips type-specific format fields whose owning ``format.type``
        doesn't apply (``unit`` outside text, ``delimiter``/``has_header``/
        ``quote`` outside csv/tsv) so a load -> dump -> load round-trip
        survives the ``before`` validator. Used by both :meth:`to_yaml`
        and the CLI override pipeline.
        """
        data = self.model_dump(by_alias=True, exclude_none=True)
        format_block = data.get("format")
        if isinstance(format_block, dict):
            type_ = format_block.get("type")
            if type_ != "text":
                for fname in _TEXT_ONLY_FIELDS:
                    format_block.pop(fname, None)
            if type_ not in _TABULAR_TYPES:
                for fname in _TABULAR_ONLY_FIELDS:
                    format_block.pop(fname, None)
        return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def strip_tz(dt: datetime) -> datetime:
    """Drop tzinfo so naive/aware mixes don't raise on comparison.

    The read layer compares record times against ``filters.time_from`` /
    ``time_to`` naively - full timezone-aware filtering is a rabbit hole
    and the v1 audience (users on a laptop) rarely needs it.
    """
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt
