"""
Format layer.

Current scope (after read-layer expansion 4):
- JSONL parsing via ``orjson`` (docs/performance.md rule 2)
- Compressed JSONL: ``.gz`` and ``.zst``, transparent (handled in sources.py)
- Plain text: ``unit: line`` or ``unit: file``
- CSV / TSV via ``polars`` streaming reader (rule 3)
- Simple text extraction (``text.fields``) AND routed text extraction
- Optional ``id.field`` resolution
- Optional ``labels.field`` populates ``Item.label``
- Normalization pipeline: lowercase / masks / collapse_whitespace / strip / trim
- Filtering: time window, labels include / exclude
- Sampling: full / head / random / stride / stratified / weighted
- **Strict mode**: per-record errors raise :class:`CorpusReadError` instead
  of being silently skipped. Default lenient.
- **Read summary**: a :class:`ReadSummary` instance can be passed in to
  capture per-stage skip counters. Filled in as the iterator is consumed.

Pipeline stages (applied in order):
  1. format-specific iterator (jsonl/csv/tsv/text) -> _PipelineRecord
  2. normalization (drops if too short after trim)
  3. filtering (time, labels)
  4. per-file cap (orthogonal to mode)
  5. sampling (one of six modes)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import orjson

from ignite_tools.core.config import (
    FiltersBlock,
    FormatConfig,
    LabelsBlock,
    NormalizeBlock,
    RouteBlock,
    SamplingBlock,
    TextBlock,
    strip_tz,
)
from ignite_tools.core.sampling import apply_sampling
from ignite_tools.core.sources import (
    DEFAULT_INCLUDES,
    iter_lines,
    list_files,
    read_file_text,
)

# Special routing key for fallback when the router value isn't in `routes`.
ROUTE_DEFAULT_KEY = "_default"

# polars batch size for tabular reads.
_TABULAR_CHUNK_SIZE = 8192


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """One unit of corpus content, ready to embed.

    The ``raw`` field is reserved for a future debugging hook; not populated
    in the current slice.
    """

    id: str
    text: str
    source_file: str
    label: Optional[str] = None
    raw: Optional[dict] = None


class CorpusReadError(Exception):
    """Raised in strict mode when a record fails to parse or extract.

    Carries ``source_file`` and ``line_no`` (best-effort) so error messages
    point at the offending record. In lenient mode the same conditions
    increment :class:`ReadSummary` counters and the record is skipped.
    """

    def __init__(
        self,
        message: str,
        *,
        source_file: Optional[str] = None,
        line_no: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.source_file = source_file
        self.line_no = line_no

    def __str__(self) -> str:
        prefix = ""
        if self.source_file:
            prefix = f"[{self.source_file}"
            if self.line_no is not None:
                prefix += f":{self.line_no}"
            prefix += "] "
        return prefix + super().__str__()


@dataclass
class ReadSummary:
    """Per-stage counters accumulated during ``read_corpus`` iteration.

    The counters are populated as the caller consumes the generator. After
    iteration completes, ``format_text()`` produces a human-readable block
    suitable for printing to stderr at end-of-run.
    """

    files_scanned: int = 0
    records_emitted: int = 0

    # Skip categories - should sum, with records_emitted, to the total
    # records seen across files (within rounding for sampling, which drops
    # records by design and isn't tracked here).
    skipped_malformed: int = 0       # JSON / row parse error
    skipped_extraction: int = 0      # no text field matched
    skipped_unrouted: int = 0        # routed extract: router value missed and no _default
    skipped_normalization: int = 0   # text dropped by trim.min_chars
    skipped_filter: int = 0          # filtered out by time / labels
    skipped_per_file_cap: int = 0    # per-file cap reached

    @property
    def total_skipped(self) -> int:
        return (
            self.skipped_malformed
            + self.skipped_extraction
            + self.skipped_unrouted
            + self.skipped_normalization
            + self.skipped_filter
            + self.skipped_per_file_cap
        )

    def format_text(self) -> str:
        """Multi-line printable summary; only nonzero categories are listed."""
        lines = [
            "Read summary:",
            f"  Files scanned:          {self.files_scanned:,}",
            f"  Records emitted:        {self.records_emitted:,}",
        ]
        if self.skipped_malformed:
            lines.append(f"  Skipped (malformed):    {self.skipped_malformed:,}")
        if self.skipped_extraction:
            lines.append(f"  Skipped (no text):      {self.skipped_extraction:,}")
        if self.skipped_unrouted:
            lines.append(f"  Skipped (unrouted):     {self.skipped_unrouted:,}")
        if self.skipped_normalization:
            lines.append(f"  Skipped (length):       {self.skipped_normalization:,}")
        if self.skipped_filter:
            lines.append(f"  Skipped (filter):       {self.skipped_filter:,}")
        if self.skipped_per_file_cap:
            lines.append(f"  Skipped (per_file_cap): {self.skipped_per_file_cap:,}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal pipeline carrier
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PipelineRecord:
    """Internal: an Item plus the metadata used by filtering and sampling."""

    item: Item
    time: Optional[datetime] = None
    group: Optional[str] = None

    def replace_text(self, text: str) -> "_PipelineRecord":
        return _PipelineRecord(
            item=replace(self.item, text=text),
            time=self.time,
            group=self.group,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_corpus(
    config: FormatConfig,
    *,
    strict: bool = False,
    summary: Optional[ReadSummary] = None,
    progress: bool = True,
) -> Iterator[Item]:
    """Yield :class:`Item` objects for every record described by ``config``.

    Pipeline order is documented at the module top. ``Item`` is the only
    type that escapes; ``_PipelineRecord`` (with ``time`` and ``group``)
    is internal.

    Parameters
    ----------
    strict
        When True, per-record errors raise :class:`CorpusReadError` instead
        of being silently skipped.
    summary
        Optional :class:`ReadSummary` to populate during iteration.
    progress
        Print progress to stderr (records read, files processed). Default True.
    """
    parser_type = config.parser.type
    if parser_type not in {"jsonl", "text", "csv", "tsv"}:
        raise NotImplementedError(
            f"Format {parser_type!r} is not in this slice yet."
        )

    if summary is None:
        summary = ReadSummary()

    files = list_files(
        config.storage,
        default_includes=DEFAULT_INCLUDES.get(parser_type),
    )
    summary.files_scanned = len(files)
    if not files:
        return

    if progress:
        total_size = sum(f.stat().st_size for f in files)
        print(
            f"  Reading {len(files)} files ({total_size / (1024*1024):.0f} MB)...",
            file=sys.stderr, flush=True,
        )

    base = _resolve_base_dir(config.storage.path)

    # Stage 1: format-specific iterator -> _PipelineRecord
    if parser_type == "jsonl":
        records: Iterator[_PipelineRecord] = _iter_jsonl(
            config, files, base, strict=strict, summary=summary, progress=progress
        )
    elif parser_type in {"csv", "tsv"}:
        records = _iter_tabular(
            config, files, base, strict=strict, summary=summary, progress=progress
        )
    else:
        records = _iter_text(
            config, files, base, strict=strict, summary=summary
        )

    # Stage 2: normalization (drops if too short after trim)
    if config.normalize is not None:
        normalize = _build_normalizer(config.normalize)
        records = _apply_normalization(records, normalize, strict, summary)

    # Stage 3: filtering
    if config.filters is not None:
        predicate = _build_filter_predicate(config.filters)
        records = _apply_filter(records, predicate, summary)

    # Stage 4: per-file cap (orthogonal to sampling mode)
    if config.sampling is not None and config.sampling.per_file_cap is not None:
        records = _apply_per_file_cap(
            records, config.sampling.per_file_cap, summary
        )

    # Stage 5: sampling
    sampled = apply_sampling(records, config.sampling)

    for record in sampled:
        summary.records_emitted += 1
        yield record.item


# ---------------------------------------------------------------------------
# Format-specific iterators (Stage 1)
# ---------------------------------------------------------------------------


def _iter_jsonl(
    config: FormatConfig,
    files: list[Path],
    base: Path,
    *,
    strict: bool,
    summary: ReadSummary,
    progress: bool = False,
) -> Iterator[_PipelineRecord]:
    text_block: TextBlock = config.text
    extract_text = _build_text_extractor(text_block)
    extract_label = _build_label_extractor(config.labels)
    extract_time = _build_time_extractor(config.filters, strict=strict)
    extract_group = _build_group_extractor(config.sampling)
    id_field = config.id_block.field if config.id_block else None
    is_routed = bool(text_block.router_field)

    records_seen = 0
    for file_idx, path in enumerate(files):
        rel = _relative_to_base(path, base)
        if progress:
            print(
                f"  [{file_idx+1}/{len(files)}] {rel} ...",
                file=sys.stderr, end="", flush=True,
            )
        file_records = 0
        for line_no, raw in enumerate(
            iter_lines(path, encoding=config.parser.encoding), start=1
        ):
            if not raw.strip():
                continue
            try:
                record = orjson.loads(raw)
            except orjson.JSONDecodeError as exc:
                summary.skipped_malformed += 1
                if strict:
                    raise CorpusReadError(
                        f"malformed JSON: {exc}", source_file=rel, line_no=line_no
                    ) from exc
                continue
            if not isinstance(record, dict):
                summary.skipped_malformed += 1
                if strict:
                    raise CorpusReadError(
                        "JSONL record is not an object",
                        source_file=rel,
                        line_no=line_no,
                    )
                continue

            text = extract_text(record)
            if text is None:
                if is_routed:
                    summary.skipped_unrouted += 1
                else:
                    summary.skipped_extraction += 1
                if strict:
                    raise CorpusReadError(
                        "no text field matched"
                        if not is_routed
                        else "router value did not match any route and no _default set",
                        source_file=rel,
                        line_no=line_no,
                    )
                continue

            records_seen += 1
            file_records += 1
            if progress and records_seen % 100000 == 0:
                print(f" {records_seen // 1000}K...", file=sys.stderr, end="", flush=True)

            yield _PipelineRecord(
                item=Item(
                    id=_resolve_id(record, id_field, rel, line_no),
                    text=text,
                    source_file=rel,
                    label=extract_label(record),
                ),
                time=extract_time(record, rel, line_no),
                group=extract_group(record),
            )

        if progress:
            print(f" {file_records:,} records", file=sys.stderr, flush=True)


def _iter_tabular(
    config: FormatConfig,
    files: list[Path],
    base: Path,
    *,
    strict: bool,
    summary: ReadSummary,
    progress: bool = False,
) -> Iterator[_PipelineRecord]:
    try:
        import polars as pl
    except ImportError:
        raise ImportError(
            "CSV/TSV format requires 'polars'. Install with: pip install ignite-tools[formats]"
        )

    text_block: TextBlock = config.text
    extract_text = _build_text_extractor(text_block)
    extract_label = _build_label_extractor(config.labels)
    extract_time = _build_time_extractor(config.filters, strict=strict)
    extract_group = _build_group_extractor(config.sampling)
    id_field = config.id_block.field if config.id_block else None
    is_routed = bool(text_block.router_field)

    parser = config.parser
    separator = parser.effective_delimiter()
    has_header = parser.has_header
    quote_char = parser.quote
    encoding = _polars_encoding(parser.encoding)

    for path in files:
        rel = _relative_to_base(path, base)
        try:
            lazy = pl.scan_csv(
                path,
                separator=separator,
                has_header=has_header,
                quote_char=quote_char,
                encoding=encoding,
                infer_schema_length=0,
            )
            row_no = 0
            for batch in lazy.collect_batches(chunk_size=_TABULAR_CHUNK_SIZE):
                for record in batch.iter_rows(named=True):
                    row_no += 1
                    text = extract_text(record)
                    if text is None:
                        if is_routed:
                            summary.skipped_unrouted += 1
                        else:
                            summary.skipped_extraction += 1
                        if strict:
                            raise CorpusReadError(
                                "no text column matched"
                                if not is_routed
                                else "router value did not match any route and no _default set",
                                source_file=rel,
                                line_no=row_no,
                            )
                        continue
                    yield _PipelineRecord(
                        item=Item(
                            id=_resolve_id(record, id_field, rel, row_no),
                            text=text,
                            source_file=rel,
                            label=extract_label(record),
                        ),
                        time=extract_time(record, rel, row_no),
                        group=extract_group(record),
                    )
        except CorpusReadError:
            raise
        except Exception as exc:
            summary.skipped_malformed += 1
            if progress:
                print(f" FAILED ({exc})", file=sys.stderr, flush=True)
            if strict:
                raise CorpusReadError(
                    f"CSV parse error: {exc}", source_file=rel
                ) from exc
            continue


def _iter_text(
    config: FormatConfig,
    files: list[Path],
    base: Path,
    *,
    strict: bool,
    summary: ReadSummary,
) -> Iterator[_PipelineRecord]:
    """Plain text - no record dict, so labels / time / group never apply."""
    encoding = config.parser.encoding

    if config.parser.unit == "file":
        for path in files:
            rel = _relative_to_base(path, base)
            content = read_file_text(path, encoding=encoding).strip()
            if not content:
                summary.skipped_extraction += 1
                if strict:
                    raise CorpusReadError(
                        "file is empty after strip", source_file=rel
                    )
                continue
            yield _PipelineRecord(
                item=Item(id=rel, text=content, source_file=rel)
            )
        return

    # unit == "line"
    for path in files:
        rel = _relative_to_base(path, base)
        for line_no, raw in enumerate(iter_lines(path, encoding=encoding), start=1):
            if not raw.strip():
                continue
            try:
                text = raw.decode(encoding)
            except UnicodeDecodeError as exc:
                summary.skipped_malformed += 1
                if strict:
                    raise CorpusReadError(
                        f"decode error: {exc}",
                        source_file=rel,
                        line_no=line_no,
                    ) from exc
                continue
            text = text.strip()
            if not text:
                continue
            yield _PipelineRecord(
                item=Item(id=f"{rel}:{line_no}", text=text, source_file=rel)
            )


# ---------------------------------------------------------------------------
# Stage 2: normalization
# ---------------------------------------------------------------------------


def _build_normalizer(block: NormalizeBlock) -> Callable[[str], Optional[str]]:
    """Return a function that applies ``normalize`` rules in fixed order.

    Returns ``None`` when ``trim.min_chars`` drops the text after processing.
    """
    lowercase = block.lowercase
    collapse = block.collapse_whitespace
    strip = block.strip
    masks: list[tuple[re.Pattern[str], str]] = (
        [(re.compile(m.pattern), m.replacement) for m in (block.masks or [])]
    )
    trim_max = block.trim.max_chars if block.trim else None
    trim_min = block.trim.min_chars if block.trim else None
    whitespace_re = re.compile(r"\s+")

    def normalize(text: str) -> Optional[str]:
        if lowercase:
            text = text.lower()
        for pattern, replacement in masks:
            text = pattern.sub(replacement, text)
        if collapse:
            text = whitespace_re.sub(" ", text)
        if strip:
            text = text.strip()
        if trim_max is not None and len(text) > trim_max:
            text = text[:trim_max]
        if trim_min is not None and len(text) < trim_min:
            return None
        return text

    return normalize


def _apply_normalization(
    records: Iterator[_PipelineRecord],
    normalize: Callable[[str], Optional[str]],
    strict: bool,
    summary: ReadSummary,
) -> Iterator[_PipelineRecord]:
    for record in records:
        new_text = normalize(record.item.text)
        if new_text is None:
            summary.skipped_normalization += 1
            if strict:
                raise CorpusReadError(
                    "text below normalize.trim.min_chars after normalization",
                    source_file=record.item.source_file,
                )
            continue
        if new_text == record.item.text:
            yield record
        else:
            yield record.replace_text(new_text)


# ---------------------------------------------------------------------------
# Stage 3: filtering
# ---------------------------------------------------------------------------


def _build_filter_predicate(
    block: FiltersBlock,
) -> Callable[[_PipelineRecord], bool]:
    """Compose label and time predicates into a single predicate."""
    predicates: list[Callable[[_PipelineRecord], bool]] = []

    if block.labels_include:
        included = set(block.labels_include)
        predicates.append(lambda r: r.item.label in included)
    if block.labels_exclude:
        excluded = set(block.labels_exclude)
        predicates.append(lambda r: r.item.label not in excluded)

    if block.time_from is not None or block.time_to is not None:
        time_from = strip_tz(block.time_from) if block.time_from else None
        time_to = strip_tz(block.time_to) if block.time_to else None

        def time_predicate(r: _PipelineRecord) -> bool:
            if r.time is None:
                return False
            t = strip_tz(r.time)
            if time_from is not None and t < time_from:
                return False
            if time_to is not None and t > time_to:
                return False
            return True

        predicates.append(time_predicate)

    if not predicates:
        return lambda r: True

    def combined(record: _PipelineRecord) -> bool:
        return all(p(record) for p in predicates)

    return combined


def _apply_filter(
    records: Iterator[_PipelineRecord],
    predicate: Callable[[_PipelineRecord], bool],
    summary: ReadSummary,
) -> Iterator[_PipelineRecord]:
    for record in records:
        if predicate(record):
            yield record
        else:
            summary.skipped_filter += 1


# ---------------------------------------------------------------------------
# Stage 4: per-file cap
# ---------------------------------------------------------------------------


def _apply_per_file_cap(
    records: Iterator[_PipelineRecord], cap: int, summary: ReadSummary
) -> Iterator[_PipelineRecord]:
    """Emit at most ``cap`` records per ``source_file``."""
    counts: dict[str, int] = {}
    for record in records:
        key = record.item.source_file
        n = counts.get(key, 0)
        if n >= cap:
            summary.skipped_per_file_cap += 1
            continue
        counts[key] = n + 1
        yield record


# ---------------------------------------------------------------------------
# Extractor builders (per-config-block factories)
# ---------------------------------------------------------------------------


def _build_text_extractor(text_block: TextBlock) -> Callable[[dict], Optional[str]]:
    """Return a callable ``record -> Optional[str]`` for the configured form."""
    if text_block.fields:
        fields = text_block.fields
        return lambda record: _extract_first_nonempty(record, fields)

    router_field = text_block.router_field
    routes: dict[str, RouteBlock] = text_block.routes or {}
    default_route = routes.get(ROUTE_DEFAULT_KEY)

    def extract_routed(record: dict) -> Optional[str]:
        router_value = _get_dotted(record, router_field)
        if isinstance(router_value, str):
            route = routes.get(router_value)
            if route is not None:
                return _extract_first_nonempty(record, route.fields)
        if default_route is not None:
            return _extract_first_nonempty(record, default_route.fields)
        return None

    return extract_routed


def _build_label_extractor(
    block: Optional[LabelsBlock],
) -> Callable[[dict], Optional[str]]:
    if block is None:
        return lambda record: None
    field = block.field

    def extract(record: dict) -> Optional[str]:
        value = _get_dotted(record, field)
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return str(value)
        return None

    return extract


def _build_time_extractor(
    block: Optional[FiltersBlock],
    *,
    strict: bool = False,
) -> Callable[[dict, Optional[str], Optional[int]], Optional[datetime]]:
    if block is None or not block.time_field:
        return lambda record, source_file=None, line_no=None: None
    field = block.time_field

    def extract(
        record: dict, source_file: Optional[str] = None, line_no: Optional[int] = None
    ) -> Optional[datetime]:
        value = _get_dotted(record, field)
        parsed = parse_time(value)
        if strict and value is not None and parsed is None:
            raise CorpusReadError(
                f"invalid timestamp in field {field!r}: {value!r}",
                source_file=source_file,
                line_no=line_no,
            )
        return parsed

    return extract


def _build_group_extractor(
    block: Optional[SamplingBlock],
) -> Callable[[dict], Optional[str]]:
    if block is None or not block.group_field:
        return lambda record: None
    field = block.group_field

    def extract(record: dict) -> Optional[str]:
        value = _get_dotted(record, field)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, (int, float)):
            return str(value)
        return None

    return extract


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolve_base_dir(path_str: str) -> Path:
    """Return the directory used to compute relative paths in Item.source_file."""
    p = Path(path_str).expanduser()
    return p if p.is_dir() else p.parent


def _relative_to_base(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _extract_first_nonempty(record: dict, fields: list[str]) -> Optional[str]:
    """Try each dotted field in order; return the first non-empty string."""
    for field in fields:
        value = _get_dotted(record, field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _resolve_id(
    record: dict, id_field: Optional[str], rel_path: str, line_no: int
) -> str:
    if id_field:
        value = _get_dotted(record, id_field)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, (int, float)):
            return str(value)
    return f"{rel_path}:{line_no}"


def _get_dotted(obj: Any, path: str) -> Any:
    """Resolve ``a.b.c`` against a nested dict; returns None if any step misses."""
    cur: Any = obj
    for seg in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(seg)
        if cur is None:
            return None
    return cur


def parse_time(value: Any) -> Optional[datetime]:
    """Parse ISO 8601 strings and Unix timestamps into datetimes."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    return None


def _polars_encoding(encoding: str) -> str:
    """Translate a Python-canonical encoding name to what polars accepts."""
    normalized = encoding.lower().replace("_", "-")
    if normalized in {"utf-8", "utf8"}:
        return "utf8"
    if normalized in {"utf-8-lossy", "utf8-lossy"}:
        return "utf8-lossy"
    raise ValueError(
        f"CSV / TSV reader supports only 'utf-8' or 'utf-8-lossy' encoding; "
        f"got {encoding!r}"
    )
