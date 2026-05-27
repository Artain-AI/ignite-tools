"""
Auto-detect format and propose an ``ignite-format.yaml``.

Used when the user runs a tool against a path with no config saved. The
sniffer looks at the data, prints what it sees, and produces a proposed
config. The CLI layer wraps this with an interactive ``[y/n/save/edit]``
prompt; this module is callable in isolation for tests and scripting.

Heuristics:
- Format: by extension first, then by content sniff if ambiguous.
- JSONL key coverage: parse first ~100 records, walk all keys (including
  dotted paths into nested dicts), report % of records each key appears in.
- CSV / TSV: read first ~100 rows; pick the longest-avg-cell column as the
  text candidate, low-cardinality columns as label / router candidates.
- Plain text: ``unit: line`` by default.
- Sniffer is conservative - it annotates guesses as such so the user can
  spot heuristics easily.

Out of scope here, by design:
- Cloud sources (the cloud download pattern is in sources.py; sniffer
  works on already-resolved local paths).
- The interactive ``[y/n/save/edit]`` prompt - that's CLI concern.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import orjson

from ignite_tools.core.config import FormatConfig
from ignite_tools.core.sources import iter_lines

# How much to look at while sniffing. 100 records OR 1 MB, whichever first.
_MAX_SNIFF_RECORDS = 100
_MAX_SNIFF_BYTES = 1 * 1024 * 1024
# Minimum coverage % for a JSONL key to be considered a "primary" candidate
# (we still surface lower-coverage keys, just not as the chosen text field).
_PRIMARY_COVERAGE = 0.5
# A field is router-candidate-shaped if it has between this many distinct
# values (low cardinality, categorical-looking).
_ROUTER_MIN_CARDINALITY = 2
_ROUTER_MAX_CARDINALITY = 20
# A router candidate must also be repeating, not unique-per-record. We require
# the distinct-values count to be a small fraction of the records inspected
# so that fields like `id` (where every value is unique) are excluded.
_ROUTER_MAX_CARDINALITY_RATIO = 0.5
# An id candidate must have *high* cardinality (most values distinct) - the
# inverse of the router rule.
_ID_MIN_CARDINALITY_RATIO = 0.8


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class SniffResult:
    """What the sniffer learned about a path.

    Carries enough information to (a) print a human-readable summary, and
    (b) build a proposed :class:`FormatConfig` via :func:`build_proposal`.
    """

    storage_type: str = "local"
    path: str = ""
    file_count: int = 0
    total_bytes: int = 0
    compression: Optional[str] = None  # "gzip", "zstd", "mixed", None
    format_type: str = "jsonl"  # jsonl | csv | tsv | text
    encoding: str = "utf-8"

    # Format-specific findings:
    jsonl_key_coverage: dict[str, float] = field(default_factory=dict)  # path -> 0.0..1.0
    jsonl_value_samples: dict[str, list[str]] = field(default_factory=dict)
    csv_columns: list[str] = field(default_factory=list)
    csv_text_guess: Optional[str] = None
    csv_label_guess: Optional[str] = None
    csv_router_guess: Optional[str] = None
    csv_router_values: list[str] = field(default_factory=list)

    # Aggregate guesses:
    text_field_guess: Optional[str] = None  # for jsonl simple form
    router_field_guess: Optional[str] = None
    router_values: list[str] = field(default_factory=list)
    per_route_text_fields: dict[str, list[str]] = field(default_factory=dict)
    label_field_guess: Optional[str] = None
    id_field_guess: Optional[str] = None

    sample_records_inspected: int = 0


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


def sniff_path(path: str | Path) -> SniffResult:
    """Inspect a local path and return a :class:`SniffResult`.

    The path may point at a single file or a directory. Sniffing only walks
    files matching common data extensions (jsonl/ndjson/csv/tsv/txt/md plus
    .gz/.zst variants). Cloud paths should be resolved to local cache by
    the caller before sniffing.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Sniffer path does not exist: {p}")

    files = _candidate_files(p)
    result = SniffResult(
        storage_type="local",
        path=str(p),
        file_count=len(files),
        total_bytes=sum(f.stat().st_size for f in files),
        compression=_detect_compression(files),
        format_type=_detect_format_type(files),
    )

    if not files:
        return result

    if result.format_type == "jsonl":
        _sniff_jsonl(files, result)
    elif result.format_type in {"csv", "tsv"}:
        _sniff_tabular(files, result)
    # text format: no per-record sniffing needed; defaults work as-is.

    return result


def build_proposal(result: SniffResult) -> dict:
    """Build a YAML-ready dict from a :class:`SniffResult`.

    The dict mirrors the ``ignite-format.yaml`` schema so the caller can
    either feed it to :class:`FormatConfig.from_dict` to validate or write
    it to disk via ``yaml.safe_dump``.
    """
    proposal: dict = {
        "storage": {
            "type": result.storage_type,
            "path": result.path,
            "recursive": True,
        },
        "format": {"type": result.format_type},
    }

    if result.format_type == "jsonl":
        if result.router_field_guess and len(result.router_values) >= _ROUTER_MIN_CARDINALITY:
            # Routed proposal: use per-route text fields if detected,
            # otherwise fall back to the global text_field_guess for all routes.
            routes = {}
            default_fields = (
                [result.text_field_guess]
                if result.text_field_guess
                else _default_text_field_candidates(result)
            )
            for value in result.router_values:
                route_fields = result.per_route_text_fields.get(value)
                if route_fields:
                    routes[value] = {"fields": route_fields}
                else:
                    routes[value] = {"fields": default_fields}
            proposal["text"] = {
                "router_field": result.router_field_guess,
                "routes": routes,
            }
        elif result.text_field_guess:
            proposal["text"] = {"fields": [result.text_field_guess]}
        else:
            proposal["text"] = {"fields": _default_text_field_candidates(result)}

        if result.label_field_guess:
            proposal["labels"] = {"field": result.label_field_guess}
        if result.id_field_guess:
            proposal["id"] = {"field": result.id_field_guess}

    elif result.format_type in {"csv", "tsv"}:
        if result.csv_text_guess:
            proposal["text"] = {"fields": [result.csv_text_guess]}
        if result.csv_label_guess:
            proposal["labels"] = {"field": result.csv_label_guess}

    return proposal


def format_human_summary(result: SniffResult) -> str:
    """Multi-line printable summary of what the sniffer found."""
    lines = [
        "Detected:",
        f"  Storage:     {result.storage_type} ({result.file_count} files, "
        f"{_format_bytes(result.total_bytes)})",
    ]
    if result.compression:
        lines.append(f"  Compression: {result.compression}")
    lines.append(
        f"  Format:      {result.format_type} (sniffed first "
        f"{result.sample_records_inspected:,} records)"
    )

    if result.format_type == "jsonl":
        if result.jsonl_key_coverage:
            lines.append("\nField coverage (top-level + nested, dotted):")
            shown = sorted(
                result.jsonl_key_coverage.items(),
                key=lambda kv: (-kv[1], kv[0]),
            )[:20]
            for key, cov in shown:
                lines.append(f"  {key}: {cov * 100:.0f}%")
        if result.router_field_guess:
            lines.append(
                f"\nRouter field guess: {result.router_field_guess} "
                f"({len(result.router_values)} distinct values: "
                f"{', '.join(result.router_values[:5])})"
            )
        if result.text_field_guess:
            lines.append(f"Text field guess:   {result.text_field_guess}")
        if result.label_field_guess:
            lines.append(f"Label field guess:  {result.label_field_guess}")
        if result.id_field_guess:
            lines.append(f"Id field guess:     {result.id_field_guess}")
    elif result.format_type in {"csv", "tsv"}:
        lines.append(f"\nColumns: {', '.join(result.csv_columns)}")
        if result.csv_text_guess:
            lines.append(f"Text column guess:  {result.csv_text_guess}")
        if result.csv_label_guess:
            lines.append(f"Label column guess: {result.csv_label_guess}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File enumeration + format detection
# ---------------------------------------------------------------------------


_DATA_SUFFIXES = (
    ".jsonl",
    ".ndjson",
    ".jsonl.gz",
    ".ndjson.gz",
    ".jsonl.zst",
    ".ndjson.zst",
    ".csv",
    ".csv.gz",
    ".csv.zst",
    ".tsv",
    ".tsv.gz",
    ".tsv.zst",
    ".txt",
    ".md",
)


def _candidate_files(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    out: list[Path] = []
    for child in p.rglob("*"):
        if not child.is_file():
            continue
        name = child.name.lower()
        if any(name.endswith(suf) for suf in _DATA_SUFFIXES):
            out.append(child)
    return sorted(out)


def _detect_compression(files: list[Path]) -> Optional[str]:
    seen = set()
    for f in files:
        name = f.name.lower()
        if name.endswith(".gz"):
            seen.add("gzip")
        elif name.endswith(".zst"):
            seen.add("zstd")
        else:
            seen.add("none")
    seen -= {"none"}
    if not seen:
        return None
    if len(seen) == 1:
        return next(iter(seen))
    return "mixed"


def _detect_format_type(files: list[Path]) -> str:
    """Pick the dominant data type across the candidate files.

    Tie-breaker order matches user expectation: jsonl > csv > tsv > text.
    Mixed extensions fall back to text (least restrictive); the user can
    override with an explicit ``--format-config``.
    """
    counts = Counter()
    for f in files:
        name = f.name.lower()
        if any(name.endswith(s) for s in (".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz", ".jsonl.zst", ".ndjson.zst")):
            counts["jsonl"] += 1
        elif any(name.endswith(s) for s in (".csv", ".csv.gz", ".csv.zst")):
            counts["csv"] += 1
        elif any(name.endswith(s) for s in (".tsv", ".tsv.gz", ".tsv.zst")):
            counts["tsv"] += 1
        elif any(name.endswith(s) for s in (".txt", ".md")):
            counts["text"] += 1
    if not counts:
        return "text"
    # Single dominant type.
    most_common, _ = counts.most_common(1)[0]
    return most_common


# ---------------------------------------------------------------------------
# JSONL sniffing
# ---------------------------------------------------------------------------


def _sniff_jsonl(files: list[Path], result: SniffResult) -> None:
    seen = 0
    bytes_seen = 0
    key_counts: Counter[str] = Counter()
    value_samples: dict[str, list[str]] = {}
    distinct_values: dict[str, set[str]] = {}
    # Full distinct count (uncapped) - needed for ID detection where high
    # cardinality is the signal. The capped set above is for router detection.
    distinct_count: Counter[str] = Counter()
    # Retain parsed records so we can do per-route analysis after the router
    # field is identified.
    sampled_records: list[dict] = []

    for path in files:
        for raw in iter_lines(path):
            if seen >= _MAX_SNIFF_RECORDS or bytes_seen >= _MAX_SNIFF_BYTES:
                break
            if not raw.strip():
                continue
            bytes_seen += len(raw)
            try:
                record = orjson.loads(raw)
            except orjson.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            seen += 1
            sampled_records.append(record)
            for path_key, value in _walk_dotted(record):
                key_counts[path_key] += 1
                if isinstance(value, str):
                    samples = value_samples.setdefault(path_key, [])
                    if len(samples) < 5:
                        samples.append(value)
                    distinct_count[path_key] += 1  # count every string occurrence
                    bucket = distinct_values.setdefault(path_key, set())
                    if len(bucket) < _ROUTER_MAX_CARDINALITY + 1:
                        bucket.add(value)
                    else:
                        # Bucket full but value might be new - track that we
                        # saw more distinct values than the bucket holds.
                        pass
        if seen >= _MAX_SNIFF_RECORDS or bytes_seen >= _MAX_SNIFF_BYTES:
            break

    result.sample_records_inspected = seen
    if seen == 0:
        return

    result.jsonl_key_coverage = {k: v / seen for k, v in key_counts.items()}
    result.jsonl_value_samples = value_samples

    # Pick a router-field candidate: a field with low-cardinality string
    # values present in most records, AND whose values repeat (cardinality
    # is small relative to records inspected - otherwise it's an id, not
    # a category).
    router_candidates: list[tuple[str, int]] = []
    for key, cov in result.jsonl_key_coverage.items():
        if cov < _PRIMARY_COVERAGE:
            continue
        bucket = distinct_values.get(key, set())
        if not (_ROUTER_MIN_CARDINALITY <= len(bucket) <= _ROUTER_MAX_CARDINALITY):
            continue
        cardinality_ratio = len(bucket) / max(1, seen)
        if cardinality_ratio > _ROUTER_MAX_CARDINALITY_RATIO:
            continue
        router_candidates.append((key, len(bucket)))
    if router_candidates:
        router_candidates.sort(key=lambda kv: (kv[1], kv[0]))
        chosen = router_candidates[0][0]
        result.router_field_guess = chosen
        result.router_values = sorted(distinct_values.get(chosen, set()))
        # Routes share a label field by convention.
        result.label_field_guess = chosen

    # Id candidate: detected BEFORE text candidates so it can be excluded
    # from the text-field search. A high-coverage field whose name looks
    # like an id and whose values are mostly distinct.
    for key, cov in result.jsonl_key_coverage.items():
        if cov < 0.95:
            continue
        if key == result.router_field_guess:
            continue
        last_segment = key.split(".")[-1]
        if last_segment not in {"id", "uid", "uuid", "_id"}:
            continue
        bucket = distinct_values.get(key, set())
        samples = value_samples.get(key, [])
        if not samples:
            result.id_field_guess = key
            break
        if len(bucket) > _ROUTER_MAX_CARDINALITY:
            result.id_field_guess = key
            break
        cardinality_ratio = len(bucket) / max(1, seen)
        if cardinality_ratio >= _ID_MIN_CARDINALITY_RATIO:
            result.id_field_guess = key
            break

    # Text candidate: highest-coverage string-valued key whose typical value
    # is reasonably long. Skip the router field, the id field, and fields
    # that look like timestamps.
    text_candidates: list[tuple[str, float, float]] = []
    for key, cov in result.jsonl_key_coverage.items():
        if key == result.router_field_guess:
            continue
        if key == result.id_field_guess:
            continue
        if cov < _PRIMARY_COVERAGE:
            continue
        samples = value_samples.get(key, [])
        if not samples:
            continue
        avg_len = sum(len(s) for s in samples) / len(samples)
        if avg_len < 4:  # filter out short ids / categorical strings
            continue
        # Skip fields whose values look like timestamps (ISO 8601).
        if _looks_like_timestamps(samples):
            continue
        text_candidates.append((key, cov, avg_len))
    if text_candidates:
        # Prefer well-known field names, then highest avg_len, then coverage.
        well_known = {"text", "body", "content", "message"}
        text_candidates.sort(
            key=lambda x: (
                0 if x[0].split(".")[-1] in well_known else 1,
                -x[2],
                -x[1],
                x[0],
            )
        )
        result.text_field_guess = text_candidates[0][0]

    # Per-route text field detection: now that we know the router field, look
    # at which text-candidate fields actually have values per route. This way
    # each route gets its own field list instead of one global guess.
    #
    # We use ALL string-valued fields (not just those above _PRIMARY_COVERAGE
    # globally) because a field like `attributes.text` might only appear in
    # one source (33% global coverage) but be 100% within that source.
    if result.router_field_guess and sampled_records:
        excluded = {result.router_field_guess}
        if result.id_field_guess:
            excluded.add(result.id_field_guess)
        # Build a wider candidate pool: any string field with avg_len >= 4,
        # not excluded, not a timestamp. No global coverage threshold here.
        wide_candidates: list[tuple[str, float, float]] = []
        for key, cov in result.jsonl_key_coverage.items():
            if key in excluded:
                continue
            samples = value_samples.get(key, [])
            if not samples:
                continue
            avg_len = sum(len(s) for s in samples) / len(samples)
            if avg_len < 4:
                continue
            if _looks_like_timestamps(samples):
                continue
            wide_candidates.append((key, cov, avg_len))
        result.per_route_text_fields = _detect_per_route_fields(
            sampled_records,
            result.router_field_guess,
            result.router_values,
            wide_candidates,
        )


def _detect_per_route_fields(
    records: list[dict],
    router_field: str,
    router_values: list[str],
    text_candidates: list[tuple[str, float, float]],
) -> dict[str, list[str]]:
    """For each route value, find which text-candidate fields are present.

    Returns ``{route_value: [field1, field2, ...]}`` where fields are ordered
    by: well-known name first, then avg length, then coverage within that
    route's records. Only string-valued non-empty fields count.
    """
    well_known = {"text", "body", "content", "message"}
    candidate_keys = {t[0] for t in text_candidates}

    # Group records by their route value.
    per_route_records: dict[str, list[dict]] = {v: [] for v in router_values}
    for record in records:
        rv = _get_dotted_value(record, router_field)
        if isinstance(rv, str) and rv in per_route_records:
            per_route_records[rv].append(record)

    per_route_fields: dict[str, list[str]] = {}
    for route_value, route_records in per_route_records.items():
        if not route_records:
            continue
        # Score each candidate field by how often it has a non-empty value
        # in this route's records.
        field_scores: list[tuple[str, float, float]] = []
        for field_key in candidate_keys:
            hits = 0
            total_len = 0
            for rec in route_records:
                val = _get_dotted_value(rec, field_key)
                if isinstance(val, str) and val.strip():
                    hits += 1
                    total_len += len(val)
            if hits == 0:
                continue
            coverage = hits / len(route_records)
            avg_len = total_len / hits
            field_scores.append((field_key, coverage, avg_len))
        # Sort: well-known name, then coverage, then length.
        field_scores.sort(
            key=lambda x: (
                0 if x[0].split(".")[-1] in well_known else 1,
                -x[1],
                -x[2],
                x[0],
            )
        )
        per_route_fields[route_value] = [f[0] for f in field_scores]

    return per_route_fields


def _get_dotted_value(obj: Any, path: str) -> Any:
    """Resolve a dotted path like 'attributes.body' against a nested dict."""
    cur = obj
    for seg in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(seg)
        if cur is None:
            return None
    return cur


def _looks_like_timestamps(samples: list[str]) -> bool:
    """Heuristic: if most samples look like ISO 8601 strings, skip the field.

    Checks for the pattern ``YYYY-MM-DD`` or ``YYYY-MM-DDTHH:`` in at
    least 80% of samples.
    """
    if not samples:
        return False
    hits = 0
    for s in samples:
        s = s.strip()
        if len(s) >= 10 and s[4] == "-" and s[7] == "-":
            hits += 1
    return hits / len(samples) >= 0.8


# ---------------------------------------------------------------------------
# CSV / TSV sniffing
# ---------------------------------------------------------------------------


def _sniff_tabular(files: list[Path], result: SniffResult) -> None:
    """Sniff a CSV / TSV file using polars at small scale.

    We use a tiny scan with a fixed batch so we don't pay for full schema
    inference. polars handles compression for us.
    """
    try:
        import polars as pl
    except ImportError:
        # polars not installed - can't sniff CSV columns.
        return

    if not files:
        return

    sep = "\t" if result.format_type == "tsv" else ","
    try:
        lazy = pl.scan_csv(
            files[0],
            separator=sep,
            has_header=True,
            infer_schema_length=0,
            n_rows=_MAX_SNIFF_RECORDS,
        )
        df = lazy.collect()
    except Exception:
        return

    result.csv_columns = list(df.columns)
    if df.height == 0:
        return
    result.sample_records_inspected = df.height

    # Text guess: longest average cell length.
    longest_avg = -1.0
    text_guess = None
    for col in df.columns:
        col_data = df.get_column(col).cast(pl.Utf8, strict=False)
        non_null = col_data.drop_nulls().to_list()
        if not non_null:
            continue
        avg_len = sum(len(s) for s in non_null) / len(non_null)
        if avg_len > longest_avg:
            longest_avg = avg_len
            text_guess = col
    result.csv_text_guess = text_guess

    # Label guess: low-cardinality non-text column.
    label_guess = None
    router_guess = None
    smallest_card = None
    for col in df.columns:
        if col == text_guess:
            continue
        col_data = df.get_column(col).cast(pl.Utf8, strict=False).drop_nulls().to_list()
        distinct = set(col_data)
        if not (_ROUTER_MIN_CARDINALITY <= len(distinct) <= _ROUTER_MAX_CARDINALITY):
            continue
        if smallest_card is None or len(distinct) < smallest_card:
            smallest_card = len(distinct)
            label_guess = col
            router_guess = col
    result.csv_label_guess = label_guess
    result.csv_router_guess = router_guess
    if router_guess:
        col_data = df.get_column(router_guess).cast(pl.Utf8, strict=False).drop_nulls().to_list()
        result.csv_router_values = sorted(set(col_data))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _walk_dotted(obj: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield ``(dotted_path, value)`` for every leaf-like field in ``obj``.

    A leaf is anything that isn't a dict; lists and other scalars are
    treated as values (their internal structure isn't recursed into for
    sniffing purposes).
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            sub = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                yield sub, v  # also report the dict-valued node
                yield from _walk_dotted(v, sub)
            else:
                yield sub, v


def _default_text_field_candidates(result: SniffResult) -> list[str]:
    """Reasonable fallback when no clear text field stood out."""
    well_known = ["text", "body", "content", "message"]
    seen = []
    for k in well_known:
        if k in result.jsonl_key_coverage:
            seen.append(k)
    if seen:
        return seen
    # Last-ditch: the highest-coverage string-valued key.
    candidates = sorted(
        result.jsonl_key_coverage.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )
    if candidates:
        return [candidates[0][0]]
    return ["text"]


def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1024
        if n < 1024:
            return f"{n:.1f} {unit}"
    return f"{n:.1f} PB"


# ---------------------------------------------------------------------------
# Convenience: validate a proposal by feeding it through FormatConfig
# ---------------------------------------------------------------------------


def proposal_to_config(proposal: dict) -> FormatConfig:
    """Validate a sniffer proposal by round-tripping through ``FormatConfig``.

    Useful for tests and callers that want to fail fast on a malformed
    proposal before trying to write it to disk.
    """
    return FormatConfig.from_dict(proposal)
