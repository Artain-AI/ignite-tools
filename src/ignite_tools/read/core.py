"""
ignite-read report engine.

Report structure (three sections):
  STANDARD - always shown, not configurable
    Configuration, Sources, Pipeline
  ANALYSIS - configurable sections, order from ReportConfig.sections
    corpus_stats, per_source, labels, top_words, patterns
  OUTPUT - always last
    Sample texts

The report is:
- Human-readable (terminal, 80-col, no color codes)
- Deterministic (same input + same config = same output)
- Inspectable (the Report dataclass is usable programmatically)
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ignite_tools.core.config import FormatConfig
from ignite_tools.core.format import Item, ReadSummary, read_corpus, parse_time
from ignite_tools.core.sources import DEFAULT_INCLUDES, list_files, iter_lines
from ignite_tools.read.config import ReportConfig


# Built-in stop words. Kept minimal - domain terms should pass through.
DEFAULT_STOP_WORDS = frozenset(
    "a an the and or but in on at to for of is it that this with from by "
    "be as are was were been has have had do does did will would can could "
    "should may might shall not no if so than then when what which who how "
    "all each every both few more most some any many much such very too also "
    "just about above after again between into through during before over "
    "i you he she we they me him her us them my your his its our their".split()
)


# ---------------------------------------------------------------------------
# Report data structures
# ---------------------------------------------------------------------------


@dataclass
class FileInfo:
    path: str
    size_bytes: int
    records: int = 0


@dataclass
class SourceStats:
    source_file: str
    record_count: int = 0
    avg_text_length: float = 0.0
    labels: dict[str, int] = field(default_factory=dict)


@dataclass
class PatternResult:
    name: str
    pattern: str
    matches: int = 0
    total: int = 0

    @property
    def pct(self) -> float:
        return (self.matches / self.total * 100) if self.total else 0.0


@dataclass
class Report:
    # Config
    config_path: Optional[str] = None
    storage_type: str = ""
    storage_path: str = ""
    format_type: str = ""
    text_mode: str = ""
    has_labels: bool = False
    has_normalize: bool = False
    has_filters: bool = False
    has_sampling: bool = False
    normalize_summary: str = ""
    filter_summary: str = ""
    sampling_summary: str = ""

    # Sources
    files: list[FileInfo] = field(default_factory=list)
    total_size_bytes: int = 0

    # Corpus stats
    items_emitted: int = 0
    text_lengths: list[int] = field(default_factory=list)
    label_distribution: dict[str, int] = field(default_factory=dict)
    source_file_distribution: dict[str, int] = field(default_factory=dict)
    per_source: list[SourceStats] = field(default_factory=list)

    # Time
    time_range_earliest: Optional[str] = None
    time_range_latest: Optional[str] = None
    time_distribution: dict[str, int] = field(default_factory=dict)

    # Languages (detected from character scripts)
    languages: dict[str, int] = field(default_factory=dict)  # script_name -> record_count

    # Content
    top_words: list[tuple[str, int]] = field(default_factory=list)
    pattern_results: list[PatternResult] = field(default_factory=list)

    # Pipeline
    summary: ReadSummary = field(default_factory=ReadSummary)

    # Sample
    sample_items: list[Item] = field(default_factory=list)

    @property
    def text_len_min(self) -> int:
        return min(self.text_lengths) if self.text_lengths else 0

    @property
    def text_len_max(self) -> int:
        return max(self.text_lengths) if self.text_lengths else 0

    @property
    def text_len_avg(self) -> float:
        return statistics.mean(self.text_lengths) if self.text_lengths else 0.0

    @property
    def text_len_median(self) -> float:
        return statistics.median(self.text_lengths) if self.text_lengths else 0.0


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def run_report(
    config: FormatConfig,
    *,
    report_config: Optional[ReportConfig] = None,
    strict: bool = False,
    config_path: Optional[str] = None,
) -> Report:
    """Execute the full read pipeline and build a :class:`Report`."""
    if report_config is None:
        report_config = ReportConfig.defaults()

    report = Report(config_path=config_path)

    # ── Config summary ───────────────────────────────────────────────────
    report.storage_type = config.storage.type
    report.storage_path = config.storage.path
    report.format_type = config.parser.type

    if config.text:
        if config.text.fields:
            report.text_mode = f"simple: {config.text.fields}"
        elif config.text.router_field:
            n_routes = len(config.text.routes) if config.text.routes else 0
            report.text_mode = f"routed via {config.text.router_field} ({n_routes} routes)"
    else:
        report.text_mode = "n/a (plain text)"

    report.has_labels = config.labels is not None
    report.has_normalize = config.normalize is not None
    report.has_filters = config.filters is not None
    report.has_sampling = config.sampling is not None

    if config.normalize:
        parts = []
        if config.normalize.lowercase:
            parts.append("lowercase")
        if config.normalize.masks:
            parts.append(f"{len(config.normalize.masks)} masks")
        if config.normalize.collapse_whitespace:
            parts.append("collapse")
        if config.normalize.strip:
            parts.append("strip")
        if config.normalize.trim:
            t = config.normalize.trim
            bounds = []
            if t.min_chars is not None:
                bounds.append(f"min={t.min_chars}")
            if t.max_chars is not None:
                bounds.append(f"max={t.max_chars}")
            parts.append(f"trim[{','.join(bounds)}]")
        report.normalize_summary = " + ".join(parts)

    if config.filters:
        parts = []
        if config.filters.time_field:
            fr = config.filters.time_from.isoformat() if config.filters.time_from else "..."
            to = config.filters.time_to.isoformat() if config.filters.time_to else "..."
            parts.append(f"time {fr}..{to}")
        if config.filters.labels_include:
            parts.append(f"labels_include={config.filters.labels_include}")
        if config.filters.labels_exclude:
            parts.append(f"labels_exclude={config.filters.labels_exclude}")
        report.filter_summary = ", ".join(parts)

    if config.sampling:
        s = config.sampling
        parts = [f"mode={s.mode}"]
        if s.total:
            parts.append(f"total={s.total}")
        if s.per_group:
            parts.append(f"per_group={s.per_group}")
        if s.group_field:
            parts.append(f"group={s.group_field}")
        if s.per_file_cap:
            parts.append(f"per_file_cap={s.per_file_cap}")
        parts.append(f"seed={s.seed}")
        report.sampling_summary = ", ".join(parts)

    # ── Source stats ─────────────────────────────────────────────────────
    try:
        files = list_files(
            config.storage,
            default_includes=DEFAULT_INCLUDES.get(config.parser.type),
        )
    except FileNotFoundError:
        files = []

    for f in files:
        report.files.append(FileInfo(
            path=str(f.relative_to(Path(config.storage.path).expanduser()))
            if f.is_relative_to(Path(config.storage.path).expanduser())
            else str(f),
            size_bytes=f.stat().st_size,
        ))
    report.total_size_bytes = sum(fi.size_bytes for fi in report.files)

    # ── Time range detection ─────────────────────────────────────────────
    time_field = None
    if config.filters and config.filters.time_field:
        time_field = config.filters.time_field
    else:
        time_field = _detect_time_field(files, config.parser.encoding)

    if time_field and files:
        earliest, latest = _scan_time_range(files, time_field, config.parser.encoding)
        report.time_range_earliest = earliest
        report.time_range_latest = latest
        report.time_distribution = _scan_time_distribution(
            files, time_field, config.parser.encoding, report_config.time.granularity
        )

    # ── Read pipeline + analysis ─────────────────────────────────────────
    summary = ReadSummary()
    label_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    word_counter: Counter[str] = Counter()
    per_source_lengths: dict[str, list[int]] = {}
    per_source_labels: dict[str, Counter[str]] = {}

    # Build word tokenizer from config.
    stop_words = _build_stop_words(report_config.top_words)
    word_re = re.compile(report_config.top_words.tokenize, re.IGNORECASE)
    min_word_len = report_config.top_words.min_length

    # Build custom pattern matchers.
    compiled_patterns = [
        (p.name, p.pattern, re.compile(p.pattern))
        for p in report_config.patterns
    ]
    pattern_hits: Counter[str] = Counter()
    pattern_total = 0

    # Language detection accumulators.
    lang_counter: Counter[str] = Counter()

    show = report_config.sample.count

    for item in read_corpus(config, strict=strict, summary=summary):
        report.items_emitted += 1
        text_len = len(item.text)
        report.text_lengths.append(text_len)

        if item.label:
            label_counter[item.label] += 1
        source_counter[item.source_file] += 1

        per_source_lengths.setdefault(item.source_file, []).append(text_len)
        if item.label:
            per_source_labels.setdefault(item.source_file, Counter())[item.label] += 1

        # Top words.
        for match in word_re.finditer(item.text.lower()):
            word = match.group()
            if word not in stop_words and len(word) >= min_word_len:
                word_counter[word] += 1

        # Custom patterns.
        pattern_total += 1
        for name, _, compiled in compiled_patterns:
            if compiled.search(item.text):
                pattern_hits[name] += 1

        # Language detection (character-script based).
        scripts = detect_scripts(item.text)
        for script in scripts:
            lang_counter[script] += 1

        if len(report.sample_items) < show:
            report.sample_items.append(item)

    report.summary = summary
    report.label_distribution = dict(sorted(label_counter.items()))
    report.source_file_distribution = dict(sorted(source_counter.items()))
    report.top_words = word_counter.most_common(report_config.top_words.count)
    report.languages = dict(sorted(lang_counter.items(), key=lambda x: -x[1]))

    # Pattern results.
    for p in report_config.patterns:
        report.pattern_results.append(PatternResult(
            name=p.name,
            pattern=p.pattern,
            matches=pattern_hits.get(p.name, 0),
            total=pattern_total,
        ))

    # Per-source stats.
    for src_file in sorted(per_source_lengths.keys()):
        lengths = per_source_lengths[src_file]
        report.per_source.append(SourceStats(
            source_file=src_file,
            record_count=len(lengths),
            avg_text_length=statistics.mean(lengths) if lengths else 0.0,
            labels=dict(sorted(per_source_labels.get(src_file, Counter()).items())),
        ))

    # Backfill per-file record counts.
    for fi in report.files:
        fi.records = source_counter.get(fi.path, 0)

    return report


# ---------------------------------------------------------------------------
# Report renderer (three-section layout)
# ---------------------------------------------------------------------------


def format_report(report: Report, report_config: Optional[ReportConfig] = None) -> str:
    """Render a :class:`Report` in the three-section layout."""
    if report_config is None:
        report_config = ReportConfig.defaults()

    w = 60
    lines: list[str] = []
    sep = "═" * w

    lines.append(sep)
    lines.append("  ignite-read report")
    lines.append(sep)

    # ══════════════════════════════════════════════════════════════════════
    # STANDARD - always shown
    # ══════════════════════════════════════════════════════════════════════

    lines.append("")
    lines.append(f"── Configuration {'─' * (w - 19)}")
    if report.config_path:
        lines.append(f"  Config file: {report.config_path}")
    lines.append(f"  Storage:     {report.storage_type}, {report.storage_path}")
    lines.append(f"  Format:      {report.format_type}")
    lines.append(f"  Text:        {report.text_mode}")
    lines.append(f"  Labels:      {'yes' if report.has_labels else 'no'}")
    lines.append(f"  Normalize:   {report.normalize_summary or 'none'}")
    lines.append(f"  Filters:     {report.filter_summary or 'none'}")
    lines.append(f"  Sampling:    {report.sampling_summary or 'none (full read)'}")

    lines.append("")
    lines.append(f"── Sources {'─' * (w - 12)}")
    lines.append(f"  Files:       {len(report.files)}")
    lines.append(f"  Total size:  {_fmt_bytes(report.total_size_bytes)}")
    if report.files:
        for fi in report.files:
            rec_note = f"  → {fi.records} emitted" if fi.records else ""
            lines.append(f"    {fi.path:<40} {_fmt_bytes(fi.size_bytes):>8}{rec_note}")

    lines.append("")
    lines.append(f"── Pipeline {'─' * (w - 13)}")
    lines.append(f"  Files scanned:          {report.summary.files_scanned}")
    lines.append(f"  Records emitted:        {report.summary.records_emitted}")
    if report.summary.skipped_malformed:
        lines.append(f"  Skipped (malformed):    {report.summary.skipped_malformed}")
    if report.summary.skipped_extraction:
        lines.append(f"  Skipped (no text):      {report.summary.skipped_extraction}")
    if report.summary.skipped_unrouted:
        lines.append(f"  Skipped (unrouted):     {report.summary.skipped_unrouted}")
    if report.summary.skipped_normalization:
        lines.append(f"  Skipped (length):       {report.summary.skipped_normalization}")
    if report.summary.skipped_filter:
        lines.append(f"  Skipped (filter):       {report.summary.skipped_filter}")
    if report.summary.skipped_per_file_cap:
        lines.append(f"  Skipped (per_file_cap): {report.summary.skipped_per_file_cap}")

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS - configurable sections, in configured order
    # ══════════════════════════════════════════════════════════════════════

    for section in report_config.sections:
        if section == "corpus_stats":
            lines.extend(_render_corpus_stats(report, report_config, w))
        elif section == "per_source":
            lines.extend(_render_per_source(report, w))
        elif section == "labels":
            lines.extend(_render_labels(report, w))
        elif section == "top_words":
            lines.extend(_render_top_words(report, w))
        elif section == "patterns":
            lines.extend(_render_patterns(report, w))

    # ══════════════════════════════════════════════════════════════════════
    # OUTPUT - always last
    # ══════════════════════════════════════════════════════════════════════

    if report.sample_items:
        trunc = report_config.sample.truncate
        lines.append("")
        n = len(report.sample_items)
        lines.append(f"── Sample texts (first {n}) {'─' * (w - 22 - len(str(n)))}")
        for item in report.sample_items:
            label_part = f" label={item.label}" if item.label else ""
            lines.append(f"  [{item.id}]{label_part}")
            text_display = item.text if len(item.text) <= trunc else item.text[: trunc - 3] + "..."
            lines.append(f"    {text_display!r}")

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis section renderers
# ---------------------------------------------------------------------------


def _render_corpus_stats(report: Report, rcfg: ReportConfig, w: int) -> list[str]:
    lines = ["", f"── Corpus stats {'─' * (w - 17)}"]
    lines.append(f"  Records emitted: {report.items_emitted}")
    if report.time_range_earliest or report.time_range_latest:
        lines.append(f"  Time range:      {report.time_range_earliest or '?'} → {report.time_range_latest or '?'}")
    if report.time_distribution:
        lines.append(f"  Time distribution ({rcfg.time.granularity}):")
        max_count = max(report.time_distribution.values()) if report.time_distribution else 1
        for bucket, count in sorted(report.time_distribution.items()):
            bar_len = int(count / max_count * 20)
            bar = "█" * bar_len
            lines.append(f"    {bucket}  {bar} {count}")
    if report.languages:
        total = sum(report.languages.values())
        lang_parts = []
        for script, count in report.languages.items():
            pct = count / total * 100
            lang_parts.append(f"{_friendly_script(script)} {pct:.0f}%")
        lines.append(f"  Languages:       {', '.join(lang_parts)}")
    if report.text_lengths:
        lines.append(f"  Text length (chars):")
        lines.append(f"    min: {report.text_len_min}  max: {report.text_len_max}  avg: {report.text_len_avg:.0f}  median: {report.text_len_median:.0f}")
    return lines


def _render_per_source(report: Report, w: int) -> list[str]:
    if not report.per_source or len(report.per_source) <= 1:
        return []
    lines = ["", f"── Per-source breakdown {'─' * (w - 25)}"]
    for src in report.per_source:
        lines.append(f"  {src.source_file}")
        lines.append(f"    Records: {src.record_count}, avg length: {src.avg_text_length:.0f} chars")
        if src.labels:
            top_labels = sorted(src.labels.items(), key=lambda x: -x[1])[:5]
            label_str = ", ".join(f"{l}={c}" for l, c in top_labels)
            lines.append(f"    Labels:  {label_str}")
    return lines


def _render_labels(report: Report, w: int) -> list[str]:
    if not report.label_distribution:
        return []
    lines = ["", f"── Labels {'─' * (w - 11)}"]
    total = sum(report.label_distribution.values())
    for label, count in report.label_distribution.items():
        pct = (count / total * 100) if total else 0
        lines.append(f"    {label:<20} {count:>4} ({pct:.1f}%)")
    return lines


def _render_top_words(report: Report, w: int) -> list[str]:
    if not report.top_words:
        return []
    lines = ["", f"── Top words {'─' * (w - 14)}"]
    for i in range(0, len(report.top_words), 2):
        left = report.top_words[i]
        left_str = f"  {left[0]:<18} {left[1]:>3}"
        if i + 1 < len(report.top_words):
            right = report.top_words[i + 1]
            right_str = f"  {right[0]:<18} {right[1]:>3}"
        else:
            right_str = ""
        lines.append(f"{left_str}{right_str}")
    return lines


def _render_patterns(report: Report, w: int) -> list[str]:
    if not report.pattern_results:
        return []
    lines = ["", f"── Patterns {'─' * (w - 13)}"]
    for pr in report.pattern_results:
        lines.append(f"  {pr.name:<20} {pr.matches:>4}/{pr.total} ({pr.pct:.1f}%)")
    return lines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_stop_words(cfg: "TopWordsConfig") -> frozenset:
    """Build the effective stop-words set from config."""
    if cfg.stop_words_replace is not None:
        return frozenset(cfg.stop_words_replace)
    return DEFAULT_STOP_WORDS | frozenset(cfg.stop_words_extend)


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit in ("KB", "MB", "GB"):
        n /= 1024
        if n < 1024:
            return f"{n:.1f} {unit}"
    return f"{n:.1f} TB"


def _detect_time_field(files: list[Path], encoding: str) -> Optional[str]:
    import orjson
    well_known = ["ts", "timestamp", "created_at", "date", "time", "datetime"]
    if not files:
        return None
    for raw in iter_lines(files[0], encoding=encoding):
        if not raw.strip():
            continue
        try:
            record = orjson.loads(raw)
        except Exception:
            continue
        if not isinstance(record, dict):
            continue
        for name in well_known:
            if name in record and parse_time(record[name]) is not None:
                return name
        break
    return None


def _scan_time_range(
    files: list[Path], time_field: str, encoding: str
) -> tuple[Optional[str], Optional[str]]:
    """Quick scan for earliest/latest timestamps. Streams first+last 5 lines."""
    import orjson
    from collections import deque

    earliest: Optional[datetime] = None
    latest: Optional[datetime] = None

    for path in files:
        # Stream first 5 and last 5 without materializing the whole file.
        first_5: list[bytes] = []
        last_5: deque[bytes] = deque(maxlen=5)
        for line in iter_lines(path, encoding=encoding):
            if len(first_5) < 5:
                first_5.append(line)
            last_5.append(line)
        sample_lines = first_5 + list(last_5)

        for raw in sample_lines:
            if not raw.strip():
                continue
            try:
                record = orjson.loads(raw)
            except Exception:
                continue
            if not isinstance(record, dict):
                continue
            val = record.get(time_field) if "." not in time_field else _get_dotted(record, time_field)
            dt = parse_time(val)
            if dt is None:
                continue
            dt = dt.replace(tzinfo=None) if dt.tzinfo else dt
            if earliest is None or dt < earliest:
                earliest = dt
            if latest is None or dt > latest:
                latest = dt

    return (
        earliest.isoformat() if earliest else None,
        latest.isoformat() if latest else None,
    )


def _scan_time_distribution(
    files: list[Path], time_field: str, encoding: str, granularity: str
) -> dict[str, int]:
    import orjson

    counts: Counter[str] = Counter()
    for path in files:
        for raw in iter_lines(path, encoding=encoding):
            if not raw.strip():
                continue
            try:
                record = orjson.loads(raw)
            except Exception:
                continue
            if not isinstance(record, dict):
                continue
            val = record.get(time_field) if "." not in time_field else _get_dotted(record, time_field)
            dt = parse_time(val)
            if dt is None:
                continue
            dt = dt.replace(tzinfo=None) if dt.tzinfo else dt
            if granularity == "day":
                key = dt.strftime("%Y-%m-%d")
            elif granularity == "week":
                key = f"{dt.year}-W{dt.isocalendar()[1]:02d}"
            else:  # month
                key = f"{dt.year}-{dt.month:02d}"
            counts[key] += 1
    return dict(counts)


def _get_dotted(obj, path: str):
    cur = obj
    for seg in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(seg)
    return cur


# ---------------------------------------------------------------------------
# Language / script detection
# ---------------------------------------------------------------------------

# Unicode block ranges for script detection. We detect the *dominant* script
# in each text. A text with 90% Latin and 10% Cyrillic is "Latin" - but the
# 10% Cyrillic will show up in the stats because we count per-record.
_SCRIPT_RANGES = [
    (0x0000, 0x007F, "Latin"),         # Basic Latin (ASCII)
    (0x0080, 0x024F, "Latin"),         # Latin Extended
    (0x0250, 0x02AF, "Latin"),         # IPA Extensions
    (0x0370, 0x03FF, "Greek"),
    (0x0400, 0x04FF, "Cyrillic"),
    (0x0500, 0x052F, "Cyrillic"),      # Cyrillic Supplement
    (0x0530, 0x058F, "Armenian"),
    (0x0590, 0x05FF, "Hebrew"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0700, 0x074F, "Syriac"),
    (0x0900, 0x097F, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Oriya"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x0E80, 0x0EFF, "Lao"),
    (0x1000, 0x109F, "Myanmar"),
    (0x10A0, 0x10FF, "Georgian"),
    (0x1100, 0x11FF, "Korean"),
    (0x3040, 0x309F, "Japanese"),      # Hiragana
    (0x30A0, 0x30FF, "Japanese"),      # Katakana
    (0x3400, 0x4DBF, "Chinese"),       # CJK Unified Ext A
    (0x4E00, 0x9FFF, "Chinese"),       # CJK Unified
    (0xAC00, 0xD7AF, "Korean"),        # Hangul Syllables
]


def detect_scripts(text: str) -> list[str]:
    """Detect which scripts are present in a text. Returns the dominant script(s).

    Fast approach: sample up to 200 non-space, non-digit, non-punctuation
    characters and classify by Unicode range. Returns all scripts that
    account for >10% of the classified characters.
    """
    counts: Counter[str] = Counter()
    sampled = 0
    for ch in text:
        if sampled >= 200:
            break
        cp = ord(ch)
        if cp < 0x21 or (0x30 <= cp <= 0x39) or cp in (0x2C, 0x2E, 0x3A, 0x3B, 0x21, 0x3F):
            continue  # skip whitespace, digits, basic punctuation
        script = _classify_codepoint(cp)
        if script:
            counts[script] += 1
            sampled += 1

    if not counts:
        return ["Latin"]  # default for empty/numeric-only texts

    total = sum(counts.values())
    # Return scripts above 10% threshold.
    return [script for script, count in counts.most_common()
            if count / total >= 0.10]


def _classify_codepoint(cp: int) -> Optional[str]:
    """Classify a Unicode codepoint into a script name."""
    for start, end, name in _SCRIPT_RANGES:
        if start <= cp <= end:
            return name
    return None


# Script-to-friendly-label mapping (shared with eval/selector.py logic).
_SCRIPT_FRIENDLY = {
    "Latin": "English",
    "Cyrillic": "Russian/Slavic",
    "Chinese": "Chinese",
    "Japanese": "Japanese",
    "Korean": "Korean",
    "Arabic": "Arabic",
    "Hebrew": "Hebrew",
    "Devanagari": "Hindi",
    "Bengali": "Bengali",
    "Thai": "Thai",
    "Greek": "Greek",
    "Georgian": "Georgian",
    "Armenian": "Armenian",
    "Tamil": "Tamil",
    "Telugu": "Telugu",
}


def _friendly_script(script: str) -> str:
    """Map a Unicode script name to a human-friendly language label."""
    return _SCRIPT_FRIENDLY.get(script, script)
