"""
Model auto-selector for ignite-eval.

Takes data signals (language mix, text length, corpus size, domain) and
picks 3-5 candidate models from the registry. Explains every choice so
non-technical users understand what's happening and why.

Philosophy: "like a doctor with a patient" - respectful, informational,
no jargon without explanation, never assumes the user knows what an
embedding model is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ignite_tools.eval.models import ModelEntry, load_registry
from ignite_tools.eval.requirements import EvalRequirements


# ---------------------------------------------------------------------------
# Data signals (input to the selector)
# ---------------------------------------------------------------------------


@dataclass
class DataSignals:
    """What we know about the user's data. Populated from the reader report."""

    # Language mix: script -> percentage (0.0-1.0)
    languages: dict[str, float] = field(default_factory=dict)

    # Text characteristics
    avg_text_length: float = 0.0
    max_text_length: int = 0
    record_count: int = 0

    # Domain hints (top words from the reader)
    top_words: list[str] = field(default_factory=list)

    # Hardware
    has_gpu: bool = False
    max_memory_gb: Optional[float] = None

    @property
    def is_multilingual(self) -> bool:
        """True if >10% of records have non-Latin script."""
        non_latin = sum(
            pct for script, pct in self.languages.items()
            if script != "Latin"
        )
        return non_latin > 0.10

    @property
    def is_short_text(self) -> bool:
        """True if avg text length is under 128 chars (~30 tokens)."""
        return self.avg_text_length < 128

    @property
    def is_long_text(self) -> bool:
        """True if avg text length exceeds 1000 chars (~250 tokens)."""
        return self.avg_text_length > 1000

    @property
    def is_large_corpus(self) -> bool:
        """True if corpus has 50K+ records (throughput starts to matter)."""
        return self.record_count > 50_000

    @property
    def is_code(self) -> bool:
        """Heuristic: top words suggest source code content."""
        code_words = {"fix", "bug", "refactor", "commit", "merge", "branch",
                      "function", "class", "import", "return", "async", "await",
                      "api", "endpoint", "deploy", "docker", "kubernetes"}
        overlap = set(self.top_words[:20]) & code_words
        return len(overlap) >= 3


# ---------------------------------------------------------------------------
# Selection result
# ---------------------------------------------------------------------------


@dataclass
class ModelChoice:
    """One selected model with the reason it was chosen."""

    model: ModelEntry
    reason: str       # plain-English explanation
    role: str         # "fast_baseline" | "quality_ceiling" | "best_fit" | "specialist"


@dataclass
class SelectionResult:
    """Output of the auto-selector."""

    candidates: list[ModelChoice] = field(default_factory=list)
    signals_summary: str = ""       # human-readable summary of detected signals
    skipped_reasons: list[str] = field(default_factory=list)  # why notable models were skipped


# ---------------------------------------------------------------------------
# The selector
# ---------------------------------------------------------------------------


def select_models(
    signals: DataSignals,
    *,
    max_candidates: int = 4,
    registry: Optional[list[ModelEntry]] = None,
    requirements: Optional[EvalRequirements] = None,
) -> SelectionResult:
    """Pick the best candidate models for this user's data.

    Selection is driven by three inputs:
    1. DATA SET signals (language, length, corpus size, domain)
    2. REQUIREMENTS (task, priority, preferences, constraints)
    3. HARDWARE (device, memory)

    Strategy:
    - Apply hard constraints first (eliminate what doesn't fit)
    - Score remaining by task fit + priority + data signals
    - Diversify by model family and size tier
    """
    if registry is None:
        registry = load_registry()
    if requirements is None:
        requirements = EvalRequirements()

    result = SelectionResult()
    result.signals_summary = _summarize_signals(signals)
    chosen_ids: set[str] = set()

    task = requirements.effective_task()
    priority = requirements.priority
    mode = requirements.mode

    # ── Determine appropriate model range ────────────────────────────────
    size_range = _determine_size_range(signals)

    # ── Hard constraints: filter registry ────────────────────────────────
    pool = list(registry)

    # Size cap from data-driven range.
    pool = [m for m in pool if m.size_mb <= size_range.max_size_mb]

    pool = _apply_hard_constraints(pool, requirements)

    # Language filter from data signals.
    # Skip this if user already set explicit language constraints above.
    if not requirements.constraints.languages:
        require_multilingual = signals.is_multilingual
        if require_multilingual:
            pool = [m for m in pool if m.multilingual]
        else:
            pool = [m for m in pool if not m.multilingual]

    # Context length for long text.
    if signals.is_long_text:
        pool = [m for m in pool if m.max_tokens >= 2048]

    # Exclude models whose avoid_when matches our data situation.
    data_tags = set()
    if not signals.is_code:
        data_tags.add("general")
    if signals.is_short_text:
        data_tags.add("short_text")
    pool = [m for m in pool if not (set(m.avoid_when) & data_tags)]

    if not pool:
        # Fallback: relax data-derived filters only, keep user hard constraints.
        pool = [m for m in registry if m.size_mb <= size_range.max_size_mb]
        pool = _apply_hard_constraints(pool, requirements)

    # ── Score and pick diverse candidates ────────────────────────────────
    chosen_families: set[str] = set()
    chosen_size_tiers: set[str] = set()

    for pick_num in range(max_candidates):
        best_score = -999.0
        best_model = None

        for m in pool:
            if m.id in chosen_ids:
                continue
            score = _fit_score_with_requirements(m, signals, task, priority, mode, requirements.prefer)

            # Diversity: penalize same model family.
            family = _model_family(m.id)
            if family in chosen_families:
                score -= 3.0

            # Diversity: penalize same size tier.
            size_tier = _size_tier(m.size_mb)
            if size_tier in chosen_size_tiers:
                score -= 1.5

            if score > best_score:
                best_score = score
                best_model = m

        if best_model is None or best_score <= -1:
            break

        result.candidates.append(ModelChoice(
            model=best_model,
            reason=_explain_choice(best_model, signals, size_range),
            role=_determine_role(best_model, pool),
        ))
        chosen_ids.add(best_model.id)
        chosen_families.add(_model_family(best_model.id))
        chosen_size_tiers.add(_size_tier(best_model.size_mb))

    # ── Explain skips ────────────────────────────────────────────────────
    result.skipped_reasons = _explain_skips(registry, chosen_ids, signals, size_range)

    return result


def _apply_hard_constraints(
    pool: list[ModelEntry], requirements: EvalRequirements
) -> list[ModelEntry]:
    """Apply user hard constraints consistently in primary and fallback paths."""
    constraints = requirements.constraints
    if constraints.max_size_mb is not None:
        pool = [m for m in pool if m.size_mb <= constraints.max_size_mb]
    if constraints.max_dim is not None:
        pool = [m for m in pool if m.dim <= constraints.max_dim]
    if constraints.min_quality is not None:
        quality_order = {"baseline": 0, "good": 1, "better": 2, "best": 3}
        min_q = quality_order.get(constraints.min_quality, 0)
        pool = [m for m in pool if quality_order.get(m.quality_tier, 0) >= min_q]
    if constraints.languages:
        requested = set(constraints.languages)
        pool = [
            m for m in pool
            if requested.issubset({
                str(lang).lower() for lang in m.languages if isinstance(lang, str)
            })
        ]
    return pool


# ---------------------------------------------------------------------------
# Size range determination
# ---------------------------------------------------------------------------


@dataclass
class SizeRange:
    """The appropriate model size range for this data."""
    max_size_mb: int
    label: str       # human-readable description
    reasoning: str   # why this range


def _determine_size_range(signals: DataSignals) -> SizeRange:
    """Determine what size of models is appropriate for this data.

    The key insight: model size should match DATA characteristics (text length,
    corpus size), not just hardware capability. A GPU makes large models
    FASTER but doesn't make them MORE APPROPRIATE for short text.

    Logic:
    - Short text + small corpus → small models (even with GPU!)
    - Short text + large corpus + GPU → small-to-base (speed still matters)
    - Long text OR quality-critical + GPU → base-to-large
    - Very large + GPU → large models with throughput consideration
    """
    # Primary driver: text length and corpus size determine model appropriateness.
    if signals.is_short_text and signals.record_count < 10_000:
        if signals.has_gpu:
            return SizeRange(
                max_size_mb=1500,
                label="small to base",
                reasoning="short texts - small models often match large ones on "
                "short text quality; including base models for comparison since GPU makes them fast",
            )
        return SizeRange(
            max_size_mb=500,
            label="small and efficient",
            reasoning="short texts on CPU - small models are fast and often just "
            "as good for short text",
        )

    if signals.record_count > 100_000:
        if signals.has_gpu:
            return SizeRange(
                max_size_mb=3000,
                label="small to large",
                reasoning="large corpus with GPU - balancing quality with throughput",
            )
        return SizeRange(
            max_size_mb=1500,
            label="fast and efficient",
            reasoning="large corpus on CPU - throughput matters, prefer fast models",
        )

    # Medium corpus, medium/long text.
    if signals.has_gpu:
        return SizeRange(
            max_size_mb=6000,
            label="all practical sizes",
            reasoning="GPU available, medium corpus - can explore large models",
        )
    return SizeRange(
        max_size_mb=1500,
        label="small to base",
        reasoning="medium corpus on CPU - base-size models are the sweet spot",
    )


# ---------------------------------------------------------------------------
# Model family detection (for diversity)
# ---------------------------------------------------------------------------


def _size_tier(size_mb: int) -> str:
    """Bucket a model's download size into a tier for diversity tracking."""
    if size_mb <= 200:
        return "small"
    if size_mb <= 600:
        return "base"
    return "large"


def _model_family(model_id: str) -> str:
    """Extract the model 'family' for diversity tracking.

    e.g. 'BAAI/bge-small-en-v1.5' → 'bge'
         'intfloat/e5-large-v2' → 'e5'
         'sentence-transformers/all-MiniLM-L6-v2' → 'minilm'
    """
    lower = model_id.lower()
    if "bge" in lower:
        return "bge"
    if "e5" in lower:
        return "e5"
    if "minilm" in lower:
        return "minilm"
    if "gte" in lower:
        return "gte"
    if "nomic" in lower:
        return "nomic"
    if "jina" in lower:
        return "jina"
    if "mpnet" in lower:
        return "mpnet"
    if "instructor" in lower:
        return "instructor"
    if "arctic" in lower:
        return "arctic"
    if "stella" in lower:
        return "stella"
    if "gist" in lower:
        return "gist"
    if "codet5" in lower:
        return "codet5"
    # Fallback: use the org name.
    return model_id.split("/")[0].lower() if "/" in model_id else model_id.lower()


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------


def _explain_choice(model: ModelEntry, signals: DataSignals, size_range: SizeRange) -> str:
    """Plain-English explanation of why this model was chosen."""
    parts = []

    # Quality statement.
    quality_adj = {"baseline": "Basic", "good": "Solid", "better": "Strong", "best": "Top-tier"}
    parts.append(f"{quality_adj.get(model.quality_tier, 'Good')} quality")

    # Speed/size fit.
    if model.speed_tier == "fast":
        parts.append("fastest in the comparison set")
    elif model.speed_tier == "medium":
        parts.append("good balance of speed and quality")

    # Specific fit reasons.
    if signals.is_short_text and model.max_tokens <= 512:
        parts.append("optimized for short text like yours")
    if signals.is_multilingual and model.multilingual:
        parts.append("handles your multilingual content")
    if model.matryoshka:
        parts.append("supports dimension reduction if you need smaller vectors later")

    # Size/practicality.
    parts.append(f"~{model.size_mb} MB download")

    return ". ".join(parts) + "."


def _determine_role(model: ModelEntry, pool: list[ModelEntry]) -> str:
    """Determine the model's role in the comparison set."""
    # Fastest in pool?
    speeds = {"fast": 0, "medium": 1, "slow": 2, "very_slow": 3}
    if all(speeds.get(model.speed_tier, 1) <= speeds.get(m.speed_tier, 1) for m in pool):
        return "fast_baseline"
    # Highest quality in pool?
    qualities = {"baseline": 0, "good": 1, "better": 2, "best": 3}
    if all(qualities.get(model.quality_tier, 1) >= qualities.get(m.quality_tier, 1) for m in pool):
        return "quality_leader"
    return "contender"


# ---------------------------------------------------------------------------
# Scoring and reasoning
# ---------------------------------------------------------------------------


def _fit_score(model: ModelEntry, signals: DataSignals) -> float:
    """Score how well a model fits the data signals. Higher = better fit."""
    return _fit_score_with_requirements(model, signals, "general", "balanced", "batch", [])


def _fit_score_with_requirements(
    model: ModelEntry,
    signals: DataSignals,
    task: str,
    priority: str,
    mode: str,
    prefer: list[str],
) -> float:
    """Score a model considering data signals + user requirements."""
    score = 0.0

    # ── Task fit: models tagged best_for this task get a big boost ────────
    task_tags = {
        "search": {"search"},
        "classify": {"classification", "clustering"},
        "cluster": {"clustering", "general"},
        "similarity": {"similarity", "general"},
        "deduplicate": {"similarity"},
        "rag": {"rag", "search", "long_documents"},
        "match": {"search", "similarity"},
        "general": {"general"},
    }
    relevant_tags = task_tags.get(task, {"general"})
    task_overlap = relevant_tags & set(model.best_for)
    if task_overlap:
        score += 2.0 * len(task_overlap)

    # ── Priority: weight quality vs speed differently ────────────────────
    quality_scores = {"baseline": 0, "good": 1, "better": 2, "best": 3}
    speed_scores = {"fast": 3, "medium": 2, "slow": 1, "very_slow": 0}

    q = quality_scores.get(model.quality_tier, 1)
    s = speed_scores.get(model.speed_tier, 1)

    if priority == "quality":
        score += q * 1.5
        score += s * 0.3
    elif priority == "throughput":
        score += s * 1.5
        score += q * 0.3
    elif priority == "size":
        score += (500 - min(model.size_mb, 500)) / 100.0
        score += q * 0.3
    else:  # balanced
        score += q * 0.8
        score += s * 0.8

    # ── Mode: batch vs realtime ──────────────────────────────────────────
    if mode == "realtime":
        # Realtime = single-query latency matters. Prefer small models that
        # load fast and respond fast per-inference. Penalize slow/large.
        if model.speed_tier == "fast":
            score += 2.0
        elif model.speed_tier == "medium":
            score += 0.5
        elif model.speed_tier in ("slow", "very_slow"):
            score -= 1.5
        # Smaller models stay in memory better for serving.
        if model.size_mb <= 200:
            score += 1.0
    # batch mode: no additional scoring (throughput is already in priority)

    # ── Soft preferences: boost models that match prefer tags ────────────
    if prefer:
        prefer_overlap = set(prefer) & set(model.best_for)
        score += 1.0 * len(prefer_overlap)

    # ── Data signal fit ──────────────────────────────────────────────────
    # Language.
    if signals.is_multilingual and model.multilingual:
        score += 2.0
    if not signals.is_multilingual and not model.multilingual:
        score += 0.5
    if not signals.is_multilingual and model.multilingual:
        score -= 0.5

    # Text length.
    if signals.is_long_text and model.max_tokens >= 8192:
        score += 2.0
    if signals.is_short_text and model.max_tokens <= 512:
        score += 0.5
    if signals.is_long_text and model.max_tokens < 512:
        score -= 2.0

    # Speed for large corpora.
    if signals.is_large_corpus and model.speed_tier in ("fast", "medium"):
        score += 1.5
    if signals.is_large_corpus and model.speed_tier == "very_slow":
        score -= 2.0

    # Code domain.
    if signals.is_code and "code" in model.best_for:
        score += 3.0

    # Matryoshka bonus.
    if model.matryoshka:
        score += 0.3

    return score


def _summarize_signals(signals: DataSignals) -> str:
    """One-paragraph summary of what we detected about the data."""
    parts = []

    # Languages.
    if signals.languages:
        top_langs = sorted(signals.languages.items(), key=lambda x: -x[1])[:3]
        lang_str = ", ".join(f"{_friendly_script(s)} ({p*100:.0f}%)" for s, p in top_langs)
        parts.append(f"Languages: {lang_str}")
        if signals.is_multilingual:
            parts.append("→ multilingual models needed")
        else:
            parts.append("→ English-only models are fine")

    # Length.
    if signals.is_short_text:
        parts.append(f"Text length: short (avg {signals.avg_text_length:.0f} chars)")
    elif signals.is_long_text:
        parts.append(f"Text length: long (avg {signals.avg_text_length:.0f} chars) → need long-context models")
    else:
        parts.append(f"Text length: medium (avg {signals.avg_text_length:.0f} chars)")

    # Corpus size.
    if signals.is_large_corpus:
        parts.append(f"Corpus: {signals.record_count:,} records → throughput matters, prefer faster models")
    else:
        parts.append(f"Corpus: {signals.record_count:,} records → quality over speed")

    # Domain.
    if signals.is_code:
        parts.append("Domain: code/engineering (detected from vocabulary)")
    elif signals.top_words:
        parts.append(f"Domain: general (top terms: {', '.join(signals.top_words[:5])})")

    # Hardware.
    if signals.has_gpu:
        parts.append("Hardware: GPU detected → large models are practical")
    else:
        parts.append("Hardware: CPU only → excluding very large models (7B+)")

    return "\n    ".join(parts)


def _explain_skips(
    registry: list[ModelEntry], chosen_ids: set[str], signals: DataSignals,
    size_range: SizeRange,
) -> list[str]:
    """Explain why notable models were NOT selected."""
    skips = []

    # Models excluded by size range.
    too_big = [m for m in registry if m.size_mb > size_range.max_size_mb and m.id not in chosen_ids]
    if too_big:
        skips.append(
            f"Larger models ({len(too_big)} available, up to {max(m.size_mb for m in too_big)/1024:.1f} GB): "
            f"excluded - {size_range.reasoning}"
        )

    # Multilingual models skipped.
    if not signals.is_multilingual:
        ml_skipped = [m for m in registry if m.multilingual and m.id not in chosen_ids
                      and m.size_mb <= size_range.max_size_mb]
        if ml_skipped:
            primary = _friendly_script(
                list(signals.languages.keys())[0] if signals.languages else "Latin"
            )
            skips.append(
                f"Multilingual models ({len(ml_skipped)} in range): skipped because "
                f"your data is {primary}-only"
            )

    # Long-context skipped.
    if not signals.is_long_text:
        long_skipped = [m for m in registry if m.max_tokens >= 8192 and m.id not in chosen_ids
                        and m.size_mb <= size_range.max_size_mb]
        if long_skipped:
            skips.append(
                f"Long-context models (8K+ tokens): not needed - your texts average "
                f"{signals.avg_text_length:.0f} chars"
            )

    return skips


# Script-to-human-language mapping. Latin script is called "English" by
# default because that's what 99% of users mean. If we later add real
# language detection (not just script detection), this gets more precise.
_SCRIPT_LABELS = {
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
    "Kannada": "Kannada",
    "Malayalam": "Malayalam",
    "Myanmar": "Burmese",
    "Lao": "Lao",
    "Gujarati": "Gujarati",
}


def _friendly_script(script: str) -> str:
    """Map a Unicode script name to a human-friendly language label."""
    return _SCRIPT_LABELS.get(script, script)
