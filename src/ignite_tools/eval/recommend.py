"""
Recommendation engine for ignite-eval Phase 4.

Given:
- Throughput results from Phase 2 (EmbedResult per model)
- Quality scores from Phase 3 (ModelScore per model)
- User requirements (task, priority, mode, constraints)
- Data signals (corpus size for scale projection)

Produces:
- A single recommended model with plain-English reasoning
- Why each other model lost
- Scale projection: "at your full corpus size, this model would take X"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ignite_tools.eval.requirements import EvalRequirements
from ignite_tools.eval.runner import EmbedResult
from ignite_tools.eval.scorer import ModelScore


@dataclass
class Recommendation:
    """The final recommendation."""

    model_id: str
    model_name: str
    reason: str               # plain-English "why this one"
    runners_up: list[str] = field(default_factory=list)  # "why not X: ..."
    scale_note: Optional[str] = None  # "at 500K texts, this takes ~X on your hardware"
    confidence: str = "high"  # high | medium | low


@dataclass(frozen=True)
class LabelQualityDecision:
    """Whether label-pair scores are safe to use as model quality."""

    use_for_recommendation: bool
    reason: str


_SEMANTIC_LABEL_PARTS = {
    "category",
    "class",
    "intent",
    "label",
    "labels",
    "tag",
    "tags",
    "topic",
}

_METADATA_LABEL_PARTS = {
    "author",
    "bucket",
    "customer",
    "domain",
    "file",
    "host",
    "origin",
    "path",
    "persona",
    "site",
    "source",
    "tenant",
    "user",
}


def assess_label_quality_signal(
    label_field: str | None,
    score_results: list[ModelScore],
) -> LabelQualityDecision:
    """Decide whether label scores should drive the final recommendation.

    Label-pair AUC is a quality signal only when labels look semantic and the
    measured similarities separate same-label from different-label pairs.
    Source/user/file metadata can be useful diagnostically, but it should not
    be treated as embedding-model quality.
    """
    valid_scores = [s for s in score_results if not s.error]
    if not valid_scores:
        return LabelQualityDecision(False, "no valid label scores were computed")

    if not label_field:
        return LabelQualityDecision(False, "no labels.field was configured")

    parts = set(re.split(r"[^a-z0-9]+", label_field.lower()))
    parts.discard("")

    metadata_hits = parts & _METADATA_LABEL_PARTS
    if metadata_hits:
        hit = sorted(metadata_hits)[0]
        return LabelQualityDecision(
            False,
            f"labels.field is {label_field!r}, which looks like {hit}/metadata",
        )

    semantic_hits = parts & _SEMANTIC_LABEL_PARTS
    if not semantic_hits:
        return LabelQualityDecision(
            False,
            f"labels.field is {label_field!r}; label meaning is unclear",
        )

    best = max(valid_scores, key=lambda s: s.auc)
    if best.auc < 0.5:
        return LabelQualityDecision(
            False,
            "all label scores are worse than random, so quality is inconclusive",
        )
    if best.separation <= 0:
        return LabelQualityDecision(
            False,
            "same-label pairs are not more similar than different-label pairs",
        )

    return LabelQualityDecision(
        True,
        f"labels.field is {label_field!r}, which looks semantic",
    )


def recommend(
    embed_results: list[EmbedResult],
    score_results: list[ModelScore],
    *,
    requirements: Optional[EvalRequirements] = None,
    corpus_size: int = 0,
    full_corpus_size: Optional[int] = None,
) -> Optional[Recommendation]:
    """Pick the best model and explain why.

    Logic depends on priority:
    - quality:    best AUC wins (if statistically significant), ties broken by throughput
    - throughput: fastest model that meets a quality floor
    - balanced:   weighted combination of quality rank + throughput rank
    - size:       smallest model that meets a quality floor
    """
    if not embed_results or not score_results:
        return None

    if requirements is None:
        requirements = EvalRequirements()

    priority = requirements.priority

    # Build a combined view per model.
    embed_by_id = {r.model_id: r for r in embed_results if not r.error}
    score_by_id = {s.model_id: s for s in score_results if not s.error}

    # Only consider models that have both throughput AND quality data.
    candidates = [
        mid for mid in embed_by_id
        if mid in score_by_id
    ]
    if not candidates:
        return None

    # ── Rank by priority ─────────────────────────────────────────────────
    if priority == "quality":
        ranked = sorted(candidates, key=lambda mid: -score_by_id[mid].auc)
    elif priority == "throughput":
        ranked = sorted(candidates, key=lambda mid: -embed_by_id[mid].throughput_warm)
    elif priority == "size":
        ranked = sorted(candidates, key=lambda mid: embed_by_id[mid].peak_memory_mb)
    else:  # balanced
        # Rank by combined: normalize AUC (0-1) and throughput (0-1), weight equally.
        max_tp = max(embed_by_id[mid].throughput_warm for mid in candidates) or 1
        max_auc = max(score_by_id[mid].auc for mid in candidates) or 1

        def balanced_score(mid: str) -> float:
            tp_norm = embed_by_id[mid].throughput_warm / max_tp
            auc_norm = score_by_id[mid].auc / max_auc
            return tp_norm * 0.4 + auc_norm * 0.6

        ranked = sorted(candidates, key=lambda mid: -balanced_score(mid))

    winner_id = ranked[0]
    winner_embed = embed_by_id[winner_id]
    winner_score = score_by_id[winner_id]

    # ── Build recommendation reason ──────────────────────────────────────
    reason = _build_reason(winner_embed, winner_score, priority, embed_by_id, score_by_id, ranked)

    # ── Explain runners-up ───────────────────────────────────────────────
    runners_up = []
    for mid in ranked[1:]:
        explanation = _explain_loss(
            mid, winner_id, embed_by_id, score_by_id, priority
        )
        runners_up.append(explanation)

    # ── Scale projection ─────────────────────────────────────────────────
    scale_note = None
    project_size = full_corpus_size or corpus_size
    if project_size > 100 and winner_embed.throughput_warm > 0:
        seconds = project_size / winner_embed.throughput_warm
        if seconds < 60:
            time_str = f"{seconds:.0f} seconds"
        elif seconds < 3600:
            time_str = f"{seconds / 60:.1f} minutes"
        else:
            time_str = f"{seconds / 3600:.1f} hours"
        scale_note = (
            f"At your full corpus ({project_size:,} texts): "
            f"~{time_str} on this hardware. "
        )
        if seconds > 300:
            scale_note += "With IgniteMS on GPU: estimated ~{:.0f}x faster.".format(
                max(seconds / 30, 2)
            )

    # ── Confidence ───────────────────────────────────────────────────────
    confidence = "high"
    if winner_score.auc < 0.5:
        confidence = "low"
    elif winner_score.auc_ci_low < 0.5:
        confidence = "medium"
    # Check if winner and runner-up CIs overlap.
    if len(ranked) > 1:
        second_score = score_by_id[ranked[1]]
        if winner_score.auc_ci_low < second_score.auc_ci_high:
            confidence = "medium" if confidence == "high" else confidence

    return Recommendation(
        model_id=winner_id,
        model_name=winner_embed.model_name or winner_id,
        reason=reason,
        runners_up=runners_up,
        scale_note=scale_note,
        confidence=confidence,
    )


def recommend_operationally(
    embed_results: list[EmbedResult],
    *,
    requirements: Optional[EvalRequirements] = None,
    corpus_size: int = 0,
    full_corpus_size: Optional[int] = None,
    note: str | None = None,
) -> Optional[Recommendation]:
    """Recommend from benchmark operational results when quality is unavailable."""
    candidates = [r for r in embed_results if not r.error]
    if not candidates:
        return None

    if requirements is None:
        requirements = EvalRequirements()

    priority = requirements.priority
    if priority == "size":
        ranked = sorted(candidates, key=lambda r: (r.embed_dim, r.peak_memory_mb))
        reason_kind = "Smallest operational choice"
    elif priority == "quality":
        ranked = sorted(candidates, key=lambda r: (-r.embed_dim, -r.throughput_warm))
        reason_kind = "Highest-capacity operational choice"
    else:
        ranked = sorted(candidates, key=lambda r: -r.throughput_warm)
        reason_kind = "Fastest operational choice"

    winner = ranked[0]
    reason_parts = [
        f"{reason_kind} among tested models",
        f"{winner.throughput_warm:.0f} texts/sec",
        f"{winner.embed_dim}d embeddings",
        f"{winner.peak_memory_mb:.0f} MB memory",
    ]
    if note:
        reason_parts.append(f"Quality ranking was not used: {note}")

    runners_up = [
        _explain_operational_loss(r, winner, priority)
        for r in ranked[1:]
    ]

    return Recommendation(
        model_id=winner.model_id,
        model_name=winner.model_name or winner.model_id,
        reason=", ".join(reason_parts) + ".",
        runners_up=runners_up,
        scale_note=_build_scale_note(
            winner, corpus_size=corpus_size, full_corpus_size=full_corpus_size
        ),
        confidence="low",
    )


# ---------------------------------------------------------------------------
# Reason builders
# ---------------------------------------------------------------------------


def _build_reason(
    winner_embed: EmbedResult,
    winner_score: ModelScore,
    priority: str,
    embed_by_id: dict,
    score_by_id: dict,
    ranked: list[str],
) -> str:
    """Plain-English reason for recommending this model."""
    parts = []

    if priority == "quality":
        parts.append(f"Best quality on your data (AUC {winner_score.auc:.3f})")
        parts.append(f"at {winner_embed.throughput_warm:.0f} texts/sec")
    elif priority == "throughput":
        parts.append(f"Fastest model ({winner_embed.throughput_warm:.0f} texts/sec)")
        parts.append(f"with AUC {winner_score.auc:.3f}")
    elif priority == "size":
        parts.append(f"Smallest memory footprint ({winner_embed.peak_memory_mb:.0f} MB)")
        parts.append(f"with AUC {winner_score.auc:.3f}")
    else:
        parts.append(f"Best overall balance of quality (AUC {winner_score.auc:.3f})")
        parts.append(f"and speed ({winner_embed.throughput_warm:.0f} texts/sec)")

    parts.append(f"{winner_embed.embed_dim}d embeddings, {winner_embed.peak_memory_mb:.0f} MB memory")

    return ", ".join(parts) + "."


def _explain_loss(
    loser_id: str,
    winner_id: str,
    embed_by_id: dict,
    score_by_id: dict,
    priority: str,
) -> str:
    """Explain why a model wasn't recommended."""
    loser_embed = embed_by_id[loser_id]
    loser_score = score_by_id[loser_id]
    winner_embed = embed_by_id[winner_id]
    winner_score = score_by_id[winner_id]

    name = loser_embed.model_name or loser_id

    if priority == "quality":
        auc_diff = winner_score.auc - loser_score.auc
        if auc_diff > 0.05:
            return f"{name}: lower quality (AUC {loser_score.auc:.3f} vs {winner_score.auc:.3f})"
        else:
            return f"{name}: similar quality but slower ({loser_embed.throughput_warm:.0f} vs {winner_embed.throughput_warm:.0f} texts/s)"
    elif priority == "throughput":
        tp_diff = winner_embed.throughput_warm - loser_embed.throughput_warm
        return f"{name}: {tp_diff:.0f} texts/s slower ({loser_embed.throughput_warm:.0f} vs {winner_embed.throughput_warm:.0f})"
    elif priority == "size":
        mem_diff = loser_embed.peak_memory_mb - winner_embed.peak_memory_mb
        return f"{name}: {mem_diff:.0f} MB more memory"
    else:
        # Balanced: explain which dimension lost.
        auc_diff = winner_score.auc - loser_score.auc
        tp_diff = winner_embed.throughput_warm - loser_embed.throughput_warm
        if auc_diff > 0.03 and tp_diff > 0:
            return f"{name}: worse on both quality and speed"
        elif auc_diff > 0.03:
            return f"{name}: lower quality (AUC {loser_score.auc:.3f}), faster but not enough to compensate"
        elif tp_diff > 100:
            return f"{name}: similar quality but {tp_diff:.0f} texts/s slower"
        else:
            return f"{name}: marginally worse overall balance"


def _explain_operational_loss(
    loser: EmbedResult,
    winner: EmbedResult,
    priority: str,
) -> str:
    """Explain fallback recommendations when quality scores are diagnostic."""
    name = loser.model_name or loser.model_id
    if priority == "size":
        dim_diff = loser.embed_dim - winner.embed_dim
        mem_diff = loser.peak_memory_mb - winner.peak_memory_mb
        if dim_diff > 0:
            return f"{name}: larger vectors ({loser.embed_dim}d vs {winner.embed_dim}d)"
        return f"{name}: {mem_diff:.0f} MB more memory"

    if priority == "quality":
        dim_diff = winner.embed_dim - loser.embed_dim
        if dim_diff > 0:
            return f"{name}: lower-capacity vectors ({loser.embed_dim}d vs {winner.embed_dim}d)"
        tp_diff = winner.throughput_warm - loser.throughput_warm
        return f"{name}: similar capacity but {tp_diff:.0f} texts/s slower"

    tp_diff = winner.throughput_warm - loser.throughput_warm
    return f"{name}: {tp_diff:.0f} texts/s slower"


def _build_scale_note(
    winner_embed: EmbedResult,
    *,
    corpus_size: int,
    full_corpus_size: Optional[int],
) -> Optional[str]:
    """Build the shared throughput projection sentence."""
    project_size = full_corpus_size or corpus_size
    if project_size <= 100 or winner_embed.throughput_warm <= 0:
        return None

    seconds = project_size / winner_embed.throughput_warm
    if seconds < 60:
        time_str = f"{seconds:.0f} seconds"
    elif seconds < 3600:
        time_str = f"{seconds / 60:.1f} minutes"
    else:
        time_str = f"{seconds / 3600:.1f} hours"

    scale_note = (
        f"At your full corpus ({project_size:,} texts): "
        f"~{time_str} on this hardware. "
    )
    if seconds > 300:
        scale_note += "With IgniteMS on GPU: estimated ~{:.0f}x faster.".format(
            max(seconds / 30, 2)
        )
    return scale_note
