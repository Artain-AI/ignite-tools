"""
Scoring engine for ignite-eval Phase 3.

Given:
- A corpus of labeled items (text + label)
- Pre-computed embedding vectors per model (.npy files from Phase 2)

Produces:
- Quality metrics per model: AUC, separation score, bootstrap CI
- Pair-level scores for debugging

Approach:
1. Generate pairs from labeled items:
   - Positive pairs: same label (random sample)
   - Hard negatives: nearest cross-label by TF-IDF (cheap, no embedding
     chicken-and-egg)
   - Random negatives: distant cross-label
2. For each model's vectors:
   - Compute cosine similarity for each pair
   - Compute AUC (can the similarity score separate positives from negatives?)
   - Compute separation (mean_positive - mean_negative)
   - Bootstrap CI on both metrics
3. Return structured results per model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score


@dataclass
class Pair:
    """One evaluation pair."""

    idx_a: int
    idx_b: int
    label: str  # "positive" | "negative"


@dataclass
class ModelScore:
    """Quality metrics for one model."""

    model_id: str
    model_name: str = ""
    auc: float = 0.0
    auc_ci_low: float = 0.0
    auc_ci_high: float = 0.0
    separation: float = 0.0
    separation_ci_low: float = 0.0
    separation_ci_high: float = 0.0
    mean_positive_sim: float = 0.0
    mean_negative_sim: float = 0.0
    num_pairs: int = 0
    error: Optional[str] = None


@dataclass
class ScoreResult:
    """Aggregate scoring results across all models."""

    scores: list[ModelScore] = field(default_factory=list)
    num_positive_pairs: int = 0
    num_negative_pairs: int = 0
    total_pairs: int = 0


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------


def generate_pairs(
    texts: list[str],
    labels: list[Optional[str]],
    *,
    max_pairs: int = 500,
    hard_negative_ratio: float = 1.0,
    seed: int = 42,
) -> list[Pair]:
    """Generate evaluation pairs from labeled items."""
    if len(texts) != len(labels):
        raise ValueError(
            f"texts and labels must have same length: {len(texts)} vs {len(labels)}"
        )

    rng = random.Random(seed)

    # Group indices by label.
    by_label: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        if label is not None:
            by_label.setdefault(label, []).append(i)

    if len(by_label) < 2:
        # Need at least 2 labels to create positive vs negative pairs.
        return []

    # ── Positive pairs (same label) ──────────────────────────────────────
    target_positives = max_pairs // 2
    positives: list[Pair] = []
    seen_pairs: set[tuple[int, int, str]] = set()

    label_keys = list(by_label.keys())
    attempts = 0
    while len(positives) < target_positives and attempts < target_positives * 10:
        attempts += 1
        label = rng.choice(label_keys)
        indices = by_label[label]
        if len(indices) < 2:
            continue
        a, b = rng.sample(indices, 2)
        key = (min(a, b), max(a, b), "positive")
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        positives.append(Pair(idx_a=a, idx_b=b, label="positive"))

    # ── Negative pairs (cross-label) ────────────────────────────────────
    target_negatives = int(len(positives) * hard_negative_ratio)

    # Hard negatives: TF-IDF nearest cross-label.
    hard_negatives = _generate_hard_negatives(
        texts, labels, by_label,
        target=target_negatives // 2,
        rng=rng,
        seen_pairs=seen_pairs,
    )
    for p in hard_negatives:
        seen_pairs.add((min(p.idx_a, p.idx_b), max(p.idx_a, p.idx_b), "negative"))

    # Random negatives: random cross-label pairs.
    random_negatives: list[Pair] = []
    attempts = 0
    target_random = target_negatives - len(hard_negatives)
    while len(random_negatives) < target_random and attempts < target_random * 10:
        attempts += 1
        label_a, label_b = rng.sample(label_keys, 2)
        a = rng.choice(by_label[label_a])
        b = rng.choice(by_label[label_b])
        key = (min(a, b), max(a, b), "negative")
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        random_negatives.append(Pair(idx_a=a, idx_b=b, label="negative"))

    return positives + hard_negatives + random_negatives


def _generate_hard_negatives(
    texts: list[str],
    labels: list[Optional[str]],
    by_label: dict[str, list[int]],
    target: int,
    rng: random.Random,
    seen_pairs: Optional[set] = None,
) -> list[Pair]:
    """Generate hard negatives using TF-IDF nearest cross-label neighbors."""
    if not texts or target <= 0:
        return []
    if seen_pairs is None:
        seen_pairs = set()

    labeled_indices = [i for i, l in enumerate(labels) if l is not None]
    if len(labeled_indices) < 10:
        return []

    labeled_texts = [texts[i] for i in labeled_indices]

    try:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(labeled_texts)
    except ValueError:
        return []

    # Use NearestNeighbors instead of full pairwise matrix (O(N*k) not O(N^2)).
    from sklearn.neighbors import NearestNeighbors
    # Query all neighbors for sampled anchors so imbalanced labels do not hide
    # the nearest cross-label candidate behind many same-label neighbors.
    n_neighbors = len(labeled_indices)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn.fit(tfidf_matrix)

    # Build a local-to-global index map.
    local_to_global = {i: labeled_indices[i] for i in range(len(labeled_indices))}

    pairs: list[Pair] = []
    anchors = rng.sample(range(len(labeled_indices)), min(target * 2, len(labeled_indices)))

    for anchor_local in anchors:
        if len(pairs) >= target:
            break

        anchor_global = local_to_global[anchor_local]
        anchor_label = labels[anchor_global]

        # Find nearest neighbors and pick first cross-label one.
        distances, indices = nn.kneighbors(tfidf_matrix[anchor_local:anchor_local+1])
        for neighbor_local in indices[0]:
            neighbor_global = local_to_global[neighbor_local]
            if labels[neighbor_global] != anchor_label:
                key = (min(anchor_global, neighbor_global), max(anchor_global, neighbor_global), "negative")
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                pairs.append(Pair(
                    idx_a=anchor_global,
                    idx_b=neighbor_global,
                    label="negative",
                ))
                break

    return pairs


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_model(
    vectors: np.ndarray,
    pairs: list[Pair],
    model_id: str,
    *,
    bootstrap_n: int = 1000,
    seed: int = 42,
) -> ModelScore:
    """Compute quality metrics for one model's embeddings.

    Computes:
    - AUC: area under ROC curve (can similarity separate pos from neg?)
    - Separation: mean(positive_sim) - mean(negative_sim)
    - Bootstrap 95% CI on both
    """
    if not pairs:
        return ModelScore(model_id=model_id, error="no pairs generated")

    if vectors.ndim != 2:
        return ModelScore(model_id=model_id, error="vectors must be a 2D array")
    if not np.isfinite(vectors).all():
        return ModelScore(model_id=model_id, error="vectors contain NaN or infinity")

    # Bounds check: ensure pair indices are valid for the vector array.
    max_idx = vectors.shape[0] - 1
    for p in pairs:
        if p.label not in {"positive", "negative"}:
            return ModelScore(
                model_id=model_id,
                error=f"invalid pair label {p.label!r}; expected 'positive' or 'negative'",
            )
        if p.idx_a < 0 or p.idx_b < 0 or p.idx_a > max_idx or p.idx_b > max_idx:
            return ModelScore(
                model_id=model_id,
                error=f"pair index out of bounds (valid 0-{max_idx}, got {p.idx_a},{p.idx_b})"
            )

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        return ModelScore(model_id=model_id, error="vectors contain zero-norm rows")
    unit_vectors = vectors / norms

    # Vectorized cosine similarity.
    idx_a = np.array([p.idx_a for p in pairs])
    idx_b = np.array([p.idx_b for p in pairs])
    sims = np.sum(unit_vectors[idx_a] * unit_vectors[idx_b], axis=1)
    pair_labels = np.array([1 if p.label == "positive" else 0 for p in pairs])

    pos_mask = pair_labels == 1
    neg_mask = pair_labels == 0

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return ModelScore(model_id=model_id, error="need both positive and negative pairs")

    # Raw metrics.
    mean_pos = float(sims[pos_mask].mean())
    mean_neg = float(sims[neg_mask].mean())
    separation = mean_pos - mean_neg

    try:
        auc = float(roc_auc_score(pair_labels, sims))
    except ValueError:
        auc = 0.5  # degenerate case

    # Bootstrap CI.
    rng = np.random.default_rng(seed)
    bootstrap_aucs = []
    bootstrap_seps = []

    for _ in range(bootstrap_n):
        idx = rng.choice(len(pairs), size=len(pairs), replace=True)
        b_sims = sims[idx]
        b_labels = pair_labels[idx]

        if b_labels.sum() == 0 or (1 - b_labels).sum() == 0:
            continue

        try:
            b_auc = roc_auc_score(b_labels, b_sims)
        except ValueError:
            continue

        b_pos = b_sims[b_labels == 1].mean()
        b_neg = b_sims[b_labels == 0].mean()
        bootstrap_aucs.append(b_auc)
        bootstrap_seps.append(b_pos - b_neg)

    auc_ci = _ci_95(bootstrap_aucs) if bootstrap_aucs else (auc, auc)
    sep_ci = _ci_95(bootstrap_seps) if bootstrap_seps else (separation, separation)

    return ModelScore(
        model_id=model_id,
        auc=round(auc, 4),
        auc_ci_low=round(auc_ci[0], 4),
        auc_ci_high=round(auc_ci[1], 4),
        separation=round(separation, 4),
        separation_ci_low=round(sep_ci[0], 4),
        separation_ci_high=round(sep_ci[1], 4),
        mean_positive_sim=round(mean_pos, 4),
        mean_negative_sim=round(mean_neg, 4),
        num_pairs=len(pairs),
    )


def score_all_models(
    vectors_by_model: dict[str, np.ndarray],
    pairs: list[Pair],
    model_names: dict[str, str],
    *,
    bootstrap_n: int = 1000,
    seed: int = 42,
) -> ScoreResult:
    """Score all models on the same pair set."""
    result = ScoreResult(
        num_positive_pairs=sum(1 for p in pairs if p.label == "positive"),
        num_negative_pairs=sum(1 for p in pairs if p.label == "negative"),
        total_pairs=len(pairs),
    )

    for model_id, vectors in vectors_by_model.items():
        ms = score_model(
            vectors, pairs, model_id,
            bootstrap_n=bootstrap_n, seed=seed,
        )
        ms.model_name = model_names.get(model_id, model_id)
        result.scores.append(ms)

    # Sort by AUC descending.
    result.scores.sort(key=lambda s: -s.auc if s.error is None else 0)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ci_95(values: list[float]) -> tuple[float, float]:
    """95% confidence interval from bootstrap samples."""
    if not values:
        return (0.0, 0.0)
    sorted_v = sorted(values)
    n = len(sorted_v)
    low_idx = int(n * 0.025)
    high_idx = int(n * 0.975)
    return (sorted_v[low_idx], sorted_v[min(high_idx, n - 1)])
