"""Tests for eval modules: scorer, selector, recommend, requirements."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ignite_tools.eval.models import ModelEntry, load_registry
from ignite_tools.eval.requirements import EvalConstraints, EvalRequirements
from ignite_tools.eval.scorer import (
    ModelScore,
    Pair,
    generate_pairs,
    score_all_models,
    score_model,
)
from ignite_tools.eval.selector import DataSignals, select_models


# ---------------------------------------------------------------------------
# Scorer: pair generation
# ---------------------------------------------------------------------------


def test_generate_pairs_basic():
    texts = [f"text {i}" for i in range(100)]
    labels = ["a"] * 50 + ["b"] * 50
    pairs = generate_pairs(texts, labels, max_pairs=50, seed=42)
    assert len(pairs) == 50
    pos = [p for p in pairs if p.label == "positive"]
    neg = [p for p in pairs if p.label == "negative"]
    assert len(pos) == 25
    assert len(neg) == 25


def test_generate_pairs_validates_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        generate_pairs(["a", "b"], ["x"], max_pairs=10)


def test_generate_pairs_no_duplicates():
    texts = [f"text {i}" for i in range(200)]
    labels = ["a"] * 100 + ["b"] * 100
    pairs = generate_pairs(texts, labels, max_pairs=100, seed=42)
    keys = set()
    for p in pairs:
        key = (min(p.idx_a, p.idx_b), max(p.idx_a, p.idx_b))
        assert key not in keys, f"duplicate pair: {key}"
        keys.add(key)


def test_generate_pairs_single_label_returns_empty():
    texts = [f"text {i}" for i in range(50)]
    labels = ["same"] * 50
    pairs = generate_pairs(texts, labels, max_pairs=20)
    assert pairs == []


def test_generate_pairs_with_none_labels():
    texts = [f"text {i}" for i in range(100)]
    labels = ["a"] * 30 + [None] * 40 + ["b"] * 30
    pairs = generate_pairs(texts, labels, max_pairs=40, seed=42)
    # Should only use labeled items.
    for p in pairs:
        assert labels[p.idx_a] is not None
        assert labels[p.idx_b] is not None


# ---------------------------------------------------------------------------
# Scorer: score_model
# ---------------------------------------------------------------------------


def test_score_model_perfect_separation():
    # Create vectors where same-label items are identical, cross-label are orthogonal.
    vectors = np.zeros((10, 4), dtype=np.float32)
    vectors[:5] = [1, 0, 0, 0]  # label "a"
    vectors[5:] = [0, 1, 0, 0]  # label "b"
    # Normalize (already unit vectors).
    pairs = [
        Pair(0, 1, "positive"),  # a-a: sim=1.0
        Pair(2, 3, "positive"),  # a-a: sim=1.0
        Pair(5, 6, "positive"),  # b-b: sim=1.0
        Pair(0, 5, "negative"),  # a-b: sim=0.0
        Pair(1, 6, "negative"),  # a-b: sim=0.0
        Pair(2, 7, "negative"),  # a-b: sim=0.0
    ]
    result = score_model(vectors, pairs, "test-model", bootstrap_n=100)
    assert result.auc == 1.0
    assert result.separation == 1.0
    assert result.error is None


def test_score_model_random_vectors():
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(100, 64)).astype(np.float32)
    # Normalize.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    labels = ["a"] * 50 + ["b"] * 50
    pairs = generate_pairs(
        [f"t{i}" for i in range(100)], labels, max_pairs=80, seed=42
    )
    result = score_model(vectors, pairs, "random", bootstrap_n=100)
    # Random vectors: AUC should be close to 0.5 (no separation).
    assert 0.3 < result.auc < 0.7
    assert result.num_pairs == len(pairs)


def test_score_model_bounds_check():
    vectors = np.ones((5, 3), dtype=np.float32)
    pairs = [Pair(0, 99, "positive")]  # index 99 is out of bounds
    result = score_model(vectors, pairs, "bad")
    assert result.error is not None
    assert "out of bounds" in result.error


def test_score_model_rejects_invalid_inputs():
    pairs = [Pair(0, 1, "maybe")]
    vectors = np.ones((2, 3), dtype=np.float32)
    result = score_model(vectors, pairs, "bad-label")
    assert result.error is not None
    assert "invalid pair label" in result.error

    result = score_model(np.ones(3, dtype=np.float32), [Pair(0, 1, "positive")], "bad-shape")
    assert result.error == "vectors must be a 2D array"

    bad = np.ones((2, 3), dtype=np.float32)
    bad[0, 0] = np.nan
    result = score_model(bad, [Pair(0, 1, "positive")], "bad-values")
    assert result.error == "vectors contain NaN or infinity"


def test_score_model_normalizes_vectors_before_cosine():
    vectors = np.array([[10, 0], [2, 0], [0, 3], [0, 4]], dtype=np.float32)
    pairs = [Pair(0, 1, "positive"), Pair(2, 3, "positive"), Pair(0, 2, "negative")]
    result = score_model(vectors, pairs, "unnormalized", bootstrap_n=10)
    assert result.error is None
    assert result.mean_positive_sim == 1.0
    assert result.mean_negative_sim == 0.0


def test_score_model_empty_pairs():
    vectors = np.ones((5, 3), dtype=np.float32)
    result = score_model(vectors, [], "empty")
    assert result.error == "no pairs generated"


# ---------------------------------------------------------------------------
# Scorer: score_all_models
# ---------------------------------------------------------------------------


def test_score_all_models_sorts_by_auc():
    rng = np.random.default_rng(42)
    # Model A: random (AUC ~0.5).
    v_a = rng.normal(size=(50, 32)).astype(np.float32)
    v_a /= np.linalg.norm(v_a, axis=1, keepdims=True)
    # Model B: perfect separation.
    v_b = np.zeros((50, 32), dtype=np.float32)
    v_b[:25, 0] = 1.0
    v_b[25:, 1] = 1.0

    labels = ["x"] * 25 + ["y"] * 25
    pairs = generate_pairs([f"t{i}" for i in range(50)], labels, max_pairs=40, seed=1)

    result = score_all_models(
        {"model_a": v_a, "model_b": v_b},
        pairs,
        {"model_a": "A", "model_b": "B"},
        bootstrap_n=50,
    )
    # B should be ranked first (higher AUC).
    assert result.scores[0].model_id == "model_b"
    assert result.scores[0].auc > result.scores[1].auc


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------


def test_selector_picks_diverse_families():
    signals = DataSignals(
        languages={"Latin": 1.0},
        avg_text_length=60,
        record_count=1000,
        top_words=["api", "fix", "deploy"],
        has_gpu=False,
    )
    result = select_models(signals, max_candidates=4)
    families = set()
    for c in result.candidates:
        family = c.model.id.split("/")[0]
        families.add(family)
    # Should have at least 3 different model providers.
    assert len(families) >= 3


def test_selector_respects_max_dim_constraint():
    signals = DataSignals(
        languages={"Latin": 1.0},
        avg_text_length=60,
        record_count=100,
        has_gpu=False,
    )
    reqs = EvalRequirements(constraints=EvalConstraints(max_dim=384))
    result = select_models(signals, max_candidates=4, requirements=reqs)
    for c in result.candidates:
        assert c.model.dim <= 384


def test_selector_requires_multilingual_when_detected():
    signals = DataSignals(
        languages={"Latin": 0.5, "Cyrillic": 0.5},
        avg_text_length=100,
        record_count=1000,
        has_gpu=False,
    )
    result = select_models(signals, max_candidates=4)
    for c in result.candidates:
        assert c.model.multilingual


def test_selector_excludes_multilingual_for_english():
    signals = DataSignals(
        languages={"Latin": 1.0},
        avg_text_length=60,
        record_count=100,
        has_gpu=False,
    )
    result = select_models(signals, max_candidates=4)
    for c in result.candidates:
        assert not c.model.multilingual


def test_selector_constraints_languages_overrides_data_filter():
    """If user says languages=[en,de], multilingual models should be selected
    even when data appears English-only."""
    signals = DataSignals(
        languages={"Latin": 1.0},  # data looks English
        avg_text_length=60,
        record_count=100,
        has_gpu=False,
    )
    reqs = EvalRequirements(constraints=EvalConstraints(languages=["en", "de"]))
    result = select_models(signals, max_candidates=4, requirements=reqs)
    # Should have multilingual models despite English-only data.
    assert any(c.model.multilingual for c in result.candidates)


def test_selector_language_constraint_matches_actual_registry_languages():
    registry = [
        ModelEntry(
            id="en-de",
            name="English German",
            params="1M",
            dim=384,
            max_tokens=512,
            multilingual=True,
            languages=["en", "de"],
            size_mb=100,
        ),
        ModelEntry(
            id="en-fr",
            name="English French",
            params="1M",
            dim=384,
            max_tokens=512,
            multilingual=True,
            languages=["en", "fr"],
            size_mb=100,
        ),
    ]
    signals = DataSignals(languages={"Latin": 1.0}, avg_text_length=60, record_count=100)
    reqs = EvalRequirements(constraints=EvalConstraints(languages=["en", "de"]))
    result = select_models(signals, max_candidates=2, registry=registry, requirements=reqs)
    assert [c.model.id for c in result.candidates] == ["en-de"]


def test_selector_unsupported_language_constraint_returns_no_models():
    registry = [
        ModelEntry(
            id="en-de",
            name="English German",
            params="1M",
            dim=384,
            max_tokens=512,
            multilingual=True,
            languages=["en", "de"],
            size_mb=100,
        )
    ]
    signals = DataSignals(languages={"Latin": 1.0}, avg_text_length=60, record_count=100)
    reqs = EvalRequirements(constraints=EvalConstraints(languages=["zz"]))
    result = select_models(signals, max_candidates=2, registry=registry, requirements=reqs)
    assert result.candidates == []


def test_selector_task_influences_selection():
    signals = DataSignals(
        languages={"Latin": 1.0},
        avg_text_length=60,
        record_count=100,
        has_gpu=True,
    )
    reqs_search = EvalRequirements(task="search")
    reqs_classify = EvalRequirements(task="classify")
    result_search = select_models(signals, max_candidates=4, requirements=reqs_search)
    result_classify = select_models(signals, max_candidates=4, requirements=reqs_classify)
    # Different tasks should produce at least partially different model sets.
    ids_search = {c.model.id for c in result_search.candidates}
    ids_classify = {c.model.id for c in result_classify.candidates}
    # Not necessarily completely different, but not identical.
    assert ids_search != ids_classify or len(ids_search) == 0


# ---------------------------------------------------------------------------
# Requirements validation
# ---------------------------------------------------------------------------


def test_requirements_invalid_task_rejected():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        EvalRequirements(task="nonexistent_task")


def test_requirements_invalid_priority_rejected():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        EvalRequirements(priority="fastest")


def test_requirements_normalizes_languages():
    reqs = EvalRequirements(constraints=EvalConstraints(languages=[" EN ", "de", "en"]))
    assert reqs.constraints.languages == ["en", "de"]


def test_requirements_rejects_empty_language():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        EvalRequirements(constraints=EvalConstraints(languages=["en", " "]))


def test_requirements_valid():
    reqs = EvalRequirements(
        task="search",
        priority="quality",
        mode="realtime",
        prefer=["cpu_friendly"],
        constraints=EvalConstraints(max_size_mb=500, max_dim=768),
    )
    assert reqs.effective_task() == "search"
    assert reqs.priority == "quality"


# ---------------------------------------------------------------------------
# Recommend
# ---------------------------------------------------------------------------


def test_recommend_picks_best_balanced():
    from ignite_tools.eval.recommend import recommend
    from ignite_tools.eval.runner import EmbedResult

    embed_results = [
        EmbedResult(model_id="fast", model_name="Fast", throughput_warm=2000, peak_memory_mb=500, embed_dim=384),
        EmbedResult(model_id="quality", model_name="Quality", throughput_warm=1800, peak_memory_mb=1500, embed_dim=1024),
    ]
    score_results = [
        ModelScore(model_id="fast", auc=0.55, auc_ci_low=0.50, auc_ci_high=0.60, separation=0.02, num_pairs=100),
        ModelScore(model_id="quality", auc=0.85, auc_ci_low=0.80, auc_ci_high=0.90, separation=0.25, num_pairs=100),
    ]
    rec = recommend(embed_results, score_results, corpus_size=10000)
    assert rec is not None
    # Quality model has much higher AUC with similar speed - should win balanced.
    assert rec.model_id == "quality"


def test_recommend_throughput_priority():
    from ignite_tools.eval.recommend import recommend
    from ignite_tools.eval.runner import EmbedResult

    embed_results = [
        EmbedResult(model_id="fast", model_name="Fast", throughput_warm=5000, peak_memory_mb=500, embed_dim=384),
        EmbedResult(model_id="slow", model_name="Slow", throughput_warm=500, peak_memory_mb=1500, embed_dim=1024),
    ]
    score_results = [
        ModelScore(model_id="fast", auc=0.60, auc_ci_low=0.55, auc_ci_high=0.65, separation=0.05, num_pairs=100),
        ModelScore(model_id="slow", auc=0.62, auc_ci_low=0.57, auc_ci_high=0.67, separation=0.06, num_pairs=100),
    ]
    reqs = EvalRequirements(priority="throughput")
    rec = recommend(embed_results, score_results, requirements=reqs, corpus_size=10000)
    assert rec is not None
    assert rec.model_id == "fast"


def test_label_quality_signal_rejects_source_metadata():
    from ignite_tools.eval.recommend import assess_label_quality_signal

    scores = [
        ModelScore(
            model_id="model",
            auc=0.85,
            auc_ci_low=0.80,
            auc_ci_high=0.90,
            separation=0.20,
            num_pairs=100,
        )
    ]

    decision = assess_label_quality_signal("attributes.source", scores)

    assert not decision.use_for_recommendation
    assert "metadata" in decision.reason


def test_label_quality_signal_accepts_semantic_labels():
    from ignite_tools.eval.recommend import assess_label_quality_signal

    scores = [
        ModelScore(
            model_id="model",
            auc=0.72,
            auc_ci_low=0.68,
            auc_ci_high=0.76,
            separation=0.12,
            num_pairs=100,
        )
    ]

    decision = assess_label_quality_signal("attributes.category", scores)

    assert decision.use_for_recommendation


def test_label_quality_signal_rejects_non_separating_scores():
    from ignite_tools.eval.recommend import assess_label_quality_signal

    scores = [
        ModelScore(
            model_id="model",
            auc=0.49,
            auc_ci_low=0.44,
            auc_ci_high=0.54,
            separation=-0.01,
            num_pairs=100,
        )
    ]

    decision = assess_label_quality_signal("category", scores)

    assert not decision.use_for_recommendation
    assert "worse than random" in decision.reason


def test_recommend_operationally_picks_fastest_by_default():
    from ignite_tools.eval.recommend import recommend_operationally
    from ignite_tools.eval.runner import EmbedResult

    embed_results = [
        EmbedResult(model_id="fast", model_name="Fast", throughput_warm=5000, peak_memory_mb=600, embed_dim=384),
        EmbedResult(model_id="slow", model_name="Slow", throughput_warm=1000, peak_memory_mb=900, embed_dim=1024),
    ]

    rec = recommend_operationally(embed_results, corpus_size=10000, note="labels are diagnostic")

    assert rec is not None
    assert rec.model_id == "fast"
    assert rec.confidence == "low"
    assert "Quality ranking was not used" in rec.reason


def test_recommend_no_results_returns_none():
    from ignite_tools.eval.recommend import recommend
    assert recommend([], []) is None
