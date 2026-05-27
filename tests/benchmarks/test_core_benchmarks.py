"""Performance smoke benchmarks for review-critical hot paths."""

from __future__ import annotations

import pytest

pytest.importorskip("pytest_benchmark")

from ignite_tools.core.config import SamplingBlock
from ignite_tools.core.sampling import apply_sampling
from ignite_tools.eval.scorer import generate_pairs


class _Rec:
    def __init__(self, group: str):
        self.group = group


def test_benchmark_weighted_sampling(benchmark):
    records = [_Rec(str(i % 10)) for i in range(100_000)]
    block = SamplingBlock(
        mode="weighted",
        total=1_000,
        group_field="group",
        weights={str(i): 1.0 for i in range(10)},
    )

    benchmark(lambda: list(apply_sampling(records, block)))


def test_benchmark_pair_generation(benchmark):
    texts = [f"text document {i} topic {i % 20}" for i in range(10_000)]
    labels = [str(i % 20) for i in range(10_000)]

    benchmark(lambda: generate_pairs(texts, labels, max_pairs=2_000, seed=42))
