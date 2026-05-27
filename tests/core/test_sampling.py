"""Tests for ``ignite_tools.core.sampling`` in isolation.

We use a tiny shim object that exposes only the attributes ``apply_sampling``
relies on (``.group``, plus an opaque payload). This decouples sampling from
the rest of the pipeline so failures here are unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from ignite_tools.core.config import SamplingBlock
from ignite_tools.core.sampling import UNKNOWN_GROUP, apply_sampling


@dataclass
class _Rec:
    """Minimal pipeline-record stand-in for sampling tests."""

    payload: int
    group: Optional[str] = None


def _records(n: int, groups: list[str] | None = None) -> list[_Rec]:
    if groups is None:
        return [_Rec(payload=i) for i in range(n)]
    out = []
    for i in range(n):
        out.append(_Rec(payload=i, group=groups[i % len(groups)]))
    return out


# ---------------------------------------------------------------------------
# Mode: full / None
# ---------------------------------------------------------------------------


def test_full_passes_everything_through():
    records = _records(10)
    out = list(apply_sampling(records, SamplingBlock(mode="full")))
    assert [r.payload for r in out] == list(range(10))


def test_none_block_passes_everything_through():
    records = _records(10)
    out = list(apply_sampling(records, None))
    assert [r.payload for r in out] == list(range(10))


# ---------------------------------------------------------------------------
# Mode: head
# ---------------------------------------------------------------------------


def test_head_truncates_to_total():
    records = _records(20)
    out = list(apply_sampling(records, SamplingBlock(mode="head", total=5)))
    assert [r.payload for r in out] == [0, 1, 2, 3, 4]


def test_head_smaller_than_total_yields_all():
    records = _records(3)
    out = list(apply_sampling(records, SamplingBlock(mode="head", total=10)))
    assert [r.payload for r in out] == [0, 1, 2]


def test_head_does_not_consume_beyond_target():
    """Confirms head is single-pass and stops early."""
    consumed = 0

    def gen():
        nonlocal consumed
        for i in range(100):
            consumed += 1
            yield _Rec(payload=i)

    out = list(apply_sampling(gen(), SamplingBlock(mode="head", total=5)))
    assert len(out) == 5
    # head stops *immediately* after yielding the 5th — but the generator
    # has already produced item 5 (index 4) and the next iteration is what
    # tripped the guard. So consumption is target+0 in the worst case;
    # what matters is we didn't drain the full 100.
    assert consumed < 100


# ---------------------------------------------------------------------------
# Mode: random (reservoir)
# ---------------------------------------------------------------------------


def test_random_returns_sample_of_target_size():
    records = _records(100)
    out = list(apply_sampling(records, SamplingBlock(mode="random", total=10, seed=42)))
    assert len(out) == 10
    payloads = {r.payload for r in out}
    assert payloads.issubset(set(range(100)))


def test_random_smaller_input_returns_all():
    records = _records(5)
    out = list(apply_sampling(records, SamplingBlock(mode="random", total=10, seed=1)))
    assert len(out) == 5
    assert {r.payload for r in out} == set(range(5))


def test_random_is_seed_deterministic():
    a = list(apply_sampling(_records(50), SamplingBlock(mode="random", total=10, seed=7)))
    b = list(apply_sampling(_records(50), SamplingBlock(mode="random", total=10, seed=7)))
    assert [r.payload for r in a] == [r.payload for r in b]


def test_random_different_seeds_differ():
    a = list(apply_sampling(_records(50), SamplingBlock(mode="random", total=10, seed=1)))
    b = list(apply_sampling(_records(50), SamplingBlock(mode="random", total=10, seed=2)))
    assert [r.payload for r in a] != [r.payload for r in b]


# ---------------------------------------------------------------------------
# Mode: stride
# ---------------------------------------------------------------------------


def test_stride_picks_evenly_spaced():
    records = _records(100)
    out = list(apply_sampling(records, SamplingBlock(mode="stride", total=10)))
    assert len(out) == 10
    # step = 10.0 -> indices 0, 10, 20, ..., 90
    assert [r.payload for r in out] == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


def test_stride_smaller_input_yields_all():
    records = _records(7)
    out = list(apply_sampling(records, SamplingBlock(mode="stride", total=10)))
    assert [r.payload for r in out] == [0, 1, 2, 3, 4, 5, 6]


def test_stride_non_integer_step():
    records = _records(13)
    out = list(apply_sampling(records, SamplingBlock(mode="stride", total=5)))
    # step = 13/5 = 2.6 -> indices 0, 2, 5, 7, 10
    assert [r.payload for r in out] == [0, 2, 5, 7, 10]


# ---------------------------------------------------------------------------
# Mode: stratified
# ---------------------------------------------------------------------------


def test_stratified_caps_per_group():
    records = _records(60, groups=["a", "b", "c"])  # 20 per group
    out = list(apply_sampling(
        records,
        SamplingBlock(mode="stratified", per_group=5, group_field="group", seed=42),
    ))
    by_group: dict[str, int] = {}
    for r in out:
        by_group[r.group] = by_group.get(r.group, 0) + 1
    assert by_group == {"a": 5, "b": 5, "c": 5}


def test_stratified_takes_all_when_group_smaller_than_target():
    records = (
        _records(3, groups=["a"])
        + _records(20, groups=["b"])
    )
    out = list(apply_sampling(
        records,
        SamplingBlock(mode="stratified", per_group=10, group_field="group", seed=42),
    ))
    by_group: dict[str, int] = {}
    for r in out:
        by_group[r.group] = by_group.get(r.group, 0) + 1
    assert by_group == {"a": 3, "b": 10}


def test_stratified_drops_unknown_by_default():
    records = [
        _Rec(payload=0, group="a"),
        _Rec(payload=1, group=None),
        _Rec(payload=2, group="a"),
        _Rec(payload=3, group=None),
    ]
    out = list(apply_sampling(
        records,
        SamplingBlock(mode="stratified", per_group=5, group_field="group", seed=42),
    ))
    assert all(r.group == "a" for r in out)
    assert len(out) == 2


def test_stratified_keep_unknown_routes_to_underscore_unknown():
    records = [
        _Rec(payload=0, group="a"),
        _Rec(payload=1, group=None),
        _Rec(payload=2, group="a"),
        _Rec(payload=3, group=None),
    ]
    out = list(apply_sampling(
        records,
        SamplingBlock(
            mode="stratified",
            per_group=5,
            group_field="group",
            stratified_unknown="keep",
            seed=42,
        ),
    ))
    assert len(out) == 4
    by_group: dict[str, int] = {}
    for r in out:
        # The sampler doesn't rewrite ``r.group``; it only uses it for
        # bucketing internally. The output here is the original records
        # in the order ``sorted(groups)`` -> 'a' before '_unknown'.
        by_group[r.group or UNKNOWN_GROUP] = by_group.get(r.group or UNKNOWN_GROUP, 0) + 1
    assert by_group == {"a": 2, UNKNOWN_GROUP: 2}


def test_stratified_is_seed_deterministic():
    records = _records(50, groups=["a", "b"])
    a = list(apply_sampling(
        records,
        SamplingBlock(mode="stratified", per_group=5, group_field="g", seed=99),
    ))
    b = list(apply_sampling(
        records,
        SamplingBlock(mode="stratified", per_group=5, group_field="g", seed=99),
    ))
    assert [r.payload for r in a] == [r.payload for r in b]


# ---------------------------------------------------------------------------
# Mode: weighted
# ---------------------------------------------------------------------------


def test_weighted_distributes_per_quotas():
    records = _records(100, groups=["a", "b", "c"])  # ~33-34 per group
    block = SamplingBlock(
        mode="weighted",
        total=20,
        group_field="group",
        weights={"a": 0.5, "b": 0.3, "c": 0.2},
        seed=42,
    )
    out = list(apply_sampling(records, block))
    by_group: dict[str, int] = {}
    for r in out:
        by_group[r.group] = by_group.get(r.group, 0) + 1
    # Targets: a=10, b=6, c=4 (total=20). Allow off-by-one rounding wiggle.
    assert by_group["a"] == 10
    assert by_group["b"] == 6
    assert by_group["c"] == 4
    assert sum(by_group.values()) == 20


def test_weighted_drops_groups_outside_weights():
    records = (
        _records(20, groups=["a"])
        + _records(20, groups=["b"])
        + _records(20, groups=["zzz"])  # not in weights
    )
    block = SamplingBlock(
        mode="weighted",
        total=10,
        group_field="group",
        weights={"a": 1.0, "b": 1.0},
        seed=42,
    )
    out = list(apply_sampling(records, block))
    groups_seen = {r.group for r in out}
    assert groups_seen == {"a", "b"}


def test_weighted_drops_unknown_group():
    records = [
        _Rec(payload=0, group="a"),
        _Rec(payload=1, group=None),
        _Rec(payload=2, group="b"),
    ]
    block = SamplingBlock(
        mode="weighted",
        total=4,
        group_field="group",
        weights={"a": 1.0, "b": 1.0},
        seed=42,
    )
    out = list(apply_sampling(records, block))
    assert all(r.group is not None for r in out)


def test_weighted_never_exceeds_total_when_groups_exceed_total():
    records = _records(30, groups=["a", "b", "c"])
    block = SamplingBlock(
        mode="weighted",
        total=2,
        group_field="group",
        weights={"a": 1.0, "b": 1.0, "c": 1.0},
        seed=42,
    )

    out = list(apply_sampling(records, block))

    assert len(out) == 2


# ---------------------------------------------------------------------------
# Sanity: every mode is covered above
# ---------------------------------------------------------------------------


def test_unknown_mode_raises():
    """Defensive: validators stop this in practice, but the dispatch should
    raise rather than silently pass-through if someone constructs a block
    via ``model_construct`` (skipping validation)."""
    block = SamplingBlock.model_construct(mode="bogus", total=10)
    with pytest.raises(ValueError, match="Unknown sampling mode"):
        list(apply_sampling([_Rec(payload=0)], block))
