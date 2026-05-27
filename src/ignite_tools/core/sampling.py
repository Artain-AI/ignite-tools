"""
Sampling modes for the read layer.

Six modes from `docs/format-config.md` and `docs/read-layer.md`. Each takes
an iterator of pipeline records and yields a (possibly smaller) iterator.

Per-mode characteristics:

- ``full``       O(N) read, O(1) memory. Pass-through.
- ``head``       O(N) read, O(1) memory. Truncates to first N.
- ``random``     O(N) read, O(N_sample) memory. Reservoir sampling.
- ``stride``     O(N) read, O(N) memory. Two-pass behavior - materializes
                 the source once because stride needs the total count to
                 compute step size. This intentionally violates rule 4
                 (no eager materialization) for stride only; the trade-off
                 is documented in `docs/performance.md` discussion of
                 "no silent eager materialization" - the materialization
                 here is named, intentional, and bounded by the upstream
                 filtering having already shrunk the corpus.
- ``stratified`` O(N) read, O(per_group * num_groups) memory. Per-group
                 reservoirs.
- ``weighted``   O(N) read, O(sum(targets)) memory. Per-group reservoirs
                 with per-group quotas computed from ``weights``.

All randomized modes seed from ``SamplingBlock.seed`` so runs are
reproducible by default.
"""

from __future__ import annotations

import itertools
import random
from typing import Iterable, Iterator, TypeVar

from ignite_tools.core.config import SamplingBlock

T = TypeVar("T")

# Bucket key used when ``stratified_unknown == "keep"`` and a record's
# group value is missing or not in the configured weights table.
UNKNOWN_GROUP = "_unknown"


def apply_sampling(
    records: Iterable[T],
    block: SamplingBlock | None,
) -> Iterator[T]:
    """Apply the configured sampling mode to ``records``.

    ``records`` must be an iterable whose elements expose a ``.group``
    attribute (only required for ``stratified`` / ``weighted``). The read
    layer wraps each item in a ``_PipelineRecord`` carrying the resolved
    group; tests can pass any object with the right shape.
    """
    if block is None or block.mode == "full":
        yield from records
        return

    rng = random.Random(block.seed)
    mode = block.mode

    if mode == "head":
        yield from _head(records, block.total)
    elif mode == "random":
        yield from _reservoir(records, block.total, rng)
    elif mode == "stride":
        yield from _stride(records, block.total)
    elif mode == "stratified":
        yield from _stratified(records, block, rng)
    elif mode == "weighted":
        yield from _weighted(records, block, rng)
    else:
        # Unreachable: validator pins ``mode`` to the closed Literal set.
        raise ValueError(f"Unknown sampling mode: {mode!r}")


# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------


def _head(records: Iterable[T], target: int) -> Iterator[T]:
    """First ``target`` records, then stop. Single-pass, no extra consumption."""
    return itertools.islice(records, target)


def _reservoir(records: Iterable[T], target: int, rng: random.Random) -> Iterator[T]:
    """Reservoir sample of size ``target`` (Algorithm R, Vitter 1985).

    Single-pass over the input. Memory bounded by ``target``.
    """
    reservoir: list[T] = []
    for i, record in enumerate(records):
        if i < target:
            reservoir.append(record)
        else:
            j = rng.randint(0, i)
            if j < target:
                reservoir[j] = record
    yield from reservoir


def _stride(records: Iterable[T], target: int) -> Iterator[T]:
    """Every k-th record where ``k = total / target``.

    Materializes the input. Stride sampling is the right tool for time-
    ordered files where ``head`` would bias the sample; it requires the
    total count, hence the materialization. Upstream filtering should
    have already shrunk the input substantially.
    """
    materialized: list[T] = list(records)
    n = len(materialized)
    if n == 0 or target <= 0:
        return
    if n <= target:
        yield from materialized
        return

    # Use a float step so we cover the full range when target doesn't
    # divide n evenly. Indices are rounded down with ``int()``.
    step = n / target
    for i in range(target):
        idx = int(i * step)
        if idx >= n:
            break
        yield materialized[idx]


def _stratified(
    records: Iterable[T], block: SamplingBlock, rng: random.Random
) -> Iterator[T]:
    """Up to ``per_group`` records per group via per-group reservoirs.

    Records whose ``.group`` is ``None`` are dropped (default) or routed to
    the ``_unknown`` bucket per ``stratified_unknown``. Output order:
    sorted-by-group, then reservoir order within each group.
    """
    target = block.per_group
    drop_unknown = block.stratified_unknown == "drop"

    reservoirs: dict[str, list[T]] = {}
    counts: dict[str, int] = {}

    for record in records:
        group = _resolve_group(record, drop_unknown)
        if group is None:
            continue

        n = counts.get(group, 0)
        bucket = reservoirs.setdefault(group, [])
        if n < target:
            bucket.append(record)
        else:
            j = rng.randint(0, n)
            if j < target:
                bucket[j] = record
        counts[group] = n + 1

    for group in sorted(reservoirs.keys()):
        yield from reservoirs[group]


def _weighted(
    records: Iterable[T], block: SamplingBlock, rng: random.Random
) -> Iterator[T]:
    """Per-group quotas computed from ``weights`` and ``total``.

    Targets sum to at most ``total`` after deterministic largest-remainder
    allocation.
    Records whose group isn't in ``weights`` are dropped - ``stratified_
    unknown`` doesn't apply because there's no weight for unknown.
    """
    weights = block.weights or {}
    total = block.total
    weight_sum = sum(weights.values())
    targets = _weighted_targets(weights, total, weight_sum)

    reservoirs: dict[str, list[T]] = {}
    counts: dict[str, int] = {}

    for record in records:
        # Always drop unknown groups for weighted: there's no quota for them.
        group = _resolve_group(record, drop_unknown=True)
        if group is None or group not in targets:
            continue

        target = targets[group]
        n = counts.get(group, 0)
        bucket = reservoirs.setdefault(group, [])
        if n < target:
            bucket.append(record)
        else:
            j = rng.randint(0, n)
            if j < target:
                bucket[j] = record
        counts[group] = n + 1

    for group in sorted(reservoirs.keys()):
        yield from reservoirs[group]


def _weighted_targets(
    weights: dict[str, float], total: int, weight_sum: float
) -> dict[str, int]:
    """Allocate exact weighted quotas using largest remainders."""
    raw = {group: total * weight / weight_sum for group, weight in weights.items()}
    targets = {group: int(quota) for group, quota in raw.items()}
    remaining = total - sum(targets.values())
    by_remainder = sorted(
        raw,
        key=lambda group: (raw[group] - targets[group], raw[group], group),
        reverse=True,
    )
    for group in by_remainder[:remaining]:
        targets[group] += 1
    return {group: target for group, target in targets.items() if target > 0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_group(record: object, drop_unknown: bool) -> str | None:
    """Return the group key for a record, or ``None`` to drop it.

    Handles three cases:
    - record has ``.group`` set      -> use as-is
    - record has ``.group`` None and drop_unknown -> drop
    - record has ``.group`` None and keep -> bucket as ``_unknown``
    """
    group = getattr(record, "group", None)
    if isinstance(group, str) and group:
        return group
    return None if drop_unknown else UNKNOWN_GROUP
