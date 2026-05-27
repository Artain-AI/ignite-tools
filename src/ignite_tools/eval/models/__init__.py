"""
Model registry loader and query interface.

The registry is a YAML file committed to the repo
(``src/ignite_tools/eval/models/registry.yaml``). This module loads it,
validates entries, and provides query functions the auto-selector uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


_REGISTRY_PATH = Path(__file__).parent / "registry.yaml"


@dataclass
class ModelEntry:
    """One model from the registry. All fields map 1:1 to the YAML schema."""

    id: str
    name: str
    params: str
    dim: int
    max_tokens: int
    multilingual: bool
    languages: list[str] = field(default_factory=list)
    speed_tier: str = "medium"      # fast / medium / slow / very_slow
    quality_tier: str = "good"      # baseline / good / better / best
    size_mb: int = 0
    matryoshka: bool = False
    prefix: str = ""
    best_for: list[str] = field(default_factory=list)
    avoid_when: list[str] = field(default_factory=list)
    description: str = ""


def load_registry(path: Optional[Path] = None) -> list[ModelEntry]:
    """Load and return all models from the registry YAML."""
    path = path or _REGISTRY_PATH
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Registry at {path} must be a YAML list")
    return [ModelEntry(**entry) for entry in raw]


def filter_registry(
    models: list[ModelEntry],
    *,
    multilingual: Optional[bool] = None,
    max_size_mb: Optional[int] = None,
    speed_tiers: Optional[list[str]] = None,
    quality_tiers: Optional[list[str]] = None,
    best_for: Optional[list[str]] = None,
    avoid_when: Optional[list[str]] = None,
    max_tokens_min: Optional[int] = None,
) -> list[ModelEntry]:
    """Filter the registry by constraints. Returns models matching ALL criteria."""
    result = list(models)

    if multilingual is not None:
        result = [m for m in result if m.multilingual == multilingual or (multilingual and m.multilingual)]

    if max_size_mb is not None:
        result = [m for m in result if m.size_mb <= max_size_mb]

    if speed_tiers is not None:
        result = [m for m in result if m.speed_tier in speed_tiers]

    if quality_tiers is not None:
        result = [m for m in result if m.quality_tier in quality_tiers]

    if best_for is not None:
        # Model must have at least one of the requested best_for tags.
        tags = set(best_for)
        result = [m for m in result if tags & set(m.best_for)]

    if avoid_when is not None:
        # Exclude models that have any of the avoid_when tags.
        tags = set(avoid_when)
        result = [m for m in result if not (tags & set(m.avoid_when))]

    if max_tokens_min is not None:
        result = [m for m in result if m.max_tokens >= max_tokens_min]

    return result


def get_model_by_id(models: list[ModelEntry], model_id: str) -> Optional[ModelEntry]:
    """Look up a specific model by its HuggingFace ID."""
    for m in models:
        if m.id == model_id:
            return m
    return None
