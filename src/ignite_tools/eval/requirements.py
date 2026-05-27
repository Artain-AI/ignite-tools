"""
Eval requirements - user-specified criteria for model selection.

These come from the ``ignite-eval:`` block in the config file and represent
what the user TELLS us about their needs (vs what we OBSERVE from their data).

Three levels:
- task:        what they're doing (mandatory mental model for the selection)
- priority:    what matters most (quality vs throughput vs balanced vs size)
- constraints: hard limits that eliminate models
- prefer:      soft signals that boost models (don't eliminate)
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# All common tasks where embeddings are the mechanism.
TASK_TYPES = [
    "search",       # query → retrieve relevant documents
    "classify",     # assign items to categories
    "cluster",      # find natural groupings
    "similarity",   # pairwise "how similar are these two?"
    "deduplicate",  # find near-duplicates above a threshold
    "rag",          # retrieval-augmented generation (search + long context)
    "match",        # two-sided matching (resumes↔jobs, products↔queries)
    "general",      # don't know yet / multiple tasks / exploratory
]

# Priority - what matters most to the user.
PRIORITY_TYPES = [
    "quality",      # best possible results, don't care about speed
    "throughput",   # fastest embedding, "good enough" quality
    "balanced",     # default - optimize both
    "size",         # smallest model/vectors (edge deployment, memory-constrained)
]

# Mode - how embeddings will be used at runtime.
MODE_TYPES = [
    "batch",        # embed a corpus once (or periodically). Throughput matters.
    "realtime",     # embed one query at a time (search, API). Single-inference latency matters.
]


class EvalConstraints(BaseModel):
    """Hard constraints that eliminate models."""

    model_config = ConfigDict(extra="forbid")

    max_size_mb: Optional[int] = None
    max_dim: Optional[int] = None
    min_quality: Optional[Literal["baseline", "good", "better", "best"]] = None
    languages: Optional[list[str]] = None

    @field_validator("max_size_mb", "max_dim")
    @classmethod
    def _positive_int(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("must be a positive integer")
        return v

    @field_validator("languages")
    @classmethod
    def _normalize_languages(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is None:
            return None
        normalized: list[str] = []
        seen: set[str] = set()
        for lang in v:
            code = lang.strip().lower()
            if not code:
                raise ValueError("languages must not contain empty values")
            if code not in seen:
                seen.add(code)
                normalized.append(code)
        return normalized


class EvalRequirements(BaseModel):
    """User-specified requirements for model selection.

    All fields are optional. When omitted, the selector relies purely on
    data signals + hardware. When specified, these override or constrain
    the data-driven selection.
    """

    model_config = ConfigDict(extra="forbid")

    task: Optional[str] = None
    priority: Literal["quality", "throughput", "balanced", "size"] = "balanced"
    mode: Literal["batch", "realtime"] = "batch"
    prefer: list[str] = Field(default_factory=list)
    constraints: EvalConstraints = Field(default_factory=EvalConstraints)

    @field_validator("task")
    @classmethod
    def _validate_task(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in TASK_TYPES:
            raise ValueError(f"task must be one of {TASK_TYPES}, got {v!r}")
        return v

    def effective_task(self) -> str:
        """Return task or 'general' as default."""
        return self.task if self.task and self.task in TASK_TYPES else "general"
