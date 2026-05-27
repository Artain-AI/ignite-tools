"""
Shared embedding layer. Two backends, one interface.

Timing is emitted via the ``ignite_tools.core.embed`` logger at INFO level.
The CLI configures this logger to print to stderr by default so the timing
math stays visible. Library users who import ``embed()`` can silence or
redirect it like any other logger.

Backend selection precedence: explicit ``backend`` argument (typically from
the CLI ``--backend`` flag) > ``IGNITE_BACKEND`` environment variable >
``sentence-transformers`` default. Never auto-detected.
"""

import logging
import os
import time
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = "sentence-transformers"
BACKEND_ENV_VAR = "IGNITE_BACKEND"

# Module-level model cache. Loading a SentenceTransformer is expensive
# (hundreds of MB, disk + tokenizer init), and tools like the evaluator
# call embed() once per candidate model, sometimes repeatedly across
# folds/samples. Keyed by (backend, model). Cleared with clear_model_cache().
_MODEL_CACHE: dict[tuple[str, str], object] = {}


def resolve_backend(backend: str | None = None) -> str:
    """Resolve the embedding backend per the documented precedence.

    Order: explicit argument > IGNITE_BACKEND env var > DEFAULT_BACKEND.
    """
    if backend is not None:
        return backend
    env = os.environ.get(BACKEND_ENV_VAR)
    if env:
        return env
    return DEFAULT_BACKEND


def embed(
    texts: list[str],
    model: str = "intfloat/e5-small-v2",
    backend: str | None = None,
) -> np.ndarray:
    backend = resolve_backend(backend)
    start = time.time()

    if backend == "sentence-transformers":
        vectors = _embed_st(texts, model)
    elif backend == "ignitems":
        vectors = _embed_ignitems(texts, model)
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Use 'sentence-transformers' or 'ignitems'."
        )

    elapsed = time.time() - start
    rate = len(texts) / elapsed if elapsed > 0 else 0
    logger.info(
        "Embedded %s texts in %.1fs (%.0f texts/s) [backend=%s, model=%s]",
        f"{len(texts):,}",
        elapsed,
        rate,
        backend,
        model,
    )

    return vectors


def clear_model_cache() -> None:
    """Drop all cached model handles. Call to reclaim memory between runs."""
    _MODEL_CACHE.clear()


def _get_st_model(model: str):
    key = ("sentence-transformers", model)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    from sentence_transformers import SentenceTransformer

    handle = SentenceTransformer(model)
    _MODEL_CACHE[key] = handle
    return handle


def _embed_st(texts: list[str], model: str) -> np.ndarray:
    prefix = _get_prefix(model)
    if prefix:
        texts = [prefix + t for t in texts]

    st_model = _get_st_model(model)
    vectors = st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vectors)


def _embed_ignitems(texts: list[str], model: str) -> np.ndarray:
    import subprocess
    import tempfile
    import orjson

    with tempfile.NamedTemporaryFile(mode='wb', suffix='.jsonl', delete=False) as f:
        for text in texts:
            f.write(orjson.dumps({"text": text}) + b"\n")
        input_path = f.name

    output_path = input_path.replace('.jsonl', '.npy')

    try:
        subprocess.run([
            "ignite-ms", "embed",
            "--model", model,
            "--input", input_path,
            "--output", output_path,
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        vectors = np.load(output_path)
    finally:
        os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)

    return vectors


def _get_prefix(model: str) -> str:
    model_lower = model.lower()
    if "e5" in model_lower:
        return "passage: "
    return ""
