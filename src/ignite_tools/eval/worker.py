"""
Embedding worker - runs as a subprocess, one per model.

Invoked by the runner as:
    python -m ignite_tools.eval.worker \
        --model <hf_id> \
        --input <corpus.jsonl> \
        --output-dir <dir> \
        --device <cpu|cuda>

The worker:
1. Loads the model (SentenceTransformer)
2. Reads the corpus from disk
3. Runs a warmup batch (first 32 texts, discarded)
4. Times the full embedding run
5. Saves vectors to <output-dir>/<model_safe_name>.npy
6. Prints a JSON result to stdout (captured by parent)

This module is both a CLI script (python -m) and importable for testing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def _safe_name(model_id: str) -> str:
    """Convert a HuggingFace model ID to a filesystem-safe name."""
    return model_id.replace("/", "__").replace(".", "_")


def _get_prefix(model_id: str) -> str:
    """Return the required input prefix for known model families."""
    lower = model_id.lower()
    if "e5" in lower and "instruct" not in lower:
        return "passage: "
    if "nomic" in lower:
        return "search_document: "
    return ""


def run_worker(
    model_id: str,
    input_path: str,
    output_dir: str,
    device: str = "cpu",
) -> dict:
    """Load model, embed corpus, return timing results.

    Returns a dict with:
    - model_id, device
    - load_time_s: time to load model into memory
    - embed_time_s: time for the full corpus (excluding warmup)
    - warmup_time_s: time for the warmup batch
    - throughput_warm: texts/sec during the timed run
    - throughput_e2e: texts/sec including model load
    - peak_memory_mb: peak RSS of this process
    - embed_dim: dimensionality of the output vectors
    - num_texts: how many texts were embedded
    - vectors_path: where the .npy file was saved
    """
    import psutil

    # ── Read corpus ──────────────────────────────────────────────────────
    import orjson
    texts = []
    with open(input_path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                return {"error": f"malformed corpus JSONL: {exc}", "model_id": model_id}
            if not isinstance(record, dict) or "text" not in record:
                return {"error": "corpus JSONL record missing 'text'", "model_id": model_id}
            text = record["text"]
            if not isinstance(text, str):
                return {"error": "corpus JSONL 'text' must be a string", "model_id": model_id}
            texts.append(text)

    if not texts:
        return {"error": "no texts in corpus file", "model_id": model_id}

    # Apply model prefix.
    prefix = _get_prefix(model_id)
    if prefix:
        prefixed = [prefix + t for t in texts]
    else:
        prefixed = texts

    # ── Load model ───────────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer

    t0 = time.perf_counter()
    model = SentenceTransformer(model_id, device=device)

    # Use half precision on CUDA (most modern GPUs have 2x fp16 throughput).
    # Skip on MPS/CPU where fp16 support varies.
    if device == "cuda":
        model = model.half()

    load_time = time.perf_counter() - t0

    # ── Determine batch size from available hardware ──────────────────────
    batch_size = 64  # safe default for CPU/MPS
    if device == "cuda":
        import torch
        props = torch.cuda.get_device_properties(0)
        vram_gb = (props.total_memory if hasattr(props, 'total_memory') else getattr(props, 'total_mem', 8 * 1024**3)) / (1024**3)
        if vram_gb >= 40:
            batch_size = 1024
        elif vram_gb >= 16:
            batch_size = 512
        elif vram_gb >= 8:
            batch_size = 256
        else:
            batch_size = 128
    elif device == "mps":
        batch_size = 128

    # ── Warmup (stabilizes CUDA kernel compilation and memory allocation) ─
    warmup_batch = prefixed[:min(batch_size, len(prefixed))]
    t0 = time.perf_counter()
    warmup_iters = 3 if device == "cuda" else 1
    for _ in range(warmup_iters):
        model.encode(warmup_batch, normalize_embeddings=True, show_progress_bar=False,
                     batch_size=batch_size)
    warmup_time = time.perf_counter() - t0

    # ── Timed run (full corpus) ──────────────────────────────────────────
    t0 = time.perf_counter()
    vectors = model.encode(
        prefixed,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=batch_size,
        convert_to_numpy=True,
    )
    embed_time = time.perf_counter() - t0

    vectors = np.array(vectors)

    # ── Save vectors ─────────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = out_dir / f"{_safe_name(model_id)}.npy"
    np.save(vectors_path, vectors)

    # ── Measure memory ───────────────────────────────────────────────────
    process = psutil.Process(os.getpid())
    peak_memory_mb = process.memory_info().rss / (1024 * 1024)

    # ── Results ──────────────────────────────────────────────────────────
    num_texts = len(texts)
    throughput_warm = num_texts / embed_time if embed_time > 0 else 0
    total_time = load_time + warmup_time + embed_time
    throughput_e2e = num_texts / total_time if total_time > 0 else 0

    # Estimate tokens (rough: 1 token ~ 4 chars for English text).
    total_chars = sum(len(t) for t in texts)
    estimated_tokens = total_chars // 4
    tokens_per_sec = estimated_tokens / embed_time if embed_time > 0 else 0

    return {
        "model_id": model_id,
        "device": device,
        "load_time_s": round(load_time, 2),
        "warmup_time_s": round(warmup_time, 2),
        "embed_time_s": round(embed_time, 2),
        "throughput_warm": round(throughput_warm, 1),
        "throughput_e2e": round(throughput_e2e, 1),
        "tokens_per_sec": round(tokens_per_sec, 0),
        "avg_text_chars": round(total_chars / num_texts, 0) if num_texts else 0,
        "peak_memory_mb": round(peak_memory_mb, 0),
        "embed_dim": int(vectors.shape[1]),
        "num_texts": num_texts,
        "vectors_path": str(vectors_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ignite-eval-worker",
        description="Embedding worker subprocess (internal, not user-facing).",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--input", required=True, help="Path to corpus.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for .npy output")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    # Redirect stdout to stderr for the duration of model loading/encoding.
    # Libraries (transformers, torch) sometimes print to stdout which would
    # corrupt our JSON result. Only our final json.dumps goes to real stdout.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    result = run_worker(
        model_id=args.model,
        input_path=args.input,
        output_dir=args.output_dir,
        device=args.device,
    )

    # Restore stdout and print the single JSON result.
    sys.stdout = real_stdout
    print(json.dumps(result))


if __name__ == "__main__":
    main()
