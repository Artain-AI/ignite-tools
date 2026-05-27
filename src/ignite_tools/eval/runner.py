"""
Embedding runner - orchestrates subprocess-per-model benchmarking.

Responsibilities:
- Write corpus texts to a temp file (one text per line)
- Detect available device (CPU/CUDA)
- Check which models need downloading; ask consent once
- Spawn a fresh subprocess per model (sequential)
- Collect JSON results from each worker
- Return structured results for the CLI to display

Process isolation is non-negotiable (docs/evaluator.md) - testing showed that
same-process sequential benchmarking contaminates throughput numbers.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EmbedResult:
    """Result from one model's embedding run."""

    model_id: str
    model_name: str = ""
    device: str = "cpu"
    load_time_s: float = 0.0
    warmup_time_s: float = 0.0
    embed_time_s: float = 0.0
    throughput_warm: float = 0.0
    throughput_e2e: float = 0.0
    tokens_per_sec: float = 0.0
    avg_text_chars: float = 0.0
    peak_memory_mb: float = 0.0
    embed_dim: int = 0
    num_texts: int = 0
    vectors_path: str = ""
    error: Optional[str] = None


@dataclass
class RunResult:
    """Aggregate results from the full embedding phase."""

    results: list[EmbedResult] = field(default_factory=list)
    corpus_path: str = ""
    output_dir: str = ""
    device: str = "cpu"
    num_texts: int = 0
    hardware_summary: str = ""


def get_hardware_summary(device: str) -> str:
    """Detect and format hardware info for the report.

    Returns a human-readable string like:
    'Apple M2 Max, 32 GB RAM, MPS (Apple Silicon GPU)'
    'Intel Xeon W-2255, 64 GB RAM, CPU only'
    'AMD EPYC 7763, 128 GB RAM, NVIDIA A100 (CUDA)'
    """
    import platform
    import psutil

    parts = []

    # CPU / chip detection.
    try:
        import subprocess as sp
        # Mac: sysctl gives the actual chip name.
        chip = sp.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
        if chip:
            parts.append(chip)
    except Exception:
        pass

    if not parts:
        # Linux / fallback: platform.processor() or /proc/cpuinfo.
        proc = platform.processor()
        if proc and proc != "arm":
            parts.append(proc)
        else:
            parts.append(f"{platform.machine()}")

    # RAM.
    try:
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        parts.append(f"{ram_gb:.0f} GB RAM")
    except Exception:
        pass

    # GPU / device.
    if device == "cuda":
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            parts.append(f"{gpu_name} x1 ({gpu_count} available)")
        except Exception:
            parts.append("NVIDIA GPU x1")
    elif device == "mps":
        parts.append("Apple Silicon GPU (MPS)")
    else:
        parts.append("CPU only")

    return ", ".join(parts)


def detect_device(force_gpu: bool = False) -> str:
    """Detect available compute device.

    Priority: cuda > mps > cpu.
    - cuda: NVIDIA GPUs (Linux/Windows)
    - mps: Apple Silicon (M1/M2/M3/M4 Macs)
    - cpu: fallback

    Prints a message showing what was detected.
    """
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"  Device: cuda ({device_name})", file=sys.stderr)
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print(f"  Device: mps (Apple Silicon GPU)", file=sys.stderr)
            return "mps"

        if force_gpu:
            print(
                "  WARNING: --gpu specified but no GPU device found (no CUDA, no MPS). "
                "Falling back to CPU.",
                file=sys.stderr,
            )
    except ImportError:
        pass

    print("  Device: cpu", file=sys.stderr)
    return "cpu"


def estimate_download_size(model_ids: list[str], registry: list) -> tuple[list[str], int]:
    """Check which models likely need downloading and estimate total size.

    Returns (models_needing_download, total_mb).
    Uses HuggingFace cache dir to detect already-cached models.
    """
    from huggingface_hub import scan_cache_dir

    try:
        cache_info = scan_cache_dir()
        cached_repos = {repo.repo_id for repo in cache_info.repos}
    except Exception:
        cached_repos = set()

    needs_download = []
    total_mb = 0
    for mid in model_ids:
        if mid not in cached_repos:
            needs_download.append(mid)
            # Look up size from registry.
            entry = next((m for m in registry if m.id == mid), None)
            total_mb += entry.size_mb if entry else 500  # default estimate

    return needs_download, total_mb


def ask_download_consent(
    models_to_download: list[str],
    total_mb: int,
    *,
    auto_yes: bool = False,
    prompt_fn=None,
) -> bool:
    """Ask the user once before downloading models.

    Returns True if consent given, False to abort.
    """
    if not models_to_download:
        return True  # nothing to download

    if auto_yes:
        print(
            f"  Downloading {len(models_to_download)} models (~{total_mb / 1024:.1f} GB)...",
            file=sys.stderr,
        )
        return True

    print(f"\n  Models to download ({len(models_to_download)}):", file=sys.stderr)
    for mid in models_to_download:
        print(f"    • {mid}", file=sys.stderr)
    print(f"  Total download: ~{total_mb / 1024:.1f} GB", file=sys.stderr)

    if prompt_fn is None:
        prompt_fn = input
    response = prompt_fn("  Proceed with download? [y/n] ").strip().lower()
    return response in {"y", "yes", ""}


def download_models(model_ids: list[str]) -> tuple[float, dict[str, bool]]:
    """Download all models before benchmarking.

    Downloads happen sequentially. Returns (total_seconds, {model_id: success}).
    Uses huggingface_hub to download without loading into memory.

    Progress bars are suppressed (HF_HUB_DISABLE_PROGRESS_BARS) to keep
    output clean. The download status is reported by our own print statements.
    """
    import time
    import logging
    from huggingface_hub import snapshot_download

    # Suppress HF's tqdm progress bars - we print our own status.
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    results: dict[str, bool] = {}
    t0 = time.perf_counter()

    for mid in model_ids:
        print(f"  Downloading {mid}...", file=sys.stderr, end=" ", flush=True)
        try:
            snapshot_download(mid, local_files_only=False)
            results[mid] = True
            print("done", file=sys.stderr)
        except Exception as exc:
            results[mid] = False
            print(f"FAILED: {_sanitize_error(str(exc))}", file=sys.stderr)

    elapsed = time.perf_counter() - t0

    # Also suppress background logging that might leak after download.
    for logger_name in ("huggingface_hub", "huggingface_hub.utils", "filelock"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    return elapsed, results


def run_embeddings(
    texts: list[str],
    model_ids: list[str],
    model_names: dict[str, str],
    *,
    device: str = "cpu",
    output_dir: Optional[str] = None,
) -> RunResult:
    """Run embedding benchmarks for all models sequentially.

    Each model gets a fresh subprocess. The corpus is written to a temp
    file once; all workers read from it.
    """
    # Create output directory.
    if output_dir is None:
        run_id = str(uuid.uuid4())[:8]
        output_dir = str(Path(tempfile.gettempdir()) / f"ignite-eval-{run_id}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Write corpus as JSONL (not raw lines - texts may contain newlines).
    corpus_path = str(Path(output_dir) / "corpus.jsonl")
    import orjson
    with open(corpus_path, "wb") as f:
        for text in texts:
            f.write(orjson.dumps({"text": text}) + b"\n")

    result = RunResult(
        corpus_path=corpus_path,
        output_dir=output_dir,
        device=device,
        num_texts=len(texts),
        hardware_summary=get_hardware_summary(device),
    )

    for model_id in model_ids:
        print(f"  Embedding with {model_names.get(model_id, model_id)}...",
              file=sys.stderr, end=" ", flush=True)

        embed_result = _run_single_model(
            model_id=model_id,
            corpus_path=corpus_path,
            output_dir=output_dir,
            device=device,
        )
        embed_result.model_name = model_names.get(model_id, model_id)
        result.results.append(embed_result)

        if embed_result.error:
            print(f"FAILED: {embed_result.error}", file=sys.stderr)
        else:
            print(
                f"done ({embed_result.throughput_warm:.0f} texts/s, "
                f"{embed_result.embed_time_s:.1f}s)",
                file=sys.stderr,
            )

    return result


def _run_single_model(
    model_id: str,
    corpus_path: str,
    output_dir: str,
    device: str,
    timeout: int = 600,
) -> EmbedResult:
    """Spawn a subprocess for one model and capture the result."""
    cmd = [
        sys.executable, "-m", "ignite_tools.eval.worker",
        "--model", model_id,
        "--input", corpus_path,
        "--output-dir", output_dir,
        "--device", device,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        detail = _bounded_output(exc.stderr or exc.output)
        msg = "timed out (>10 min)"
        if detail:
            msg += f": {detail}"
        return EmbedResult(model_id=model_id, error=_sanitize_error(msg))
    except Exception as exc:
        return EmbedResult(model_id=model_id, error=_sanitize_error(f"subprocess failed: {exc}"))

    if proc.returncode != 0:
        # Try to extract a useful error from stderr.
        err_msg = proc.stderr.strip().split("\n")[-1] if proc.stderr else "unknown error"
        err_msg = _sanitize_error(err_msg)
        return EmbedResult(model_id=model_id, error=err_msg)

    # Parse JSON from stdout.
    try:
        data = json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        return EmbedResult(
            model_id=model_id,
            error=_sanitize_error(f"worker produced invalid JSON: {proc.stdout[:200]!r}"),
        )

    if "error" in data:
        return EmbedResult(model_id=model_id, error=_sanitize_error(str(data["error"])))

    return EmbedResult(
        model_id=data.get("model_id", model_id),
        device=data.get("device", device),
        load_time_s=data.get("load_time_s", 0),
        warmup_time_s=data.get("warmup_time_s", 0),
        embed_time_s=data.get("embed_time_s", 0),
        throughput_warm=data.get("throughput_warm", 0),
        throughput_e2e=data.get("throughput_e2e", 0),
        tokens_per_sec=data.get("tokens_per_sec", 0),
        avg_text_chars=data.get("avg_text_chars", 0),
        peak_memory_mb=data.get("peak_memory_mb", 0),
        embed_dim=data.get("embed_dim", 0),
        num_texts=data.get("num_texts", 0),
        vectors_path=data.get("vectors_path", ""),
    )


def _bounded_output(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return str(value).strip()[:500]


def _sanitize_error(message: str) -> str:
    """Redact common token-bearing fragments before printing user-facing errors."""
    patterns = [
        (r"hf_[A-Za-z0-9_]+", "hf_[REDACTED]"),
        (r"(?i)(token|access_token|sig|signature)=([^&\s]+)", r"\1=[REDACTED]"),
        (r"(?i)(authorization:\s*bearer\s+)[^\s]+", r"\1[REDACTED]"),
    ]
    for pattern, replacement in patterns:
        message = re.sub(pattern, replacement, message)
    return message
