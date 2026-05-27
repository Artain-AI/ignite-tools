"""Focused tests for eval runner/worker error paths."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from ignite_tools.eval.runner import _run_single_model, download_models
from ignite_tools.eval.worker import run_worker


def test_worker_rejects_malformed_jsonl(tmp_path: Path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("{not json}\n", encoding="utf-8")

    result = run_worker("model", str(corpus), str(tmp_path), device="cpu")

    assert "malformed corpus JSONL" in result["error"]


def test_worker_rejects_missing_text(tmp_path: Path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"id": "x"}\n', encoding="utf-8")

    result = run_worker("model", str(corpus), str(tmp_path), device="cpu")

    assert "missing 'text'" in result["error"]


def test_run_single_model_reports_timeout_context():
    exc = subprocess.TimeoutExpired(
        cmd=["python", "-m", "ignite_tools.eval.worker"],
        timeout=1,
        stderr="last stderr line",
    )
    with patch("ignite_tools.eval.runner.subprocess.run", side_effect=exc):
        result = _run_single_model("m", "input.jsonl", "out", "cpu", timeout=1)

    assert result.error is not None
    assert "timed out" in result.error
    assert "last stderr line" in result.error


def test_download_models_sanitizes_token_in_error(capsys):
    with patch("huggingface_hub.snapshot_download") as snapshot_download:
        snapshot_download.side_effect = RuntimeError("bad token=hf_secret123 access_token=abc")
        _, results = download_models(["repo/model"])

    assert results == {"repo/model": False}
    err = capsys.readouterr().err
    assert "abc" not in err
    assert "hf_secret123" not in err
    assert "[REDACTED]" in err
