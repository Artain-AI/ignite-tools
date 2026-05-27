"""Tests for cloud storage materialization (S3 + Azure).

We don't hit real cloud services. Instead we mock ``shutil.which`` to
simulate the s5cmd / azcopy presence-or-absence flag and mock
``subprocess.run`` (or the SDK clients) to "download" by writing files
into the cache dir directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ignite_tools.core.config import StorageBlock
from ignite_tools.core.sources import (
    DEFAULT_INCLUDES,
    _cache_key,
    _parse_azure_uri,
    _parse_s3_uri,
    cache_dir_for,
    list_files,
)


# ---------------------------------------------------------------------------
# URI parsing
# ---------------------------------------------------------------------------


def test_parse_s3_uri_with_prefix():
    bucket, prefix = _parse_s3_uri("s3://my-bucket/some/prefix/")
    assert bucket == "my-bucket"
    assert prefix == "some/prefix/"


def test_parse_s3_uri_bucket_only():
    bucket, prefix = _parse_s3_uri("s3://my-bucket")
    assert bucket == "my-bucket"
    assert prefix == ""


def test_parse_s3_uri_rejects_non_s3():
    with pytest.raises(ValueError):
        _parse_s3_uri("/local/path")


def test_parse_azure_uri_full():
    account, container, prefix = _parse_azure_uri(
        "azure://acct/cont/some/prefix/"
    )
    assert account == "acct"
    assert container == "cont"
    assert prefix == "some/prefix/"


def test_parse_azure_uri_no_prefix():
    account, container, prefix = _parse_azure_uri("azure://acct/cont")
    assert account == "acct"
    assert container == "cont"
    assert prefix == ""


def test_parse_azure_uri_rejects_missing_container():
    with pytest.raises(ValueError):
        _parse_azure_uri("azure://acct")


# ---------------------------------------------------------------------------
# Cache key + cache_dir_for
# ---------------------------------------------------------------------------


def test_cache_key_is_deterministic():
    a = _cache_key("s3://bucket/prefix/")
    b = _cache_key("s3://bucket/prefix/")
    assert a == b
    assert len(a) == 32


def test_cache_key_differs_between_uris():
    a = _cache_key("s3://bucket/prefix/")
    b = _cache_key("s3://bucket/other/")
    assert a != b


def test_cache_dir_for_local_returns_path(tmp_path: Path):
    storage = StorageBlock(type="local", path=str(tmp_path))
    assert cache_dir_for(storage) == tmp_path


def test_cache_dir_for_s3_uses_explicit_cache_dir(tmp_path: Path):
    storage = StorageBlock(
        type="s3",
        path="s3://my-bucket/x/",
        cache_dir=str(tmp_path),
    )
    cd = cache_dir_for(storage)
    assert cd.parent == tmp_path
    assert cd.name == _cache_key("s3://my-bucket/x/")


# ---------------------------------------------------------------------------
# StorageBlock validation
# ---------------------------------------------------------------------------


def test_s3_storage_requires_s3_uri():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="s3://"):
        StorageBlock(type="s3", path="/local/path")


def test_azure_storage_requires_azure_uri():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="azure://"):
        StorageBlock(type="azure", path="/local/path")


def test_local_storage_rejects_cloud_uri():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="cloud URI"):
        StorageBlock(type="local", path="s3://bucket/x/")


def test_region_only_valid_for_s3():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="only valid when storage.type is 's3'"):
        StorageBlock(type="azure", path="azure://a/c/", region="us-east-1")


# ---------------------------------------------------------------------------
# S3 materialization
# ---------------------------------------------------------------------------


def _populate_cache(cache_dir: Path) -> None:
    """Helper: simulate a download by writing a couple of fixture files."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    p1 = cache_dir / "a.jsonl"
    p1.write_text('{"id":"a","text":"hello"}\n', encoding="utf-8")
    sub = cache_dir / "sub"
    sub.mkdir()
    p2 = sub / "b.jsonl"
    p2.write_text('{"id":"b","text":"world"}\n', encoding="utf-8")
    (cache_dir / ".complete").write_text("ok\n", encoding="utf-8")


def test_s3_materialize_uses_s5cmd_when_available(tmp_path: Path):
    """When s5cmd is on PATH, we shell out to it."""
    storage = StorageBlock(
        type="s3",
        path="s3://bucket/prefix/",
        cache_dir=str(tmp_path / "cache"),
    )

    target_cache = cache_dir_for(storage)
    invocations: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        invocations.append(cmd)
        # Pretend s5cmd downloaded something into the cache dir.
        _populate_cache(Path(cmd[3]))
        # Mimic CompletedProcess
        from subprocess import CompletedProcess
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch(
        "ignite_tools.core.sources.shutil.which",
        side_effect=lambda name: "/usr/local/bin/s5cmd" if name == "s5cmd" else None,
    ), patch("ignite_tools.core.sources.subprocess.run", side_effect=fake_run):
        files = list_files(storage, default_includes=DEFAULT_INCLUDES["jsonl"])

    # Exactly one subprocess invocation, and it was s5cmd.
    assert len(invocations) == 1
    assert invocations[0][0] == "s5cmd"
    assert invocations[0][1] == "cp"
    assert invocations[0][2].startswith("s3://bucket/prefix/")
    # Cache populated; both files discovered.
    assert {f.name for f in files} == {"a.jsonl", "b.jsonl"}


def test_s3_materialize_falls_back_to_boto3_with_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    """When s5cmd is missing, we use boto3 and emit a loud warning."""
    storage = StorageBlock(
        type="s3",
        path="s3://bucket/x/",
        cache_dir=str(tmp_path / "cache"),
    )

    boto3_called = []

    def fake_boto3_download(uri, cache_dir, region):
        boto3_called.append((uri, cache_dir, region))
        _populate_cache(cache_dir)

    with patch(
        "ignite_tools.core.sources.shutil.which",
        side_effect=lambda name: None,  # nothing on PATH
    ), patch(
        "ignite_tools.core.sources._boto3_download",
        side_effect=fake_boto3_download,
    ):
        files = list_files(storage, default_includes=DEFAULT_INCLUDES["jsonl"])

    err = capsys.readouterr().err
    assert "s5cmd not found" in err
    assert "boto3" in err
    assert "10-30x slower" in err
    assert len(boto3_called) == 1
    assert {f.name for f in files} == {"a.jsonl", "b.jsonl"}


def test_s3_cache_hit_skips_download(tmp_path: Path):
    """If the cache dir is already populated, we don't re-download."""
    storage = StorageBlock(
        type="s3",
        path="s3://bucket/y/",
        cache_dir=str(tmp_path / "cache"),
    )
    _populate_cache(cache_dir_for(storage))

    invocations: list = []

    with patch(
        "ignite_tools.core.sources.shutil.which",
        side_effect=lambda name: "/x/s5cmd" if name == "s5cmd" else None,
    ), patch(
        "ignite_tools.core.sources.subprocess.run",
        side_effect=lambda *a, **kw: invocations.append(a) or None,
    ):
        files = list_files(storage, default_includes=DEFAULT_INCLUDES["jsonl"])

    assert invocations == []  # no subprocess call
    assert {f.name for f in files} == {"a.jsonl", "b.jsonl"}


def test_s3_partial_cache_without_complete_marker_redownloads(tmp_path: Path):
    storage = StorageBlock(
        type="s3",
        path="s3://bucket/partial/",
        cache_dir=str(tmp_path / "cache"),
    )
    cache = cache_dir_for(storage)
    cache.mkdir(parents=True)
    (cache / "partial.jsonl").write_text("{}\n", encoding="utf-8")
    invocations: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        invocations.append(cmd)
        _populate_cache(Path(cmd[3]))
        from subprocess import CompletedProcess
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch(
        "ignite_tools.core.sources.shutil.which",
        side_effect=lambda name: "/x/s5cmd" if name == "s5cmd" else None,
    ), patch("ignite_tools.core.sources.subprocess.run", side_effect=fake_run):
        files = list_files(storage, default_includes=DEFAULT_INCLUDES["jsonl"])

    assert len(invocations) == 1
    assert {f.name for f in files} == {"a.jsonl", "b.jsonl"}


def test_s3_no_cache_forces_redownload(tmp_path: Path):
    """``no_cache=True`` wipes the cache dir before downloading."""
    storage = StorageBlock(
        type="s3",
        path="s3://bucket/z/",
        cache_dir=str(tmp_path / "cache"),
    )
    target_cache = cache_dir_for(storage)
    _populate_cache(target_cache)
    # Add an extra file so we can detect the wipe.
    (target_cache / "stale.jsonl").write_text("{}\n")

    def fake_run(cmd, **kwargs):
        # Simulate a fresh download — re-create the standard fixture set.
        _populate_cache(Path(cmd[3]))
        from subprocess import CompletedProcess
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch(
        "ignite_tools.core.sources.shutil.which",
        side_effect=lambda name: "/x/s5cmd" if name == "s5cmd" else None,
    ), patch(
        "ignite_tools.core.sources.subprocess.run",
        side_effect=fake_run,
    ):
        files = list_files(
            storage,
            default_includes=DEFAULT_INCLUDES["jsonl"],
            no_cache=True,
        )

    # ``stale.jsonl`` got wiped; only the standard fixtures remain.
    assert {f.name for f in files} == {"a.jsonl", "b.jsonl"}


def test_s3_subprocess_arguments(tmp_path: Path):
    """Region propagates as AWS_REGION env var, pattern uses '**' suffix."""
    storage = StorageBlock(
        type="s3",
        path="s3://bucket/p",
        region="eu-west-1",
        cache_dir=str(tmp_path / "cache"),
    )

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        _populate_cache(Path(cmd[3]))
        from subprocess import CompletedProcess
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch(
        "ignite_tools.core.sources.shutil.which",
        side_effect=lambda name: "/x/s5cmd" if name == "s5cmd" else None,
    ), patch("ignite_tools.core.sources.subprocess.run", side_effect=fake_run):
        list_files(storage, default_includes=DEFAULT_INCLUDES["jsonl"])

    assert captured["cmd"][0] == "s5cmd"
    assert captured["cmd"][1] == "cp"
    # Trailing slash added; pattern is ``**`` for recursive.
    assert captured["cmd"][2] == "s3://bucket/p/**"
    assert captured["env"].get("AWS_REGION") == "eu-west-1"


# ---------------------------------------------------------------------------
# Azure materialization
# ---------------------------------------------------------------------------


def test_azure_materialize_uses_azcopy_when_available(tmp_path: Path):
    storage = StorageBlock(
        type="azure",
        path="azure://acct/cont/prefix/",
        cache_dir=str(tmp_path / "cache"),
    )
    target_cache = cache_dir_for(storage)
    invocations: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        invocations.append(cmd)
        _populate_cache(Path(cmd[3]))
        from subprocess import CompletedProcess
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch(
        "ignite_tools.core.sources.shutil.which",
        side_effect=lambda name: "/x/azcopy" if name == "azcopy" else None,
    ), patch("ignite_tools.core.sources.subprocess.run", side_effect=fake_run):
        files = list_files(storage, default_includes=DEFAULT_INCLUDES["jsonl"])

    assert invocations[0][0] == "azcopy"
    assert invocations[0][1] == "copy"
    assert invocations[0][2].startswith(
        "https://acct.blob.core.windows.net/cont/prefix/"
    )
    assert {f.name for f in files} == {"a.jsonl", "b.jsonl"}


def test_azure_materialize_falls_back_to_sdk_with_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    storage = StorageBlock(
        type="azure",
        path="azure://acct/cont/",
        cache_dir=str(tmp_path / "cache"),
    )
    sdk_calls = []

    def fake_sdk(account, container, prefix, cache_dir):
        sdk_calls.append((account, container, prefix, cache_dir))
        _populate_cache(cache_dir)

    with patch(
        "ignite_tools.core.sources.shutil.which",
        side_effect=lambda name: None,
    ), patch(
        "ignite_tools.core.sources._azure_sdk_download",
        side_effect=fake_sdk,
    ):
        files = list_files(storage, default_includes=DEFAULT_INCLUDES["jsonl"])

    err = capsys.readouterr().err
    assert "azcopy not found" in err
    assert "azure-storage-blob" in err
    assert sdk_calls[0][:3] == ("acct", "cont", "")
    assert {f.name for f in files} == {"a.jsonl", "b.jsonl"}


# ---------------------------------------------------------------------------
# End-to-end via read_corpus
# ---------------------------------------------------------------------------


def test_read_corpus_against_cloud_cache(tmp_path: Path):
    """Once materialized, cloud sources behave identically to local for
    everything downstream of list_files."""
    from ignite_tools.core.config import FormatConfig
    from ignite_tools.core.format import read_corpus

    storage = StorageBlock(
        type="s3",
        path="s3://bucket/full/",
        cache_dir=str(tmp_path / "cache"),
    )
    _populate_cache(cache_dir_for(storage))

    cfg = FormatConfig.from_dict({
        "storage": {
            "type": "s3",
            "path": "s3://bucket/full/",
            "cache_dir": str(tmp_path / "cache"),
            "recursive": True,
        },
        "format": {"type": "jsonl"},
        "text": {"fields": ["text"]},
        "id": {"field": "id"},
    })

    with patch(
        "ignite_tools.core.sources.shutil.which",
        side_effect=lambda name: "/x/s5cmd" if name == "s5cmd" else None,
    ):
        items = list(read_corpus(cfg))

    assert {it.id for it in items} == {"a", "b"}
