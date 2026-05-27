"""
Storage layer.

Scope at this slice:
- Local filesystem
- S3 (``s3://bucket/prefix/``) - bulk download via ``s5cmd`` subprocess,
  with ``boto3`` SDK fallback when the tool is not on PATH
- Azure Blob (``azure://account/container/prefix/``) - bulk download via
  ``azcopy`` subprocess, with ``azure-storage-blob`` SDK fallback
- JSONL / CSV / TSV / plain text, optionally compressed with .gz or .zst
- Recursive walk with sorted output for determinism
- include / exclude glob filtering

Cloud strategy (per docs/performance.md rule 1):
- Cache to disk first, then read locally. No streaming-from-cloud.
- Cache dir derived from ``sha256(uri)`` so multiple corpora coexist.
- Detect ``s5cmd`` / ``azcopy`` on PATH at materialization time. When the
  external tool is missing we fall back to the SDK with a LOUD warning to
  stderr - never silently slow.
- ``no_cache=True`` wipes the cache dir before downloading.

Credentials NEVER live in the format config. Cloud SDKs use their default
credential chains: env vars, ``~/.aws/credentials`` / ``az login``, IMDS,
managed identity. The format config is safe to commit.

Public surface:
- ``list_files(storage, default_includes, no_cache)`` -> sorted ``list[Path]``
- ``iter_lines(path, encoding)`` -> generator of stripped ``bytes`` per line
- ``read_file_text(path, encoding)`` -> whole-file content as ``str``
- ``cache_dir_for(storage)`` -> ``Path`` of where this corpus would cache to
"""

from __future__ import annotations

import fnmatch
import gzip
import hashlib
import io
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator

from ignite_tools.core.config import StorageBlock

# ---------------------------------------------------------------------------
# Default include patterns by format type
# ---------------------------------------------------------------------------

DEFAULT_INCLUDES: dict[str, list[str]] = {
    "jsonl": [
        "*.jsonl",
        "*.ndjson",
        "*.jsonl.gz",
        "*.ndjson.gz",
        "*.jsonl.zst",
        "*.ndjson.zst",
    ],
    "csv": [
        "*.csv",
        "*.csv.gz",
        "*.csv.zst",
    ],
    "tsv": [
        "*.tsv",
        "*.tsv.gz",
        "*.tsv.zst",
    ],
    "text": ["*.txt", "*.md"],
}

# Where cloud sources cache to by default. Overridable via storage.cache_dir
# or the ``--cache-dir`` CLI flag.
_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "ignite-tools"
_CACHE_COMPLETE = ".complete"


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def list_files(
    storage: StorageBlock,
    default_includes: list[str] | None = None,
    *,
    no_cache: bool = False,
) -> list[Path]:
    """Resolve a :class:`StorageBlock` into a deterministic, sorted list of files.

    For ``storage.type == "local"`` this walks the filesystem directly.
    For cloud types, files are downloaded to a local cache dir first
    (``~/.cache/ignite-tools/<hash>/`` by default) and the cache directory
    is then walked. Per ``docs/performance.md`` rule 1, bulk download uses
    ``s5cmd`` / ``azcopy`` as subprocesses when available; SDK fallback
    emits a LOUD warning so users know they're on the slow path.
    """
    if storage.type == "local":
        local_root = Path(storage.path).expanduser()
    elif storage.type == "s3":
        local_root = _materialize_s3(storage, no_cache=no_cache)
    elif storage.type == "azure":
        local_root = _materialize_azure(storage, no_cache=no_cache)
    else:
        raise NotImplementedError(
            f"Storage type {storage.type!r} is not supported."
        )

    return _walk_local(local_root, storage, default_includes)


def cache_dir_for(storage: StorageBlock) -> Path:
    """Return the cache directory used for this storage block (cloud only).

    Useful for tests and tooling that want to inspect / pre-populate the
    cache without going through ``list_files``.
    """
    if storage.type == "local":
        return Path(storage.path).expanduser()
    cache_root = Path(storage.cache_dir).expanduser() if storage.cache_dir else _DEFAULT_CACHE_ROOT
    return cache_root / _cache_key(storage.path)


def iter_lines(path: Path, encoding: str = "utf-8") -> Iterator[bytes]:
    """Yield raw lines as ``bytes`` with trailing CR/LF stripped.

    Streaming generator (rule 4): never loads the file in full. Compression
    is auto-detected by suffix (``.gz`` -> gzip, ``.zst`` -> zstandard).

    Note: the ``encoding`` parameter is accepted for API compatibility but
    lines are always yielded as raw bytes. Callers that need strings should
    decode explicitly. JSON parsing (orjson) works directly on bytes.
    """
    with _open_binary_lines(path) as fh:
        for line in fh:
            yield line.rstrip(b"\r\n")


def read_file_text(path: Path, encoding: str = "utf-8") -> str:
    """Read a file's full content as a decoded string."""
    with _open_binary_lines(path) as fh:
        raw = fh.read()
    return raw.decode(encoding, errors="replace")


# ---------------------------------------------------------------------------
# Local walking
# ---------------------------------------------------------------------------


def _walk_local(
    base: Path,
    storage: StorageBlock,
    default_includes: list[str] | None,
) -> list[Path]:
    if not base.exists():
        raise FileNotFoundError(f"Storage path does not exist: {base}")

    if base.is_file():
        return [base]

    includes = storage.include or default_includes
    excludes = storage.exclude or []
    walker = base.rglob("*") if storage.recursive else base.glob("*")
    files: list[Path] = []
    for path in walker:
        if not path.is_file():
            continue
        if path.name == _CACHE_COMPLETE:
            continue
        rel = str(path.relative_to(base))
        if includes and not (_matches_any(path.name, includes) or _matches_any(rel, includes)):
            continue
        if excludes and _matches_any(rel, excludes):
            continue
        files.append(path)

    return sorted(files)


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


# ---------------------------------------------------------------------------
# Compression-aware line opener
# ---------------------------------------------------------------------------


@contextmanager
def _open_binary_lines(path: Path) -> Iterator[IO[bytes]]:
    """Yield a binary file-like object usable for line iteration."""
    suffix = path.suffix.lower()

    if suffix == ".gz":
        with gzip.open(path, "rb") as fh:
            yield fh
        return

    if suffix == ".zst":
        # Prefer zstd CLI (much faster than Python library for large files).
        if shutil.which("zstd"):
            proc = subprocess.Popen(
                ["zstd", "-d", "-c", str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                yield proc.stdout
            finally:
                proc.stdout.close()
                rc = proc.wait()
                if rc != 0:
                    stderr = proc.stderr.read().decode(errors="replace")[:200]
                    raise IOError(
                        f"zstd decompression failed for {path} (exit {rc}): {stderr}"
                    )
                proc.stderr.close()
            return
        # Fallback to Python zstandard.
        try:
            import zstandard as zstd_lib
        except ImportError:
            raise ImportError(
                ".zst files require 'zstandard'. Install with: pip install ignite-tools[formats]"
            )
        with path.open("rb") as raw:
            dctx = zstd_lib.ZstdDecompressor()
            with dctx.stream_reader(raw) as reader:
                yield io.BufferedReader(reader, buffer_size=128 * 1024)
        return

    with path.open("rb") as fh:
        yield fh


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_key(uri: str) -> str:
    """Stable cache key for a cloud URI. Truncated sha256 to keep paths short."""
    return hashlib.sha256(uri.encode("utf-8")).hexdigest()[:32]


def _resolve_cache_dir(storage: StorageBlock) -> Path:
    cache_root = (
        Path(storage.cache_dir).expanduser()
        if storage.cache_dir
        else _DEFAULT_CACHE_ROOT
    )
    return cache_root / _cache_key(storage.path)


def _safe_cache_path(cache_dir: Path, rel: str) -> Path:
    """Resolve a relative path safely inside cache_dir. Rejects traversal."""
    # Reject absolute paths and any ".." component outright.
    if rel.startswith("/") or ".." in Path(rel).parts:
        raise ValueError(
            f"Unsafe path rejected: {rel!r} (absolute or contains '..')"
        )
    dest = (cache_dir / rel).resolve()
    # Use relative_to for robust boundary check (raises ValueError if outside).
    try:
        dest.relative_to(cache_dir.resolve())
    except ValueError:
        raise ValueError(
            f"Path traversal detected: {rel!r} resolves outside cache dir"
        )
    return dest


def _prepare_cache_dir(cache_dir: Path, no_cache: bool) -> bool:
    """Make sure cache_dir exists. Returns True if a download is needed."""
    if no_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
    if cache_dir.exists() and (cache_dir / _CACHE_COMPLETE).exists():
        return False
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    return True


def _tmp_cache_dir(cache_dir: Path) -> Path:
    return cache_dir.with_name(f"{cache_dir.name}.tmp-{os.getpid()}-{time.time_ns()}")


def _validate_cache_contents(cache_dir: Path) -> None:
    root = cache_dir.resolve()
    for path in cache_dir.rglob("*"):
        try:
            path.resolve().relative_to(root)
        except ValueError:
            raise ValueError(f"download produced path outside cache dir: {path}")


def _promote_cache(tmp_dir: Path, cache_dir: Path) -> None:
    _validate_cache_contents(tmp_dir)
    (tmp_dir / _CACHE_COMPLETE).write_text("ok\n", encoding="utf-8")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    tmp_dir.replace(cache_dir)


@contextmanager
def _cache_lock(cache_dir: Path, timeout_s: float = 300.0) -> Iterator[None]:
    """Single-host lock so concurrent processes do not share partial caches."""
    lock_path = cache_dir.with_name(f"{cache_dir.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii"))
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for cache lock: {lock_path}")
            time.sleep(0.1)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------


def _materialize_s3(storage: StorageBlock, *, no_cache: bool) -> Path:
    cache_dir = _resolve_cache_dir(storage)
    with _cache_lock(cache_dir):
        needs_download = _prepare_cache_dir(cache_dir, no_cache)
        if not needs_download:
            return cache_dir

        tmp_dir = _tmp_cache_dir(cache_dir)
        tmp_dir.mkdir(parents=True, exist_ok=False)
        try:
            if shutil.which("s5cmd"):
                _s5cmd_download(storage.path, tmp_dir, region=storage.region)
            else:
                _print_sdk_fallback_warning(
                    tool="s5cmd",
                    sdk="boto3",
                    install_hint="https://github.com/peak/s5cmd",
                )
                _boto3_download(storage.path, tmp_dir, region=storage.region)
            _promote_cache(tmp_dir, cache_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
    return cache_dir


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 URI: {uri!r}")
    rest = uri[len("s3://"):]
    if "/" in rest:
        bucket, prefix = rest.split("/", 1)
    else:
        bucket, prefix = rest, ""
    return bucket, prefix


def _s5cmd_download(uri: str, cache_dir: Path, region: str | None) -> None:
    """Download all files under ``uri`` into ``cache_dir`` via s5cmd subprocess.

    s5cmd's wildcard semantics: ``s3://bucket/prefix/**`` recursively matches
    everything under the prefix; ``s3://bucket/prefix/*`` matches one level.
    We use ``**`` since the user can still apply ``include`` / ``exclude``
    globs at list time.
    """
    if not uri.endswith("/"):
        uri += "/"
    pattern = f"{uri}**"
    env = os.environ.copy()
    if region:
        env["AWS_REGION"] = region

    # Trailing slash on the destination tells s5cmd to treat it as a directory.
    dst = str(cache_dir).rstrip("/") + "/"
    subprocess.run(
        ["s5cmd", "cp", pattern, dst],
        check=True,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _boto3_download(uri: str, cache_dir: Path, region: str | None) -> None:
    """SDK fallback: list and download keys one at a time via boto3."""
    import boto3

    bucket, prefix = _parse_s3_uri(uri)
    client = boto3.client("s3", region_name=region) if region else boto3.client("s3")

    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            rel = key
            if prefix:
                rel = key[len(prefix):].lstrip("/") if key.startswith(prefix) else key
            if not rel:
                continue
            dest = _safe_cache_path(cache_dir, rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(dest))


# ---------------------------------------------------------------------------
# Azure Blob
# ---------------------------------------------------------------------------


def _materialize_azure(storage: StorageBlock, *, no_cache: bool) -> Path:
    cache_dir = _resolve_cache_dir(storage)
    with _cache_lock(cache_dir):
        needs_download = _prepare_cache_dir(cache_dir, no_cache)
        if not needs_download:
            return cache_dir

        account, container, prefix = _parse_azure_uri(storage.path)
        tmp_dir = _tmp_cache_dir(cache_dir)
        tmp_dir.mkdir(parents=True, exist_ok=False)
        try:
            if shutil.which("azcopy"):
                _azcopy_download(account, container, prefix, tmp_dir)
            else:
                _print_sdk_fallback_warning(
                    tool="azcopy",
                    sdk="azure-storage-blob",
                    install_hint="https://github.com/Azure/azure-storage-azcopy",
                )
                _azure_sdk_download(account, container, prefix, tmp_dir)
            _promote_cache(tmp_dir, cache_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
    return cache_dir


def _parse_azure_uri(uri: str) -> tuple[str, str, str]:
    """Parse ``azure://account/container/prefix/`` -> (account, container, prefix)."""
    if not uri.startswith("azure://"):
        raise ValueError(f"not an azure URI: {uri!r}")
    rest = uri[len("azure://"):]
    parts = rest.split("/", 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(
            "azure URI must be 'azure://<account>/<container>[/<prefix>]', "
            f"got {uri!r}"
        )
    account = parts[0]
    container = parts[1]
    prefix = parts[2] if len(parts) > 2 else ""
    return account, container, prefix


def _azcopy_download(
    account: str, container: str, prefix: str, cache_dir: Path
) -> None:
    """Download all blobs under ``prefix`` via azcopy subprocess."""
    suffix = "" if not prefix or prefix.endswith("/") else "/"
    src = f"https://{account}.blob.core.windows.net/{container}/{prefix}{suffix}*"
    subprocess.run(
        ["azcopy", "copy", src, str(cache_dir), "--recursive=true"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _azure_sdk_download(
    account: str, container: str, prefix: str, cache_dir: Path
) -> None:
    """SDK fallback: list blobs and stream-download via azure-storage-blob."""
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    credential = DefaultAzureCredential()
    service = BlobServiceClient(
        account_url=f"https://{account}.blob.core.windows.net",
        credential=credential,
    )
    container_client = service.get_container_client(container)
    for blob in container_client.list_blobs(name_starts_with=prefix):
        rel = blob.name
        if prefix:
            rel = blob.name[len(prefix):].lstrip("/") if blob.name.startswith(prefix) else blob.name
        if not rel:
            continue
        dest = _safe_cache_path(cache_dir, rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as fh:
            stream = container_client.download_blob(blob)
            stream.readinto(fh)


# ---------------------------------------------------------------------------
# Loud SDK-fallback warning
# ---------------------------------------------------------------------------


def _print_sdk_fallback_warning(*, tool: str, sdk: str, install_hint: str) -> None:
    """Print a clear, repeated warning to stderr.

    We print directly (not via ``warnings.warn``) so users get the message
    on every run, not just the first import - and so it shows up alongside
    other CLI output without needing logging configuration.
    """
    print(
        f"WARNING: {tool} not found on PATH. Falling back to {sdk} for bulk download.",
        file=sys.stderr,
    )
    print(
        f"         This will be 10-30x slower for large corpora.",
        file=sys.stderr,
    )
    print(
        f"         Install: {install_hint}",
        file=sys.stderr,
    )
