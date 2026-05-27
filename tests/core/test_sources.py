"""Tests for ``ignite_tools.core.sources``."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest
import zstandard as zstd

from ignite_tools.core.config import StorageBlock
from ignite_tools.core.sources import (
    DEFAULT_INCLUDES,
    iter_lines,
    list_files,
    read_file_text,
)


def _storage(path: Path, **kwargs) -> StorageBlock:
    return StorageBlock(type="local", path=str(path), **kwargs)


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------


def test_list_files_single_file_returns_only_that_file(fixtures_root: Path):
    files = list_files(_storage(fixtures_root / "simple.jsonl"))
    assert files == [fixtures_root / "simple.jsonl"]


def test_list_files_recursive_walk(fixtures_root: Path):
    files = list_files(
        _storage(fixtures_root, recursive=True),
        default_includes=DEFAULT_INCLUDES["jsonl"],
    )
    names = [f.name for f in files]
    assert "simple.jsonl" in names
    assert "noisy.jsonl" in names
    assert "file_a.jsonl" in names
    assert "file_b.jsonl" in names
    assert "simple.jsonl.gz" in names
    assert "simple.jsonl.zst" in names
    # text fixtures should not be picked up by jsonl default includes
    assert "doc1.txt" not in names
    # sorted for determinism
    assert files == sorted(files)


def test_list_files_text_default_includes(fixtures_root: Path):
    files = list_files(
        _storage(fixtures_root, recursive=True),
        default_includes=DEFAULT_INCLUDES["text"],
    )
    names = {f.name for f in files}
    assert names == {"doc1.txt", "doc2.md"}


def test_list_files_csv_default_includes(fixtures_root: Path):
    files = list_files(
        _storage(fixtures_root, recursive=True),
        default_includes=DEFAULT_INCLUDES["csv"],
    )
    names = {f.name for f in files}
    assert names == {
        "products.csv",
        "products.csv.gz",
        "products.csv.zst",
        "events_routed.csv",
        "quirky.csv",
    }


def test_list_files_tsv_default_includes(fixtures_root: Path):
    files = list_files(
        _storage(fixtures_root, recursive=True),
        default_includes=DEFAULT_INCLUDES["tsv"],
    )
    names = {f.name for f in files}
    assert names == {"products.tsv"}


def test_list_files_flat_walk_skips_subdirs(fixtures_root: Path):
    files = list_files(
        _storage(fixtures_root, recursive=False),
        default_includes=DEFAULT_INCLUDES["jsonl"],
    )
    names = {f.name for f in files}
    assert "simple.jsonl" in names
    assert "file_a.jsonl" not in names  # under nested/, recursion off


def test_list_files_include_filter(fixtures_root: Path):
    files = list_files(
        _storage(fixtures_root, recursive=True, include=["file_*.jsonl"])
    )
    names = {f.name for f in files}
    assert names == {"file_a.jsonl", "file_b.jsonl"}


def test_list_files_exclude_filter(fixtures_root: Path):
    files = list_files(
        _storage(
            fixtures_root,
            recursive=True,
            include=["*.jsonl"],
            exclude=["**/file_b.jsonl"],
        )
    )
    names = {f.name for f in files}
    assert "file_a.jsonl" in names
    assert "file_b.jsonl" not in names


def test_list_files_missing_path_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        list_files(_storage(tmp_path / "does-not-exist"))


def test_list_files_genuinely_unknown_type_not_implemented():
    """Defensive: a hypothetical future type bypasses validation but is rejected."""
    storage = StorageBlock.model_construct(type="ftp", path="ftp://x/y")
    with pytest.raises(NotImplementedError):
        list_files(storage)


# ---------------------------------------------------------------------------
# iter_lines: raw, gzip, zst
# ---------------------------------------------------------------------------


def test_iter_lines_streams_bytes(fixtures_root: Path):
    path = fixtures_root / "simple.jsonl"
    lines = list(iter_lines(path))
    assert all(isinstance(line, bytes) for line in lines)
    assert all(not line.endswith(b"\r") and not line.endswith(b"\n") for line in lines)
    assert len(lines) == 50


def test_iter_lines_handles_crlf(tmp_path: Path):
    path = tmp_path / "crlf.jsonl"
    path.write_bytes(b'{"text": "a"}\r\n{"text": "b"}\r\n')
    lines = list(iter_lines(path))
    assert lines == [b'{"text": "a"}', b'{"text": "b"}']


def test_iter_lines_gzip(fixtures_root: Path):
    path = fixtures_root / "compressed" / "simple.jsonl.gz"
    lines = list(iter_lines(path))
    assert len(lines) == 15
    assert all(isinstance(line, bytes) for line in lines)
    assert lines[0].startswith(b'{"id": "rec-0000"')


def test_iter_lines_zstd(fixtures_root: Path):
    path = fixtures_root / "compressed" / "simple.jsonl.zst"
    lines = list(iter_lines(path))
    assert len(lines) == 15
    assert all(isinstance(line, bytes) for line in lines)
    assert lines[0].startswith(b'{"id": "rec-0000"')


def test_iter_lines_gzip_and_raw_yield_same_bytes(tmp_path: Path):
    raw_path = tmp_path / "data.jsonl"
    raw_path.write_text("alpha\nbeta\ngamma\n")

    gz_path = tmp_path / "data.jsonl.gz"
    with gzip.open(gz_path, "wb") as f:
        f.write(b"alpha\nbeta\ngamma\n")

    assert list(iter_lines(raw_path)) == list(iter_lines(gz_path))


def test_iter_lines_zst_and_raw_yield_same_bytes(tmp_path: Path):
    raw_path = tmp_path / "data.jsonl"
    raw_path.write_text("alpha\nbeta\ngamma\n")

    zst_path = tmp_path / "data.jsonl.zst"
    cctx = zstd.ZstdCompressor()
    zst_path.write_bytes(cctx.compress(b"alpha\nbeta\ngamma\n"))

    assert list(iter_lines(raw_path)) == list(iter_lines(zst_path))


# ---------------------------------------------------------------------------
# read_file_text
# ---------------------------------------------------------------------------


def test_read_file_text_decodes_utf8(fixtures_root: Path):
    content = read_file_text(fixtures_root / "text" / "doc1.txt")
    assert content.startswith("first paragraph")
    assert "third line" in content


def test_read_file_text_replaces_bad_bytes(tmp_path: Path):
    path = tmp_path / "bad.txt"
    # Latin-1 byte that's invalid as UTF-8 lead byte
    path.write_bytes(b"hello \xff world")
    content = read_file_text(path)
    assert "hello" in content
    assert "world" in content
    # Default replacement char is U+FFFD
    assert "\ufffd" in content


def test_read_file_text_handles_gzip(tmp_path: Path):
    path = tmp_path / "doc.txt.gz"
    with gzip.open(path, "wb") as f:
        f.write("compressed text content\nsecond line\n".encode("utf-8"))
    content = read_file_text(path)
    assert "compressed text content" in content
    assert "second line" in content
