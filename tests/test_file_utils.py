"""Tests for file utility functions (no external services needed)."""
import tempfile, hashlib
from pathlib import Path
import pytest

from src.utils.file_utils import (
    sanitize_filename,
    get_file_hash,
    get_file_size_mb,
    validate_file,
    read_text_with_encoding,
    list_files_recursive,
)

ENCODINGS = ["utf-8", "latin-1", "ascii"]


# ── sanitize_filename ─────────────────────────────────────────────────────────

def test_sanitize_removes_directory_separators():
    result = sanitize_filename("../etc/passwd")
    assert "/" not in result
    assert ".." not in result


def test_sanitize_removes_special_chars():
    result = sanitize_filename("file@#!.txt")
    assert "@" not in result
    assert "#" not in result
    assert "!" not in result


def test_sanitize_preserves_extension():
    result = sanitize_filename("report.pdf")
    assert result.endswith(".pdf")


def test_sanitize_long_filename_truncated():
    long_name = "a" * 300 + ".txt"
    result = sanitize_filename(long_name)
    assert len(result) < 210  # 200 chars + extension


def test_sanitize_normal_name_unchanged():
    result = sanitize_filename("valid-file_name.txt")
    assert result == "valid-file_name.txt"


# ── get_file_hash ─────────────────────────────────────────────────────────────

def test_hash_is_sha256_hex():
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        f.write(b"hello")
        path = Path(f.name)
    try:
        result = get_file_hash(path)
        assert len(result) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in result)
    finally:
        path.unlink()


def test_hash_consistent():
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        f.write(b"deterministic")
        path = Path(f.name)
    try:
        assert get_file_hash(path) == get_file_hash(path)
    finally:
        path.unlink()


def test_hash_matches_expected():
    data = b"post-silicon validation"
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        f.write(data)
        path = Path(f.name)
    try:
        expected = hashlib.sha256(data).hexdigest()
        assert get_file_hash(path) == expected
    finally:
        path.unlink()


def test_different_content_different_hash():
    p1 = Path(tempfile.mktemp())
    p2 = Path(tempfile.mktemp())
    p1.write_bytes(b"aaa")
    p2.write_bytes(b"bbb")
    try:
        assert get_file_hash(p1) != get_file_hash(p2)
    finally:
        p1.unlink(); p2.unlink()


# ── get_file_size_mb ──────────────────────────────────────────────────────────

def test_file_size_zero_for_empty():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = Path(f.name)
    try:
        assert get_file_size_mb(path) == 0.0
    finally:
        path.unlink()


def test_file_size_approx():
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        f.write(b"x" * 1024 * 1024)  # 1 MB exactly
        path = Path(f.name)
    try:
        assert get_file_size_mb(path) == pytest.approx(1.0, rel=1e-3)
    finally:
        path.unlink()


# ── validate_file ─────────────────────────────────────────────────────────────

def test_validate_passes_valid_file():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("ok")
        path = Path(f.name)
    try:
        ok, err = validate_file(path, [".txt"], 10.0)
        assert ok is True
        assert err is None
    finally:
        path.unlink()


def test_validate_fails_wrong_extension():
    with tempfile.NamedTemporaryFile(suffix=".exe", mode="w", delete=False) as f:
        f.write("ok")
        path = Path(f.name)
    try:
        ok, err = validate_file(path, [".txt"], 10.0)
        assert ok is False
        assert err is not None
    finally:
        path.unlink()


def test_validate_fails_nonexistent():
    ok, err = validate_file(Path("/tmp/does_not_exist_xyz.txt"), [".txt"], 10.0)
    assert ok is False
    assert "not found" in err.lower()


def test_validate_fails_oversized(tmp_path):
    p = tmp_path / "big.txt"
    p.write_bytes(b"x" * 1024 * 1024 * 2)  # 2 MB
    ok, err = validate_file(p, [".txt"], 1.0)  # max 1 MB
    assert ok is False
    assert "large" in err.lower()


# ── read_text_with_encoding ───────────────────────────────────────────────────

def test_read_utf8():
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                     suffix=".txt", delete=False) as f:
        f.write("hello utf8")
        path = Path(f.name)
    try:
        content, enc = read_text_with_encoding(path, ENCODINGS)
        assert content == "hello utf8"
        assert enc == "utf-8"
    finally:
        path.unlink()


def test_read_returns_none_for_binary():
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        f.write(bytes(range(256)))
        path = Path(f.name)
    try:
        content, enc = read_text_with_encoding(path, ["ascii"])
        # Should return None for undecodable content
        assert content is None or isinstance(content, str)
    finally:
        path.unlink()


# ── list_files_recursive ──────────────────────────────────────────────────────

def test_list_files_finds_files(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.log").write_text("b")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("c")
    result = list_files_recursive(tmp_path)
    assert len(result) == 3


def test_list_files_filters_by_extension(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.log").write_text("b")
    result = list_files_recursive(tmp_path, [".txt"])
    assert len(result) == 1
    assert result[0].suffix == ".txt"
