"""
Unit tests for file utilities.
"""

import pytest
from pathlib import Path
import tempfile

from src.utils.file_utils import (
    sanitize_filename,
    get_file_hash,
    get_file_size_mb,
    validate_file,
    read_text_with_encoding
)


class TestSanitizeFilename:
    """Test filename sanitization."""
    
    def test_basic_sanitization(self):
        assert sanitize_filename("test file.txt") == "test_file.txt"
    
    def test_remove_special_chars(self):
        assert sanitize_filename("test@#$%file.txt") == "test_file.txt"
    
    def test_path_traversal_prevention(self):
        assert sanitize_filename("../../../etc/passwd") == "etc_passwd"
    
    def test_multiple_dots(self):
        assert sanitize_filename("test...file.txt") == "test.file.txt"


class TestFileHash:
    """Test file hashing."""
    
    def test_hash_consistency(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            f.flush()
            path = Path(f.name)
        
        hash1 = get_file_hash(path)
        hash2 = get_file_hash(path)
        
        assert hash1 == hash2
        path.unlink()
    
    def test_different_content_different_hash(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f1:
            f1.write("content1")
            f1.flush()
            path1 = Path(f1.name)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f2:
            f2.write("content2")
            f2.flush()
            path2 = Path(f2.name)
        
        hash1 = get_file_hash(path1)
        hash2 = get_file_hash(path2)
        
        assert hash1 != hash2
        path1.unlink()
        path2.unlink()


class TestFileSize:
    """Test file size calculation."""
    
    def test_small_file(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("x" * 1000)
            f.flush()
            path = Path(f.name)
        
        size = get_file_size_mb(path)
        assert size < 0.01
        path.unlink()


class TestReadTextWithEncoding:
    """Test text reading with encoding detection."""
    
    def test_utf8_file(self):
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
            f.write("Hello World")
            f.flush()
            path = Path(f.name)
        
        content = read_text_with_encoding(path)
        assert content == "Hello World"
        path.unlink()
