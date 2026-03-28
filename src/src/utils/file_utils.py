"""
File Utilities
Helper functions for file operations.
"""

import hashlib
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False

from src.utils.logger import get_logger

logger = get_logger()


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal attacks."""
    # Remove directory separators
    filename = os.path.basename(filename)
    
    # Remove potentially dangerous characters
    filename = re.sub(r'[^\w\s\-\.]', '_', filename)
    
    # Limit length
    name, ext = os.path.splitext(filename)
    if len(name) > 200:
        name = name[:200]
    
    return f"{name}{ext}"


def get_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of file."""
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()


def get_file_mime_type(file_path: Path) -> str:
    """Get MIME type of file."""
    try:
        if HAS_MAGIC:
            mime = magic.Magic(mime=True)
            return mime.from_file(str(file_path))
        else:
            # Fallback to mimetypes module
            import mimetypes
            mime_type, _ = mimetypes.guess_type(str(file_path))
            return mime_type or "application/octet-stream"
    except Exception as e:
        logger.warning(f"Could not determine MIME type for {file_path}: {e}")
        return "application/octet-stream"


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def validate_file(
    file_path: Path,
    allowed_extensions: List[str],
    max_size_mb: float
) -> Tuple[bool, Optional[str]]:
    """
    Validate file.
    
    Returns:
        (is_valid, error_message)
    """
    # Check existence
    if not file_path.exists():
        return False, f"File not found: {file_path}"
    
    # Check extension
    ext = file_path.suffix.lower()
    if ext not in allowed_extensions:
        return False, f"Unsupported file format: {ext}. Allowed: {', '.join(allowed_extensions)}"
    
    # Check size
    size_mb = get_file_size_mb(file_path)
    if size_mb > max_size_mb:
        return False, f"File too large: {size_mb:.2f}MB. Max: {max_size_mb}MB"
    
    return True, None


def ensure_directory(directory: Path) -> None:
    """Ensure directory exists."""
    directory.mkdir(parents=True, exist_ok=True)


def list_files_recursive(directory: Path, extensions: Optional[List[str]] = None) -> List[Path]:
    """List all files in directory recursively."""
    files = []
    
    for item in directory.rglob("*"):
        if item.is_file():
            if extensions is None or item.suffix.lower() in extensions:
                files.append(item)
    
    return files


def read_text_with_encoding(file_path: Path, encodings: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Read text file with multiple encoding attempts.
    
    Returns:
        (text_content, encoding_used)
    """
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            logger.debug(f"Successfully read {file_path} with {encoding} encoding")
            return content, encoding
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Error reading {file_path} with {encoding}: {e}")
            continue
    
    return None, None
