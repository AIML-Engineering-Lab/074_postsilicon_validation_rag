"""
Unit tests for document loaders.
"""

import pytest
from pathlib import Path
import tempfile

from src.loaders.document_loader import (
    TextDocumentLoader,
    DocumentLoaderFactory
)


class TestTextDocumentLoader:
    """Test text document loader."""
    
    def test_load_txt_file(self):
        """Test loading .txt file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test validation log\nLine 2\nLine 3")
            f.flush()
            path = Path(f.name)
        
        loader = TextDocumentLoader(str(path))
        documents = loader.load()
        
        assert len(documents) > 0
        assert "Test validation log" in documents[0].page_content
        assert documents[0].metadata['file_type'] == 'txt'
        
        path.unlink()
    
    def test_log_file(self):
        """Test loading .log file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("INFO: Test started\nERROR: Test failed")
            f.flush()
            path = Path(f.name)
        
        loader = TextDocumentLoader(str(path))
        documents = loader.load()
        
        assert len(documents) > 0
        assert "INFO: Test started" in documents[0].page_content
        
        path.unlink()


class TestDocumentLoaderFactory:
    """Test document loader factory."""
    
    def test_txt_loader_creation(self):
        """Test creating loader for .txt file."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            path = Path(f.name)
        
        factory = DocumentLoaderFactory()
        loader = factory.create_loader(path)
        
        assert isinstance(loader, TextDocumentLoader)
        path.unlink()
    
    def test_unsupported_extension(self):
        """Test unsupported file extension."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            path = Path(f.name)
        
        factory = DocumentLoaderFactory()
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            factory.create_loader(path)
        
        path.unlink()
