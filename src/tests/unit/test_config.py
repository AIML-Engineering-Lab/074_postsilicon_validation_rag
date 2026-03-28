"""
Unit tests for configuration manager.
"""

import pytest
from pathlib import Path

from src.utils.config import get_config, ConfigManager


class TestConfigManager:
    """Test configuration manager."""
    
    def test_singleton_pattern(self):
        """Test that ConfigManager is a singleton."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
    
    def test_config_structure(self):
        """Test config has required sections."""
        config = get_config()
        assert hasattr(config, 'app')
        assert hasattr(config, 'embeddings')
        assert hasattr(config, 'vectorstore')
        assert hasattr(config, 'llm')
        assert hasattr(config, 'document_processing')
        assert hasattr(config, 'retrieval')
    
    def test_app_config(self):
        """Test app configuration."""
        config = get_config()
        assert config.app.name == "Post-Silicon Validation RAG Platform"
        assert config.app.version == "1.0.0"
        assert isinstance(config.app.port, int)
    
    def test_embeddings_config(self):
        """Test embeddings configuration."""
        config = get_config()
        assert config.embeddings.model_name == "hkunlp/instructor-large"
        assert config.embeddings.device in ["cpu", "cuda"]
    
    def test_vectorstore_config(self):
        """Test vectorstore configuration."""
        config = get_config()
        assert config.vectorstore.provider == "chromadb"
        assert "chromadb" in config.vectorstore.chroma.persist_directory
    
    def test_llm_config(self):
        """Test LLM configuration."""
        config = get_config()
        assert config.llm.provider in ["ollama", "openai"]
