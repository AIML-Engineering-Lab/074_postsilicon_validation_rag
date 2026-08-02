"""Tests for Config Pydantic models (no external services needed)."""
from src.utils.config import (
    AppConfig, EmbeddingsConfig, VectorStoreConfig, LLMConfig,
    DocumentProcessingConfig, RetrievalConfig, ConversationConfig,
    Config, get_config, ConfigManager,
)


def test_app_config_defaults():
    cfg = AppConfig()
    assert cfg.name == "Post-Silicon Validation RAG Platform"
    assert cfg.version == "1.0.0"
    assert cfg.port == 8501
    assert cfg.debug is False


def test_embeddings_config_defaults():
    cfg = EmbeddingsConfig()
    assert "instructor" in cfg.model_name.lower()
    assert cfg.device == "cpu"
    assert cfg.normalize is True


def test_vectorstore_config_defaults():
    cfg = VectorStoreConfig()
    assert cfg.type == "chromadb"
    assert "chromadb" in cfg.persist_directory
    assert cfg.collection_name == "postsilicon_docs"
    assert cfg.distance_metric == "cosine"


def test_llm_config_defaults():
    cfg = LLMConfig()
    assert cfg.provider in ("ollama", "openai")
    assert cfg.ollama.temperature == 0.7
    assert cfg.openai.temperature == 0.7


def test_document_processing_config():
    cfg = DocumentProcessingConfig()
    assert cfg.chunk_size == 500
    assert cfg.chunk_overlap == 50
    assert ".pdf" in cfg.supported_formats
    assert ".txt" in cfg.supported_formats


def test_retrieval_config_defaults():
    cfg = RetrievalConfig()
    assert cfg.top_k == 5
    assert cfg.score_threshold == 0.7
    assert cfg.search_type == "similarity"


def test_conversation_config_defaults():
    cfg = ConversationConfig()
    assert cfg.max_history_length == 50
    assert "markdown" in cfg.export_formats
    assert "json" in cfg.export_formats


def test_get_config_returns_config():
    cfg = get_config()
    assert isinstance(cfg, Config)
    assert hasattr(cfg, "app")
    assert hasattr(cfg, "embeddings")
    assert hasattr(cfg, "vectorstore")


def test_get_config_singleton():
    cfg1 = get_config()
    cfg2 = get_config()
    assert cfg1 is cfg2


def test_config_manager_dot_get():
    mgr = ConfigManager()
    val = mgr.get("app.name")
    assert val == "Post-Silicon Validation RAG Platform"


def test_config_manager_get_missing_returns_default():
    mgr = ConfigManager()
    val = mgr.get("nonexistent.key", default="fallback")
    assert val == "fallback"
