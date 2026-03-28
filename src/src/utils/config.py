"""
Configuration Manager
Loads and manages application configuration from YAML and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class EmbeddingsConfig(BaseModel):
    """Embeddings configuration."""
    model_name: str = "hkunlp/instructor-large"
    device: str = "cpu"
    normalize: bool = True
    cache_folder: str = "./models/instructor-large"
    instruction: str = "Represent the semiconductor test document for retrieval:"


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    type: str = "chromadb"
    persist_directory: str = "./data/chromadb"
    collection_name: str = "postsilicon_docs"
    distance_metric: str = "cosine"


class OllamaConfig(BaseModel):
    """Ollama LLM configuration."""
    model: str = "llama2"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 2048


class OpenAIConfig(BaseModel):
    """OpenAI LLM configuration."""
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 2048
    api_key_env: str = "OPENAI_API_KEY"


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str = "ollama"
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)


class DocumentProcessingConfig(BaseModel):
    """Document processing configuration."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_file_size_mb: int = 1024
    max_corpus_size_gb: int = 5
    supported_formats: List[str] = [".txt", ".log", ".report", ".csv", ".xlsx", ".pdf", ".docx"]
    encodings: List[str] = ["utf-8", "latin-1", "ascii", "cp1252"]


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    top_k: int = 5
    score_threshold: float = 0.7
    search_type: str = "similarity"
    mmr_lambda: float = 0.5


class ConversationConfig(BaseModel):
    """Conversation configuration."""
    save_directory: str = "./data/conversations"
    auto_save_interval: int = 5
    max_history_length: int = 50
    export_formats: List[str] = ["markdown", "json"]


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    log_directory: str = "./logs"
    log_file: str = "rag_platform.log"
    rotation: str = "10 MB"
    retention: str = "30 days"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"


class AppConfig(BaseModel):
    """Application configuration."""
    name: str = "Post-Silicon Validation RAG Platform"
    version: str = "1.0.0"
    port: int = 8501
    debug: bool = False


class Config(BaseSettings):
    """Main configuration class."""
    
    app: AppConfig = Field(default_factory=AppConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    document_processing: DocumentProcessingConfig = Field(default_factory=DocumentProcessingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    model_config = {
        "extra": "allow",  # Allow extra fields
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }


class ConfigManager:
    """Manages application configuration."""
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[Config] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load_config()
    
    def load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file and environment variables."""
        # Load environment variables
        load_dotenv()
        
        # Determine config file path
        if config_path is None:
            config_path = os.getenv("CONFIG_FILE", "./config/config.yaml")
        
        config_file = Path(config_path)
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Merge with environment variables
            self._config = Config(**config_dict)
        else:
            # Use defaults
            self._config = Config()
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self._config.vectorstore.persist_directory,
            self._config.conversation.save_directory,
            self._config.logging.log_directory,
            self._config.embeddings.cache_folder,
            "./data/raw",
            "./data/processed"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @property
    def config(self) -> Config:
        """Get configuration object."""
        if self._config is None:
            self.load_config()
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            else:
                return default
        
        return value
    
    def reload(self, config_path: Optional[str] = None) -> None:
        """Reload configuration."""
        self._config = None
        self.load_config(config_path)


# Singleton instance
config_manager = ConfigManager()


def get_config() -> Config:
    """Get configuration instance."""
    return config_manager.config
