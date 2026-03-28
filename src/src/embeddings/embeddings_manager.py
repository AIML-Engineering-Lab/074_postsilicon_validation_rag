"""
Embeddings Manager
Handles document embeddings using instructor-large model.
"""

from typing import List, Optional
from pathlib import Path

from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from InstructorEmbedding import INSTRUCTOR

from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger()


class EmbeddingsManager:
    """Manages document embeddings."""
    
    _instance: Optional['EmbeddingsManager'] = None
    _embeddings: Optional[HuggingFaceInstructEmbeddings] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._embeddings is None:
            self._initialize_embeddings()
    
    def _initialize_embeddings(self) -> None:
        """Initialize instructor-large embeddings."""
        config = get_config()
        emb_config = config.embeddings
        
        logger.info(f"Initializing embeddings: {emb_config.model_name}")
        logger.info(f"Device: {emb_config.device}")
        
        try:
            # Ensure cache directory exists
            cache_dir = Path(emb_config.cache_folder)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            self._embeddings = HuggingFaceInstructEmbeddings(
                model_name=emb_config.model_name,
                model_kwargs={
                    "device": emb_config.device
                },
                encode_kwargs={
                    "normalize_embeddings": emb_config.normalize
                },
                cache_folder=str(cache_dir)
            )
            
            logger.info("Embeddings initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def get_embeddings(self) -> HuggingFaceInstructEmbeddings:
        """Get embeddings instance."""
        if self._embeddings is None:
            self._initialize_embeddings()
        return self._embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed list of documents."""
        return self._embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query."""
        return self._embeddings.embed_query(text)


# Singleton instance
embeddings_manager = EmbeddingsManager()


def get_embeddings():
    """Get embeddings instance."""
    return embeddings_manager.get_embeddings()
