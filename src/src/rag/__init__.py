"""
RAG Module
Contains RAG pipeline, ingestion, and conversation management.
"""

from src.rag.rag_pipeline import get_rag_pipeline, RAGPipeline, RAGResponse
from src.rag.ingestion import get_ingestion_pipeline, DocumentIngestionPipeline
from src.rag.conversation import get_conversation_manager, ConversationManager, Message

__all__ = [
    'get_rag_pipeline',
    'RAGPipeline',
    'RAGResponse',
    'get_ingestion_pipeline',
    'DocumentIngestionPipeline',
    'get_conversation_manager',
    'ConversationManager',
    'Message'
]
