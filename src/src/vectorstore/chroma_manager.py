"""
Vector Store Manager
Manages ChromaDB vector store for document embeddings.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.embeddings.embeddings_manager import get_embeddings
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger()


class VectorStoreManager:
    """Manages ChromaDB vector store."""
    
    _instance: Optional['VectorStoreManager'] = None
    _vectorstore: Optional[Chroma] = None
    _text_splitter: Optional[RecursiveCharacterTextSplitter] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._vectorstore is None:
            self._initialize_vectorstore()
            self._initialize_text_splitter()
    
    def _initialize_vectorstore(self) -> None:
        """Initialize ChromaDB vector store."""
        config = get_config()
        vs_config = config.vectorstore
        
        logger.info(f"Initializing ChromaDB: {vs_config.collection_name}")
        
        try:
            persist_dir = Path(vs_config.persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            embeddings = get_embeddings()
            
            self._vectorstore = Chroma(
                collection_name=vs_config.collection_name,
                embedding_function=embeddings,
                persist_directory=str(persist_dir)
            )
            
            logger.info("ChromaDB initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _initialize_text_splitter(self) -> None:
        """Initialize text splitter."""
        config = get_config()
        doc_config = config.document_processing
        
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=doc_config.chunk_size,
            chunk_overlap=doc_config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info(f"Text splitter initialized: chunk_size={doc_config.chunk_size}, "
                   f"overlap={doc_config.chunk_overlap}")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to vector store.
        
        Returns:
            List of document IDs
        """
        try:
            # Split documents into chunks
            split_docs = self._text_splitter.split_documents(documents)
            
            logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
            
            # Add to vector store
            ids = self._vectorstore.add_documents(split_docs)
            
            # Persist
            self._vectorstore.persist()
            
            logger.info(f"Added {len(ids)} chunks to vector store")
            
            return ids
        
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Metadata filter
        
        Returns:
            List of similar documents
        """
        try:
            results = self._vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            
            logger.info(f"Found {len(results)} results for query")
            
            return results
        
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        Search for similar documents with relevance scores.
        
        Returns:
            List of (document, score) tuples
        """
        try:
            results = self._vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            
            logger.info(f"Found {len(results)} results with scores")
            
            return results
        
        except Exception as e:
            logger.error(f"Error during similarity search with score: {e}")
            return []
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Get retriever for RAG pipeline.
        
        Args:
            search_kwargs: Search parameters (k, filter, etc.)
        
        Returns:
            Retriever instance
        """
        if search_kwargs is None:
            config = get_config()
            search_kwargs = {
                "k": config.retrieval.top_k
            }
        
        return self._vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents by IDs."""
        try:
            self._vectorstore.delete(ids=ids)
            self._vectorstore.persist()
            logger.info(f"Deleted {len(ids)} documents from vector store")
        
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise
    
    def delete_by_source(self, source: str) -> None:
        """Delete all documents from a specific source file."""
        try:
            # Get all documents with this source
            results = self._vectorstore.get(where={"source": source})
            
            if results and results['ids']:
                self.delete_documents(results['ids'])
                logger.info(f"Deleted all documents from source: {source}")
        
        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            collection = self._vectorstore._collection
            count = collection.count()
            
            return {
                "total_documents": count,
                "collection_name": self._vectorstore._collection.name
            }
        
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def list_sources(self) -> List[str]:
        """List all unique source files in the collection."""
        try:
            results = self._vectorstore.get()
            
            if not results or 'metadatas' not in results:
                return []
            
            sources = set()
            for metadata in results['metadatas']:
                if 'source' in metadata:
                    sources.add(metadata['source'])
            
            return sorted(list(sources))
        
        except Exception as e:
            logger.error(f"Error listing sources: {e}")
            return []


# Singleton instance
vectorstore_manager = VectorStoreManager()


def get_vectorstore():
    """Get vectorstore manager instance."""
    return vectorstore_manager
