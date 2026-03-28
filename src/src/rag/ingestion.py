"""
Document Ingestion Pipeline
Orchestrates document loading, embedding, and storage.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from src.loaders.document_loader import DocumentLoaderFactory
from src.vectorstore.chroma_manager import get_vectorstore
from src.utils.logger import get_logger
from src.utils.config import get_config
from src.utils.file_utils import (
    validate_file,
    get_file_hash,
    get_file_size_mb
)

logger = get_logger()


class DocumentIngestionPipeline:
    """Pipeline for ingesting documents."""
    
    def __init__(self):
        self.config = get_config()
        self.vectorstore = get_vectorstore()
    
    def ingest_file(self, file_path: str) -> Dict:
        """
        Ingest a single file.
        
        Args:
            file_path: Path to file
        
        Returns:
            Dict with ingestion results
        """
        try:
            logger.info(f"Ingesting file: {file_path}")
            
            # Validate file
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Get validation parameters from config
            allowed_extensions = self.config.document_processing.supported_formats
            max_size_mb = self.config.document_processing.max_file_size_mb
            
            is_valid, error_msg = validate_file(file_path, allowed_extensions, max_size_mb)
            if not is_valid:
                raise ValueError(error_msg)
            
            # Get file metadata
            file_hash = get_file_hash(file_path)
            file_size = get_file_size_mb(file_path)
            
            # Load documents using factory
            documents = DocumentLoaderFactory.load_document(file_path)
            
            if not documents:
                raise ValueError(f"No content extracted from file: {file_path}")
            
            # Add to vectorstore
            doc_ids = self.vectorstore.add_documents(documents)
            
            result = {
                "status": "success",
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_hash": file_hash,
                "file_size_mb": file_size,
                "num_documents": len(documents),
                "num_chunks": len(doc_ids),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"File ingested successfully: {file_path.name} ({len(doc_ids)} chunks)")
            
            return result
        
        except Exception as e:
            logger.error(f"Error ingesting file {file_path}: {e}")
            return {
                "status": "error",
                "file_path": str(file_path),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def ingest_directory(self, directory_path: str, recursive: bool = True) -> List[Dict]:
        """
        Ingest all files in a directory.
        
        Args:
            directory_path: Path to directory
            recursive: Process subdirectories
        
        Returns:
            List of ingestion results
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        logger.info(f"Ingesting directory: {directory_path}")
        
        # Get all supported files
        pattern = "**/*" if recursive else "*"
        supported_extensions = ['.txt', '.log', '.report', '.csv', '.xlsx', '.pdf', '.docx']
        
        files = []
        for ext in supported_extensions:
            files.extend(directory_path.glob(f"{pattern}{ext}"))
        
        logger.info(f"Found {len(files)} files to ingest")
        
        # Ingest each file
        results = []
        for file_path in files:
            result = self.ingest_file(str(file_path))
            results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - successful
        
        logger.info(f"Directory ingestion complete: {successful} successful, {failed} failed")
        
        return results
    
    def ingest_text_content(self, text: str, source_name: str = "pasted_content") -> Dict:
        """
        Ingest text content directly.
        
        Args:
            text: Text content
            source_name: Name for this content
        
        Returns:
            Dict with ingestion results
        """
        try:
            from langchain_core.documents import Document
            
            logger.info(f"Ingesting text content: {source_name}")
            
            # Create document
            document = Document(
                page_content=text,
                metadata={
                    "source": source_name,
                    "file_type": "text",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Add to vectorstore
            doc_ids = self.vectorstore.add_documents([document])
            
            result = {
                "status": "success",
                "source_name": source_name,
                "num_chunks": len(doc_ids),
                "content_length": len(text),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Text content ingested: {source_name} ({len(doc_ids)} chunks)")
            
            return result
        
        except Exception as e:
            logger.error(f"Error ingesting text content: {e}")
            return {
                "status": "error",
                "source_name": source_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def delete_document(self, source_name: str) -> Dict:
        """
        Delete a document by source name.
        
        Args:
            source_name: Source name or file path
        
        Returns:
            Dict with deletion results
        """
        try:
            logger.info(f"Deleting document: {source_name}")
            
            self.vectorstore.delete_by_source(source_name)
            
            result = {
                "status": "success",
                "source_name": source_name,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Document deleted: {source_name}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error deleting document {source_name}: {e}")
            return {
                "status": "error",
                "source_name": source_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def list_documents(self) -> List[str]:
        """List all ingested documents."""
        return self.vectorstore.list_sources()
    
    def get_stats(self) -> Dict:
        """Get ingestion statistics."""
        return self.vectorstore.get_collection_stats()


# Singleton instance
ingestion_pipeline = DocumentIngestionPipeline()


def get_ingestion_pipeline() -> DocumentIngestionPipeline:
    """Get ingestion pipeline instance."""
    return ingestion_pipeline
