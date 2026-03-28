"""
Document Loaders
Handles loading and processing of various document formats.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    PyPDFLoader,
    Docx2txtLoader
)

from src.utils.logger import get_logger
from src.utils.file_utils import read_text_with_encoding, get_file_hash
from src.utils.config import get_config

logger = get_logger()


class BaseDocumentLoader(ABC):
    """Base class for document loaders."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.config = get_config()
    
    @abstractmethod
    def load(self) -> List[Document]:
        """Load documents from file."""
        pass
    
    def _add_metadata(self, documents: List[Document]) -> List[Document]:
        """Add common metadata to documents."""
        file_hash = get_file_hash(self.file_path)
        file_size = self.file_path.stat().st_size / (1024 * 1024)  # MB
        
        for i, doc in enumerate(documents):
            doc.metadata.update({
                "source": str(self.file_path.name),
                "file_path": str(self.file_path),
                "file_type": self.file_path.suffix,
                "file_hash": file_hash,
                "file_size_mb": round(file_size, 2),
                "chunk_id": i,
                "total_chunks": len(documents)
            })
        
        return documents


class TextDocumentLoader(BaseDocumentLoader):
    """Loader for text files (.txt, .log, .report)."""
    
    def load(self) -> List[Document]:
        """Load text file."""
        encodings = self.config.document_processing.encodings
        content, encoding = read_text_with_encoding(self.file_path, encodings)
        
        if content is None:
            logger.error(f"Failed to read {self.file_path} with any encoding")
            return []
        
        doc = Document(
            page_content=content,
            metadata={"encoding": encoding}
        )
        
        return self._add_metadata([doc])


class CSVDocumentLoader(BaseDocumentLoader):
    """Loader for CSV files."""
    
    def load(self) -> List[Document]:
        """Load CSV file."""
        try:
            df = pd.read_csv(self.file_path)
            
            # Convert to document
            # Option 1: Each row as a document
            documents = []
            for idx, row in df.iterrows():
                content = "\n".join([f"{col}: {val}" for col, val in row.items()])
                doc = Document(
                    page_content=content,
                    metadata={"row_number": idx}
                )
                documents.append(doc)
            
            return self._add_metadata(documents)
        
        except Exception as e:
            logger.error(f"Error loading CSV {self.file_path}: {e}")
            return []


class ExcelDocumentLoader(BaseDocumentLoader):
    """Loader for Excel files (.xlsx)."""
    
    def load(self) -> List[Document]:
        """Load Excel file."""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(self.file_path)
            documents = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(self.file_path, sheet_name=sheet_name)
                
                # Convert each row to document
                for idx, row in df.iterrows():
                    content = "\n".join([f"{col}: {val}" for col, val in row.items()])
                    doc = Document(
                        page_content=content,
                        metadata={
                            "sheet_name": sheet_name,
                            "row_number": idx
                        }
                    )
                    documents.append(doc)
            
            return self._add_metadata(documents)
        
        except Exception as e:
            logger.error(f"Error loading Excel {self.file_path}: {e}")
            return []


class PDFDocumentLoader(BaseDocumentLoader):
    """Loader for PDF files."""
    
    def load(self) -> List[Document]:
        """Load PDF file."""
        try:
            loader = PyPDFLoader(str(self.file_path))
            documents = loader.load()
            
            # Add page numbers
            for i, doc in enumerate(documents):
                doc.metadata["page_number"] = i + 1
            
            return self._add_metadata(documents)
        
        except Exception as e:
            logger.error(f"Error loading PDF {self.file_path}: {e}")
            return []


class DocxDocumentLoader(BaseDocumentLoader):
    """Loader for Word documents (.docx)."""
    
    def load(self) -> List[Document]:
        """Load DOCX file."""
        try:
            loader = Docx2txtLoader(str(self.file_path))
            documents = loader.load()
            
            return self._add_metadata(documents)
        
        except Exception as e:
            logger.error(f"Error loading DOCX {self.file_path}: {e}")
            return []


class DocumentLoaderFactory:
    """Factory for creating document loaders."""
    
    _loaders = {
        '.txt': TextDocumentLoader,
        '.log': TextDocumentLoader,
        '.report': TextDocumentLoader,
        '.csv': CSVDocumentLoader,
        '.xlsx': ExcelDocumentLoader,
        '.pdf': PDFDocumentLoader,
        '.docx': DocxDocumentLoader
    }
    
    @classmethod
    def get_loader(cls, file_path: Path) -> Optional[BaseDocumentLoader]:
        """Get appropriate loader for file type."""
        ext = file_path.suffix.lower()
        loader_class = cls._loaders.get(ext)
        
        if loader_class is None:
            logger.error(f"No loader found for {ext} files")
            return None
        
        return loader_class(file_path)
    
    @classmethod
    def load_document(cls, file_path: Path) -> List[Document]:
        """Load document using appropriate loader."""
        loader = cls.get_loader(file_path)
        
        if loader is None:
            return []
        
        try:
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} chunks from {file_path.name}")
            return documents
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []
