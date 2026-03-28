"""
RAG Pipeline
Orchestrates the retrieval-augmented generation pipeline.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from src.vectorstore.chroma_manager import get_vectorstore
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger()


@dataclass
class RAGResponse:
    """RAG response with metadata."""
    answer: str
    source_documents: List[Dict[str, Any]]
    confidence: str  # "low", "medium", "high"
    model_used: str


class RAGPipeline:
    """RAG pipeline for question answering."""
    
    _instance: Optional['RAGPipeline'] = None
    _chain: Optional[RetrievalQA] = None
    _llm: Optional[Any] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._chain is None:
            self._initialize_llm()
            self._initialize_chain()
    
    def _initialize_llm(self) -> None:
        """Initialize LLM."""
        config = get_config()
        llm_config = config.llm
        
        logger.info(f"Initializing LLM: {llm_config.provider}")
        
        try:
            if llm_config.provider == "ollama":
                self._llm = Ollama(
                    model=llm_config.ollama.model,
                    base_url=llm_config.ollama.base_url,
                    temperature=llm_config.ollama.temperature
                )
                logger.info(f"Ollama LLM initialized: {llm_config.ollama.model}")
            
            elif llm_config.provider == "openai":
                import os
                api_key = os.getenv(llm_config.openai.api_key_env)
                
                if not api_key:
                    raise ValueError(f"OpenAI API key not found in {llm_config.openai.api_key_env}")
                
                self._llm = ChatOpenAI(
                    model=llm_config.openai.model,
                    temperature=llm_config.openai.temperature,
                    max_tokens=llm_config.openai.max_tokens
                )
                logger.info(f"OpenAI LLM initialized: {llm_config.openai.model}")
            
            else:
                raise ValueError(f"Unknown LLM provider: {llm_config.provider}")
        
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _initialize_chain(self) -> None:
        """Initialize RAG chain."""
        try:
            vectorstore = get_vectorstore()
            retriever = vectorstore.get_retriever()
            
            # Custom prompt template
            template = """You are an expert assistant for semiconductor post-silicon validation engineers.
Use the following context to answer the question. If you don't know the answer based on the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer: Let me help you with that. """
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            self._chain = RetrievalQA.from_chain_type(
                llm=self._llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            
            logger.info("RAG chain initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize RAG chain: {e}")
            raise
    
    def query(self, question: str) -> RAGResponse:
        """
        Query the RAG pipeline.
        
        Args:
            question: User question
        
        Returns:
            RAGResponse with answer and metadata
        """
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            result = self._chain({"query": question})
            
            # Extract answer and source documents
            answer = result["result"]
            source_docs = result["source_documents"]
            
            # Calculate confidence based on source relevance
            confidence = self._calculate_confidence(source_docs)
            
            # Format source documents
            formatted_sources = []
            for doc in source_docs:
                formatted_sources.append({
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", 0),
                    "page_number": doc.metadata.get("page_number"),
                    "file_type": doc.metadata.get("file_type", "")
                })
            
            config = get_config()
            model_used = f"{config.llm.provider}/{config.llm.ollama.model if config.llm.provider == 'ollama' else config.llm.openai.model}"
            
            response = RAGResponse(
                answer=answer,
                source_documents=formatted_sources,
                confidence=confidence,
                model_used=model_used
            )
            
            logger.info(f"Query processed successfully. Confidence: {confidence}")
            
            return response
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def _calculate_confidence(self, source_docs: List) -> str:
        """Calculate confidence level based on source documents."""
        if not source_docs:
            return "low"
        
        if len(source_docs) >= 3:
            return "high"
        elif len(source_docs) >= 2:
            return "medium"
        else:
            return "low"
    
    def reload(self) -> None:
        """Reload the RAG pipeline (useful after configuration changes)."""
        self._chain = None
        self._llm = None
        self._initialize_llm()
        self._initialize_chain()
        logger.info("RAG pipeline reloaded")


# Singleton instance
rag_pipeline = RAGPipeline()


def get_rag_pipeline() -> RAGPipeline:
    """Get RAG pipeline instance."""
    return rag_pipeline
