"""Pytest configuration: add inner src/ package to sys.path and stub heavy deps."""
import sys, os
from unittest.mock import MagicMock

# src/src/ is the real package root; adding src/ lets `from src.utils.*` work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

# Stub langchain/chromadb/streamlit before any src.rag.* import triggers them
for _mod in [
    "langchain_classic",
    "langchain_classic.chains",
    "langchain_classic.chains.retrieval_qa",
    "langchain_classic.chains.retrieval_qa.base",
    "langchain_community",
    "langchain_community.llms",
    "langchain_community.chat_models",
    "langchain_community.vectorstores",
    "langchain_community.embeddings",
    "langchain_community.document_loaders",
    "langchain_core",
    "langchain_core.prompts",
    "langchain_core.documents",
    "langchain_text_splitters",
    "chromadb",
    "streamlit",
    "sentence_transformers",
    "InstructorEmbedding",
    "magic",
]:
    sys.modules.setdefault(_mod, MagicMock())
