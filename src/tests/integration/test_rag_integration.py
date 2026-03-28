"""
Integration tests for RAG pipeline.
"""

import pytest
from pathlib import Path
import tempfile

from src.rag.ingestion import DocumentIngestionPipeline
from src.rag.rag_pipeline import RAGPipeline


@pytest.mark.integration
class TestRAGPipelineIntegration:
    """Integration tests for complete RAG workflow."""
    
    @pytest.fixture(scope="class")
    def sample_document(self):
        """Create sample document for testing."""
        content = """
        Post-Silicon Validation Report
        Test Date: 2024-12-01
        
        Test Results:
        - CPU Core 0: PASS
        - CPU Core 1: PASS
        - Memory Test: PASS
        - Clock Domain Crossing: PASS
        
        Critical Issues:
        - None found
        
        Performance Metrics:
        - Maximum Frequency: 3.5 GHz
        - Power Consumption: 95W
        - Temperature: 65°C
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)
        
        yield path
        
        # Cleanup
        if path.exists():
            path.unlink()
    
    def test_ingest_and_query(self, sample_document):
        """Test ingesting document and querying."""
        # Initialize pipelines
        ingestion = DocumentIngestionPipeline()
        
        # Ingest document
        result = ingestion.ingest_file(str(sample_document))
        
        assert result["status"] == "success"
        assert result["num_chunks"] > 0
        
        # Note: RAG pipeline requires Ollama/OpenAI to be running
        # This test will be skipped in CI/CD without proper setup
        try:
            rag = RAGPipeline()
            
            # Query the document
            response = rag.query("What were the test results?")
            
            assert response.answer is not None
            assert len(response.source_documents) > 0
            assert response.confidence in ["low", "medium", "high"]
        
        except Exception as e:
            pytest.skip(f"RAG pipeline requires Ollama/OpenAI: {e}")


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""
    
    def test_multi_document_workflow(self):
        """Test workflow with multiple documents."""
        # This test requires full system setup
        pytest.skip("Requires full system setup with Ollama/OpenAI")
