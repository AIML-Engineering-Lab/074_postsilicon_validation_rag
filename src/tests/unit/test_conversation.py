"""
Unit tests for conversation manager.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from src.rag.conversation import ConversationManager, Message


class TestConversationManager:
    """Test conversation manager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for conversations."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)
    
    @pytest.fixture
    def manager(self, temp_dir, monkeypatch):
        """Create conversation manager with temp directory."""
        from src.utils import config
        
        # Mock config to use temp directory
        original_get_config = config.get_config
        
        class MockConversationConfig:
            save_directory = temp_dir
            auto_save_interval = 5
        
        class MockConfig:
            conversation = MockConversationConfig()
        
        monkeypatch.setattr(config, 'get_config', lambda: MockConfig())
        
        manager = ConversationManager()
        
        # Restore original
        monkeypatch.setattr(config, 'get_config', original_get_config)
        
        return manager
    
    def test_add_message(self, manager):
        """Test adding messages."""
        manager.add_message("user", "Test question")
        manager.add_message("assistant", "Test answer")
        
        messages = manager.get_messages()
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Test question"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Test answer"
    
    def test_clear_messages(self, manager):
        """Test clearing messages."""
        manager.add_message("user", "Question")
        assert len(manager.get_messages()) == 1
        
        manager.clear_messages()
        assert len(manager.get_messages()) == 0
    
    def test_save_and_load_conversation(self, manager):
        """Test saving and loading conversation."""
        manager.add_message("user", "Question 1")
        manager.add_message("assistant", "Answer 1")
        
        # Save
        filepath = manager.save_conversation("test_convo.json")
        assert Path(filepath).exists()
        
        # Clear and load
        manager.clear_messages()
        assert len(manager.get_messages()) == 0
        
        manager.load_conversation("test_convo.json")
        messages = manager.get_messages()
        
        assert len(messages) == 2
        assert messages[0].content == "Question 1"
        assert messages[1].content == "Answer 1"
    
    def test_export_to_markdown(self, manager):
        """Test exporting to Markdown."""
        manager.add_message("user", "Test question")
        manager.add_message("assistant", "Test answer", sources=[{"source": "test.txt", "chunk_id": 1, "content": "snippet"}])
        
        filepath = manager.export_to_markdown("test_export.md")
        
        assert Path(filepath).exists()
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        assert "Test question" in content
        assert "Test answer" in content
        assert "test.txt" in content
