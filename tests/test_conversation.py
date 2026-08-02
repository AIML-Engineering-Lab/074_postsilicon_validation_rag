"""Tests for ConversationManager (no external services needed)."""
import json
import tempfile
from pathlib import Path

import pytest
from src.rag.conversation import ConversationManager, Message


@pytest.fixture
def mgr(tmp_path, monkeypatch):
    """ConversationManager with isolated tmp save directory."""
    monkeypatch.setenv("CONFIG_FILE", "")
    cm = ConversationManager.__new__(ConversationManager)
    cm.save_directory = tmp_path / "convos"
    cm.save_directory.mkdir()
    cm.messages = []
    cm.conversation_id = "test_session"
    cm.auto_save_counter = 0
    # Patch auto_save_interval so we don't trigger file writes
    from src.utils.config import get_config
    cm.config = get_config()
    return cm


def test_message_dataclass():
    m = Message(role="user", content="hello", timestamp="2024-01-01T00:00:00")
    assert m.role == "user"
    assert m.content == "hello"
    assert m.sources is None
    assert m.confidence is None


def test_message_with_sources():
    m = Message(role="assistant", content="answer", timestamp="t",
                sources=[{"doc": "x"}], confidence="high")
    assert m.sources == [{"doc": "x"}]
    assert m.confidence == "high"


def test_add_message_appends(mgr):
    mgr.add_message("user", "test query")
    assert len(mgr.messages) == 1
    assert mgr.messages[0].role == "user"
    assert mgr.messages[0].content == "test query"


def test_add_multiple_messages(mgr):
    mgr.add_message("user", "q1")
    mgr.add_message("assistant", "a1", sources=[{"src": "doc.txt"}])
    assert len(mgr.messages) == 2
    assert mgr.messages[1].role == "assistant"


def test_get_messages_returns_list(mgr):
    mgr.add_message("user", "hi")
    result = mgr.get_messages()
    assert isinstance(result, list)
    assert len(result) == 1


def test_clear_messages(mgr):
    mgr.add_message("user", "hi")
    mgr.clear_messages()
    assert mgr.messages == []


def test_clear_resets_counter(mgr):
    mgr.add_message("user", "hi")
    mgr.auto_save_counter = 3
    mgr.clear_messages()
    assert mgr.auto_save_counter == 0


def test_timestamp_set_on_message(mgr):
    mgr.add_message("user", "hi")
    assert mgr.messages[0].timestamp != ""


def test_export_markdown(mgr):
    mgr.add_message("user", "What is yield?")
    mgr.add_message("assistant", "Yield is the ratio of good dies.")
    result = mgr.export_to_markdown()
    # Returns the file path; verify file exists and contains messages
    p = Path(result)
    assert p.exists()
    content = p.read_text()
    assert "What is yield?" in content


def test_get_messages_count(mgr):
    mgr.add_message("user", "hi")
    mgr.add_message("assistant", "hello")
    assert len(mgr.get_messages()) == 2
