"""
Conversation History Manager
Manages conversation history and exports.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger()


@dataclass
class Message:
    """Chat message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    sources: Optional[List[Dict]] = None
    confidence: Optional[str] = None


class ConversationManager:
    """Manages conversation history."""
    
    def __init__(self):
        self.config = get_config()
        self.save_directory = Path(self.config.conversation.save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)
        
        self.messages: List[Message] = []
        self.conversation_id = self._generate_conversation_id()
        self.auto_save_counter = 0
    
    def _generate_conversation_id(self) -> str:
        """Generate unique conversation ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_message(self, role: str, content: str, sources: Optional[List[Dict]] = None, confidence: Optional[str] = None) -> None:
        """
        Add a message to conversation.
        
        Args:
            role: "user" or "assistant"
            content: Message content
            sources: Source documents (for assistant messages)
            confidence: Confidence level (for assistant messages)
        """
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            sources=sources,
            confidence=confidence
        )
        
        self.messages.append(message)
        
        # Auto-save
        self.auto_save_counter += 1
        if self.auto_save_counter >= self.config.conversation.auto_save_interval:
            self.save_conversation()
            self.auto_save_counter = 0
    
    def get_messages(self) -> List[Message]:
        """Get all messages."""
        return self.messages
    
    def clear_messages(self) -> None:
        """Clear all messages."""
        self.messages = []
        self.conversation_id = self._generate_conversation_id()
        self.auto_save_counter = 0
        logger.info("Conversation cleared")
    
    def save_conversation(self, filename: Optional[str] = None) -> str:
        """
        Save conversation to file.
        
        Args:
            filename: Custom filename (optional)
        
        Returns:
            Path to saved file
        """
        if not self.messages:
            logger.warning("No messages to save")
            return ""
        
        if filename is None:
            filename = f"conversation_{self.conversation_id}.json"
        
        filepath = self.save_directory / filename
        
        # Convert messages to dict
        messages_dict = [asdict(msg) for msg in self.messages]
        
        data = {
            "conversation_id": self.conversation_id,
            "created_at": self.messages[0].timestamp,
            "last_updated": datetime.now().isoformat(),
            "num_messages": len(self.messages),
            "messages": messages_dict
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Conversation saved: {filepath}")
        
        return str(filepath)
    
    def load_conversation(self, filename: str) -> None:
        """
        Load conversation from file.
        
        Args:
            filename: Filename to load
        """
        filepath = self.save_directory / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Conversation file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.conversation_id = data["conversation_id"]
        self.messages = [Message(**msg) for msg in data["messages"]]
        
        logger.info(f"Conversation loaded: {filepath} ({len(self.messages)} messages)")
    
    def export_to_markdown(self, filename: Optional[str] = None) -> str:
        """
        Export conversation to Markdown.
        
        Args:
            filename: Custom filename (optional)
        
        Returns:
            Path to exported file
        """
        if not self.messages:
            logger.warning("No messages to export")
            return ""
        
        if filename is None:
            filename = f"conversation_{self.conversation_id}.md"
        
        filepath = self.save_directory / filename
        
        lines = [
            f"# Conversation Export",
            f"",
            f"**Conversation ID:** {self.conversation_id}",
            f"**Date:** {self.messages[0].timestamp.split('T')[0]}",
            f"**Messages:** {len(self.messages)}",
            f"",
            "---",
            ""
        ]
        
        for i, msg in enumerate(self.messages, 1):
            role_label = "👤 User" if msg.role == "user" else "🤖 Assistant"
            lines.append(f"## Message {i}: {role_label}")
            lines.append(f"**Time:** {msg.timestamp}")
            
            if msg.role == "assistant" and msg.confidence:
                lines.append(f"**Confidence:** {msg.confidence}")
            
            lines.append("")
            lines.append(msg.content)
            
            if msg.role == "assistant" and msg.sources:
                lines.append("")
                lines.append("### Sources")
                for j, source in enumerate(msg.sources, 1):
                    lines.append(f"{j}. **{source['source']}** (chunk {source['chunk_id']})")
                    lines.append(f"   > {source['content']}")
            
            lines.append("")
            lines.append("---")
            lines.append("")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Conversation exported to Markdown: {filepath}")
        
        return str(filepath)
    
    def list_saved_conversations(self) -> List[Dict]:
        """List all saved conversations."""
        conversations = []
        
        for filepath in self.save_directory.glob("conversation_*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                conversations.append({
                    "filename": filepath.name,
                    "conversation_id": data.get("conversation_id", ""),
                    "created_at": data.get("created_at", ""),
                    "num_messages": data.get("num_messages", 0)
                })
            except Exception as e:
                logger.warning(f"Error reading conversation file {filepath}: {e}")
        
        # Sort by creation time (newest first)
        conversations.sort(key=lambda x: x["created_at"], reverse=True)
        
        return conversations


# Singleton instance
conversation_manager = ConversationManager()


def get_conversation_manager() -> ConversationManager:
    """Get conversation manager instance."""
    return conversation_manager
