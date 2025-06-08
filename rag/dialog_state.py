from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid

class DialogStateManager:
    """Manages conversation history for multi-turn dialogues."""
    
    def __init__(self):
        """Initialize an empty dialog state manager."""
        self.conversations = {}  # Dictionary to store conversations by session_id
        
    def create_session(self) -> str:
        """
        Create a new conversation session.
        
        Returns:
            A unique session ID
        """
        session_id = str(uuid.uuid4())
        self.conversations[session_id] = []
        return session_id
        
    def add_to_history(self, session_id: str, role: str, message: str) -> None:
        """
        Add a message to the conversation history for a specific session.
        
        Args:
            session_id: Unique identifier for the conversation session
            role: Role of the message sender ('user' or 'assistant')
            message: Content of the message
        """
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        self.conversations[session_id].append({
            "role": role,  # "user" or "assistant"
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
    def get_history(self, session_id: str, max_turns: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the conversation history for a specific session.
        
        Args:
            session_id: Unique identifier for the conversation session
            max_turns: Maximum number of turns to return (each turn is a user-assistant exchange)
            
        Returns:
            List of messages in the conversation history
        """
        if session_id not in self.conversations:
            return []
        
        history = self.conversations[session_id]
        if max_turns and len(history) > max_turns * 2:
            # Return the most recent max_turns turns
            return history[-max_turns*2:]  # *2 because each turn has user + assistant message
        return history
    
    def clear_history(self, session_id: str) -> None:
        """
        Clear the conversation history for a specific session.
        
        Args:
            session_id: Unique identifier for the conversation session
        """
        if session_id in self.conversations:
            self.conversations[session_id] = []
            
    def format_history_for_llm(self, session_id: str, include_last_n_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Format the conversation history for input to an LLM.
        
        Args:
            session_id: Unique identifier for the conversation session
            include_last_n_turns: Number of recent turns to include
            
        Returns:
            List of formatted messages for LLM input
        """
        history = self.get_history(session_id, include_last_n_turns)
        formatted_messages = []
        
        for msg in history:
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
            
        return formatted_messages
    
    def get_last_n_exchanges_as_string(self, session_id: str, n: int = 3) -> str:
        """
        Get the last n conversation exchanges as a formatted string.
        
        Args:
            session_id: Unique identifier for the conversation session
            n: Number of recent exchanges to include
            
        Returns:
            Formatted string representation of conversation history
        """
        history = self.get_history(session_id, n)
        if not history:
            return ""
        
        result = "Previous conversation:\n"
        for msg in history:
            speaker = "User" if msg["role"] == "user" else "Assistant"
            result += f"{speaker}: {msg['content']}\n"
        
        return result
