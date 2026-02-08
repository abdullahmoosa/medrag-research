#!/usr/bin/env python3
"""
Conversation History Manager
Handles session management and conversation persistence
"""

import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import asyncio

@dataclass
class ConversationMessage:
    """Single message in a conversation"""
    id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float
    sources: List[Dict] = None

@dataclass 
class ConversationSession:
    """Complete conversation session"""
    conversation_id: str
    created_at: float
    last_updated: float
    messages: List[ConversationMessage]
    user_ip: Optional[str] = None
    
class ConversationManager:
    """Manages conversation history and sessions"""
    
    def __init__(self, storage_dir: str = "conversations", max_sessions: int = 1000):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.max_sessions = max_sessions
        
        # In-memory cache for active sessions
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.session_timeout = 3600  # 1 hour timeout
        
        # Load existing sessions
        self._load_recent_sessions()
    
    def _load_recent_sessions(self):
        """Load recent sessions from disk"""
        try:
            # Load sessions from last 24 hours
            cutoff_time = time.time() - 86400  # 24 hours
            
            for session_file in self.storage_dir.glob("session_*.json"):
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if data.get('last_updated', 0) > cutoff_time:
                        session = self._dict_to_session(data)
                        self.active_sessions[session.conversation_id] = session
                        
                except Exception as e:
                    print(f"Warning: Could not load session {session_file}: {e}")
                    
        except Exception as e:
            print(f"Warning: Error loading sessions: {e}")
    
    def _dict_to_session(self, data: Dict) -> ConversationSession:
        """Convert dictionary to ConversationSession"""
        messages = []
        for msg_data in data.get('messages', []):
            msg = ConversationMessage(
                id=msg_data['id'],
                role=msg_data['role'], 
                content=msg_data['content'],
                timestamp=msg_data['timestamp'],
                sources=msg_data.get('sources')
            )
            messages.append(msg)
        
        return ConversationSession(
            conversation_id=data['conversation_id'],
            created_at=data['created_at'],
            last_updated=data['last_updated'],
            messages=messages,
            user_ip=data.get('user_ip')
        )
    
    def _session_to_dict(self, session: ConversationSession) -> Dict:
        """Convert ConversationSession to dictionary"""
        return {
            'conversation_id': session.conversation_id,
            'created_at': session.created_at,
            'last_updated': session.last_updated,
            'user_ip': session.user_ip,
            'messages': [asdict(msg) for msg in session.messages]
        }
    
    def create_session(self, user_ip: Optional[str] = None) -> str:
        """Create a new conversation session"""
        conversation_id = str(uuid.uuid4())
        current_time = time.time()
        
        session = ConversationSession(
            conversation_id=conversation_id,
            created_at=current_time,
            last_updated=current_time,
            messages=[],
            user_ip=user_ip
        )
        
        self.active_sessions[conversation_id] = session
        self._save_session(session)
        self._cleanup_old_sessions()
        
        return conversation_id
    
    def add_message(self, conversation_id: str, role: str, content: str, 
                   sources: List[Dict] = None, user_ip: Optional[str] = None) -> str:
        """Add a message to the conversation"""
        
        # Create session if it doesn't exist
        if conversation_id not in self.active_sessions:
            if conversation_id == "default":
                conversation_id = self.create_session(user_ip)
            else:
                # Try to load from disk
                session_file = self.storage_dir / f"session_{conversation_id}.json"
                if session_file.exists():
                    with open(session_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    session = self._dict_to_session(data)
                    self.active_sessions[conversation_id] = session
                else:
                    # Create new session with specified ID
                    current_time = time.time()
                    session = ConversationSession(
                        conversation_id=conversation_id,
                        created_at=current_time,
                        last_updated=current_time,
                        messages=[],
                        user_ip=user_ip
                    )
                    self.active_sessions[conversation_id] = session
        
        # Add message
        session = self.active_sessions[conversation_id]
        message_id = str(uuid.uuid4())
        
        message = ConversationMessage(
            id=message_id,
            role=role,
            content=content,
            timestamp=time.time(),
            sources=sources
        )
        
        session.messages.append(message)
        session.last_updated = time.time()
        
        # Update user IP if provided
        if user_ip:
            session.user_ip = user_ip
        
        # Save to disk
        self._save_session(session)
        
        return conversation_id
    
    def get_conversation_history(self, conversation_id: str, max_messages: int = 50) -> List[ConversationMessage]:
        """Get conversation history"""
        if conversation_id not in self.active_sessions:
            return []
        
        session = self.active_sessions[conversation_id]
        return session.messages[-max_messages:] if max_messages else session.messages
    
    def get_context_for_llm(self, conversation_id: str, max_context: int = 10) -> str:
        """Get formatted context for LLM"""
        history = self.get_conversation_history(conversation_id, max_context)
        
        if not history:
            return ""
        
        context_parts = []
        for msg in history[-max_context:]:
            role_label = "User" if msg.role == "user" else "Assistant" 
            context_parts.append(f"{role_label}: {msg.content}")
        
        return "\n\n".join(context_parts)
    
    def _save_session(self, session: ConversationSession):
        """Save session to disk"""
        try:
            session_file = self.storage_dir / f"session_{session.conversation_id}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(self._session_to_dict(session), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Could not save session {session.conversation_id}: {e}")
    
    def _cleanup_old_sessions(self):
        """Clean up old sessions to manage memory and storage"""
        current_time = time.time()
        cutoff_time = current_time - self.session_timeout
        
        # Remove from memory
        expired_sessions = [
            conv_id for conv_id, session in self.active_sessions.items()
            if session.last_updated < cutoff_time
        ]
        
        for conv_id in expired_sessions:
            del self.active_sessions[conv_id]
        
        # Clean up old files (keep last 7 days)
        file_cutoff = current_time - (7 * 86400)  # 7 days
        try:
            for session_file in self.storage_dir.glob("session_*.json"):
                if session_file.stat().st_mtime < file_cutoff:
                    session_file.unlink()
        except Exception as e:
            print(f"Warning: Error cleaning up old session files: {e}")
    
    def list_sessions(self, limit: int = 20) -> List[Dict]:
        """List recent sessions"""
        sessions = list(self.active_sessions.values())
        sessions.sort(key=lambda x: x.last_updated, reverse=True)
        
        result = []
        for session in sessions[:limit]:
            result.append({
                'conversation_id': session.conversation_id,
                'created_at': datetime.fromtimestamp(session.created_at).isoformat(),
                'last_updated': datetime.fromtimestamp(session.last_updated).isoformat(),
                'message_count': len(session.messages),
                'user_ip': session.user_ip
            })
        
        return result
    
    def delete_session(self, conversation_id: str) -> bool:
        """Delete a conversation session"""
        try:
            # Remove from memory
            if conversation_id in self.active_sessions:
                del self.active_sessions[conversation_id]
            
            # Remove from disk
            session_file = self.storage_dir / f"session_{conversation_id}.json"
            if session_file.exists():
                session_file.unlink()
            
            return True
        except Exception as e:
            print(f"Error deleting session {conversation_id}: {e}")
            return False

# Global conversation manager instance
conversation_manager = ConversationManager()
