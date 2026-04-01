"""
Session Manager

High-level session management with SessionContext objects.
Provides CRUD operations and session lifecycle management.
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from database.session_db import session_db

logger = logging.getLogger(__name__)


class SessionContext:
    """
    Top-level session container (implements OBJECT_SCHEMAS.md spec).
    
    All pipeline phases read/write to this object.
    Persists across app restarts via SessionDatabase.
    """
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        project_name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """Initialize a new session context."""
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.user_id = user_id
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.status = "active"
        
        # Metadata
        self.project_name = project_name
        self.description = description
        
        # Dataset Management
        self.active_dataset_ids: List[str] = []
        self.cached_dataset_ids: List[str] = []
        self.dataset_profiles: Dict[str, Any] = {}
        
        # Global Intelligence
        self.global_schema: Optional[Dict[str, Any]] = None
        self.global_target: Optional[str] = None
        self.primary_dataset_id: Optional[str] = None
        
        # User Overrides
        self.overrides: Dict[str, Any] = {}
        
        # Execution State
        self.execution_context: Optional[Dict[str, Any]] = None
        self.execution_history: List[Dict[str, Any]] = []
        
        # Training State
        self.current_task_id: Optional[str] = None
        self.trained_model_ids: List[str] = []
        
        # Version Control
        self.version = "1.0"
        self.context_hash: Optional[str] = None
        self.update_hash()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON/DB storage."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status,
            "project_name": self.project_name,
            "description": self.description,
            "active_dataset_ids": self.active_dataset_ids,
            "cached_dataset_ids": self.cached_dataset_ids,
            "dataset_profiles": self.dataset_profiles,
            "global_schema": self.global_schema,
            "global_target": self.global_target,
            "primary_dataset_id": self.primary_dataset_id,
            "overrides": self.overrides,
            "execution_context": self.execution_context,
            "execution_history": self.execution_history,
            "current_task_id": self.current_task_id,
            "trained_model_ids": self.trained_model_ids,
            "version": self.version,
            "context_hash": self.context_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionContext':
        """Deserialize from dict."""
        ctx = cls.__new__(cls)
        
        ctx.session_id = data["session_id"]
        ctx.user_id = data.get("user_id")
        ctx.created_at = datetime.fromisoformat(data["created_at"])
        ctx.updated_at = datetime.fromisoformat(data["updated_at"])
        ctx.status = data.get("status", "active")
        
        ctx.project_name = data.get("project_name")
        ctx.description = data.get("description")
        
        ctx.active_dataset_ids = data.get("active_dataset_ids", [])
        ctx.cached_dataset_ids = data.get("cached_dataset_ids", [])
        ctx.dataset_profiles = data.get("dataset_profiles", {})
        
        ctx.global_schema = data.get("global_schema")
        ctx.global_target = data.get("global_target")
        ctx.primary_dataset_id = data.get("primary_dataset_id")
        
        ctx.overrides = data.get("overrides", {})
        
        ctx.execution_context = data.get("execution_context")
        ctx.execution_history = data.get("execution_history", [])
        
        ctx.current_task_id = data.get("current_task_id")
        ctx.trained_model_ids = data.get("trained_model_ids", [])
        
        ctx.version = data.get("version", "1.0")
        ctx.context_hash = data.get("context_hash")
        
        return ctx
    
    def update_hash(self):
        """Recompute context hash (changes only when decisions change)."""
        serialized = json.dumps(self.to_dict(), sort_keys=True)
        self.context_hash = hashlib.sha256(serialized.encode()).hexdigest()[:16]
    
    def add_execution_event(self, phase: str, decisions: Dict[str, Any]):
        """Log a pipeline phase execution."""
        self.execution_history.append({
            "phase": phase,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decisions": decisions
        })
        self.updated_at = datetime.now(timezone.utc)
        self.update_hash()
    
    def add_dataset(self, dataset_id: str):
        """Add a dataset to the active session."""
        if dataset_id not in self.active_dataset_ids:
            self.active_dataset_ids.append(dataset_id)
            self.updated_at = datetime.now(timezone.utc)
            self.update_hash()
            logger.info(f"Added dataset {dataset_id} to session {self.session_id}")
    
    def remove_dataset(self, dataset_id: str):
        """Remove a dataset from the active session (keeps in cache)."""
        if dataset_id in self.active_dataset_ids:
            self.active_dataset_ids.remove(dataset_id)
            # Move to cached list if not already there
            if dataset_id not in self.cached_dataset_ids:
                self.cached_dataset_ids.append(dataset_id)
            self.updated_at = datetime.now(timezone.utc)
            self.update_hash()
            logger.info(f"Removed dataset {dataset_id} from session {self.session_id}")


class SessionManager:
    """
    High-level session management service.
    
    Provides CRUD operations for SessionContext objects.
    Singleton pattern ensures consistent state.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize session manager."""
        if self._initialized:
            return
        
        self.db = session_db
        self._initialized = True
        logger.info("SessionManager initialized")
    
    def create_session(
        self,
        user_id: Optional[str] = None,
        project_name: Optional[str] = None,
        description: Optional[str] = None
    ) -> SessionContext:
        """
        Create a new session.
        
        Args:
            user_id: Optional user identifier
            project_name: Optional project name
            description: Optional description
        
        Returns:
            SessionContext: Created session
        """
        ctx = SessionContext(
            user_id=user_id,
            project_name=project_name,
            description=description
        )
        
        # Persist to database
        self.db.create_session(ctx.to_dict())
        
        logger.info(f"Created session {ctx.session_id}")
        return ctx
    
    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """
        Retrieve a session by ID.
        
        Args:
            session_id: Session identifier
        
        Returns:
            SessionContext or None if not found
        """
        data = self.db.get_session(session_id)
        if data:
            return SessionContext.from_dict(data)
        return None
    
    def update_session(self, ctx: SessionContext) -> bool:
        """
        Update an existing session.
        
        Args:
            ctx: SessionContext to update
        
        Returns:
            True if updated, False if not found
        """
        ctx.updated_at = datetime.now(timezone.utc)
        ctx.update_hash()
        return self.db.update_session(ctx.session_id, ctx.to_dict())
    
    def list_sessions(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List sessions with optional filtering.
        
        Args:
            user_id: Filter by user ID
            status: Filter by status (active, closed, error)
            limit: Maximum sessions to return
            offset: Pagination offset
        
        Returns:
            List of session summaries
        """
        return self.db.list_sessions(user_id, status, limit, offset)
    
    def close_session(self, session_id: str) -> bool:
        """
        Mark a session as closed.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if closed, False if not found
        """
        return self.db.close_session(session_id)
    
    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> SessionContext:
        """
        Get existing session or create new one.
        
        Args:
            session_id: Optional session ID to retrieve
            user_id: Optional user ID for new session
        
        Returns:
            SessionContext (existing or new)
        """
        if session_id:
            ctx = self.get_session(session_id)
            if ctx:
                return ctx
        
        # Create new session
        return self.create_session(user_id=user_id)
    
    def add_dataset_to_session(
        self,
        session_id: str,
        dataset_id: str
    ) -> bool:
        """
        Add a dataset to a session's active list.
        
        Args:
            session_id: Session identifier
            dataset_id: Dataset identifier
        
        Returns:
            True if added, False if session not found
        """
        ctx = self.get_session(session_id)
        if not ctx:
            return False
        
        ctx.add_dataset(dataset_id)
        return self.update_session(ctx)
    
    def remove_dataset_from_session(
        self,
        session_id: str,
        dataset_id: str
    ) -> bool:
        """
        Remove a dataset from a session's active list.
        
        Args:
            session_id: Session identifier
            dataset_id: Dataset identifier
        
        Returns:
            True if removed, False if session not found
        """
        ctx = self.get_session(session_id)
        if not ctx:
            return False
        
        ctx.remove_dataset(dataset_id)
        return self.update_session(ctx)


# Singleton instance
session_manager = SessionManager()
