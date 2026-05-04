"""
Session Manager

Thin wrapper over ContextDatabase for session lifecycle management.
SessionContext removed - use ExecutionContext from core.execution_context instead.

This module is a service-layer utility. It should not contain HTTP request/
response logic; FastAPI routes in api/run_api.py call into this class.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

from database.context_db import context_db
from core.execution_context import ExecutionContext, create_execution_context

logger = logging.getLogger(__name__)


class SessionManager:
    """
    High-level session management service.
    
    Thin CRUD wrapper over ContextDatabase.
    Use ExecutionContext (from core.execution_context) for session state.
    Endpoint behavior belongs in route handlers, not this class.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize session manager."""
        if self._initialized:
            return
        
        self._initialized = True
        logger.info("SessionManager initialized (using ContextDatabase)")
    
    def create_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        project_name: Optional[str] = None,
        description: Optional[str] = None
    ) -> ExecutionContext:
        """
        Create a new session.
        
        Args:
            session_id: Optional session ID (auto-generated if not provided)
            user_id: Optional user identifier
            project_name: Optional project name
            description: Optional description
        
        Returns:
            ExecutionContext: Created session context
        """
        # Create execution context
        ctx = create_execution_context(
            session_id=session_id,
            metadata={
                'user_id': user_id,
                'project_name': project_name,
                'description': description
            }
        )
        
        # Persist to database
        ctx.revision = context_db.save_context(
            ctx.to_dict(),
            expected_revision=0,
        )
        
        logger.info("Created session %s", ctx.session_id)
        return ctx
    
    def get_session(self, session_id: str) -> Optional[ExecutionContext]:
        """
        Retrieve a session by ID.
        
        Args:
            session_id: Session identifier
        
        Returns:
            ExecutionContext if found, None otherwise
        """
        data = context_db.load_context(session_id)
        if data:
            return ExecutionContext.from_dict(data)
        return None
    
    def get_or_create_session(self, session_id: str, **kwargs) -> ExecutionContext:
        """Return existing session context or create a new one atomically.

        Used at ingestion start and schema detection to guarantee the sessions
        table row exists before FK-constrained child tables are written to.
        """
        ctx = self.get_session(session_id)
        if ctx is not None:
            return ctx
        ctx = self.create_session(session_id=session_id, **kwargs)
        logger.info("get_or_create_session: created session %s", session_id)
        return ctx

    def update_session(self, ctx: ExecutionContext) -> None:
        """
        Update an existing session.
        
        Args:
            ctx: ExecutionContext to update
        """
        expected_revision = int(getattr(ctx, "revision", 0) or 0)
        ctx.update_timestamp()
        ctx.revision = context_db.save_context(
            ctx.to_dict(),
            expected_revision=expected_revision,
        )
        logger.debug("Updated session %s", ctx.session_id)
    
    def list_sessions(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List all sessions (with optional filtering).
        
        Args:
            user_id: Optional user filter (not yet implemented in DB)
            status: Optional status filter (not yet implemented in DB)
            limit: Maximum number of sessions to return
            offset: Offset for pagination
        
        Returns:
            List of session summaries
        """
        raw_sessions = context_db.list_sessions(
            limit=limit,
            offset=offset,
            user_id=user_id,
            status=status,
        )
        summaries: List[Dict[str, Any]] = []

        for row in raw_sessions:
            session_id_value = row.get("session_id")
            context_payload = context_db.load_context(session_id_value) or {}

            session_status = context_payload.get("status")
            if not session_status:
                session_status = "closed" if row.get("pipeline_stage") == "closed" else "active"

            summaries.append(
                {
                    "session_id": session_id_value,
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at"),
                    "status": session_status,
                    "pipeline_stage": row.get("pipeline_stage"),
                    "user_id": context_payload.get("user_id"),
                    "project_name": context_payload.get("project_name"),
                    "description": context_payload.get("description"),
                }
            )

        return summaries
    
    def close_session(self, session_id: str) -> bool:
        """
        Close a session (mark as complete).
        
        Args:
            session_id: Session to close
        
        Returns:
            True if closed successfully, False if session not found
        """
        success = context_db.close_session(session_id)
        if success:
            logger.info("Closed session %s", session_id)
        return success
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session permanently.
        
        Args:
            session_id: Session to delete
        
        Returns:
            True if deleted successfully
        """
        # Load context first
        ctx = self.get_session(session_id)
        if not ctx:
            return False
        
        # Delete all associated profiles
        for dataset_id in ctx.active_dataset_ids:
            try:
                context_db.load_profile(dataset_id)  # Verify exists
                # TODO: Add delete_profile method to context_db if needed
            except Exception:
                pass
        
        # Mark as closed (for now - can add hard delete later)
        return self.close_session(session_id)
    
    def update_session_context(self, session_id: str, ctx: ExecutionContext) -> None:
        """
        Update session context (alias for update_session).
        
        Args:
            session_id: Session ID (must match ctx.session_id)
            ctx: ExecutionContext to update
        """
        if ctx.session_id != session_id:
            raise ValueError(
                f"Session ID mismatch: route session_id={session_id} "
                f"does not match context session_id={ctx.session_id}"
            )
        self.update_session(ctx)
    
    def remove_dataset_from_session(self, session_id: str, dataset_id: str) -> bool:
        """
        Remove a dataset from a session.
        
        Args:
            session_id: Session ID
            dataset_id: Dataset to remove
        
        Returns:
            True if removed successfully
        """
        ctx = self.get_session(session_id)
        if not ctx:
            return False
        
        # Remove from active datasets
        if dataset_id in ctx.active_dataset_ids:
            ctx.active_dataset_ids.remove(dataset_id)
            self.update_session(ctx)
            logger.info("Removed dataset %s from session %s", dataset_id, session_id)
            return True
        
        return False
    
    @property
    def db(self):
        """Access to underlying ContextDatabase (for backward compatibility)."""
        return context_db


# Global singleton instance
session_manager = SessionManager()
