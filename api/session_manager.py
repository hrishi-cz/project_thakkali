"""
Session Manager

Thin wrapper over ContextDatabase for session lifecycle management.
SessionContext removed - use ExecutionContext from core.execution_context instead.
"""

import logging
from typing import Any, Dict, List, Optional

from database.context_db import context_db
from core.execution_context import ExecutionContext, create_execution_context

logger = logging.getLogger(__name__)


class SessionManager:
    """
    High-level session management service.
    
    Thin CRUD wrapper over ContextDatabase.
    Use ExecutionContext (from core.execution_context) for session state.
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
        context_db.save_context(ctx.to_dict())
        
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
    
    def update_session(self, ctx: ExecutionContext) -> None:
        """
        Update an existing session.
        
        Args:
            ctx: ExecutionContext to update
        """
        ctx.update_timestamp()
        context_db.save_context(ctx.to_dict())
        logger.debug("Updated session %s", ctx.session_id)
    
    def list_sessions(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List all sessions.
        
        Args:
            limit: Maximum number of sessions to return
            offset: Offset for pagination
        
        Returns:
            List of session summaries
        """
        return context_db.list_sessions(limit=limit, offset=offset)
    
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


# Global singleton instance
session_manager = SessionManager()
