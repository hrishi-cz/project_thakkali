"""
Session Database Manager

Provides persistent SQLite storage for SessionContext objects.
Replaces in-memory _session_store with durable storage.
"""

import json
import sqlite3
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from contextlib import contextmanager
import threading

logger = logging.getLogger(__name__)


class SessionDatabase:
    """
    Thread-safe SQLite database for session persistence.
    
    Stores SessionContext objects as JSON blobs with indexed metadata.
    Singleton pattern ensures one connection per process.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path: Optional[Path] = None):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database connection."""
        if self._initialized:
            return
            
        self.db_path = db_path or Path("./data/sessions.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread-local connections (one per thread for safety)
        self._local = threading.local()
        
        # Initialize schema
        self._init_schema()
        self._initialized = True
        logger.info(f"SessionDatabase initialized at {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=10.0
            )
            self._local.conn.row_factory = sqlite3.Row
        
        try:
            yield self._local.conn
        except Exception as e:
            self._local.conn.rollback()
            raise
        else:
            self._local.conn.commit()
    
    def _init_schema(self):
        """Initialize database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    project_name TEXT,
                    context_json TEXT NOT NULL,
                    context_hash TEXT
                )
            """)
            
            # Indices for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_user 
                ON sessions(user_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_created 
                ON sessions(created_at DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_status 
                ON sessions(status)
            """)
            
            # Dataset profiles table (for Phase 3)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dataset_profiles (
                    dataset_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    cached_at TEXT NOT NULL,
                    file_size_mb REAL,
                    row_count INTEGER,
                    column_count INTEGER,
                    profile_json TEXT
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_profiles_source 
                ON dataset_profiles(source)
            """)
            
            # Session-to-Dataset mapping table (for explicit active/cached tracking)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_datasets (
                    session_id TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    added_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, dataset_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                    FOREIGN KEY (dataset_id) REFERENCES dataset_profiles(dataset_id) ON DELETE CASCADE
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_datasets_session 
                ON session_datasets(session_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_datasets_status 
                ON session_datasets(status)
            """)
            
            conn.commit()
            logger.info("Database schema initialized")
    
    def create_session(self, session_data: Dict[str, Any]) -> str:
        """
        Create a new session.
        
        Args:
            session_data: SessionContext as dict (from to_dict())
        
        Returns:
            session_id: Created session ID
        """
        session_id = session_data.get("session_id")
        if not session_id:
            raise ValueError("session_data must contain session_id")
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO sessions (
                    session_id, user_id, created_at, updated_at, 
                    status, project_name, context_json, context_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                session_data.get("user_id"),
                session_data.get("created_at"),
                session_data.get("updated_at"),
                session_data.get("status", "active"),
                session_data.get("project_name"),
                json.dumps(session_data),
                session_data.get("context_hash")
            ))
            
            logger.info(f"Created session {session_id}")
            return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a session by ID.
        
        Args:
            session_id: Session identifier
        
        Returns:
            SessionContext as dict, or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT context_json FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return json.loads(row["context_json"])
            return None
    
    def update_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """
        Update an existing session.
        
        Args:
            session_id: Session identifier
            session_data: Updated SessionContext as dict
        
        Returns:
            True if updated, False if session not found
        """
        # Ensure updated_at is current
        session_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE sessions 
                SET user_id = ?,
                    updated_at = ?,
                    status = ?,
                    project_name = ?,
                    context_json = ?,
                    context_hash = ?
                WHERE session_id = ?
            """, (
                session_data.get("user_id"),
                session_data["updated_at"],
                session_data.get("status", "active"),
                session_data.get("project_name"),
                json.dumps(session_data),
                session_data.get("context_hash"),
                session_id
            ))
            
            if cursor.rowcount > 0:
                logger.info(f"Updated session {session_id}")
                return True
            return False
    
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
            List of session dicts (summary view, not full context)
        """
        query = "SELECT session_id, user_id, created_at, updated_at, status, project_name FROM sessions"
        params = []
        conditions = []
        
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        
        if status:
            conditions.append("status = ?")
            params.append(status)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    def close_session(self, session_id: str) -> bool:
        """
        Mark a session as closed.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if closed, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions 
                SET status = 'closed', updated_at = ?
                WHERE session_id = ?
            """, (datetime.utcnow().isoformat(), session_id))
            
            if cursor.rowcount > 0:
                logger.info(f"Closed session {session_id}")
                return True
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        Permanently delete a session (use with caution).
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            
            if cursor.rowcount > 0:
                logger.warning(f"Deleted session {session_id}")
                return True
            return False
    
    def get_session_count(self, user_id: Optional[str] = None, status: Optional[str] = None) -> int:
        """Get total session count with optional filters."""
        query = "SELECT COUNT(*) as count FROM sessions"
        params = []
        conditions = []
        
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        
        if status:
            conditions.append("status = ?")
            params.append(status)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return row["count"] if row else 0
    
    def cleanup_old_sessions(self, days: int = 30) -> int:
        """
        Delete sessions older than N days with status 'closed'.
        
        Args:
            days: Age threshold in days
        
        Returns:
            Number of sessions deleted
        """
        from datetime import timedelta
        
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM sessions 
                WHERE status = 'closed' AND updated_at < ?
            """, (cutoff,))
            
            deleted = cursor.rowcount
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old sessions")
            return deleted


# Singleton instance
session_db = SessionDatabase()
