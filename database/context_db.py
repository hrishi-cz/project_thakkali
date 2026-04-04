"""
Unified Context Database - Single Persistence Layer
Merged from database/dataset_profile_db.py and database/session_db.py

Provides unified storage for:
- ExecutionContext (session-level intelligence)
- DatasetProfile (per-dataset intelligence)
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


class ContextDatabase:
    """
    Unified database for ExecutionContext and DatasetProfile.
    
    Thread-safe singleton with one connection per thread.
    Replaces separate dataset_profile_db and session_db.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path: str = "./data/sessions.db"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_path: str = "./data/sessions.db"):
        if self._initialized:
            return
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._create_tables()
        self._initialized = True
        logger.info("ContextDatabase initialized at %s", self.db_path)
    
    @contextmanager
    def _get_connection(self):
        """Thread-safe connection manager."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        
        try:
            yield self._local.conn
        except Exception:
            self._local.conn.rollback()
            raise
        else:
            self._local.conn.commit()
    
    def _create_tables(self):
        """Create unified schema for sessions and dataset profiles."""
        with self._get_connection() as conn:
            # Sessions table (stores ExecutionContext)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    pipeline_stage TEXT,
                    context_json TEXT NOT NULL,
                    context_hash TEXT
                )
            """)
            
            # Dataset profiles table (stores DatasetProfile)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dataset_profiles (
                    dataset_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    source_url TEXT,
                    file_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Schema detection
                    schema_detected BOOLEAN DEFAULT 0,
                    schema_result TEXT,
                    schema_confidence REAL,
                    schema_evidence TEXT,
                    
                    -- Target detection
                    target_detected BOOLEAN DEFAULT 0,
                    target_candidates TEXT,
                    chosen_target TEXT,
                    target_locked BOOLEAN DEFAULT 0,
                    target_override_reason TEXT,
                    
                    -- Modality
                    modality_breakdown TEXT,
                    
                    -- Compatibility
                    global_compatible BOOLEAN DEFAULT 0,
                    compatibility_score REAL,
                    compatibility_notes TEXT,
                    
                    -- Preprocessing
                    preprocessing_plan TEXT,
                    
                    -- Embeddings
                    embeddings_cached BOOLEAN DEFAULT 0,
                    embedding_refs TEXT,
                    
                    -- User overrides
                    user_overrides TEXT,
                    
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            
            # Indices for fast lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_updated 
                ON sessions(updated_at DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_profiles_session 
                ON dataset_profiles(session_id)
            """)
            
            logger.info("Database schema created")
    
    # ===== ExecutionContext Operations =====
    
    def save_context(self, context_dict: Dict[str, Any]) -> None:
        """Save ExecutionContext to sessions table."""
        session_id = context_dict.get('session_id')
        if not session_id:
            raise ValueError("context_dict must contain session_id")
        
        with self._get_connection() as conn:
            # Check if exists
            cursor = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            exists = cursor.fetchone() is not None
            
            if exists:
                # Update
                conn.execute("""
                    UPDATE sessions
                    SET updated_at = ?,
                        pipeline_stage = ?,
                        context_json = ?,
                        context_hash = ?
                    WHERE session_id = ?
                """, (
                    context_dict.get('updated_at', datetime.utcnow().isoformat()),
                    context_dict.get('pipeline_stage'),
                    json.dumps(context_dict),
                    context_dict.get('version'),
                    session_id
                ))
                logger.debug("Updated context for session %s", session_id)
            else:
                # Insert
                conn.execute("""
                    INSERT INTO sessions (session_id, created_at, updated_at, pipeline_stage, context_json, context_hash)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    context_dict.get('created_at', datetime.utcnow().isoformat()),
                    context_dict.get('updated_at', datetime.utcnow().isoformat()),
                    context_dict.get('pipeline_stage'),
                    json.dumps(context_dict),
                    context_dict.get('version')
                ))
                logger.debug("Created context for session %s", session_id)
    
    def load_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load ExecutionContext from sessions table."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT context_json FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return json.loads(row['context_json'])
            return None
    
    def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List all sessions (summary view)."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT session_id, created_at, updated_at, pipeline_stage
                FROM sessions
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_session_count(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> int:
        """
        Get total number of sessions (for pagination).
        
        Args:
            user_id: Optional user filter (not yet implemented)
            status: Optional status filter (not yet implemented)
        
        Returns:
            Total count of sessions
        """
        # TODO: Add user_id and status filtering
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM sessions")
            row = cursor.fetchone()
            return row['count'] if row else 0
    
    def close_session(self, session_id: str) -> bool:
        """Mark a session as closed."""
        with self._get_connection() as conn:
            # Load context, mark as closed, save back
            cursor = conn.execute(
                "SELECT context_json FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return False
            
            context_dict = json.loads(row['context_json'])
            context_dict['pipeline_stage'] = 'closed'
            context_dict['updated_at'] = datetime.utcnow().isoformat()
            
            conn.execute("""
                UPDATE sessions
                SET context_json = ?, updated_at = ?, pipeline_stage = 'closed'
                WHERE session_id = ?
            """, (json.dumps(context_dict), context_dict['updated_at'], session_id))
            
            logger.info("Closed session %s", session_id)
            return True
    
    # ===== DatasetProfile Operations =====
    
    def save_profile(self, profile_dict: Dict[str, Any], session_id: str) -> None:
        """Save DatasetProfile to dataset_profiles table."""
        dataset_id = profile_dict.get('dataset_id')
        if not dataset_id:
            raise ValueError("profile_dict must contain dataset_id")
        
        with self._get_connection() as conn:
            # Check if exists
            cursor = conn.execute(
                "SELECT dataset_id FROM dataset_profiles WHERE dataset_id = ?",
                (dataset_id,)
            )
            exists = cursor.fetchone() is not None
            
            if exists:
                # Update
                conn.execute("""
                    UPDATE dataset_profiles
                    SET updated_at = ?,
                        schema_detected = ?,
                        schema_result = ?,
                        schema_confidence = ?,
                        schema_evidence = ?,
                        target_detected = ?,
                        target_candidates = ?,
                        chosen_target = ?,
                        target_locked = ?,
                        target_override_reason = ?,
                        modality_breakdown = ?,
                        global_compatible = ?,
                        compatibility_score = ?,
                        preprocessing_plan = ?,
                        embeddings_cached = ?,
                        embedding_refs = ?,
                        user_overrides = ?
                    WHERE dataset_id = ?
                """, (
                    datetime.utcnow(),
                    int(profile_dict.get('schema_detected', False)),
                    json.dumps(profile_dict.get('schema_result')) if profile_dict.get('schema_result') else None,
                    profile_dict.get('schema_confidence', 0.0),
                    profile_dict.get('schema_evidence'),
                    int(profile_dict.get('target_detected', False)),
                    json.dumps(profile_dict.get('target_candidates', [])),
                    profile_dict.get('chosen_target'),
                    int(profile_dict.get('target_locked', False)),
                    profile_dict.get('target_override_reason'),
                    json.dumps(profile_dict.get('modality_breakdown', {})),
                    int(profile_dict.get('global_compatible', False)),
                    profile_dict.get('compatibility_score', 0.0),
                    json.dumps(profile_dict.get('preprocessing_plan')) if profile_dict.get('preprocessing_plan') else None,
                    int(profile_dict.get('embeddings_cached', False)),
                    json.dumps(profile_dict.get('embedding_refs')) if profile_dict.get('embedding_refs') else None,
                    json.dumps(profile_dict.get('user_overrides', {})),
                    dataset_id
                ))
                logger.debug("Updated profile for dataset %s", dataset_id)
            else:
                # Insert
                conn.execute("""
                    INSERT INTO dataset_profiles (
                        dataset_id, session_id, source_url, file_path,
                        schema_detected, schema_result, schema_confidence, schema_evidence,
                        target_detected, target_candidates, chosen_target, target_locked,
                        modality_breakdown, global_compatible, preprocessing_plan, user_overrides
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    dataset_id,
                    session_id,
                    profile_dict.get('source_url'),
                    profile_dict.get('file_path'),
                    int(profile_dict.get('schema_detected', False)),
                    json.dumps(profile_dict.get('schema_result')) if profile_dict.get('schema_result') else None,
                    profile_dict.get('schema_confidence', 0.0),
                    profile_dict.get('schema_evidence'),
                    int(profile_dict.get('target_detected', False)),
                    json.dumps(profile_dict.get('target_candidates', [])),
                    profile_dict.get('chosen_target'),
                    int(profile_dict.get('target_locked', False)),
                    json.dumps(profile_dict.get('modality_breakdown', {})),
                    int(profile_dict.get('global_compatible', False)),
                    json.dumps(profile_dict.get('preprocessing_plan')) if profile_dict.get('preprocessing_plan') else None,
                    json.dumps(profile_dict.get('user_overrides', {}))
                ))
                logger.debug("Created profile for dataset %s", dataset_id)
    
    def load_profile(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Load DatasetProfile from dataset_profiles table."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM dataset_profiles WHERE dataset_id = ?",
                (dataset_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_profile_dict(row)
    
    def load_session_profiles(self, session_id: str) -> List[Dict[str, Any]]:
        """Load all DatasetProfiles for a session."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM dataset_profiles WHERE session_id = ? ORDER BY created_at",
                (session_id,)
            )
            
            return [self._row_to_profile_dict(row) for row in cursor.fetchall()]
    
    def get_session_profiles(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Alias for load_session_profiles (backward compatibility).
        
        Get all dataset profiles for a session.
        """
        return self.load_session_profiles(session_id)
    
    def _row_to_profile_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to profile dict, parsing JSON fields."""
        data = dict(row)
        
        # Parse JSON fields
        json_fields = [
            'schema_result', 'target_candidates', 'modality_breakdown',
            'preprocessing_plan', 'embedding_refs', 'user_overrides'
        ]
        for field in json_fields:
            if data.get(field):
                try:
                    data[field] = json.loads(data[field])
                except (json.JSONDecodeError, TypeError):
                    data[field] = None
        
        # Convert booleans
        bool_fields = ['schema_detected', 'target_detected', 'target_locked',
                       'global_compatible', 'embeddings_cached']
        for field in bool_fields:
            if field in data and data[field] is not None:
                data[field] = bool(data[field])
        
        return data


# Global singleton instance
context_db = ContextDatabase()
