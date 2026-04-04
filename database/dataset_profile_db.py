"""
Dataset Profile Database - Phase 3
Stores per-dataset intelligence: schema, target, modality, embeddings, preprocessing plans.
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


class DatasetProfileDB:
    """
    Database for dataset profiles (Phase 3-5).
    Thread-safe singleton with one connection per thread.
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
        logger.info("DatasetProfileDB initialized at %s", self.db_path)
    
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
    
    def _create_tables(self):
        """Create dataset_profiles table if not exists."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dataset_profiles (
                    dataset_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    source_url TEXT,
                    file_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Schema detection (Phase 3)
                    schema_detected BOOLEAN DEFAULT 0,
                    schema_result TEXT,  -- JSON
                    schema_confidence REAL,
                    schema_evidence TEXT,
                    
                    -- Target detection (Phase 4)
                    target_detected BOOLEAN DEFAULT 0,
                    target_candidates TEXT,  -- JSON: [{name, score, reason}, ...]
                    chosen_target TEXT,
                    target_locked BOOLEAN DEFAULT 0,
                    target_override_reason TEXT,
                    
                    -- Global aggregation (Phase 5)
                    global_compatible BOOLEAN DEFAULT 0,
                    compatibility_score REAL,
                    compatibility_notes TEXT,
                    
                    -- Modality breakdown
                    modality_breakdown TEXT,  -- JSON: {tabular: 0.8, text: 0.2, ...}
                    
                    -- Embeddings (Phase 8)
                    embeddings_cached BOOLEAN DEFAULT 0,
                    embedding_refs TEXT,  -- JSON: {col_embeddings: path, ...}
                    
                    -- Preprocessing plan (Phase 7)
                    preprocessing_plan TEXT,  -- JSON
                    
                    -- User overrides
                    user_overrides TEXT,  -- JSON
                    
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            
            # Indices for fast lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_dataset_profiles_session 
                ON dataset_profiles(session_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_dataset_profiles_created 
                ON dataset_profiles(created_at)
            """)
            
            conn.commit()
            logger.info("dataset_profiles table ready")
    
    # ------------------------------------------------------------------ #
    # CRUD Operations
    # ------------------------------------------------------------------ #
    
    def create_profile(
        self,
        dataset_id: str,
        session_id: str,
        source_url: Optional[str] = None,
        file_path: Optional[str] = None,
        modality_breakdown: Optional[Dict[str, float]] = None
    ) -> bool:
        """Create a new dataset profile."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO dataset_profiles 
                    (dataset_id, session_id, source_url, file_path, modality_breakdown, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    dataset_id,
                    session_id,
                    source_url,
                    file_path,
                    json.dumps(modality_breakdown or {}),
                    datetime.utcnow(),
                    datetime.utcnow()
                ))
                conn.commit()
                logger.info("Created dataset profile: %s in session %s", dataset_id, session_id)
                return True
        except sqlite3.IntegrityError as e:
            logger.warning("Profile already exists: %s", dataset_id)
            return False
        except Exception as e:
            logger.error("Failed to create profile: %s", e, exc_info=True)
            return False
    
    def get_profile(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get a dataset profile by ID."""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM dataset_profiles WHERE dataset_id = ?
            """, (dataset_id,)).fetchone()
            
            if not row:
                return None
            
            return self._row_to_dict(row)
    
    def get_session_profiles(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all dataset profiles for a session."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM dataset_profiles 
                WHERE session_id = ?
                ORDER BY created_at DESC
            """, (session_id,)).fetchall()
            
            return [self._row_to_dict(row) for row in rows]
    
    def update_schema(
        self,
        dataset_id: str,
        schema_result: Dict[str, Any],
        confidence: float,
        evidence: Optional[str] = None
    ) -> bool:
        """Update schema detection results (Phase 3)."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE dataset_profiles
                    SET schema_detected = 1,
                        schema_result = ?,
                        schema_confidence = ?,
                        schema_evidence = ?,
                        updated_at = ?
                    WHERE dataset_id = ?
                """, (
                    json.dumps(schema_result),
                    confidence,
                    evidence,
                    datetime.utcnow(),
                    dataset_id
                ))
                conn.commit()
                logger.info("Updated schema for dataset: %s (confidence: %.2f)", dataset_id, confidence)
                return True
        except Exception as e:
            logger.error("Failed to update schema: %s", e, exc_info=True)
            return False
    
    def update_target(
        self,
        dataset_id: str,
        target_candidates: List[Dict[str, Any]],
        chosen_target: Optional[str] = None,
        locked: bool = False,
        override_reason: Optional[str] = None
    ) -> bool:
        """Update target detection results (Phase 4)."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE dataset_profiles
                    SET target_detected = 1,
                        target_candidates = ?,
                        chosen_target = ?,
                        target_locked = ?,
                        target_override_reason = ?,
                        updated_at = ?
                    WHERE dataset_id = ?
                """, (
                    json.dumps(target_candidates),
                    chosen_target,
                    1 if locked else 0,
                    override_reason,
                    datetime.utcnow(),
                    dataset_id
                ))
                conn.commit()
                logger.info("Updated target for dataset: %s -> %s", dataset_id, chosen_target)
                return True
        except Exception as e:
            logger.error("Failed to update target: %s", e, exc_info=True)
            return False
    
    def update_preprocessing_plan(
        self,
        dataset_id: str,
        preprocessing_plan: Dict[str, Any]
    ) -> bool:
        """Update preprocessing plan (Phase 7)."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE dataset_profiles
                    SET preprocessing_plan = ?,
                        updated_at = ?
                    WHERE dataset_id = ?
                """, (
                    json.dumps(preprocessing_plan),
                    datetime.utcnow(),
                    dataset_id
                ))
                conn.commit()
                logger.info("Updated preprocessing plan for dataset: %s", dataset_id)
                return True
        except Exception as e:
            logger.error("Failed to update preprocessing plan: %s", e, exc_info=True)
            return False
    
    def update_compatibility(
        self,
        dataset_id: str,
        compatible: bool,
        score: float,
        notes: Optional[str] = None
    ) -> bool:
        """Update global compatibility (Phase 5)."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE dataset_profiles
                    SET global_compatible = ?,
                        compatibility_score = ?,
                        compatibility_notes = ?,
                        updated_at = ?
                    WHERE dataset_id = ?
                """, (
                    1 if compatible else 0,
                    score,
                    notes,
                    datetime.utcnow(),
                    dataset_id
                ))
                conn.commit()
                logger.info("Updated compatibility for dataset: %s (score: %.2f)", dataset_id, score)
                return True
        except Exception as e:
            logger.error("Failed to update compatibility: %s", e, exc_info=True)
            return False
    
    def delete_profile(self, dataset_id: str) -> bool:
        """Delete a dataset profile."""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM dataset_profiles WHERE dataset_id = ?", (dataset_id,))
                conn.commit()
                logger.info("Deleted dataset profile: %s", dataset_id)
                return True
        except Exception as e:
            logger.error("Failed to delete profile: %s", e, exc_info=True)
            return False
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to dict, parsing JSON fields."""
        data = dict(row)
        
        # Parse JSON fields
        json_fields = [
            'schema_result', 'target_candidates', 'modality_breakdown',
            'embedding_refs', 'preprocessing_plan', 'user_overrides'
        ]
        for field in json_fields:
            if data.get(field):
                try:
                    data[field] = json.loads(data[field])
                except (json.JSONDecodeError, TypeError):
                    data[field] = None
        
        # Convert booleans
        bool_fields = ['schema_detected', 'target_detected', 'global_compatible', 
                       'target_locked', 'embeddings_cached']
        for field in bool_fields:
            if field in data and data[field] is not None:
                data[field] = bool(data[field])
        
        return data


# Global singleton instance
dataset_profile_db = DatasetProfileDB()
