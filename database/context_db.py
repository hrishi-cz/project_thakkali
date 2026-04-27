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
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging

from config.paths import SESSION_DB_PATH

logger = logging.getLogger(__name__)


class OptimisticLockError(RuntimeError):
    """Raised when a session write detects a stale revision."""


class ContextDatabase:
    """
    Unified database for ExecutionContext and DatasetProfile.
    
    Thread-safe singleton with one connection per thread.
    Replaces separate dataset_profile_db and session_db.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path: str = str(SESSION_DB_PATH)):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_path: str = str(SESSION_DB_PATH)):
        if self._initialized:
            return
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._op_lock = threading.RLock()
        self._create_tables()
        self._initialized = True
        logger.info("ContextDatabase initialized at %s", self.db_path)
    
    @contextmanager
    def _get_connection(self):
        """Thread-safe connection manager."""
        with self._op_lock:
            if not hasattr(self._local, 'conn') or self._local.conn is None:
                self._local.conn = sqlite3.connect(
                    str(self.db_path),
                    timeout=30.0,
                    check_same_thread=False
                )
                # WAL mode is safer for concurrent readers/writers in API workloads.
                self._local.conn.execute("PRAGMA journal_mode=WAL")
                self._local.conn.execute("PRAGMA busy_timeout=5000")
                self._local.conn.execute("PRAGMA foreign_keys=ON")
                self._local.conn.execute("PRAGMA synchronous=NORMAL")
                self._local.conn.row_factory = sqlite3.Row

            try:
                yield self._local.conn
            except Exception:
                self._local.conn.rollback()
                raise
            else:
                for attempt in range(3):
                    try:
                        self._local.conn.commit()
                        break
                    except sqlite3.OperationalError as exc:
                        if "locked" in str(exc).lower() and attempt < 2:
                            time.sleep(0.05 * (attempt + 1))
                            continue
                        raise
    
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
                    context_hash TEXT,
                    revision INTEGER NOT NULL DEFAULT 0
                )
            """)

            self._ensure_sessions_revision_column(conn)
            
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
                    revision INTEGER NOT NULL DEFAULT 0,
                    
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            try:
                conn.execute(
                    "ALTER TABLE dataset_profiles ADD COLUMN revision INTEGER NOT NULL DEFAULT 0"
                )
            except sqlite3.OperationalError:
                pass

            conn.execute("""
                CREATE TABLE IF NOT EXISTS probe_samples (
                    session_id TEXT PRIMARY KEY,
                    data BLOB NOT NULL,
                    updated_at TEXT NOT NULL,
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
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_probe_samples_updated
                ON probe_samples(updated_at DESC)
            """)
            
            logger.info("Database schema created")

    @staticmethod
    def _json_default(value: Any) -> Any:
        """Best-effort serializer for numpy / datetime / path-like values."""
        try:
            import numpy as np

            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
        except Exception:
            pass

        if isinstance(value, (datetime, Path)):
            return str(value)

        return str(value)

    @staticmethod
    def _ensure_sessions_revision_column(conn: sqlite3.Connection) -> None:
        """Ensure optimistic-lock revision column exists on legacy DBs."""
        try:
            columns = {
                str(row["name"]).lower()
                for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
            }
        except Exception:
            columns = set()

        if "revision" in columns:
            return

        conn.execute(
            "ALTER TABLE sessions ADD COLUMN revision INTEGER NOT NULL DEFAULT 0"
        )

        rows = conn.execute(
            "SELECT session_id, context_json FROM sessions"
        ).fetchall()
        for row in rows:
            revision = 0
            try:
                payload = json.loads(row["context_json"] or "{}")
                revision = int(payload.get("revision", 0) or 0)
            except Exception:
                revision = 0
            if revision <= 0:
                revision = 1
            conn.execute(
                "UPDATE sessions SET revision = ? WHERE session_id = ?",
                (revision, row["session_id"]),
            )

        logger.info("ContextDatabase: added sessions.revision column for optimistic locking")
    
    # ===== ExecutionContext Operations =====
    
    def save_context(
        self,
        context_dict: Dict[str, Any],
        expected_revision: Optional[int] = None,
    ) -> int:
        """Save ExecutionContext to sessions table with optimistic locking."""
        session_id = context_dict.get('session_id')
        if not session_id:
            raise ValueError("context_dict must contain session_id")

        now_iso = datetime.now(timezone.utc).isoformat()
        context_dict.setdefault('created_at', now_iso)
        context_dict.setdefault('updated_at', now_iso)
        
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT revision FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()

            if row is not None:
                current_revision = int(row["revision"] or 0)
                if expected_revision is None:
                    raw_expected = context_dict.get("revision", current_revision)
                else:
                    raw_expected = expected_revision

                try:
                    expected = int(raw_expected)
                except Exception:
                    expected = current_revision

                if expected != current_revision:
                    raise OptimisticLockError(
                        f"Session '{session_id}' is stale (expected revision {expected}, "
                        f"current revision {current_revision})."
                    )

                new_revision = current_revision + 1
                context_dict["revision"] = new_revision

                update_cursor = conn.execute(
                    """
                    UPDATE sessions
                    SET updated_at = ?,
                        pipeline_stage = ?,
                        context_json = ?,
                        context_hash = ?,
                        revision = ?
                    WHERE session_id = ? AND revision = ?
                    """,
                    (
                        context_dict.get("updated_at", now_iso),
                        context_dict.get("pipeline_stage"),
                        json.dumps(context_dict, default=self._json_default),
                        context_dict.get("version"),
                        new_revision,
                        session_id,
                        current_revision,
                    ),
                )

                if int(update_cursor.rowcount or 0) != 1:
                    raise OptimisticLockError(
                        f"Session '{session_id}' update lost due to concurrent write."
                    )

                logger.debug("Updated context for session %s", session_id)
            else:
                new_revision = int(context_dict.get("revision", 0) or 0)
                if new_revision <= 0:
                    new_revision = 1
                context_dict["revision"] = new_revision

                conn.execute(
                    """
                    INSERT INTO sessions (
                        session_id,
                        created_at,
                        updated_at,
                        pipeline_stage,
                        context_json,
                        context_hash,
                        revision
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        context_dict.get("created_at", now_iso),
                        context_dict.get("updated_at", now_iso),
                        context_dict.get("pipeline_stage"),
                        json.dumps(context_dict, default=self._json_default),
                        context_dict.get("version"),
                        new_revision,
                    ),
                )
                logger.debug("Created context for session %s", session_id)

        return int(context_dict.get("revision", 0) or 0)
    
    def session_exists(self, session_id: str) -> bool:
        """Return True if a session row exists (regardless of ExecutionContext state)."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ? LIMIT 1",
                (session_id,),
            )
            return cursor.fetchone() is not None

    def load_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load ExecutionContext from sessions table."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT context_json, revision FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()

            if row:
                payload = json.loads(row['context_json'])
                if "revision" not in payload:
                    payload["revision"] = int(row["revision"] or 0)
                return payload
            return None
    
    def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all sessions (summary view)."""
        with self._get_connection() as conn:
            safe_limit = max(0, int(limit))
            safe_offset = max(0, int(offset))

            if user_id is None and status is None:
                cursor = conn.execute(
                    """
                    SELECT session_id, created_at, updated_at, pipeline_stage
                    FROM sessions
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (safe_limit, safe_offset),
                )
                return [dict(row) for row in cursor.fetchall()]

            status_expr = (
                "COALESCE("
                "NULLIF(json_extract(context_json, '$.status'), ''), "
                "CASE WHEN pipeline_stage = 'closed' THEN 'closed' ELSE 'active' END"
                ")"
            )
            query = (
                "SELECT session_id, created_at, updated_at, pipeline_stage "
                "FROM sessions WHERE 1=1"
            )
            params: List[Any] = []

            if user_id is not None:
                query += " AND json_extract(context_json, '$.user_id') = ?"
                params.append(user_id)

            if status is not None:
                query += f" AND {status_expr} = ?"
                params.append(status)

            query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
            params.extend([safe_limit, safe_offset])

            try:
                cursor = conn.execute(query, tuple(params))
                return [dict(row) for row in cursor.fetchall()]
            except sqlite3.OperationalError as exc:
                if "json_extract" not in str(exc).lower():
                    raise
                logger.warning(
                    "JSON1 functions unavailable in SQLite; falling back to Python filtering for session list"
                )
                return self._list_sessions_python(
                    conn,
                    user_id=user_id,
                    status=status,
                    limit=safe_limit,
                    offset=safe_offset,
                )
    
    def get_session_count(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> int:
        """
        Get total number of sessions (for pagination).
        
        Args:
            user_id: Optional user filter (matched from context_json.user_id)
            status: Optional status filter (matched against pipeline_stage)
        
        Returns:
            Total count of sessions
        """
        with self._get_connection() as conn:
            if user_id is None and status is None:
                cursor = conn.execute("SELECT COUNT(*) as count FROM sessions")
                row = cursor.fetchone()
                return row['count'] if row else 0

            status_expr = (
                "COALESCE(" 
                "NULLIF(json_extract(context_json, '$.status'), ''), "
                "CASE WHEN pipeline_stage = 'closed' THEN 'closed' ELSE 'active' END"
                ")"
            )

            query = "SELECT COUNT(*) as count FROM sessions WHERE 1=1"
            params: List[Any] = []

            if user_id is not None:
                query += " AND json_extract(context_json, '$.user_id') = ?"
                params.append(user_id)

            if status is not None:
                query += f" AND {status_expr} = ?"
                params.append(status)

            try:
                cursor = conn.execute(query, tuple(params))
                row = cursor.fetchone()
                return int(row['count']) if row else 0
            except sqlite3.OperationalError as exc:
                # Fallback for SQLite builds without JSON1 support.
                if "json_extract" not in str(exc).lower():
                    raise
                logger.warning(
                    "JSON1 functions unavailable in SQLite; falling back to Python filtering for session count"
                )
                return self._count_sessions_python(conn, user_id=user_id, status=status)

    @staticmethod
    def _count_sessions_python(
        conn: sqlite3.Connection,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> int:
        """Python fallback for filtered session counts when JSON1 is unavailable."""
        cursor = conn.execute("SELECT pipeline_stage, context_json FROM sessions")
        count = 0

        for row in cursor.fetchall():
            try:
                ctx = json.loads(row["context_json"])
            except (json.JSONDecodeError, TypeError):
                ctx = {}

            if user_id is not None and ctx.get("user_id") != user_id:
                continue

            session_status = ctx.get("status")
            if not session_status:
                session_status = "closed" if row["pipeline_stage"] == "closed" else "active"

            if status is not None and session_status != status:
                continue

            count += 1

        return count

    @staticmethod
    def _list_sessions_python(
        conn: sqlite3.Connection,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Python fallback for session listing when JSON1 functions are unavailable."""
        cursor = conn.execute(
            """
            SELECT session_id, created_at, updated_at, pipeline_stage, context_json
            FROM sessions
            ORDER BY updated_at DESC
            """
        )

        filtered_rows: List[Dict[str, Any]] = []
        for row in cursor.fetchall():
            try:
                ctx = json.loads(row["context_json"])
            except (json.JSONDecodeError, TypeError):
                ctx = {}

            if user_id is not None and ctx.get("user_id") != user_id:
                continue

            session_status = ctx.get("status")
            if not session_status:
                session_status = "closed" if row["pipeline_stage"] == "closed" else "active"

            if status is not None and session_status != status:
                continue

            filtered_rows.append(
                {
                    "session_id": row["session_id"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "pipeline_stage": row["pipeline_stage"],
                }
            )

        safe_offset = max(0, int(offset))
        safe_limit = max(0, int(limit))
        if safe_limit == 0:
            return []
        return filtered_rows[safe_offset:safe_offset + safe_limit]
    
    def close_session(self, session_id: str) -> bool:
        """Mark a session as closed."""
        context_dict = self.load_context(session_id)
        if not context_dict:
            return False

        expected_revision = int(context_dict.get("revision", 0) or 0)
        context_dict['pipeline_stage'] = 'closed'
        context_dict['status'] = 'closed'
        context_dict['updated_at'] = datetime.now(timezone.utc).isoformat()

        self.save_context(context_dict, expected_revision=expected_revision)
        logger.info("Closed session %s", session_id)
        return True

    def cleanup_stale_sessions(self, max_age_hours: int = 24) -> int:
        """
        Delete sessions that have not been updated in ``max_age_hours`` hours.

        Called on API startup and periodically to prevent unbounded DB growth.
        Returns the number of sessions deleted.
        """
        cutoff = (
            datetime.now(timezone.utc)
            .replace(tzinfo=None)  # SQLite stores naive UTC strings
        )
        import datetime as _dt
        cutoff_str = (
            datetime.now(timezone.utc) - _dt.timedelta(hours=max_age_hours)
        ).isoformat()

        try:
            with self._get_connection() as conn:
                # Collect stale session IDs first
                stale_ids = [
                    row[0] for row in conn.execute(
                        """
                        SELECT session_id FROM sessions
                        WHERE pipeline_stage = 'closed'
                           OR (updated_at < ? AND pipeline_stage NOT IN ('phase5_complete', 'phase7_complete'))
                        """,
                        (cutoff_str,),
                    ).fetchall()
                ]
                if not stale_ids:
                    return 0
                placeholders = ",".join("?" * len(stale_ids))
                # Delete child rows first to satisfy foreign key constraints
                conn.execute(f"DELETE FROM dataset_profiles WHERE session_id IN ({placeholders})", stale_ids)
                conn.execute(f"DELETE FROM probe_samples WHERE session_id IN ({placeholders})", stale_ids)
                cursor = conn.execute(f"DELETE FROM sessions WHERE session_id IN ({placeholders})", stale_ids)
                deleted = cursor.rowcount
                conn.commit()
            if deleted:
                logger.info("cleanup_stale_sessions: deleted %d stale sessions", deleted)
            return deleted
        except Exception as exc:
            logger.warning("cleanup_stale_sessions failed: %s", exc)
            return 0

    def save_probe_sample(self, session_id: str, X: Any, y: Any) -> None:
        """Persist tabular probe sample (X, y) as compressed bytes."""
        import gzip
        import io

        import numpy as np

        if not session_id:
            raise ValueError("session_id is required")

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, X=X_arr, y=y_arr)
        payload = gzip.compress(buffer.getvalue())

        with self._get_connection() as conn:
            session_row = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session_row is None:
                raise ValueError(
                    f"Cannot save probe sample: session '{session_id}' is not persisted"
                )

            conn.execute(
                """
                INSERT INTO probe_samples (session_id, data, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    data = excluded.data,
                    updated_at = excluded.updated_at
                """,
                (session_id, payload, datetime.now(timezone.utc).isoformat()),
            )

    def load_probe_sample(self, session_id: str) -> Optional[tuple[Any, Any]]:
        """Load persisted probe sample for a session, if available."""
        import gzip
        import io

        import numpy as np

        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT data FROM probe_samples WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        if row is None:
            return None

        raw = row["data"] if isinstance(row, sqlite3.Row) else row[0]
        buffer = io.BytesIO(gzip.decompress(raw))
        loaded = np.load(buffer, allow_pickle=False)
        return loaded["X"], loaded["y"]
    
    # ===== DatasetProfile Operations =====
    
    def save_profile(
        self,
        profile_dict: Dict[str, Any],
        session_id: str,
        expected_revision: Optional[int] = None,
    ) -> None:
        """Save DatasetProfile to dataset_profiles table with optional optimistic locking."""
        dataset_id = profile_dict.get('dataset_id')
        if not dataset_id:
            raise ValueError("profile_dict must contain dataset_id")

        now_iso = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")

            row = conn.execute(
                "SELECT revision FROM dataset_profiles WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()

            if row is not None:
                current_revision = int(row['revision'] or 0)

                if expected_revision is not None:
                    try:
                        expected = int(expected_revision)
                    except Exception:
                        expected = current_revision
                else:
                    expected = current_revision

                update_sql = """
                    UPDATE dataset_profiles
                    SET session_id = ?,
                        updated_at = ?,
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
                        user_overrides = ?,
                        revision = revision + 1
                    WHERE dataset_id = ?
                """
                params: List[Any] = [
                    session_id,  # update session_id so profile moves to the current session
                    now_iso,
                    int(profile_dict.get('schema_detected', False)),
                    json.dumps(profile_dict.get('schema_result'), default=self._json_default) if profile_dict.get('schema_result') else None,
                    profile_dict.get('schema_confidence', 0.0),
                    profile_dict.get('schema_evidence'),
                    int(profile_dict.get('target_detected', False)),
                    json.dumps(profile_dict.get('target_candidates', []), default=self._json_default),
                    profile_dict.get('chosen_target'),
                    int(profile_dict.get('target_locked', False)),
                    profile_dict.get('target_override_reason'),
                    json.dumps(profile_dict.get('modality_breakdown', {}), default=self._json_default),
                    int(profile_dict.get('global_compatible', False)),
                    profile_dict.get('compatibility_score', 0.0),
                    json.dumps(profile_dict.get('preprocessing_plan'), default=self._json_default) if profile_dict.get('preprocessing_plan') else None,
                    int(profile_dict.get('embeddings_cached', False)),
                    json.dumps(profile_dict.get('embedding_refs'), default=self._json_default) if profile_dict.get('embedding_refs') else None,
                    json.dumps(profile_dict.get('user_overrides', {}), default=self._json_default),
                    dataset_id,
                ]

                if expected_revision is not None:
                    update_sql += " AND revision = ?"
                    params.append(expected)

                update_cursor = conn.execute(update_sql, tuple(params))

                if expected_revision is not None and int(update_cursor.rowcount or 0) != 1:
                    raise OptimisticLockError(
                        f"Profile '{dataset_id}' update lost due to concurrent write "
                        f"(expected revision {expected})."
                    )

                profile_dict['revision'] = current_revision + 1
                logger.debug("Updated profile for dataset %s", dataset_id)
            else:
                new_revision = int(profile_dict.get('revision', 0) or 0)
                if new_revision <= 0:
                    new_revision = 1
                profile_dict['revision'] = new_revision

                conn.execute("""
                    INSERT INTO dataset_profiles (
                        dataset_id, session_id, source_url, file_path,
                        schema_detected, schema_result, schema_confidence, schema_evidence,
                        target_detected, target_candidates, chosen_target, target_locked,
                        modality_breakdown, global_compatible, preprocessing_plan, user_overrides,
                        revision
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    dataset_id,
                    session_id,
                    profile_dict.get('source_url'),
                    profile_dict.get('file_path'),
                    int(profile_dict.get('schema_detected', False)),
                    json.dumps(profile_dict.get('schema_result'), default=self._json_default) if profile_dict.get('schema_result') else None,
                    profile_dict.get('schema_confidence', 0.0),
                    profile_dict.get('schema_evidence'),
                    int(profile_dict.get('target_detected', False)),
                    json.dumps(profile_dict.get('target_candidates', []), default=self._json_default),
                    profile_dict.get('chosen_target'),
                    int(profile_dict.get('target_locked', False)),
                    json.dumps(profile_dict.get('modality_breakdown', {}), default=self._json_default),
                    int(profile_dict.get('global_compatible', False)),
                    json.dumps(profile_dict.get('preprocessing_plan'), default=self._json_default) if profile_dict.get('preprocessing_plan') else None,
                    json.dumps(profile_dict.get('user_overrides', {}), default=self._json_default),
                    new_revision,
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
