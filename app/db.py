#!/usr/bin/env python3
"""
db.py — Per-session SQLite database management for Agent-StateSync.

Each chat session gets its own SQLite database file in the dbs/ directory.
Tables:
  - sessions:       Session metadata
  - world_state:    Current key-value state (single source of truth)
  - state_changes:  Audit log keyed by (message_id, swipe_index)
  - message_log:    Optional log of all messages
"""

import sqlite3
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

_SESSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id         TEXT PRIMARY KEY,
    character_name     TEXT DEFAULT '',
    character_description  TEXT DEFAULT '',
    character_personality TEXT DEFAULT '',
    character_scenario TEXT DEFAULT '',
    character_first_mes TEXT DEFAULT '',
    character_mes_example TEXT DEFAULT '',
    persona_name       TEXT DEFAULT '',
    persona_description TEXT DEFAULT '',
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    initialized        BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS world_state (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    category    TEXT DEFAULT 'general',
    source      TEXT DEFAULT 'auto',
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS state_changes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id   INTEGER NOT NULL,
    swipe_index  INTEGER NOT NULL DEFAULT 0,
    field        TEXT NOT NULL,
    old_value    TEXT,
    new_value    TEXT,
    applied      BOOLEAN DEFAULT TRUE,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(message_id, swipe_index, field)
);

CREATE TABLE IF NOT EXISTS message_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id   INTEGER NOT NULL,
    swipe_index  INTEGER NOT NULL DEFAULT 0,
    role         TEXT NOT NULL,
    content      TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class Database:
    """Per-session SQLite database manager."""

    def __init__(self, db_dir: str = "./dbs"):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Database directory: {self.db_dir.resolve()}")

    def _get_conn(self, session_id: str) -> sqlite3.Connection:
        db_path = self.db_dir / f"{session_id}.db"
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ── Session Management ─────────────────────────────────────

    def create_session(self, session_id: str) -> bool:
        conn = self._get_conn(session_id)
        try:
            conn.executescript(_SESSION_SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)",
                (session_id,),
            )
            conn.commit()
            logger.info(f"Created session DB: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {e}")
            return False
        finally:
            conn.close()

    def init_session(self, session_id: str, character_data: dict) -> bool:
        self.create_session(session_id)
        conn = self._get_conn(session_id)
        try:
            conn.execute(
                """
                UPDATE sessions SET
                    character_name = COALESCE(?, character_name),
                    character_description = COALESCE(?, character_description),
                    character_personality = COALESCE(?, character_personality),
                    character_scenario = COALESCE(?, character_scenario),
                    character_first_mes = COALESCE(?, character_first_mes),
                    character_mes_example = COALESCE(?, character_mes_example),
                    persona_name = COALESCE(?, persona_name),
                    persona_description = COALESCE(?, persona_description),
                    initialized = TRUE
                WHERE session_id = ?
                """,
                (
                    character_data.get("character_name"),
                    character_data.get("character_description"),
                    character_data.get("character_personality"),
                    character_data.get("character_scenario"),
                    character_data.get("character_first_mes"),
                    character_data.get("character_mes_example"),
                    character_data.get("persona_name"),
                    character_data.get("persona_description"),
                    session_id,
                ),
            )
            conn.commit()
            logger.info(f"Initialized session {session_id} with character data")
            return True
        except Exception as e:
            logger.error(f"Failed to init session {session_id}: {e}")
            return False
        finally:
            conn.close()

    def is_initialized(self, session_id: str) -> bool:
        try:
            conn = self._get_conn(session_id)
            try:
                row = conn.execute(
                    "SELECT initialized FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                return bool(row and row["initialized"])
            finally:
                conn.close()
        except sqlite3.OperationalError:
            return False

    def get_session(self, session_id: str) -> Optional[dict]:
        try:
            conn = self._get_conn(session_id)
            try:
                row = conn.execute(
                    "SELECT * FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()
        except sqlite3.OperationalError:
            return None

    # ── World State ────────────────────────────────────────────

    def get_world_state(self, session_id: str) -> Dict[str, Any]:
        try:
            conn = self._get_conn(session_id)
            try:
                rows = conn.execute(
                    "SELECT key, value, category, source FROM world_state ORDER BY key"
                ).fetchall()
                state = {}
                for row in rows:
                    try:
                        state[row["key"]] = json.loads(row["value"])
                    except (json.JSONDecodeError, TypeError):
                        state[row["key"]] = row["value"]
                return state
            finally:
                conn.close()
        except sqlite3.OperationalError:
            return {}

    def update_world_state(
        self,
        session_id: str,
        changes: Dict[str, Any],
        message_id: int,
        swipe_index: int,
    ) -> int:
        conn = self._get_conn(session_id)
        count = 0
        try:
            for field, new_value in changes.items():
                row = conn.execute(
                    "SELECT value FROM world_state WHERE key = ?", (field,)
                ).fetchone()
                old_value = row["value"] if row else None

                if isinstance(new_value, (dict, list)):
                    value_str = json.dumps(new_value, ensure_ascii=False)
                else:
                    value_str = str(new_value)

                conn.execute(
                    """
                    INSERT INTO world_state (key, value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (field, value_str),
                )

                conn.execute(
                    """
                    INSERT OR REPLACE INTO state_changes
                        (message_id, swipe_index, field, old_value, new_value)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (message_id, swipe_index, field, old_value, value_str),
                )
                count += 1

            conn.commit()
            logger.info(
                f"Updated {count} state fields for session {session_id}, "
                f"msg {message_id}, swipe {swipe_index}"
            )
        except Exception as e:
            logger.error(f"Failed to update world state: {e}")
            conn.rollback()
        finally:
            conn.close()

        return count

    # ── Swipe / Redo Reversion ─────────────────────────────────

    def revert_swipe(
        self, session_id: str, message_id: int, swipe_index: int
    ) -> int:
        conn = self._get_conn(session_id)
        reverted = 0
        try:
            changes = conn.execute(
                """
                SELECT id, field, old_value, new_value, swipe_index
                FROM state_changes
                WHERE message_id = ? AND swipe_index >= ? AND applied = TRUE
                ORDER BY id DESC
                """,
                (message_id, swipe_index),
            ).fetchall()

            seen = set()
            for change in changes:
                field = change["field"]
                if field in seen:
                    continue
                seen.add(field)

                if change["old_value"] is not None:
                    conn.execute(
                        "UPDATE world_state SET value = ?, updated_at = CURRENT_TIMESTAMP WHERE key = ?",
                        (change["old_value"], field),
                    )
                else:
                    conn.execute(
                        "DELETE FROM world_state WHERE key = ?", (field,)
                    )

                conn.execute(
                    "UPDATE state_changes SET applied = FALSE WHERE message_id = ? AND field = ? AND applied = TRUE",
                    (message_id, field),
                )
                reverted += 1

            conn.commit()
            if reverted:
                logger.info(
                    f"Reverted {reverted} fields for swipe at msg {message_id}, "
                    f"swipe >= {swipe_index}"
                )
        except Exception as e:
            logger.error(f"Failed to revert swipe: {e}")
            conn.rollback()
        finally:
            conn.close()

        return reverted

    def revert_from_message(self, session_id: str, message_id: int) -> int:
        conn = self._get_conn(session_id)
        reverted = 0
        try:
            changes = conn.execute(
                """
                SELECT id, message_id AS mid, swipe_index, field, old_value, new_value
                FROM state_changes
                WHERE message_id >= ? AND applied = TRUE
                ORDER BY mid DESC, swipe_index DESC, id DESC
                """,
                (message_id,),
            ).fetchall()

            seen = set()
            for change in changes:
                key = (change["mid"], change["swipe_index"], change["field"])
                if key in seen:
                    continue
                seen.add(key)

                field = change["field"]
                if change["old_value"] is not None:
                    conn.execute(
                        "UPDATE world_state SET value = ?, updated_at = CURRENT_TIMESTAMP WHERE key = ?",
                        (change["old_value"], field),
                    )
                else:
                    conn.execute(
                        "DELETE FROM world_state WHERE key = ?", (field,)
                    )

                conn.execute(
                    "UPDATE state_changes SET applied = FALSE WHERE message_id = ? AND swipe_index = ? AND field = ?",
                    (change["mid"], change["swipe_index"], field),
                )
                reverted += 1

            conn.commit()
            if reverted:
                logger.info(
                    f"Redo revert: undone {reverted} changes from msg {message_id}+"
                )
        except Exception as e:
            logger.error(f"Failed to revert from message: {e}")
            conn.rollback()
        finally:
            conn.close()

        return reverted

    # ── Message Log ────────────────────────────────────────────

    def log_message(
        self,
        session_id: str,
        message_id: int,
        swipe_index: int,
        role: str,
        content: str,
    ):
        try:
            conn = self._get_conn(session_id)
            try:
                conn.execute(
                    "INSERT INTO message_log (message_id, swipe_index, role, content) VALUES (?, ?, ?, ?)",
                    (message_id, swipe_index, role, content),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.debug(f"Message log failed (non-critical): {e}")

    # ── Maintenance ────────────────────────────────────────────

    def list_sessions(self) -> List[str]:
        sessions = []
        for db_file in self.db_dir.glob("*.db"):
            sessions.append(db_file.stem)
        return sorted(sessions)

    def delete_session(self, session_id: str) -> bool:
        db_path = self.db_dir / f"{session_id}.db"
        try:
            if db_path.exists():
                db_path.unlink()
                logger.info(f"Deleted session DB: {session_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
        return False

    def get_session_stats(self, session_id: str) -> Optional[dict]:
        try:
            conn = self._get_conn(session_id)
            try:
                state_count = conn.execute(
                    "SELECT COUNT(*) as c FROM world_state"
                ).fetchone()["c"]
                change_count = conn.execute(
                    "SELECT COUNT(*) as c FROM state_changes WHERE applied = TRUE"
                ).fetchone()["c"]
                msg_count = conn.execute(
                    "SELECT COUNT(*) as c FROM message_log"
                ).fetchone()["c"]
                return {
                    "session_id": session_id,
                    "world_state_fields": state_count,
                    "applied_changes": change_count,
                    "logged_messages": msg_count,
                }
            finally:
                conn.close()
        except sqlite3.OperationalError:
            return None