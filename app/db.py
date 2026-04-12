#!/usr/bin/env python3
"""
db.py — Per-session SQLite database management for Agent-StateSync.

Each chat session gets its own SQLite database file in the dbs/ directory.
Tables:
  - sessions:       Session metadata (character, persona, init status)
  - world_state:    Current key-value state of the world (single source of truth)
  - state_changes:  Audit log of every state change, keyed by (message_id, swipe_index)
  - message_log:    Optional log of all messages for debugging/replay

Swipe handling:
  - When the user swipes (regenerates) a response, the previous state changes
    for that message_id at the previous swipe_index are reverted (applied=FALSE).
  - When the user edits a previous message (redo), all state changes from
    that message_id onwards are reverted.
"""

import sqlite3
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# SQL schema — executed per session database
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
    mode               TEXT DEFAULT 'character',
    multi_character    BOOLEAN DEFAULT FALSE,
    tracked_characters TEXT DEFAULT '',
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
        """Open a connection to the session's database."""
        db_path = self.db_dir / f"{session_id}.db"
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ── Session Management ─────────────────────────────────────

    # Migration: add columns that didn't exist in older schema versions.
    # SQLite doesn't support ALTER TABLE ADD COLUMN IF NOT EXISTS, so we
    # catch the duplicate-column error and ignore it.
    _MIGRATION_COLUMNS = [
        ("mode", "TEXT DEFAULT 'character'"),
        ("multi_character", "BOOLEAN DEFAULT FALSE"),
        ("tracked_characters", "TEXT DEFAULT ''"),
    ]

    def _migrate_session(self, conn: sqlite3.Connection):
        """Add any missing columns to the sessions table."""
        for col_name, col_def in self._MIGRATION_COLUMNS:
            try:
                conn.execute(f"ALTER TABLE sessions ADD COLUMN {col_name} {col_def}")
            except sqlite3.OperationalError:
                pass  # Column already exists — ignore

    def create_session(self, session_id: str) -> bool:
        """Create a new session database with schema."""
        conn = self._get_conn(session_id)
        try:
            conn.executescript(_SESSION_SCHEMA)
            self._migrate_session(conn)
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
        """Initialize session with character card and persona data.

        Called by POST /api/sessions/{id}/init when the extension sends
        character info on first load.
        """
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
                    mode = COALESCE(?, mode),
                    multi_character = COALESCE(?, multi_character),
                    tracked_characters = COALESCE(?, tracked_characters),
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
                    character_data.get("mode"),
                    character_data.get("multi_character"),
                    json.dumps(character_data.get("tracked_characters", []))
                        if isinstance(character_data.get("tracked_characters"), list)
                        else character_data.get("tracked_characters", ""),
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
        """Check if a session has been initialized with character data."""
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
        """Retrieve session metadata."""
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
        """Get the current world state as a dictionary.

        Values that are JSON objects/arrays are parsed; plain strings
        are kept as-is.
        """
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
        """Apply state changes and log them.

        For each changed field:
          1. Record the old value from world_state
          2. Upsert the new value into world_state
          3. Insert an audit row into state_changes

        Returns the number of fields updated.
        """
        conn = self._get_conn(session_id)
        count = 0
        try:
            for field, new_value in changes.items():
                # Get old value
                row = conn.execute(
                    "SELECT value FROM world_state WHERE key = ?", (field,)
                ).fetchone()
                old_value = row["value"] if row else None

                # Serialize new value
                if isinstance(new_value, (dict, list)):
                    value_str = json.dumps(new_value, ensure_ascii=False)
                else:
                    value_str = str(new_value)

                # Upsert world_state
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

                # Log the change
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
        """Revert state changes for a specific swipe.

        When the user swipes to a new generation, the changes from the
        previous swipe_index for this message_id need to be undone.
        """
        conn = self._get_conn(session_id)
        reverted = 0
        try:
            # Get changes at or above this swipe_index for this message
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

                # Restore old value or delete the key
                if change["old_value"] is not None:
                    conn.execute(
                        """
                        UPDATE world_state
                        SET value = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE key = ?
                        """,
                        (change["old_value"], field),
                    )
                else:
                    conn.execute(
                        "DELETE FROM world_state WHERE key = ?", (field,)
                    )

                # Mark all matching changes as not applied
                conn.execute(
                    """
                    UPDATE state_changes SET applied = FALSE
                    WHERE message_id = ? AND field = ? AND applied = TRUE
                    """,
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
        """Revert all state changes from a specific message onwards.

        Used for "redo" — when the user edits a previous message, all
        state changes derived from that message and subsequent messages
        must be undone because the context has fundamentally changed.
        """
        conn = self._get_conn(session_id)
        reverted = 0
        try:
            # Get all applied changes from this message onwards
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
                        """
                        UPDATE world_state
                        SET value = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE key = ?
                        """,
                        (change["old_value"], field),
                    )
                else:
                    conn.execute(
                        "DELETE FROM world_state WHERE key = ?", (field,)
                    )

                conn.execute(
                    """
                    UPDATE state_changes SET applied = FALSE
                    WHERE message_id = ? AND swipe_index = ? AND field = ?
                    """,
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
        """Store a message in the log for debugging/replay."""
        try:
            conn = self._get_conn(session_id)
            try:
                conn.execute(
                    """
                    INSERT INTO message_log
                        (message_id, swipe_index, role, content)
                    VALUES (?, ?, ?, ?)
                    """,
                    (message_id, swipe_index, role, content),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.debug(f"Message log failed (non-critical): {e}")

    # ── Maintenance ────────────────────────────────────────────

    def list_sessions(self) -> List[str]:
        """List all session IDs that have database files."""
        sessions = []
        for db_file in self.db_dir.glob("*.db"):
            sessions.append(db_file.stem)
        return sorted(sessions)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session's database file."""
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
        """Get basic statistics for a session."""
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