"""
SQLite database for structured character state storage.

Stores concrete facts like:
  - character location: "Elena is at the tavern"
  - inventory: "Elena has a silver dagger"
  - relationships: "Elena trusts Marcus"
  - status: "Elena is injured"

Each fact has a character_id, category, key/value, source, and timestamp.
"""

import sqlite3
import json
import time
from pathlib import Path
from typing import Optional
from agent.config import SQLITE_PATH, DB_DIR


class StateDB:
    """SQLite-backed structured state database for character facts."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or SQLITE_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        """Create the database schema if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS characters (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                first_seen TEXT DEFAULT (datetime('now')),
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id TEXT NOT NULL,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                source TEXT DEFAULT 'logic_llm',
                confidence REAL DEFAULT 1.0,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (character_id) REFERENCES characters(id),
                UNIQUE(character_id, category, key)
            );

            CREATE INDEX IF NOT EXISTS idx_facts_character ON facts(character_id);
            CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(character_id, category);
            CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(character_id, key);

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (character_id) REFERENCES characters(id)
            );

            CREATE INDEX IF NOT EXISTS idx_convos_character ON conversations(character_id);
        """)
        self.conn.commit()

    # ============================================================
    # Character Management
    # ============================================================

    def ensure_character(self, character_id: str, name: str = "", metadata: dict = None):
        """Create a character if it doesn't exist, or update metadata."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO characters (id, name, metadata) VALUES (?, ?, ?)",
            (character_id, name or character_id, json.dumps(metadata or {}))
        )
        if metadata:
            cursor.execute(
                "UPDATE characters SET metadata = ? WHERE id = ?",
                (json.dumps(metadata), character_id)
            )
        self.conn.commit()

    def get_character(self, character_id: str) -> Optional[dict]:
        """Get a character's basic info."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM characters WHERE id = ?", (character_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def list_characters(self) -> list[dict]:
        """List all known characters."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM characters ORDER BY first_seen")
        return [dict(row) for row in cursor.fetchall()]

    # ============================================================
    # Fact Management (the core of the DB)
    # ============================================================

    def upsert_fact(self, character_id: str, category: str, key: str,
                    value: str, source: str = "logic_llm", confidence: float = 1.0):
        """Insert or update a character fact. Uses UPSERT so latest value wins."""
        self.ensure_character(character_id)
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO facts (character_id, category, key, value, source, confidence, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(character_id, category, key)
            DO UPDATE SET value = excluded.value,
                          source = excluded.source,
                          confidence = excluded.confidence,
                          updated_at = datetime('now')
        """, (character_id, category, key, value, source, confidence))
        self.conn.commit()

    def remove_fact(self, character_id: str, category: str, key: str):
        """Delete a specific fact."""
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM facts WHERE character_id = ? AND category = ? AND key = ?",
            (character_id, category, key)
        )
        self.conn.commit()

    def get_facts(self, character_id: str, category: Optional[str] = None) -> list[dict]:
        """Get all facts for a character, optionally filtered by category."""
        cursor = self.conn.cursor()
        if category:
            cursor.execute(
                "SELECT * FROM facts WHERE character_id = ? AND category = ? ORDER BY updated_at DESC",
                (character_id, category)
            )
        else:
            cursor.execute(
                "SELECT * FROM facts WHERE character_id = ? ORDER BY category, updated_at DESC",
                (character_id,)
            )
        return [dict(row) for row in cursor.fetchall()]

    def search_facts(self, character_id: str, query: str, limit: int = 10) -> list[dict]:
        """Full-text search across all facts for a character."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM facts
            WHERE character_id = ?
              AND (key LIKE ? OR value LIKE ?)
            ORDER BY updated_at DESC
            LIMIT ?
        """, (character_id, f"%{query}%", f"%{query}%", limit))
        return [dict(row) for row in cursor.fetchall()]

    def get_context_for_prompt(self, character_id: str, categories: Optional[list[str]] = None) -> str:
        """Build a human-readable context string from all character facts for injection into prompts."""
        facts = self.get_facts(character_id)
        if categories:
            facts = [f for f in facts if f["category"] in categories]

        if not facts:
            return ""

        lines = []
        current_category = None
        for fact in facts:
            if fact["category"] != current_category:
                current_category = fact["category"]
                lines.append(f"\n[{current_category.upper()}]")
            lines.append(f"  {fact['key']}: {fact['value']}")

        return "\n".join(lines)

    # ============================================================
    # Bulk Operations (for Logic LLM output)
    # ============================================================

    def apply_state_changes(self, changes: list[dict]):
        """
        Apply a batch of state changes from the Logic LLM output.
        
        Each change should have: character_id, category, action, key, value, confidence
        Actions: "set" (upsert), "add" (append to existing), "remove" (delete)
        """
        applied = []
        errors = []

        for change in changes:
            try:
                action = change.get("action", "set")
                character_id = change.get("character_id", "unknown")
                category = change.get("category", "general")
                key = change.get("key", "")
                value = change.get("value", "")
                confidence = float(change.get("confidence", 0.8))
                source = change.get("reasoning", "logic_llm")[:500]  # truncate

                if action == "remove":
                    self.remove_fact(character_id, category, key)
                    applied.append({"action": "removed", "character": character_id, "key": key})

                elif action == "set":
                    self.upsert_fact(character_id, category, key, value, source, confidence)
                    applied.append({"action": "set", "character": character_id, "key": key, "value": value})

                elif action == "add":
                    # Append to existing value or create new
                    existing = self.get_facts(character_id, category)
                    existing_fact = next((f for f in existing if f["key"] == key), None)
                    if existing_fact:
                        new_value = f"{existing_fact['value']}; {value}"
                    else:
                        new_value = value
                    self.upsert_fact(character_id, category, key, new_value, source, confidence)
                    applied.append({"action": "added", "character": character_id, "key": key})

                elif action == "update":
                    self.upsert_fact(character_id, category, key, value, source, confidence)
                    applied.append({"action": "updated", "character": character_id, "key": key, "value": value})

            except Exception as e:
                errors.append(f"Error applying change {change}: {str(e)}")

        return {"applied": applied, "errors": errors}

    # ============================================================
    # Conversation History
    # ============================================================

    def save_message(self, character_id: str, role: str, content: str):
        """Save a message to the conversation log."""
        self.ensure_character(character_id)
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (character_id, role, content) VALUES (?, ?, ?)",
            (character_id, role, content)
        )
        self.conn.commit()

    def get_recent_messages(self, character_id: str, limit: int = 20) -> list[dict]:
        """Get recent conversation messages for context."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM conversations WHERE character_id = ? ORDER BY timestamp DESC LIMIT ?",
            (character_id, limit)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in reversed(rows)]  # oldest first

    # ============================================================
    # Maintenance
    # ============================================================

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def get_stats(self) -> dict:
        """Get database statistics."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM characters")
        characters = cursor.fetchone()["count"]
        cursor.execute("SELECT COUNT(*) as count FROM facts")
        facts = cursor.fetchone()["count"]
        cursor.execute("SELECT COUNT(*) as count FROM conversations")
        messages = cursor.fetchone()["count"]
        return {"characters": characters, "facts": facts, "messages": messages}
