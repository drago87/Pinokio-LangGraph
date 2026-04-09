"""
Unified database interface.

Provides a single interface to both SQLite (structured facts) and ChromaDB (semantic memory).
"""

from agent.db.sqlite_store import StateDB
from agent.db.chroma_store import MemoryDB


class Database:
    """
    Unified database combining structured state storage and semantic memory.
    
    Usage:
        db = Database()
        db.ensure_character("elena", "Elena Stormweaver")
        db.upsert_fact("elena", "location", "current", "The Sleeping Dragon Inn")
        db.add_narrative("elena", "Elena approached the bar cautiously...")
        context = db.get_full_context("elena", "What does Elena know about the dragon?")
    """

    def __init__(self):
        self.state = StateDB()      # SQLite - structured facts
        self.memory = MemoryDB()    # ChromaDB - semantic narrative memory

    # ============================================================
    # Character Management
    # ============================================================

    def ensure_character(self, character_id: str, name: str = ""):
        """Register a character in the database."""
        self.state.ensure_character(character_id, name)

    def get_full_context(self, character_id: str, query: str = "") -> dict:
        """
        Get all relevant context for a character, combining:
        1. Structured facts from SQLite
        2. Semantic memories from ChromaDB (if query provided)
        
        Returns a dict with 'facts' and 'memories' lists.
        """
        facts = self.state.get_facts(character_id)
        
        memories = []
        if query:
            memories = self.memory.search(character_id, query)
        
        # Build a summary string for prompt injection
        summary_parts = []
        
        # Structured facts summary
        if facts:
            lines = ["[CHARACTER STATE]"]
            current_cat = None
            for f in facts:
                if f["category"] != current_cat:
                    current_cat = f["category"]
                    lines.append(f"  {current_cat.upper()}:")
                lines.append(f"    - {f['key']}: {f['value']}")
            summary_parts.append("\n".join(lines))
        
        # Semantic memory summary
        if memories:
            mem_lines = ["[RELEVANT PAST EVENTS]"]
            for m in memories[:5]:
                text = m["text"][:300]  # truncate long passages
                mem_lines.append(f"  - {text}")
            summary_parts.append("\n".join(mem_lines))
        
        return {
            "facts": facts,
            "memories": memories,
            "summary": "\n\n".join(summary_parts)
        }

    # ============================================================
    # Fact shortcuts (delegates to StateDB)
    # ============================================================

    def upsert_fact(self, character_id: str, category: str, key: str, value: str,
                    source: str = "logic_llm", confidence: float = 1.0):
        self.state.upsert_fact(character_id, category, key, value, source, confidence)

    def remove_fact(self, character_id: str, category: str, key: str):
        self.state.remove_fact(character_id, category, key)

    def get_facts(self, character_id: str, category: str = None) -> list:
        return self.state.get_facts(character_id, category)

    # ============================================================
    # Memory shortcuts (delegates to MemoryDB)
    # ============================================================

    def add_narrative(self, character_id: str, text: str, source: str = "narrative"):
        """Store a narrative passage in semantic memory."""
        self.memory.add_memory(character_id, text, source)

    def search_memories(self, character_id: str, query: str, n: int = 5) -> list:
        """Search semantic memory for relevant past events."""
        return self.memory.search(character_id, query, n)

    # ============================================================
    # Bulk Operations (from Logic LLM)
    # ============================================================

    def apply_state_changes(self, changes: list[dict]) -> dict:
        """Apply a batch of state changes from the Logic LLM."""
        return self.state.apply_state_changes(changes)

    # ============================================================
    # Conversation History
    # ============================================================

    def save_message(self, character_id: str, role: str, content: str):
        self.state.save_message(character_id, role, content)

    def get_recent_messages(self, character_id: str, limit: int = 20) -> list:
        return self.state.get_recent_messages(character_id, limit)

    # ============================================================
    # Stats
    # ============================================================

    def get_stats(self) -> dict:
        return {
            "state_db": self.state.get_stats(),
            "memory_db": self.memory.get_stats()
        }

    def close(self):
        self.state.close()
