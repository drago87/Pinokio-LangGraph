"""
ChromaDB for semantic memory storage and retrieval.

Stores narrative text as embeddings, allowing the agent to
semantically search past events. For example:
  "Has Elena ever mentioned dragons?" → finds relevant past passages.
"""

from pathlib import Path
from typing import Optional
from agent.config import CHROMA_DIR, DB_RETRIEVAL_COUNT


class MemoryDB:
    """
    ChromaDB-backed semantic memory for narrative text.
    
    Unlike SQLite (which stores structured facts), this stores
    raw narrative passages and allows semantic similarity search.
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or CHROMA_DIR
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self._init_client()

    def _init_client(self):
        """Initialize ChromaDB with a persistent store."""
        import chromadb
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Default embedding function uses ChromaDB's built-in model
        # (no external API calls needed)
        self.embedding_fn = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
        
        # Collection for narrative memory
        self.collection = self.client.get_or_create_collection(
            name="narrative_memory",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn
        )

    def add_memory(self, character_id: str, text: str, source: str = "narrative",
                   metadata: Optional[dict] = None):
        """
        Store a narrative passage as a memory entry.
        
        Args:
            character_id: The character this memory relates to.
            text: The narrative text to store (e.g., a RP response).
            source: Where this came from ("narrative", "user_input", etc.)
            metadata: Additional metadata (role, turn number, etc.)
        """
        import time
        entry_id = f"{character_id}_{int(time.time() * 1000)}"
        
        meta = metadata or {}
        meta["character_id"] = character_id
        meta["source"] = source
        meta["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        
        # ChromaDB has a 1000 char limit by default, but we store full text
        self.collection.upsert(
            ids=[entry_id],
            documents=[text],
            metadatas=[meta]
        )

    def search(self, character_id: str, query: str, n_results: Optional[int] = None) -> list[dict]:
        """
        Semantically search narrative memory.
        
        Args:
            character_id: Filter results to this character.
            query: The search query (natural language).
            n_results: Max results to return.
        
        Returns:
            List of matching memory entries with text and metadata.
        """
        n = n_results or DB_RETRIEVAL_COUNT
        
        results = self.collection.query(
            query_texts=[query],
            where={"character_id": character_id},
            n_results=min(n, self.collection.count()) if self.collection.count() > 0 else n
        )
        
        memories = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                entry = {
                    "id": doc_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0
                }
                memories.append(entry)
        
        return memories

    def get_recent(self, character_id: str, limit: int = 5) -> list[dict]:
        """Get the most recently stored memories for a character."""
        all_memories = self.collection.get(
            where={"character_id": character_id},
            include=["documents", "metadatas"]
        )
        
        if not all_memories or not all_memories["ids"]:
            return []
        
        # Sort by timestamp descending
        entries = []
        for i, doc_id in enumerate(all_memories["ids"]):
            meta = all_memories["metadatas"][i] if all_memories["metadatas"] else {}
            entries.append({
                "id": doc_id,
                "text": all_memories["documents"][i] if all_memories["documents"] else "",
                "metadata": meta
            })
        
        entries.sort(key=lambda x: x["metadata"].get("timestamp", ""), reverse=True)
        return entries[:limit]

    def get_stats(self) -> dict:
        """Get collection statistics."""
        return {
            "total_memories": self.collection.count(),
            "persist_dir": self.persist_dir
        }
