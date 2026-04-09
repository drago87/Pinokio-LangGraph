"""
Node: Retrieve Context from Database.

Queries both SQLite (structured facts) and ChromaDB (semantic memories)
to build context that will be injected into the RP LLM prompt.
"""

from agent.state import AgentState, DBContext, CharacterFact
from agent.config import CONTEXT_WINDOW_MESSAGES


def retrieve_context(state: AgentState, db) -> AgentState:
    """
    Graph Node: Retrieve relevant context from the database.
    
    This runs BEFORE the RP LLM call to enrich the prompt with:
    - Current character state (location, inventory, relationships, etc.)
    - Relevant past narrative events (semantic search)
    - Recent conversation history
    
    Args:
        state: Current agent state.
        db: Database instance.
    
    Returns:
        Updated state with db_context populated.
    """
    character_id = state.get("character_id", "default")
    user_message = state.get("user_message", "")
    
    # 1. Get structured facts + semantic memories
    full_context = db.get_full_context(character_id, user_message)
    
    # 2. Get recent conversation history for context window
    recent_msgs = db.get_recent_messages(character_id, limit=CONTEXT_WINDOW_MESSAGES)
    
    # 3. Build conversation context string
    conv_lines = []
    for msg in recent_msgs:
        role = msg["role"].upper()
        content = msg["content"][:500]  # truncate
        conv_lines.append(f"{role}: {content}")
    conversation_context = "\n".join(conv_lines)
    
    # 4. Compose the full DBContext
    db_context = DBContext(
        character_facts=[
            CharacterFact(
                character_id=f.get("character_id", ""),
                category=f.get("category", ""),
                key=f.get("key", ""),
                value=f.get("value", ""),
                source=f.get("source", ""),
                last_updated=f.get("updated_at", "")
            )
            for f in full_context["facts"]
        ],
        summary=full_context["summary"]
    )
    
    return {
        **state,
        "db_context": db_context,
        "debug_info": {
            **(state.get("debug_info") or {}),
            "retrieved_facts_count": len(full_context["facts"]),
            "retrieved_memories_count": len(full_context["memories"]),
            "recent_messages_count": len(recent_msgs),
            "context_summary_preview": (full_context["summary"] or "(none)")[:200]
        }
    }
