"""
Node: Update the database with extracted state changes.

Takes the Logic LLM output and writes the changes to both
SQLite (structured facts) and ChromaDB (narrative memory).
"""

from agent.config import DEBUG_MODE


def update_database(state: dict, db) -> dict:
    """
    Graph Node: Apply state changes to the database.
    
    This runs AFTER the Logic LLM has extracted state changes.
    It writes:
    1. Structured facts to SQLite
    2. The RP narrative to ChromaDB (for future semantic search)
    
    Args:
        state: Current agent state (must have logic_output, character_id, rp_response).
        db: Database instance.
    
    Returns:
        Updated state with errors list and debug info.
    """
    logic_output = state.get("logic_output", {})
    character_id = state.get("character_id", "unknown")
    rp_response = state.get("rp_response", "")
    user_message = state.get("user_message", "")
    debug = state.get("debug_info", {})
    errors = list(state.get("errors", []))
    
    # 1. Apply structured state changes to SQLite
    changes = logic_output.get("changes", [])
    if changes:
        result = db.apply_state_changes(changes)
        errors.extend(result.get("errors", []))
        debug["db_applied_changes"] = result.get("applied", [])
        debug["db_apply_errors"] = result.get("errors", [])
    
    # 2. Save the user message and RP response as narrative memory
    if user_message:
        db.save_message(character_id, "user", user_message)
        db.add_narrative(character_id, user_message, source="user_input")
    
    if rp_response and not rp_response.startswith("[ERROR"):
        db.save_message(character_id, "assistant", rp_response)
        db.add_narrative(character_id, rp_response, source="narrative")
    
    return {
        **state,
        "errors": errors,
        "debug_info": debug
    }
