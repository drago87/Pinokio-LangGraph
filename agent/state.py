"""
State definitions for the LangGraph agent.

Defines all the data structures that flow through the graph nodes.
"""

from typing import Annotated, TypedDict, Optional
from langgraph.graph.message import add_messages


class CharacterFact(TypedDict):
    """A single fact about a character stored in the DB."""
    character_id: str
    category: str        # "location", "knowledge", "relationship", "inventory", "status", "personality", "history"
    key: str             # short identifier, e.g. "current_location", "knows_about_dragon"
    value: str           # the actual value, e.g. "tavern", "heard rumor about dragon in mountains"
    source: str          # where this came from: "narrative", "user_input", "logic_llm"
    last_updated: str    # timestamp


class DBContext(TypedDict):
    """Context retrieved from the database before calling the RP LLM."""
    character_facts: list[CharacterFact]
    summary: str         # human-readable summary of relevant context


class StateChange(TypedDict):
    """A single state change extracted by the Logic LLM."""
    character_id: str
    category: str        # "location", "knowledge", "relationship", "inventory", "status"
    action: str          # "set", "add", "remove", "update"
    key: str
    value: str
    reasoning: str       # why this change was made
    confidence: float    # 0.0 - 1.0


class LogicLLMOutput(TypedDict):
    """The full structured output from the Logic LLM."""
    changes: list[StateChange]
    narrative_summary: str   # brief summary of what happened in the RP response


class AgentState(TypedDict):
    """
    The master state that flows through the LangGraph pipeline.
    
    Flow:
        START → retrieve_context → call_rp_llm → call_logic_llm → update_db → END
    """
    # Core inputs/outputs
    messages: Annotated[list, add_messages]    # full conversation history
    character_id: str                          # which character this is about
    user_message: str                          # latest user message (from ST)
    rp_response: str                           # the roleplay response (sent back to ST)
    
    # Intermediate data
    db_context: Optional[DBContext]            # context pulled from DB
    enriched_prompt: str                       # prompt sent to RP LLM (context + message)
    logic_output: Optional[LogicLLMOutput]     # structured state changes
    
    # Metadata
    errors: list[str]                          # any errors that occurred
    debug_info: dict                           # debug info when DEBUG_MODE is on
