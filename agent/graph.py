"""
The main LangGraph state machine.

Defines the complete graph that orchestrates the SillyTavern integration:

    START → retrieve_context → call_rp_llm → call_logic_llm → update_db → END
            (query DB)         (creative LLM)    (extract state)      (save changes)

This graph is exposed as a FastAPI endpoint that SillyTavern can call
using the standard OpenAI /v1/chat/completions format.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agent.state import AgentState
from agent.nodes.retrieve import retrieve_context
from agent.nodes.rp_llm import call_rp_llm
from agent.nodes.logic_llm import call_logic_llm
from agent.nodes.db_update import update_database


def build_agent_graph(db):
    """
    Build and compile the LangGraph agent.
    
    The graph processes each SillyTavern message through 5 stages:
    
    1. RETRIEVE CONTEXT: Query SQLite + ChromaDB for character state
       and relevant past events
    2. CALL RP LLM: Send enriched prompt (context + user message) to 
       the creative roleplay model
    3. CALL LOGIC LLM: Send the RP response to a smaller model that 
       extracts structured state changes as JSON
    4. UPDATE DB: Write state changes to SQLite, store narrative in ChromaDB
    5. Return the RP response to SillyTavern
    
    Args:
        db: The unified Database instance (SQLite + ChromaDB).
    
    Returns:
        A compiled LangGraph runnable.
    """
    
    # ----- Node Functions -----
    
    def node_retrieve_context(state: AgentState) -> dict:
        """Query the database for relevant context."""
        return retrieve_context(state, db)
    
    def node_call_rp_llm(state: AgentState) -> dict:
        """Send enriched prompt to the creative RP model."""
        return call_rp_llm(state)
    
    def node_call_logic_llm(state: AgentState) -> dict:
        """Extract state changes from the RP response."""
        return call_logic_llm(state)
    
    def node_update_database(state: AgentState) -> dict:
        """Save state changes and narrative to the database."""
        return update_database(state, db)
    
    # ----- Build the Graph -----
    
    graph = StateGraph(AgentState)
    
    # Add all nodes
    graph.add_node("retrieve_context", node_retrieve_context)
    graph.add_node("call_rp_llm", node_call_rp_llm)
    graph.add_node("call_logic_llm", node_call_logic_llm)
    graph.add_node("update_db", node_update_database)
    
    # Define the linear flow
    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "call_rp_llm")
    graph.add_edge("call_rp_llm", "call_logic_llm")
    graph.add_edge("call_logic_llm", "update_db")
    graph.add_edge("update_db", END)
    
    return graph.compile()


def prepare_state(messages: list, character_id: str = "default") -> dict:
    """
    Convert incoming SillyTavern/OpenAI messages to AgentState format.
    
    Args:
        messages: List of OpenAI-format messages: [{"role": "user/assistant/system", "content": "..."}]
        character_id: Character identifier for DB lookups.
    
    Returns:
        AgentState dict ready to be passed to the graph.
    """
    # Convert to LangChain message objects
    lc_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
    
    # Get the latest user message
    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break
    
    return {
        "messages": lc_messages,
        "character_id": character_id,
        "user_message": user_message,
        "rp_response": "",
        "db_context": None,
        "enriched_prompt": "",
        "logic_output": None,
        "errors": [],
        "debug_info": {}
    }
