"""
FastAPI server exposing an OpenAI-compatible /v1/chat/completions endpoint.

SillyTavern connects to this server as if it were an OpenAI API backend.
Each message flows through the full LangGraph pipeline:

    SillyTavern → POST /v1/chat/completions
                    → retrieve context from DB
                    → call RP LLM (KoboldCPP/Ollama)
                    → call Logic LLM (extract state changes)
                    → update DB
                    → return RP response to SillyTavern
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from agent.config import SERVER_HOST, SERVER_PORT, DEBUG_MODE
import agent.config as agent_cfg
from agent.db import Database
from agent.graph import build_agent_graph, prepare_state


# ============================================================
# Globals
# ============================================================

db: Optional[Database] = None
graph = None
start_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB and graph on startup."""
    global db, graph, start_time
    start_time = time.time()
    db = Database()
    graph = build_agent_graph(db)
    print(f"\n{'='*60}")
    print(f"  LangGraph ST Agent — Ready")
    print(f"  OpenAI-compatible endpoint: http://{SERVER_HOST}:{SERVER_PORT}/v1/chat/completions")
    print(f"  Dashboard: http://localhost:{SERVER_PORT}")
    print(f"{'='*60}\n")
    yield
    if db:
        db.close()


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="LangGraph ST Agent",
    description="SillyTavern-compatible agent middleware with DB-backed context and state extraction",
    version="2.0.0",
    lifespan=lifespan
)

# Allow SillyTavern (which runs on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request/Response Models (OpenAI-compatible)
# ============================================================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "langgraph-agent"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    # Custom fields for our agent
    character_id: Optional[str] = "default"

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = "langgraph-st"
    object: str = "chat.completion"
    created: int = 0
    model: str = "langgraph-agent"
    choices: List[ChatCompletionChoice]
    usage: Optional[dict] = None


# ============================================================
# Endpoints
# ============================================================

@app.get("/")
async def dashboard():
    """Simple dashboard page showing DB stats and status."""
    stats = db.get_stats() if db else {}
    uptime = int(time.time() - start_time) if start_time else 0
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head><title>LangGraph ST Agent</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
               max-width: 800px; margin: 40px auto; padding: 0 20px; background: #1a1a2e; color: #eee; }}
        h1 {{ color: #e94560; }}
        h2 {{ color: #0f3460; background: #16213e; padding: 10px 15px; border-radius: 8px; }}
        .card {{ background: #16213e; padding: 15px; margin: 10px 0; border-radius: 8px; }}
        .stat {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #0f3460; }}
        .stat:last-child {{ border-bottom: none; }}
        code {{ background: #0f3460; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
        .endpoint {{ background: #0f3460; padding: 10px; border-radius: 6px; font-family: monospace; }}
        pre {{ background: #0f3460; padding: 15px; border-radius: 8px; overflow-x: auto; }}
        .tag {{ display: inline-block; background: #e94560; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; }}
    </style></head>
    <body>
        <h1>🔗 LangGraph ST Agent</h1>
        <p>SillyTavern-compatible agent with DB-backed context and state extraction</p>
        
        <h2>📡 Endpoints</h2>
        <div class="card">
            <div class="endpoint">POST http://localhost:{SERVER_PORT}/v1/chat/completions</div>
            <p style="margin-top:8px;color:#aaa;">Point SillyTavern's OpenAI API to this URL</p>
        </div>
        
        <h2>📊 Database Stats</h2>
        <div class="card">
            <div class="stat"><span>Characters</span><span>{stats.get('state_db', {{}}).get('characters', 0)}</span></div>
            <div class="stat"><span>Facts Stored</span><span>{stats.get('state_db', {{}}).get('facts', 0)}</span></div>
            <div class="stat"><span>Messages Logged</span><span>{stats.get('state_db', {{}}).get('messages', 0)}</span></div>
            <div class="stat"><span>Narrative Memories</span><span>{stats.get('memory_db', {{}}).get('total_memories', 0)}</span></div>
            <div class="stat"><span>Uptime</span><span>{uptime}s</span></div>
        </div>
        
        <h2>🔧 Configuration</h2>
        <div class="card">
            <pre>RP LLM: {agent_cfg.RP_LLM_BASE_URL} ({agent_cfg.RP_LLM_BACKEND})
Logic LLM: {agent_cfg.LOGIC_LLM_MODEL} @ {agent_cfg.LOGIC_LLM_BASE_URL}
Server Port: {SERVER_PORT}</pre>
        </div>
        
        <h2>🏗️ Architecture</h2>
        <div class="card">
            <pre>
SillyTavern → <span class="tag">/v1/chat/completions</span> → LangGraph
    ↓
<span class="tag">retrieve_context</span>  → Query SQLite + ChromaDB
    ↓
<span class="tag">call_rp_llm</span>      → Send to RP LLM (KoboldCPP/Ollama)
    ↓
<span class="tag">call_logic_llm</span>   → Extract state changes (Logic LLM)
    ↓
<span class="tag">update_db</span>        → Save changes + narrative to DB
    ↓
Response → SillyTavern</pre>
        </div>
        
        <h2>📚 API Docs</h2>
        <div class="card">
            <a href="/docs" style="color:#e94560;">Swagger API Documentation →</a>
        </div>
    </body></html>
    """
    return HTMLResponse(html)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    
    SillyTavern sends POST requests here. The message flows through
    the full LangGraph pipeline and the RP response is returned.
    """
    if not graph or not db:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Convert messages to LangChain format
        messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
        character_id = request.character_id or "default"
        
        # Prepare state for the graph
        state = prepare_state(messages_dict, character_id)
        
        # Run the graph
        result = await asyncio.to_thread(graph.invoke, state)
        
        # Extract the RP response
        rp_response = result.get("rp_response", "[No response generated]")
        
        # Build OpenAI-compatible response
        response = ChatCompletionResponse(
            id=f"langgraph-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=rp_response),
                finish_reason="stop"
            )],
            usage={
                "prompt_tokens": sum(len(m.content.split()) for m in request.messages),
                "completion_tokens": len(rp_response.split()),
                "total_tokens": sum(len(m.content.split()) for m in request.messages) + len(rp_response.split())
            }
        )
        
        # Include debug info if in debug mode
        if DEBUG_MODE:
            debug_info = result.get("debug_info", {})
            logic_output = result.get("logic_output", {})
            response_dict = response.model_dump()
            response_dict["debug"] = {
                "graph_debug": debug_info,
                "logic_output": logic_output,
                "db_stats": db.get_stats()
            }
            return JSONResponse(content=response_dict)
        
        return response
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


# ============================================================
# DB Management Endpoints (for the setup UI)
# ============================================================

@app.get("/api/stats")
async def get_stats():
    """Get database statistics."""
    if not db:
        raise HTTPException(status_code=503, detail="DB not initialized")
    return db.get_stats()


@app.get("/api/characters")
async def list_characters():
    """List all known characters."""
    if not db:
        raise HTTPException(status_code=503, detail="DB not initialized")
    return {"characters": db.state.list_characters()}


@app.get("/api/characters/{character_id}/facts")
async def get_character_facts(character_id: str, category: Optional[str] = None):
    """Get all facts for a character."""
    if not db:
        raise HTTPException(status_code=503, detail="DB not initialized")
    return {"facts": db.get_facts(character_id, category)}


@app.get("/api/characters/{character_id}/memories")
async def search_character_memories(character_id: str, q: str = ""):
    """Search narrative memories for a character."""
    if not db:
        raise HTTPException(status_code=503, detail="DB not initialized")
    if q:
        return {"memories": db.search_memories(character_id, q)}
    return {"memories": db.memory.get_recent(character_id)}


@app.get("/api/config")
async def get_config():
    """Get current configuration (without API keys)."""
    from agent import config as agent_cfg
    return {
        "rp_llm_base_url": agent_cfg.RP_LLM_BASE_URL,
        "rp_llm_model": agent_cfg.RP_LLM_MODEL,
        "rp_llm_backend": agent_cfg.RP_LLM_BACKEND,
        "logic_llm_base_url": agent_cfg.LOGIC_LLM_BASE_URL,
        "logic_llm_model": agent_cfg.LOGIC_LLM_MODEL,
        "logic_llm_backend": agent_cfg.LOGIC_LLM_BACKEND,
        "server_port": agent_cfg.SERVER_PORT,
        "context_window_messages": agent_cfg.CONTEXT_WINDOW_MESSAGES,
        "db_retrieval_count": agent_cfg.DB_RETRIEVAL_COUNT,
        "debug_mode": agent_cfg.DEBUG_MODE,
    }


@app.post("/api/config")
async def update_config(request: Request):
    """Update configuration values."""
    from agent.config import save_config
    data = await request.json()
    
    # Map frontend keys to env var keys
    env_map = {
        "rp_llm_base_url": "RP_LLM_BASE_URL",
        "rp_llm_model": "RP_LLM_MODEL",
        "rp_llm_backend": "RP_LLM_BACKEND",
        "rp_llm_api_key": "RP_LLM_API_KEY",
        "logic_llm_base_url": "LOGIC_LLM_BASE_URL",
        "logic_llm_model": "LOGIC_LLM_MODEL",
        "logic_llm_backend": "LOGIC_LLM_BACKEND",
        "logic_llm_api_key": "LOGIC_LLM_API_KEY",
        "server_port": "SERVER_PORT",
        "context_window_messages": "CONTEXT_WINDOW_MESSAGES",
        "db_retrieval_count": "DB_RETRIEVAL_COUNT",
        "debug_mode": "DEBUG_MODE",
    }
    
    config_updates = {}
    for frontend_key, env_key in env_map.items():
        if frontend_key in data:
            config_updates[env_key] = str(data[frontend_key])
    
    if config_updates:
        save_config(config_updates)
        return {"status": "updated", "changes": config_updates}
    
    return {"status": "no changes"}


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  LangGraph ST Agent for SillyTavern")
    print("=" * 60)
    print(f"  Starting server on http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"  SillyTavern endpoint: http://localhost:{SERVER_PORT}/v1/chat/completions")
    print("=" * 60)
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
