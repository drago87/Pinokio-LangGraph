#!/usr/bin/env python3
"""
server.py — FastAPI application for Agent-StateSync.

Acts as the middle layer between SillyTavern (via extension) and the
RP/Instruct LLMs. Handles:

Endpoints:
  POST /v1/chat/completions  — Main pipeline (receives from ST extension)
  POST /api/sessions          — Create a new session
  POST /api/sessions/{id}/init — Initialize session with character data
  POST /api/config            — Receive config from ST extension
  POST /api/stop              — Stop current generation (called by ST extension)
  GET  /health                — Health check
  GET  /api/sessions          — List sessions
  GET  /api/sessions/{id}     — Get session info/stats
  GET  /api/debug/state       — Get debug stepping state
  POST /api/debug/continue    — Resume paused debug pipeline
  POST /api/debug/toggle      — Enable/disable debug stepping

Main Pipeline (POST /v1/chat/completions):
  1. Parse [SYSTEM_META] from messages[0]
  2. Handle swipe/redo (revert previous DB changes)
  3. Translate world state (JSON → natural language)
  4. Inject world state as hidden system message
  5. Format messages with selected template
  6. Optional: thinking passes (RP LLM internal planning)
  7. Stream RP LLM response via SSE → SillyTavern
  8. Optional: refinement passes (post-generation review)
  9. After stream: background task runs Instruct LLM extraction
 10. Update session DB with new world state
"""

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse

from config import ConfigManager
from parser import strip_meta_from_messages, SystemMeta
from db import Database
from llm_client import LLMClientManager
from templates import format_messages, inject_world_state_context
from agent import (
    ExtractionPipeline,
    TranslationPipeline,
    run_thinking_passes,
    run_refinement_passes,
)

# ── Logging ───────────────────────────────────────────────────
# Base level is INFO. If debug_mode is enabled in config.ini (or
# pushed by the extension), the lifespan handler flips everything
# to DEBUG so you see full message payloads, parsed meta, DB ops, etc.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent-statesync")

# Suppress httpx/httpcore noise at import time, BEFORE any probe runs.
# _probe_llm() hits /v1/models every 3 seconds — the DEBUG-level wire
# logs from httpcore.connection / httpcore.http11 / httpx would flood
# the terminal.  _apply_debug_level() re-applies these on toggle.
for _ln in ("httpx", "httpcore", "hpack", "anyio"):
    logging.getLogger(_ln).setLevel(logging.WARNING)

# ── Globals ───────────────────────────────────────────────────

config_manager: Optional[ConfigManager] = None
database: Optional[Database] = None
client_manager: Optional[LLMClientManager] = None

# Pending background tasks (for tracking + cancellation)
_background_tasks: set = set()

# Abort event: set by /api/stop to signal active streams to halt
_abort_event: asyncio.Event = asyncio.Event()

# ── Pipeline Tracker ──────────────────────────────────────
# Stores recent pipeline runs in memory for the dashboard.
# Each run records every step of the chat completion pipeline.

class PipelineTracker:
    def __init__(self, max_runs: int = 50):
        self.runs: deque = deque(maxlen=max_runs)
        self._current: Optional[Dict[str, Any]] = None

    def start_run(self, session_id: str, message_id: int, msg_type: str):
        self._current = {
            "id": f"run_{int(time.time() * 1000)}",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "session_id": session_id[:8] if session_id else "?",
            "message_id": message_id,
            "type": msg_type,
            "start_time": time.time(),
            "steps": [],
            "status": "running",
            "duration_ms": None,
            "response_preview": "",
        }

    def step(self, name: str, label: str, data: dict = None, status: str = "done", preview: str = None, changes: dict = None):
        if self._current is None:
            return
        entry = {
            "name": name,
            "label": label,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "data": data or {},
            "status": status,
            "preview": preview,
            "changes": changes,
        }
        self._current["steps"].append(entry)

    def finish(self, status: str = "completed", response_preview: str = ""):
        if self._current is None:
            return
        self._current["status"] = status
        self._current["duration_ms"] = int((time.time() - self._current["start_time"]) * 1000)
        self._current["response_preview"] = (response_preview or "")[:500]
        self.runs.appendleft(self._current)
        self._current = None

    def to_dict(self) -> dict:
        result = [dict(r) for r in self.runs]
        if self._current:
            running = dict(self._current)
            running["duration_ms"] = int((time.time() - running["start_time"]) * 1000)
            result.insert(0, running)
        return {"runs": result}


pipeline_tracker = PipelineTracker()


# ── Debug Gate ──────────────────────────────────────────────
# When debug stepping is enabled, the pipeline pauses at each
# checkpoint and waits for the user to click "Continue" in the
# dashboard.  Health checks, pings, and other non-pipeline
# endpoints are NOT affected because asyncio.Event.wait() only
# blocks the specific coroutine — the event loop keeps running.

class DebugGate:
    """Manages debug stepping for the pipeline.

    When enabled, the pipeline pauses at each checkpoint and waits
    for the user to click "Continue" in the dashboard.  Health checks,
    pings, and other non-pipeline endpoints are NOT affected because
    asyncio.Event.wait() only blocks the calling coroutine — the
    event loop continues to serve other requests.

    Only one pipeline can be debugged at a time.  If a second pipeline
    starts while one is paused, the second skips all debug checkpoints.
    """

    def __init__(self):
        self.enabled: bool = False
        self._event: asyncio.Event = asyncio.Event()
        self._event.set()  # Start unblocked
        self._current: Optional[Dict[str, Any]] = None
        self._completed: List[Dict[str, Any]] = []
        self._active: bool = False  # True while a pipeline is being debugged
        self._lock: asyncio.Lock = asyncio.Lock()

    async def wait(self, incoming: str, outgoing: str, data: dict = None):
        """Pause the pipeline at a debug checkpoint.

        Records the incoming/outgoing labels and waits for the user
        to click "Continue" in the dashboard.  If debug stepping is
        disabled, or another pipeline is already being debugged,
        this returns immediately.
        """
        if not self.enabled:
            return

        async with self._lock:
            if self._active:
                return  # Another pipeline already holds the debug gate
            self._active = True

        step = {
            "incoming": incoming,
            "outgoing": outgoing,
            "data": data or {},
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }

        self._current = step
        self._event.clear()  # Block

        logger.info(
            f"[DEBUG GATE] PAUSED — "
            f"Incoming: {incoming}  |  Outgoing: {outgoing}"
        )

        try:
            await self._event.wait()
        finally:
            self._completed.append(step)
            self._current = None
            self._active = False

        logger.info(f"[DEBUG GATE] RESUMED — proceeding with: {outgoing}")

    def continue_pipeline(self):
        """Signal the waiting pipeline to proceed."""
        if self._current is not None:
            logger.info("[DEBUG GATE] User clicked Continue")
        self._event.set()

    def reset(self):
        """Clear all debug state (completed steps, current step)."""
        self._completed.clear()
        self._current = None
        self._active = False
        self._event.set()

    def toggle(self, enabled: bool):
        """Enable or disable debug stepping."""
        self.enabled = enabled
        if not enabled:
            self.reset()
        state = "ON" if enabled else "OFF"
        logger.info(f"[DEBUG GATE] Debug stepping: {state}")

    def to_dict(self) -> dict:
        """Return debug state for the dashboard."""
        return {
            "enabled": self.enabled,
            "active": self._active,
            "waiting": self._current is not None,
            "current": self._current,
            "completed": list(self._completed),
        }


debug_gate = DebugGate()


def _apply_debug_level(debug: bool):
    """Flip the root logger between INFO and DEBUG.

    When debug_mode is ON, the agent-statesync logger gets DEBUG so you
    see full message payloads, parsed meta, DB ops, extraction results, etc.
    However, httpx/httpcore are kept at WARNING even in debug mode —
    their per-request wire logs are pure noise (especially since _probe_llm
    runs every few seconds for the status indicators).
    """
    target = logging.DEBUG if debug else logging.INFO
    logging.getLogger().setLevel(target)
    # Our app logger + langgraph internals get full debug
    for name in ("agent-statesync", "langgraph", "asyncio"):
        logging.getLogger(name).setLevel(target)
    # Keep httpx/httpcore at WARNING to avoid probe spam
    for name in ("httpx", "httpcore", "hpack", "anyio"):
        logging.getLogger(name).setLevel(logging.WARNING)
    level_name = "DEBUG" if debug else "INFO"
    logger.info(f"Log level set to {level_name}")


# ── Lifespan ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup on startup/shutdown."""
    global config_manager, database, client_manager

    config_manager = ConfigManager()
    database = Database(str(config_manager.db_dir_path))
    client_manager = LLMClientManager()

    # Apply debug mode from config.ini (extension can change it later)
    _apply_debug_level(config_manager.config.debug_mode)

    # Apply debug stepping from config.ini
    debug_stepping = getattr(config_manager.config, 'debug_stepping', False)
    if debug_stepping:
        debug_gate.toggle(True)

    logger.info("=" * 56)
    logger.info("  Agent-StateSync starting...")
    logger.info(f"  Port:            {config_manager.port}")
    logger.info(f"  RP LLM:          {config_manager.config.rp_llm_url}")
    logger.info(f"  RP disabled:     {config_manager.config.rp_llm_disabled}")
    logger.info(f"  Instruct LLM:     {config_manager.config.instruct_llm_url}")
    logger.info(f"  Instruct dis.:    {config_manager.config.instruct_llm_disabled}")
    logger.info(f"  Debug mode:      {config_manager.config.debug_mode}")
    logger.info(f"  Debug stepping:  {debug_stepping}")
    logger.info(f"  Dry run:         {config_manager.config.dry_run}")
    logger.info(f"  DB directory:    {config_manager.db_dir_path}")
    logger.info(f"  Prompts dir:     {config_manager.prompts_dir_path}")
    logger.info("=" * 56)

    yield

    logger.info("Agent-StateSync shutting down.")


# ── App ───────────────────────────────────────────────────────

app = FastAPI(
    title="Agent-StateSync",
    description="FastAPI + LangGraph agent for SillyTavern RP state management",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    cfg = config_manager.config if config_manager else None
    return {
        "status": "ok",
        "version": "2.0.0",
        "sessions": len(database.list_sessions()) if database else 0,
        "debug_mode": cfg.debug_mode if cfg else False,
        "debug_stepping": debug_gate.enabled,
        "dry_run": cfg.dry_run if cfg else False,
        "rp_llm_disabled": cfg.rp_llm_disabled if cfg else False,
        "instruct_llm_disabled": cfg.instruct_llm_disabled if cfg else False,
    }


# ── Dashboard ─────────────────────────────────────────────

@app.get("/")
async def dashboard():
    """Serve the dashboard HTML page."""
    html_path = Path(__file__).parent / "static" / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Dashboard not found</h1><p>static/dashboard.html is missing.</p>", status_code=404)


@app.get("/api/dashboard/pipeline")
async def dashboard_pipeline():
    """Return recent pipeline runs for the dashboard."""
    return pipeline_tracker.to_dict()


@app.get("/api/dashboard/sessions")
async def dashboard_sessions():
    """Return all sessions with world state summaries for the dashboard."""
    if not database:
        return {"sessions": []}

    sessions = []
    for sid in database.list_sessions():
        meta = database.get_session(sid)
        stats = database.get_session_stats(sid) or {}
        ws = database.get_world_state(sid)
        sessions.append({
            "session_id": sid,
            "character_name": meta.get("character_name", "") if meta else "",
            "message_count": stats.get("logged_messages", 0),
            "world_state": ws,
            "initialized": database.is_initialized(sid),
        })
    return {"sessions": sessions}


@app.get("/api/dashboard/config")
async def dashboard_config():
    """Return current config values for the dashboard."""
    if not config_manager:
        return {}
    cfg = config_manager.config
    return {
        "port": cfg.port,
        "debug_mode": cfg.debug_mode,
        "debug_stepping": debug_gate.enabled,
        "dry_run": cfg.dry_run,
        "rp_llm_url": cfg.rp_llm_url,
        "rp_llm_backend": cfg.rp_llm_backend,
        "rp_llm_model": cfg.rp_llm_model or "",
        "rp_llm_disabled": cfg.rp_llm_disabled,
        "instruct_llm_url": cfg.instruct_llm_url,
        "instruct_llm_backend": cfg.instruct_llm_backend,
        "instruct_llm_model": cfg.instruct_llm_model or "",
        "instruct_llm_disabled": cfg.instruct_llm_disabled,
        "thinking_steps": cfg.thinking_steps,
        "refinement_steps": cfg.refinement_steps,
        "history_count": cfg.history_count,
        "rp_template": cfg.rp_template,
        "db_dir": cfg.db_dir,
        "prompts_dir": cfg.prompts_dir,
    }


# ── Extension Ping & Dashboard Status ────────────────────────

# Timestamp of the last ping from the ST extension.
# The dashboard polls /api/dashboard/status to show a live indicator.
_last_extension_ping: float = 0.0


@app.post("/api/ping")
async def extension_ping():
    """Heartbeat ping from the SillyTavern extension.

    The extension sends this every 30 seconds as part of its health
    check routine.  We record the timestamp so the dashboard can show
    whether the extension is currently connected.
    """
    global _last_extension_ping
    _last_extension_ping = time.time()
    return {"status": "ok"}


@app.get("/api/dashboard/status")
async def dashboard_status():
    """Return live status indicators for the dashboard header.

    * extension_connected — True if the ST extension pinged recently
      (within the last 45 seconds).
    * debug_mode — current debug flag from config.
    * debug_stepping — True if debug gate stepping is active.
    * debug_waiting — True if a pipeline is currently paused at a debug checkpoint.
    * rp_llm_connected — True if the RP LLM backend responds to /v1/models.
    * instruct_llm_connected — True if the Instruct LLM backend responds to /v1/models.
    """
    connected = (time.time() - _last_extension_ping) < 45
    if not config_manager:
        return {
            "extension_connected": False,
            "debug_mode": False,
            "debug_stepping": False,
            "debug_waiting": False,
            "rp_llm_connected": False,
            "rp_llm_disabled": False,
            "instruct_llm_connected": False,
            "instruct_llm_disabled": False,
        }

    cfg = config_manager.config
    urls = config_manager.get_effective_urls()

    # Probe RP LLM
    rp_connected = False
    if not cfg.rp_llm_disabled:
        rp_connected = await _probe_llm(urls["rp_llm_url"])

    # Probe Instruct LLM
    instruct_connected = False
    if not cfg.instruct_llm_disabled:
        instruct_connected = await _probe_llm(urls["instruct_llm_url"])

    return {
        "extension_connected": connected,
        "debug_mode": cfg.debug_mode,
        "debug_stepping": debug_gate.enabled,
        "debug_waiting": debug_gate._current is not None,
        "rp_llm_connected": rp_connected,
        "rp_llm_disabled": cfg.rp_llm_disabled,
        "instruct_llm_connected": instruct_connected,
        "instruct_llm_disabled": cfg.instruct_llm_disabled,
    }


async def _probe_llm(base_url: str, timeout_s: float = 3.0) -> bool:
    """Check if an LLM backend is reachable by hitting /v1/models.

    Returns True if we get any HTTP response within the timeout.
    The URL may or may not include a path already — we only append
    /v1/models if the base doesn't end with a slash-path.
    """
    import httpx
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    url += "/models"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
            resp = await client.get(url)
            return resp.status_code == 200
    except Exception:
        return False


# ── Debug Stepping Endpoints ────────────────────────────────

@app.get("/api/debug/state")
async def debug_state():
    """Return the current debug stepping state for the dashboard."""
    return debug_gate.to_dict()


@app.post("/api/debug/continue")
async def debug_continue():
    """Resume a paused debug pipeline.

    Called when the user clicks "Continue" in the dashboard's
    debug panel.  If no pipeline is currently paused, this is a no-op.
    """
    if debug_gate._current is None:
        return {"status": "idle", "message": "No pipeline is currently paused"}
    debug_gate.continue_pipeline()
    step = debug_gate._completed[-1] if debug_gate._completed else None
    return {
        "status": "continued",
        "message": f"Proceeding with: {step['outgoing']}" if step else "Continued",
    }


@app.post("/api/debug/toggle")
async def debug_toggle(request: Request):
    """Enable or disable debug stepping.

    Body: {"enabled": true/false}

    When disabled, any currently paused pipeline is immediately
    resumed so it doesn't hang.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    enabled = bool(body.get("enabled", False))
    debug_gate.toggle(enabled)

    # Persist to config.ini
    if config_manager:
        config_manager.config.debug_stepping = enabled
        config_manager.save()

    return {
        "status": "ok",
        "debug_stepping": debug_gate.enabled,
    }


@app.post("/api/debug/reset")
async def debug_reset():
    """Reset the debug gate (clear completed steps, unblock if waiting)."""
    had_current = debug_gate._current is not None
    debug_gate.reset()
    return {
        "status": "ok",
        "message": "Debug state reset" + (" (resumed paused pipeline)" if had_current else ""),
    }


# ── Delete Session Endpoint ──────────────────────────────────

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its database file.

    Returns 404 if the session does not exist.
    """
    if not database:
        raise HTTPException(503, "Database not initialized")

    # Verify the session exists
    meta = database.get_session(session_id)
    if not meta:
        raise HTTPException(404, "Session not found")

    char_name = meta.get("character_name", "") or "Unnamed"
    success = database.delete_session(session_id)

    if not success:
        raise HTTPException(500, "Failed to delete session")

    logger.info(f"Deleted session {session_id} ({char_name})")
    return {"status": "deleted", "session_id": session_id, "character_name": char_name}


@app.get("/v1/models")
async def list_models():
    """Proxy /v1/models — returns combined models from both LLMs."""
    if not config_manager or not client_manager:
        raise HTTPException(503, "Agent not ready")

    cfg = config_manager.config
    urls = config_manager.get_effective_urls()

    models = []

    if not cfg.rp_llm_disabled:
        rp_client = client_manager.get_rp_client(
            urls["rp_llm_url"], cfg.rp_template, cfg.rp_llm_model
        )
        models.extend(await rp_client.list_models())

    if not cfg.instruct_llm_disabled:
        instruct_client = client_manager.get_instruct_client(
            urls["instruct_llm_url"], cfg.instruct_template, cfg.instruct_llm_model
        )
        models.extend(await instruct_client.list_models())

    all_models = list(set(models))
    return {
        "data": [{"id": m, "object": "model", "owned_by": "local"} for m in all_models],
    }


# ── Session Endpoints ─────────────────────────────────────────

@app.post("/api/sessions")
async def create_session(request: Request):
    """Create a new session. Returns a session_id UUID.

    Optional body field ``st_chat_id`` links the session to a
    SillyTavern chat so the extension can look it up later via
    GET /api/sessions/by-chat.
    """
    if not database:
        raise HTTPException(503, "Database not initialized")

    # Parse optional body (may be empty or omitted entirely)
    st_chat_id = ""
    try:
        body = await request.json()
        st_chat_id = (body or {}).get("st_chat_id", "") or ""
    except Exception:
        pass

    await debug_gate.wait(
        "Request to create session",
        "Creating session in database",
        {"st_chat_id": st_chat_id or "none"},
    )

    session_id = str(uuid.uuid4())
    success = database.create_session(session_id, st_chat_id=st_chat_id)

    if not success:
        raise HTTPException(500, "Failed to create session")

    logger.info(f"Session created: {session_id} (st_chat_id={st_chat_id or 'none'})")

    await debug_gate.wait(
        f"Session created: {session_id[:8]}",
        "Returning session_id to ST Extension",
        {"session_id": session_id},
    )

    return {"session_id": session_id, "st_chat_id": st_chat_id}


@app.get("/api/sessions")
async def list_sessions():
    """List all sessions and their basic stats."""
    if not database:
        raise HTTPException(503, "Database not initialized")

    sessions = []
    for sid in database.list_sessions():
        stats = database.get_session_stats(sid)
        meta = database.get_session(sid)
        sessions.append({
            "session_id": sid,
            "character_name": meta.get("character_name", "") if meta else "",
            "initialized": database.is_initialized(sid),
            **(stats or {}),
        })

    return {"sessions": sessions}


@app.get("/api/sessions/by-chat")
async def find_session_by_chat(st_chat_id: str):
    """Look up a session by its linked SillyTavern chat ID.

    Returns ``{session_id: "..."}`` if found, or 404.
    NOTE: This route MUST be defined before /api/sessions/{session_id}
          so FastAPI doesn't match "by-chat" as a session_id.
    """
    if not database:
        raise HTTPException(503, "Database not initialized")

    if not st_chat_id:
        raise HTTPException(400, "st_chat_id query parameter is required")

    session_id = database.find_session_by_st_chat_id(st_chat_id)
    if not session_id:
        raise HTTPException(404, "No session linked to this chat ID")

    return {"session_id": session_id, "st_chat_id": st_chat_id}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session metadata and stats."""
    if not database:
        raise HTTPException(503, "Database not initialized")

    meta = database.get_session(session_id)
    if not meta:
        raise HTTPException(404, "Session not found")

    stats = database.get_session_stats(session_id)
    world_state = database.get_world_state(session_id)

    return {
        **meta,
        **(stats or {}),
        "world_state": world_state,
    }


@app.post("/api/sessions/{session_id}/link-chat")
async def link_session_chat(session_id: str, request: Request):
    """Link an existing session to a SillyTavern chat ID.

    Used when the extension detects that a local session exists
    but the Agent does not yet know about the chat ID mapping.
    """
    if not database:
        raise HTTPException(503, "Database not initialized")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    st_chat_id = (body or {}).get("st_chat_id", "") or ""
    if not st_chat_id:
        raise HTTPException(400, "st_chat_id is required")

    success = database.link_session_chat(session_id, st_chat_id)
    if not success:
        raise HTTPException(500, "Failed to link session")

    logger.info(f"Session {session_id} linked to st_chat_id {st_chat_id}")
    return {
        "session_id": session_id,
        "st_chat_id": st_chat_id,
        "status": "linked",
    }


@app.post("/api/sessions/{session_id}/init")
async def init_session(session_id: str, request: Request):
    """Initialize a session with character card data."""
    if not database:
        raise HTTPException(503, "Database not initialized")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    if not body:
        raise HTTPException(400, "Empty request body")

    # --- Group detection ---
    is_group = body.get("is_group", False)
    group_name = body.get("group_name", "")
    group_members = body.get("group_members", [])
    group_disabled = body.get("group_disabled_members", [])
    char_name = body.get("character_name", "Unknown")

    # For group chats, use group_name as the display name
    display_name = group_name if (is_group and group_name) else char_name

    logger.info(f"Initializing session {session_id}: {display_name} (is_group={is_group}, members={len(group_members) if group_members else 0})")
    logger.debug(f"  Character data keys: {list(body.keys())}")
    success = database.init_session(session_id, body)

    if not success:
        raise HTTPException(500, "Failed to initialize session")

    await debug_gate.wait(
        f"Data received for: {display_name}",
        "Session initialized in database",
        {
            "is_group": is_group,
            "name": display_name,
            "member_count": len(group_members) if group_members else 0,
        },
    )

    cfg = config_manager.config

    # Skip initial extraction if dry_run or instruct disabled
    if cfg.dry_run or cfg.instruct_llm_disabled:
        reason = "DRY RUN" if cfg.dry_run else "INSTRUCT LLM DISABLED"
        logger.info(f"[{reason}] Skipping initial state extraction")
        await debug_gate.wait(
            f"Session initialized ({reason})",
            "Skipping extraction — returning to ST Extension",
            {"reason": reason},
        )
    else:
        # Build extraction data outside the async function
        urls = config_manager.get_effective_urls()
        instruct_client = client_manager.get_instruct_client(
            urls["instruct_llm_url"], cfg.instruct_template, cfg.instruct_llm_model
        )
        pipeline = ExtractionPipeline(
            database, instruct_client, str(config_manager.prompts_dir_path)
        )

        desc = body.get("character_description", "")
        scenario = body.get("character_scenario", "")
        first_mes = body.get("character_first_mes", "")
        combined = f"Character: {display_name}\n{desc}\n{scenario}\n{first_mes}"

        # Build extraction params
        extract_params = {
            "character_name": display_name,
            "tracked_characters": group_members if (is_group and group_members) else [],
            "mode": body.get("mode", "character"),
            "is_initial": True,
        }

        async def _extract_initial_state():
            """Run initial state extraction with debug checkpoints.

            For group chats: runs extraction ONCE with all members listed,
            so the Instruct LLM can extract state for each character.
            """
            try:
                await debug_gate.wait(
                    "Session ready for extraction",
                    "Sending character data to Instruct LLM",
                    {
                        "character": display_name,
                        "is_group": is_group,
                        "members": extract_params["tracked_characters"],
                        "data_length": len(combined),
                        "instruct_url": urls["instruct_llm_url"],
                    },
                )

                logger.info(f"[INIT EXTRACT] Starting initial state extraction for session {session_id[:8]}...")
                logger.info(f"[INIT EXTRACT] Instruct LLM URL: {urls['instruct_llm_url']}")
                logger.info(f"[INIT EXTRACT] Instruct model: {cfg.instruct_llm_model or '(default)'}")
                if is_group:
                    logger.info(f"[INIT EXTRACT] GROUP MODE: {display_name} — members: {extract_params['tracked_characters']}")

                result = await pipeline.run(
                    session_id=session_id,
                    message_id=0,
                    swipe_index=0,
                    assistant_response=combined,
                    conversation_context="Initial character data extraction",
                    **extract_params,
                )

                await debug_gate.wait(
                    "Instruct LLM response received",
                    "Writing state changes to database",
                    {
                        "success": result.get("success", False),
                        "changes": result.get("changes_applied", 0),
                        "raw_response_length": len(result.get("raw_response", "")),
                    },
                )

                logger.info(
                    f"[INIT EXTRACT] Done: "
                    f"{'success' if result.get('success') else 'failed'}, "
                    f"{result.get('changes_applied', 0)} fields, "
                    f"raw_response length: {len(result.get('raw_response', ''))} chars"
                )

                await debug_gate.wait(
                    "Database write complete",
                    "Reporting success to ST Extension",
                    {
                        "fields_written": result.get("changes_applied", 0),
                        "session_id": session_id[:8],
                    },
                )
            except Exception as e:
                logger.error(f"[INIT EXTRACT] Initial extraction failed: {e}", exc_info=True)

        if debug_gate.enabled:
            # Run inline so debug gate can pause each step
            await _extract_initial_state()
        else:
            # Run as background task (normal mode)
            task = asyncio.create_task(_extract_initial_state())
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

    return {
        "session_id": session_id,
        "status": "initialized",
        "message": "Session initialized.",
    }


# ── Config Endpoint ───────────────────────────────────────────

@app.post("/api/config")
async def receive_config(request: Request):
    """Receive configuration from the SillyTavern extension.

    Extension values OVERWRITE config.ini values.
    """
    if not config_manager:
        raise HTTPException(503, "Agent not ready")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    logger.info(f"[EXTENSION CONFIG] Received: {json.dumps(body, indent=2)}")

    # Check if debug_mode changed — apply it immediately
    old_debug = config_manager.config.debug_mode
    applied = config_manager.update_from_extension(body)

    if config_manager.config.debug_mode != old_debug:
        _apply_debug_level(config_manager.config.debug_mode)

    # Update client manager snapshot
    cfg = config_manager.config
    urls = config_manager.get_effective_urls()
    client_manager.update_config({
        "rp": {
            "url": urls["rp_llm_url"],
            "template": cfg.rp_template,
            "model": cfg.rp_llm_model,
        },
        "instruct": {
            "url": urls["instruct_llm_url"],
            "template": cfg.instruct_template,
            "model": cfg.instruct_llm_model,
        },
    })

    return {"status": "ok", "applied": list(applied.keys())}


# ── Stop Generation Endpoint ──────────────────────────────

@app.post("/api/stop")
async def stop_generation():
    """Stop the current generation.

    Called by the SillyTavern extension when the user presses Stop.
    Sets the abort event (checked by active SSE streams each iteration)
    and optionally calls the RP LLM's native abort endpoint.

    KoboldCPP: POST /api/extra/abort
    Ollama:    Closing the upstream connection is sufficient
    Generic:   Closing the upstream connection is sufficient
    """
    global _abort_event

    # Set the abort signal so the streaming generator stops on next iteration
    _abort_event.set()
    logger.info("[STOP] Abort signal sent — active streams will halt")

    # Also try to call the RP LLM's native abort endpoint
    if config_manager and not config_manager.config.rp_llm_disabled:
        cfg = config_manager.config
        urls = config_manager.get_effective_urls()
        rp_url = urls["rp_llm_url"]

        try:
            import httpx
            abort_urls = {
                "kobold": f"{rp_url}/api/extra/abort",
            }
            backend = cfg.rp_llm_backend.lower() if hasattr(cfg, 'rp_llm_backend') else ""

            if backend in abort_urls:
                async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                    resp = await client.post(abort_urls[backend])
                    logger.info(f"[STOP] Sent native abort to {abort_urls[backend]} → {resp.status_code}")
            else:
                # For Ollama and generic backends, the upstream connection closing
                # when we break out of the stream loop is sufficient to stop generation
                logger.debug(f"[STOP] No native abort endpoint for backend '{backend}' — upstream close will handle it")
        except Exception as e:
            logger.warning(f"[STOP] Failed to call native abort: {e}")

    # Reset the event after a brief delay so future generations aren't immediately aborted
    async def _reset_abort():
        await asyncio.sleep(0.5)
        _abort_event.clear()
        logger.debug("[STOP] Abort event cleared — ready for next generation")

    asyncio.create_task(_reset_abort())

    return {"status": "stopped", "message": "Generation abort signal sent"}


# ── Dry-run helpers ───────────────────────────────────────────

def _log_dry_run_receive(body: dict, meta: SystemMeta, clean_messages: list):
    """Log what the agent received from SillyTavern."""
    logger.info("=" * 56)
    logger.info("[DRY RUN] === WHAT THE AGENT RECEIVED FROM ST ===")
    logger.info(f"[DRY RUN] Raw request body keys: {list(body.keys())}")
    logger.info(f"[DRY RUN]   temperature: {body.get('temperature', 0.8)}")
    logger.info(f"[DRY RUN]   max_tokens:   {body.get('max_tokens', 2048)}")
    logger.info(f"[DRY RUN]   stream:       {body.get('stream', True)}")
    if meta:
        logger.info(f"[DRY RUN] Parsed [SYSTEM_META]:")
        logger.info(f"[DRY RUN]   session_id:   {meta.session_id}")
        logger.info(f"[DRY RUN]   message_id:   {meta.message_id}")
        logger.info(f"[DRY RUN]   type:         {meta.type}")
        logger.info(f"[DRY RUN]   swipe_index:  {meta.swipe_index}")
    else:
        logger.info("[DRY RUN] No [SYSTEM_META] found (passthrough mode)")
    logger.info(f"[DRY RUN] Cleaned messages ({len(clean_messages)}):")
    for i, msg in enumerate(clean_messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        preview = content[:200] + ("..." if len(content) > 200 else "")
        logger.info(f"[DRY RUN]   [{i}] {role}: {preview}")


def _log_dry_run_send(messages: list, target: str, url: str, temperature: float, max_tokens: int):
    """Log what the agent WOULD send to an LLM."""
    logger.info(f"[DRY RUN] === WOULD SEND TO {target.upper()} LLM ===")
    logger.info(f"[DRY RUN]   URL:         {url}")
    logger.info(f"[DRY RUN]   temperature: {temperature}")
    logger.info(f"[DRY RUN]   max_tokens:  {max_tokens}")
    logger.info(f"[DRY RUN]   Messages ({len(messages)}):")
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        preview = content[:300] + ("..." if len(content) > 300 else "")
        logger.info(f"[DRY RUN]     [{i}] {role}: {preview}")
    logger.info("=" * 56)


def _dry_run_stream(text: str, model: str = "dry-run"):
    """Yield a fake SSE stream for dry-run mode."""
    # Send as one big chunk so SillyTavern gets a response
    payload = json.dumps({
        "choices": [{
            "delta": {"content": text},
            "finish_reason": None,
            "index": 0,
        }],
        "model": model,
        "object": "chat.completion.chunk",
    })
    yield f"data: {payload}\n\n"
    # Finish
    yield f"data: {json.dumps({'choices': [{'delta': {'content': ''}, 'finish_reason': 'stop', 'index': 0}], 'model': model, 'object': 'chat.completion.chunk'})}\n\n"
    yield "data: [DONE]\n\n"


# ── Main Chat Completion Pipeline ─────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Main chat completion endpoint."""
    if not config_manager or not database or not client_manager:
        raise HTTPException(503, "Agent not ready")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(400, "No messages in request")

    cfg = config_manager.config
    urls = config_manager.get_effective_urls()

    # ── Step 1: Parse [SYSTEM_META] ────────────────────────────
    meta, clean_messages = strip_meta_from_messages(messages)

    if not meta:
        logger.warning("No [SYSTEM_META] found in request — passing through to RP LLM directly")
        if cfg.dry_run or cfg.rp_llm_disabled:
            _log_dry_run_receive(body, None, clean_messages)
            _log_dry_run_send(
                clean_messages, "rp", urls["rp_llm_url"],
                body.get("temperature", 0.8), body.get("max_tokens", 2048),
            )
            label = "DRY RUN" if cfg.dry_run else "RP LLM DISABLED"
            return StreamingResponse(
                _dry_run_stream(f"[{label}] Passthrough — no SYSTEM_META, would forward {len(clean_messages)} messages to RP LLM at {urls['rp_llm_url']}"),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        rp_client = client_manager.get_rp_client(
            urls["rp_llm_url"], cfg.rp_template, cfg.rp_llm_model
        )
        return StreamingResponse(
            _passthrough_stream(rp_client, clean_messages, body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    logger.info(
        f"[Pipeline] session={meta.session_id[:8]}... "
        f"msg={meta.message_id} type={meta.type} "
        f"swipe={meta.swipe_index}"
    )

    # ── Track pipeline run for dashboard ────────────────────
    pipeline_tracker.start_run(meta.session_id, meta.message_id, meta.type)
    pipeline_tracker.step("received", "Received from SillyTavern", data={"messages": len(clean_messages)})
    user_preview = ""
    for m in reversed(clean_messages):
        if m.get("role") == "user" and m.get("content"):
            user_preview = m["content"][:300]
            break
    if user_preview:
        pipeline_tracker.step("user_message", "User Message", preview=user_preview)

    # ── DRY RUN: show what was received ───────────────────────
    if cfg.dry_run:
        _log_dry_run_receive(body, meta, clean_messages)

    # ── Debug: pause after receiving request ────────────────
    await debug_gate.wait(
        "Chat request from SillyTavern",
        f"Parse [SYSTEM_META] — type={meta.type}, msg={meta.message_id}, swipe={meta.swipe_index}",
        {"session_id": meta.session_id[:8], "messages": len(clean_messages), "type": meta.type},
    )

    # ── Step 2: Handle swipe/redo ─────────────────────────────
    if meta.type == "swipe":
        reverted = database.revert_swipe(
            meta.session_id, meta.message_id, meta.swipe_index
        )
        if reverted:
            logger.info(f"Reverted {reverted} state fields for swipe")
        pipeline_tracker.step("swipe", "Swipe Handling", data={"reverted": reverted or 0}, status="warn" if reverted else "done")

    elif meta.type == "redo":
        reverted = database.revert_from_message(
            meta.session_id, meta.message_id
        )
        if reverted:
            logger.info(f"Reverted {reverted} state fields for redo")
        pipeline_tracker.step("redo", "Redo Handling", data={"reverted": reverted or 0}, status="warn" if reverted else "done")

    # ── Debug: pause after swipe/redo ──────────────────────
    await debug_gate.wait(
        f"Message type: {meta.type} (reverted {reverted or 0} fields)" if meta.type in ("swipe", "redo") else f"Message type: {meta.type}",
        "Translate world state via Instruct LLM" if not (cfg.dry_run or cfg.instruct_llm_disabled) else "Skip world state translation",
        {"type": meta.type, "reverted": reverted or 0},
    )

    # ── Step 3: Translate world state ─────────────────────────
    instruct_client = client_manager.get_instruct_client(
        urls["instruct_llm_url"], cfg.instruct_template, cfg.instruct_llm_model
    )

    world_summary = ""
    if cfg.dry_run or cfg.instruct_llm_disabled:
        world_state = database.get_world_state(meta.session_id)
        if world_state:
            world_summary = f"[DRY RUN / INSTRUCT DISABLED] Would translate world state: {json.dumps(world_state, indent=2, ensure_ascii=False)[:500]}"
            logger.info(f"[DRY RUN] World state exists ({len(world_state)} fields) — would send to Instruct LLM for translation")
            pipeline_tracker.step("world_state", "World State Translation", data={"fields": len(world_state)}, status="warn", preview=world_summary[:200])
    else:
        translation_pipe = TranslationPipeline(
            database, instruct_client, str(config_manager.prompts_dir_path)
        )
        t_start = time.time()
        try:
            world_summary = await translation_pipe.translate(meta.session_id)
            t_ms = int((time.time() - t_start) * 1000)
            pipeline_tracker.step("world_state", "World State Translation", data={"fields": "n/a", "duration_ms": t_ms}, preview=world_summary[:200] if world_summary else "(empty)")
        except Exception as e:
            logger.warning(f"World state translation failed: {e}")
            pipeline_tracker.step("world_state", "World State Translation", status="warn", preview=f"Failed: {e}")

    # ── Step 4: Inject world state context ────────────────────
    if world_summary:
        clean_messages = inject_world_state_context(
            clean_messages, world_summary
        )
        logger.debug("Injected world state context into messages")

    # ── Step 5: Format with template ───────────────────────────
    formatted_messages = format_messages(clean_messages, cfg.rp_template)
    pipeline_tracker.step("template", "Template Formatting", data={"template": cfg.rp_template, "messages": len(formatted_messages)})

    # ── Debug: pause before RP LLM generation ───────────────
    await debug_gate.wait(
        f"World state translated ({len(world_summary)} chars), messages formatted ({cfg.rp_template})",
        "Send to RP LLM for narrative generation",
        {
            "world_state_length": len(world_summary),
            "template": cfg.rp_template,
            "formatted_messages": len(formatted_messages),
            "thinking_steps": cfg.thinking_steps,
        },
    )

    # ── Step 6: Thinking passes (optional) ────────────────────
    rp_client = client_manager.get_rp_client(
        urls["rp_llm_url"], cfg.rp_template, cfg.rp_llm_model
    )

    thinking_notes = []
    if cfg.thinking_steps > 0 and meta.type not in ("continue", "swipe"):
        if cfg.dry_run or cfg.rp_llm_disabled:
            logger.info(f"[DRY RUN / RP DISABLED] Would run {cfg.thinking_steps} thinking passes")
        else:
            logger.info(f"Running {cfg.thinking_steps} thinking passes...")
            try:
                thinking_notes = await run_thinking_passes(
                    rp_client, formatted_messages, cfg.thinking_steps
                )
                if thinking_notes:
                    notes_text = "\n\n[Internal planning notes — do not include in response]\n" + "\n".join(
                        f"- {note}" for note in thinking_notes
                    )
                    for i in range(len(formatted_messages) - 1, -1, -1):
                        if formatted_messages[i].get("role") == "user":
                            formatted_messages[i]["content"] += notes_text
                            break
                    logger.info(f"Thinking complete: {len(thinking_notes)} notes generated")
                    pipeline_tracker.step("thinking", "Thinking Passes", data={"notes": len(thinking_notes)}, preview="\n".join(f"- {n}" for n in thinking_notes[:3]))
            except Exception as e:
                logger.warning(f"Thinking pipeline failed: {e}")
                pipeline_tracker.step("thinking", "Thinking Passes", status="warn", preview=f"Failed: {e}")

    # ── Step 7-8: Stream + Refine ─────────────────────────────
    stream_temperature = body.get("temperature", 0.8)
    stream_max_tokens = body.get("max_tokens", 2048)

    # DRY RUN: show what we WOULD send to RP LLM
    if cfg.dry_run:
        _log_dry_run_send(
            formatted_messages, "rp", urls["rp_llm_url"],
            stream_temperature, stream_max_tokens,
        )
        # Also show what we WOULD send to Instruct LLM for extraction
        _log_dry_run_extraction(formatted_messages, clean_messages)

        return StreamingResponse(
            _dry_run_stream(
                f"[DRY RUN] Pipeline completed. No LLM calls were made.\n\n"
                f"Session: {meta.session_id[:8]}...\n"
                f"Message: {meta.message_id} | Type: {meta.type} | Swipe: {meta.swipe_index}\n"
                f"Messages prepared: {len(formatted_messages)}\n"
                f"World state injected: {'yes' if world_summary else 'no'}\n"
                f"Template: {cfg.rp_template}\n"
                f"RP LLM target: {urls['rp_llm_url']} (disabled={cfg.rp_llm_disabled})\n"
                f"Instruct LLM target: {urls['instruct_llm_url']} (disabled={cfg.instruct_llm_disabled})\n"
                f"\nCheck the Pinokio terminal for full message logs.",
                model=cfg.rp_llm_model or "dry-run",
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # RP LLM DISABLED (but not dry_run — use empty placeholder response)
    if cfg.rp_llm_disabled:
        logger.info("[RP LLM DISABLED] Returning placeholder response")
        placeholder = "[RP LLM DISABLED] The RP LLM is turned off in config.ini. No narrative was generated."
        return StreamingResponse(
            _dry_run_stream(placeholder, model=cfg.rp_llm_model or "rp-disabled"),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── LIVE: actual streaming ────────────────────────────────
    async def generate():
        """SSE generator that streams RP LLM output.

        Checks _abort_event each iteration — if set (by /api/stop),
        the stream stops cleanly without dropping the connection.
        """
        full_response = []
        aborted = False

        pipeline_tracker.step("rp_send", "Sent to RP LLM", data={"url": urls["rp_llm_url"], "messages": len(formatted_messages), "temperature": stream_temperature, "max_tokens": stream_max_tokens})
        rp_start_time = time.time()

        try:
            async for chunk in rp_client.chat_stream(
                messages=formatted_messages,
                temperature=stream_temperature,
                max_tokens=stream_max_tokens,
            ):
                # Check if /api/stop was called (ST extension sent abort)
                if _abort_event.is_set():
                    logger.info("[STOP] Abort signal received — halting generation")
                    aborted = True
                    break

                full_response.append(chunk)
                payload = json.dumps({
                    "choices": [{
                        "delta": {"content": chunk},
                        "finish_reason": None,
                        "index": 0,
                    }],
                    "model": cfg.rp_llm_model or "agent-statesync",
                    "object": "chat.completion.chunk",
                })
                yield f"data: {payload}\n\n"

            # If aborted, send finish_reason=stop and skip post-processing
            if aborted:
                logger.info("[STOP] Skipping refinement, logging, and state extraction")
                pipeline_tracker.finish("aborted", "".join(full_response))
                finish = json.dumps({
                    "choices": [{
                        "delta": {"content": ""},
                        "finish_reason": "stop",
                        "index": 0,
                    }],
                    "model": cfg.rp_llm_model or "agent-statesync",
                    "object": "chat.completion.chunk",
                })
                yield f"data: {finish}\n\n"
                yield "data: [DONE]\n\n"
                return

            response_text = "".join(full_response)
            rp_elapsed = int((time.time() - rp_start_time) * 1000)
            pipeline_tracker.step("rp_response", "RP LLM Response", data={"chars": len(response_text), "duration_ms": rp_elapsed}, preview=response_text[:300])

            # ── Debug: pause after RP LLM response ──────────
            await debug_gate.wait(
                f"RP LLM response complete ({len(response_text)} chars, {rp_elapsed}ms)",
                "Send conversation to Instruct LLM for state extraction",
                {"response_length": len(response_text), "duration_ms": rp_elapsed},
            )

            # ── Step 8: Refinement (optional, blocking) ──────────
            if cfg.refinement_steps > 0 and response_text:
                logger.info(f"Running {cfg.refinement_steps} refinement pass(es)...")
                try:
                    refined = await run_refinement_passes(
                        rp_client,
                        formatted_messages,
                        response_text,
                        cfg.refinement_steps,
                    )
                    if refined and refined != response_text:
                        logger.info(
                            f"Refinement complete: "
                            f"{len(response_text)} -> {len(refined)} chars"
                        )
                        response_text = refined
                        pipeline_tracker.step("refinement", "Refinement", data={"chars": len(refined), "duration_ms": int((time.time() - rp_elapsed - rp_start_time) * 1000)})
                except Exception as e:
                    logger.warning(f"Refinement pipeline failed: {e}")
                    pipeline_tracker.step("refinement", "Refinement", status="warn", preview=f"Failed: {e}")

            # ── Log message to DB ──────────────────────────────
            database.log_message(
                meta.session_id, meta.message_id, meta.swipe_index,
                "assistant", response_text,
            )

            # ── Step 9: State extraction ────────────────────────
            if cfg.instruct_llm_disabled:
                logger.info("[INSTRUCT LLM DISABLED] Skipping state extraction")
                pipeline_tracker.step("extraction", "State Extraction", status="warn", preview="Instruct LLM disabled — skipped")
            else:
                extraction_pipe = ExtractionPipeline(
                    database, instruct_client, str(config_manager.prompts_dir_path)
                )

                conv_context = ""
                user_msgs = [m for m in clean_messages if m.get("role") == "user"]
                if user_msgs:
                    conv_context = f"Latest user message:\n{user_msgs[-1].get('content', '')}"

                async def _extract_state():
                    try:
                        # Build extraction params with character/group info from META tag
                        extract_params = {
                            "session_id": meta.session_id,
                            "message_id": meta.message_id,
                            "swipe_index": meta.swipe_index,
                            "assistant_response": response_text,
                            "conversation_context": conv_context,
                            "mode": "character",
                            "is_initial": False,
                        }

                        # Pass group metadata if available (from [SYSTEM_META] tag)
                        if meta.members:
                            extract_params["tracked_characters"] = meta.members
                            extract_params["character_name"] = meta.members[0] if len(meta.members) == 1 else (meta.group_name or "")
                        else:
                            # Single character: get name from session metadata
                            session_meta = database.get_session(meta.session_id)
                            if session_meta:
                                extract_params["character_name"] = session_meta.get("character_name", "")

                        logger.debug(f"[EXTRACT] Params: char={extract_params.get('character_name', '')}, tracked={extract_params.get('tracked_characters', [])}")

                        result = await extraction_pipe.run(**extract_params)

                        # Track extraction result for dashboard
                        if result.get("success"):
                            ws = database.get_world_state(meta.session_id)
                            pipeline_tracker.step("extraction", "State Extraction", data={"changes": result.get("changes_applied", 0)}, changes=ws)
                        else:
                            pipeline_tracker.step("extraction", "State Extraction", preview="No changes extracted")

                        if result.get("success"):
                            logger.info(
                                f"State extracted: {result.get('changes_applied', 0)} fields "
                                f"(msg={meta.message_id}, swipe={meta.swipe_index})"
                            )
                        else:
                            logger.debug(
                                f"No state changes extracted "
                                f"(msg={meta.message_id})"
                            )
                    except Exception as e:
                        logger.error(f"Background state extraction error: {e}")
                        pipeline_tracker.step("extraction", "State Extraction", status="warn", preview=f"Failed: {e}")

                if debug_gate.enabled:
                    # Run inline so debug gate can pause it
                    await _extract_state()
                else:
                    task = asyncio.create_task(_extract_state())
                    _background_tasks.add(task)
                    task.add_done_callback(_background_tasks.discard)

            pipeline_tracker.finish("completed", response_text)

        except ConnectionError as e:
            logger.error(f"LLM connection error: {e}")
            error_payload = json.dumps({
                "error": {
                    "message": str(e),
                    "type": "connection_error",
                }
            })
            yield f"data: {error_payload}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            error_payload = json.dumps({
                "error": {
                    "message": f"Agent error: {str(e)}",
                    "type": "agent_error",
                }
            })
            yield f"data: {error_payload}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _log_dry_run_extraction(formatted_messages: list, clean_messages: list):
    """Log what the extraction pipeline WOULD receive."""
    user_msgs = [m for m in clean_messages if m.get("role") == "user"]
    conv_context = ""
    if user_msgs:
        conv_context = f"Latest user message:\n{user_msgs[-1].get('content', '')[:200]}"

    logger.info(f"[DRY RUN] === WOULD SEND TO INSTRUCT LLM (extraction) ===")
    logger.info(f"[DRY RUN]   (runs as background task AFTER RP response)")
    logger.info(f"[DRY RUN]   conversation_context: {conv_context[:300]}")
    logger.info(f"[DRY RUN]   assistant_response: (would be whatever RP LLM returns)")
    logger.info("=" * 56)


async def _passthrough_stream(
    rp_client, messages: list, body: dict
):
    """Fallback stream when no [SYSTEM_META] is present."""
    temperature = body.get("temperature", 0.8)
    max_tokens = body.get("max_tokens", 2048)

    try:
        async for chunk in rp_client.chat_stream(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            # Check for abort signal
            if _abort_event.is_set():
                logger.info("[STOP] Abort signal received during passthrough")
                break

            payload = json.dumps({
                "choices": [{
                    "delta": {"content": chunk},
                    "finish_reason": None,
                    "index": 0,
                }],
                "object": "chat.completion.chunk",
            })
            yield f"data: {payload}\n\n"
    except Exception as e:
        logger.error(f"Passthrough stream error: {e}")
        error_payload = json.dumps({
            "error": {"message": str(e), "type": "passthrough_error"}
        })
        yield f"data: {error_payload}\n\n"

    yield "data: [DONE]\n\n"


# ── Non-Streaming Fallback ────────────────────────────────────

@app.post("/v1/completions")
async def legacy_completions(request: Request):
    """Legacy text completions endpoint (minimal support)."""
    return JSONResponse(
        {
            "error": {
                "message": "Use /v1/chat/completions instead. "
                "Configure SillyTavern to use Chat Completion mode.",
                "type": "unsupported",
            }
        },
        status_code=400,
    )


# ── Entry Point ───────────────────────────────────────────────

if __name__ == "__main__":
    cfg = ConfigManager()
    _port = cfg.port

    # Print a localhost URL BEFORE uvicorn starts.
    # Pinokio's start.json uses regex /http://\S+/ to capture the first
    # URL from stdout and sets it as the dashboard button link.
    # uvicorn prints "http://0.0.0.0:PORT" which is not a valid browser
    # URL. By printing localhost first, Pinokio captures our line instead.
    print(f"http://localhost:{_port}")

    uvicorn.run(
        "server:app",
        host=cfg.host,
        port=_port,
        reload=False,
        log_level="debug" if cfg.config.debug_mode else "info",
        access_log=False,
    )