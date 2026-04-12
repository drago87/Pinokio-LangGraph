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

# ── Globals ───────────────────────────────────────────────────

config_manager: Optional[ConfigManager] = None
database: Optional[Database] = None
client_manager: Optional[LLMClientManager] = None

# Pending background tasks (for tracking + cancellation)
_background_tasks: set = set()

# Abort event: set by /api/stop to signal active streams to halt
_abort_event: asyncio.Event = asyncio.Event()

# ── Debug Stepping ──────────────────────────────────────────
# When debug_mode is enabled in config.ini, the pipeline pauses at
# each breakpoint and waits for the user to click "Play" in the
# dashboard before continuing to the next step.

_debug_continue: asyncio.Event = asyncio.Event()
_debug_state: Dict[str, Any] = {
    "active": False,
    "paused": False,
    "step": 0,
    "total_steps": 3,
    "done_text": "",
    "next_text": "",
    "data": {},
    "log": [],
}


async def _debug_wait(done_text: str, next_text: str, data: dict = None):
    """Pause pipeline execution when debug mode is enabled.

    Updates the debug state (visible in the dashboard) and blocks
    until the user clicks the Play button (POST /api/debug/continue).
    """
    if not config_manager or not config_manager.config.debug_mode:
        return

    _debug_state["active"] = True
    _debug_state["paused"] = True
    _debug_state["step"] += 1
    _debug_state["done_text"] = done_text
    _debug_state["next_text"] = next_text
    _debug_state["data"] = data or {}
    _debug_state["log"].append({
        "step": _debug_state["step"],
        "done": done_text,
        "time": datetime.now().strftime("%H:%M:%S"),
    })

    _debug_continue.clear()
    logger.info(
        f"[DEBUG] PAUSED at step {_debug_state['step']}/{_debug_state['total_steps']} "
        f"— {next_text}"
    )

    try:
        await _debug_continue.wait()
    except asyncio.CancelledError:
        logger.info("[DEBUG] Wait cancelled (request disconnected)")
        _debug_state["paused"] = False
        raise

    logger.info("[DEBUG] RESUMED — proceeding")
    _debug_state["paused"] = False

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


def _apply_debug_level(debug: bool):
    """Flip the root logger between INFO and DEBUG."""
    target = logging.DEBUG if debug else logging.INFO
    logging.getLogger().setLevel(target)
    # Also flip our own logger and any child loggers (httpx, etc.)
    for name in ("agent-statesync", "httpx", "httpcore", "asyncio"):
        logging.getLogger(name).setLevel(target)
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

    logger.info("=" * 56)
    logger.info("  Agent-StateSync starting...")
    logger.info(f"  Port:           {config_manager.port}")
    logger.info(f"  RP LLM:         {config_manager.config.rp_llm_url}")
    logger.info(f"  RP disabled:    {config_manager.config.rp_llm_disabled}")
    logger.info(f"  Instruct LLM:   {config_manager.config.instruct_llm_url}")
    logger.info(f"  Instruct dis.:  {config_manager.config.instruct_llm_disabled}")
    logger.info(f"  Debug mode:     {config_manager.config.debug_mode}")
    logger.info(f"  Dry run:        {config_manager.config.dry_run}")
    logger.info(f"  DB directory:   {config_manager.db_dir_path}")
    logger.info(f"  Prompts dir:    {config_manager.prompts_dir_path}")
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
            "message_count": stats.get("message_count", 0),
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
        "dry_run": cfg.dry_run,
        "rp_llm_url": cfg.rp_llm_url,
        "rp_llm_backend": cfg.rp_llm_backend,
        "rp_llm_model": cfg.rp_model or "",
        "rp_llm_disabled": cfg.rp_llm_disabled,
        "instruct_llm_url": cfg.instruct_llm_url,
        "instruct_llm_backend": cfg.instruct_llm_backend,
        "instruct_llm_model": cfg.instruct_model or "",
        "instruct_llm_disabled": cfg.instruct_llm_disabled,
        "thinking_steps": cfg.thinking_steps,
        "refinement_steps": cfg.refinement_steps,
        "history_count": cfg.history_count,
        "rp_template": cfg.rp_template,
        "db_dir": cfg.db_dir,
        "prompts_dir": cfg.prompts_dir,
    }


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
            urls["rp_llm_url"], cfg.rp_template, cfg.rp_model
        )
        models.extend(await rp_client.list_models())

    if not cfg.instruct_llm_disabled:
        instruct_client = client_manager.get_instruct_client(
            urls["instruct_llm_url"], cfg.instruct_template, cfg.instruct_model
        )
        models.extend(await instruct_client.list_models())

    all_models = list(set(models))
    return {
        "data": [{"id": m, "object": "model", "owned_by": "local"} for m in all_models],
    }


# ── Session Endpoints ─────────────────────────────────────────

@app.post("/api/sessions")
async def create_session():
    """Create a new session. Returns a session_id UUID."""
    if not database:
        raise HTTPException(503, "Database not initialized")

    session_id = str(uuid.uuid4())
    success = database.create_session(session_id)

    if not success:
        raise HTTPException(500, "Failed to create session")

    logger.info(f"Session created: {session_id}")
    return {"session_id": session_id}


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

    logger.info(f"Initializing session {session_id} with character data...")
    logger.debug(f"  Character data keys: {list(body.keys())}")
    mode = body.get("mode", "character")
    multi_char = body.get("multi_character", False)
    tracked = body.get("tracked_characters", [])
    logger.info(f"  Mode: {mode}, Multi-character: {multi_char}, Tracked: {tracked}")
    success = database.init_session(session_id, body)

    if not success:
        raise HTTPException(500, "Failed to initialize session")

    cfg = config_manager.config

    # Skip initial extraction if dry_run or instruct disabled
    if cfg.dry_run or cfg.instruct_llm_disabled:
        logger.info(
            f"[DRY RUN / INSTRUCT DISABLED] Skipping initial state extraction"
        )
    else:
        async def _extract_initial_state():
            try:
                urls = config_manager.get_effective_urls()
                instruct_client = client_manager.get_instruct_client(
                    urls["instruct_llm_url"], cfg.instruct_template, cfg.instruct_model
                )
                pipeline = ExtractionPipeline(
                    database, instruct_client, str(config_manager.prompts_dir_path)
                )

                desc = body.get("character_description", "")
                scenario = body.get("character_scenario", "")
                first_mes = body.get("character_first_mes", "")
                combined = f"Character: {body.get('character_name', '')}\n{desc}\n{scenario}\n{first_mes}"

                result = await pipeline.run(
                    session_id=session_id,
                    message_id=0,
                    swipe_index=0,
                    assistant_response=combined,
                    conversation_context="Initial character data extraction",
                )
                logger.info(
                    f"Initial state extraction: "
                    f"{'success' if result.get('success') else 'failed'}, "
                    f"{result.get('changes_applied', 0)} fields"
                )
            except Exception as e:
                logger.error(f"Background initial extraction failed: {e}")

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
            "model": cfg.rp_model,
        },
        "instruct": {
            "url": urls["instruct_llm_url"],
            "template": cfg.instruct_template,
            "model": cfg.instruct_model,
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


# ── Debug Stepping Endpoints ──────────────────────────────────

@app.get("/api/debug/status")
async def debug_status():
    """Return current debug stepping state for the dashboard.

    The dashboard polls this endpoint to show the current breakpoint,
    what was just completed, what's next, and the Play button.
    """
    return {
        "debug_mode": config_manager.config.debug_mode if config_manager else False,
        **_debug_state,
    }


@app.post("/api/debug/continue")
async def debug_continue():
    """Resume a paused pipeline step.

    Called by the dashboard Play button. Sets the continue event so
    the pipeline's _debug_wait() call unblocks.
    """
    if not _debug_state["paused"]:
        return {"status": "not_paused", "message": "Pipeline is not currently paused"}
    _debug_continue.set()
    return {"status": "resumed", "message": "Pipeline resumed"}


@app.post("/api/debug/reset")
async def debug_reset():
    """Reset debug state (clear log, step counter, etc.)."""
    _debug_state["active"] = False
    _debug_state["paused"] = False
    _debug_state["step"] = 0
    _debug_state["done_text"] = ""
    _debug_state["next_text"] = ""
    _debug_state["data"] = {}
    _debug_state["log"] = []
    return {"status": "reset"}


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
            urls["rp_llm_url"], cfg.rp_template, cfg.rp_model
        )
        return StreamingResponse(
            _passthrough_stream(rp_client, clean_messages, body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    logger.info(
        f"[Pipeline] session={meta.session_id[:8]}... "
        f"msg={meta.message_id} type={meta.type} "
        f"swipe={meta.swipe_index} char={meta.character_name} "
        f"mode={meta.mode} tracked={meta.tracked_list}"
    )

    # ── Track pipeline run for dashboard ────────────────────
    pipeline_tracker.start_run(meta.session_id, meta.message_id, meta.type)
    pipeline_tracker.step("received", "Received from SillyTavern", data={
        "messages": len(clean_messages),
        "character_name": meta.character_name,
        "persona_name": meta.persona_name,
        "mode": meta.mode,
        "tracked": meta.tracked_list,
    })
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

    # ── DEBUG BREAKPOINT 1: After parsing ──────────────────────
    await _debug_wait(
        f"Received data from ST — {len(clean_messages)} messages, session={meta.session_id[:8]}",
        f"Preparing to handle {meta.type} (swipe/redo check)",
        {"messages": len(clean_messages), "session_id": meta.session_id[:8], "type": meta.type, "character": meta.character_name, "mode": meta.mode},
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

    # ── DEBUG BREAKPOINT 2: After swipe/redo ─────────────────
    await _debug_wait(
        f"Swipe/redo handled — {reverted or 0} fields reverted" if reverted else "No swipe/redo needed",
        "Preparing to translate world state (Instruct LLM)",
        {"reverted": reverted or 0},
    )

    # ── Step 3: Translate world state ─────────────────────────
    instruct_client = client_manager.get_instruct_client(
        urls["instruct_llm_url"], cfg.instruct_template, cfg.instruct_model
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

    # ── Step 6: Thinking passes (optional) ────────────────────
    rp_client = client_manager.get_rp_client(
        urls["rp_llm_url"], cfg.rp_template, cfg.rp_model
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

    # ── DEBUG BREAKPOINT 3: After formatting, before send ──────
    await _debug_wait(
        f"Messages prepared — {len(formatted_messages)} formatted msgs, template={cfg.rp_template}"
        + (f", {len(thinking_notes)} thinking notes" if thinking_notes else ""),
        f"Preparing to send to RP LLM at {urls['rp_llm_url']}",
        {"messages": len(formatted_messages), "template": cfg.rp_template, "thinking_notes": len(thinking_notes), "rp_url": urls["rp_llm_url"], "temperature": stream_temperature, "max_tokens": stream_max_tokens},
    )

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
                model=cfg.rp_model or "dry-run",
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # RP LLM DISABLED (but not dry_run — use empty placeholder response)
    if cfg.rp_llm_disabled:
        logger.info("[RP LLM DISABLED] Returning placeholder response")
        placeholder = "[RP LLM DISABLED] The RP LLM is turned off in config.ini. No narrative was generated."
        return StreamingResponse(
            _dry_run_stream(placeholder, model=cfg.rp_model or "rp-disabled"),
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
                    "model": cfg.rp_model or "agent-statesync",
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
                    "model": cfg.rp_model or "agent-statesync",
                    "object": "chat.completion.chunk",
                })
                yield f"data: {finish}\n\n"
                yield "data: [DONE]\n\n"
                return

            response_text = "".join(full_response)
            rp_elapsed = int((time.time() - rp_start_time) * 1000)
            pipeline_tracker.step("rp_response", "RP LLM Response", data={"chars": len(response_text), "duration_ms": rp_elapsed}, preview=response_text[:300])

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

            # ── Step 9: Background state extraction ────────────
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

                # Get session mode and tracked characters from DB
                session_meta = database.get_session(meta.session_id)
                session_mode = meta.mode  # From the current request
                if session_meta:
                    # If the request didn't include mode, fall back to session
                    if not session_mode or session_mode == "character":
                        session_mode = session_meta.get("mode", "character")

                # Parse tracked characters from session DB or current meta
                tracked_chars = meta.tracked_list
                if not tracked_chars and session_meta:
                    tc_str = session_meta.get("tracked_characters", "")
                    if tc_str:
                        try:
                            tracked_chars = json.loads(tc_str) if isinstance(tc_str, str) else tc_str
                        except (json.JSONDecodeError, TypeError):
                            tracked_chars = [t.strip() for t in tc_str.split(",") if t.strip()]
                if not tracked_chars and meta.character_name:
                    tracked_chars = [meta.character_name]

                async def _extract_state():
                    try:
                        result = await extraction_pipe.run(
                            session_id=meta.session_id,
                            message_id=meta.message_id,
                            swipe_index=meta.swipe_index,
                            assistant_response=response_text,
                            conversation_context=conv_context,
                            mode=session_mode,
                            character_name=meta.character_name,
                            persona_name=meta.persona_name,
                            tracked_characters=tracked_chars,
                            is_initial=False,
                        )

                        # Track extraction result for dashboard
                        if result.get("success"):
                            ws = database.get_world_state(meta.session_id)
                            pipeline_tracker.step("extraction", "State Extraction",
                                data={
                                    "changes": result.get("changes_applied", 0),
                                    "mode": session_mode,
                                    "tracked": tracked_chars,
                                },
                                changes=ws,
                            )
                            logger.info(
                                f"State extracted: {result.get('changes_applied', 0)} fields "
                                f"(msg={meta.message_id}, swipe={meta.swipe_index}, "
                                f"mode={session_mode}, tracked={tracked_chars})"
                            )
                        else:
                            pipeline_tracker.step("extraction", "State Extraction",
                                preview="No changes extracted",
                                data={"mode": session_mode, "tracked": tracked_chars},
                            )
                            logger.debug(
                                f"No state changes extracted "
                                f"(msg={meta.message_id}, mode={session_mode})"
                            )
                    except Exception as e:
                        logger.error(f"Background state extraction error: {e}")
                        pipeline_tracker.step("extraction", "State Extraction",
                            status="warn", preview=f"Failed: {e}")

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
    uvicorn.run(
        "server:app",
        host=cfg.host,
        port=cfg.port,
        reload=False,
        log_level="debug" if cfg.config.debug_mode else "info",
        access_log=False,
    )