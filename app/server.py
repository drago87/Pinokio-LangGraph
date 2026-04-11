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
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

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

# Pending background tasks (for tracking)
_background_tasks: set = set()


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
        f"swipe={meta.swipe_index}"
    )

    # ── DRY RUN: show what was received ───────────────────────
    if cfg.dry_run:
        _log_dry_run_receive(body, meta, clean_messages)

    # ── Step 2: Handle swipe/redo ─────────────────────────────
    if meta.type == "swipe":
        reverted = database.revert_swipe(
            meta.session_id, meta.message_id, meta.swipe_index
        )
        if reverted:
            logger.info(f"Reverted {reverted} state fields for swipe")

    elif meta.type == "redo":
        reverted = database.revert_from_message(
            meta.session_id, meta.message_id
        )
        if reverted:
            logger.info(f"Reverted {reverted} state fields for redo")

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
    else:
        translation_pipe = TranslationPipeline(
            database, instruct_client, str(config_manager.prompts_dir_path)
        )
        try:
            world_summary = await translation_pipe.translate(meta.session_id)
        except Exception as e:
            logger.warning(f"World state translation failed: {e}")

    # ── Step 4: Inject world state context ────────────────────
    if world_summary:
        clean_messages = inject_world_state_context(
            clean_messages, world_summary
        )
        logger.debug("Injected world state context into messages")

    # ── Step 5: Format with template ───────────────────────────
    formatted_messages = format_messages(clean_messages, cfg.rp_template)

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
            except Exception as e:
                logger.warning(f"Thinking pipeline failed: {e}")

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
        """SSE generator that streams RP LLM output, then optionally refines."""
        full_response = []

        try:
            async for chunk in rp_client.chat_stream(
                messages=formatted_messages,
                temperature=stream_temperature,
                max_tokens=stream_max_tokens,
            ):
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

            response_text = "".join(full_response)

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
                except Exception as e:
                    logger.warning(f"Refinement pipeline failed: {e}")

            # ── Log message to DB ──────────────────────────────
            database.log_message(
                meta.session_id, meta.message_id, meta.swipe_index,
                "assistant", response_text,
            )

            # ── Step 9: Background state extraction ────────────
            if cfg.instruct_llm_disabled:
                logger.info("[INSTRUCT LLM DISABLED] Skipping state extraction")
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
                        result = await extraction_pipe.run(
                            session_id=meta.session_id,
                            message_id=meta.message_id,
                            swipe_index=meta.swipe_index,
                            assistant_response=response_text,
                            conversation_context=conv_context,
                        )
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

                task = asyncio.create_task(_extract_state())
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)

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