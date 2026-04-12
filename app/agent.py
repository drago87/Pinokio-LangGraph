#!/usr/bin/env python3
"""
agent.py — LangGraph-based orchestration for Agent-StateSync.

Implements two LangGraph pipelines:

1. State Extraction Pipeline (runs after each RP LLM response):
   format_prompt → extract_state → validate_changes → update_database
   
2. World State Translation Pipeline (runs before each RP LLM request):
   load_state → format_narrative → return_summary

Also handles the thinking/refinement pipeline for the RP LLM:
- Thinking steps: internal planning passes before generating
- Refinement steps: post-generation review and improvement

All pipelines are async and designed to run as background tasks
so they don't block the SSE stream to SillyTavern.
"""

import json
import logging
from pathlib import Path
from typing import TypedDict, Optional, Dict, Any, List, Callable
from datetime import datetime

from langgraph.graph import StateGraph, END

from db import Database
from llm_client import LLMClient

logger = logging.getLogger(__name__)


# ── State Types ───────────────────────────────────────────────

class ExtractionState(TypedDict):
    """State for the state extraction pipeline."""
    session_id: str
    message_id: int
    swipe_index: int
    assistant_response: str
    conversation_context: str
    world_state: Dict[str, Any]
    formatted_messages: List[Dict[str, str]]
    raw_response: str
    extracted_changes: Dict[str, Any]
    success: bool
    error: Optional[str]
    changes_applied: int
    # New fields for mode-aware extraction
    mode: str                    # "character" or "scenario"
    character_name: str          # Active character name (from extension)
    persona_name: str            # User persona name
    tracked_characters: List[str]  # Characters to track (for multi-char cards)
    character_description: str   # Character card description (for initial extraction)
    is_initial: bool             # Whether this is an initial card extraction vs ongoing


class TranslationState(TypedDict):
    """State for the world state → narrative translation pipeline."""
    session_id: str
    world_state: Dict[str, Any]
    session_meta: Optional[dict]
    formatted_messages: List[Dict[str, str]]
    narrative_summary: str
    success: bool


class ThinkingState(TypedDict):
    """State for the thinking/refinement pipeline."""
    messages: List[Dict[str, str]]
    thinking_notes: List[str]
    refined_response: str
    pass_number: int
    max_passes: int


# ── Prompt Loaders ────────────────────────────────────────────

def load_prompt_file(prompt_path: str, fallback: str) -> str:
    """Load a prompt template from a YAML or text file.

    Falls back to the provided default string if the file doesn't exist.
    """
    from pathlib import Path

    path = Path(prompt_path)
    if path.exists():
        try:
            content = path.read_text(encoding="utf-8")
            # If it's YAML, extract the system_prompt field
            if path.suffix in (".yml", ".yaml"):
                try:
                    import yaml
                    data = yaml.safe_load(content)
                    if isinstance(data, dict) and "system_prompt" in data:
                        return data["system_prompt"]
                    elif isinstance(data, str):
                        return data
                except ImportError:
                    pass
            return content.strip()
        except Exception as e:
            logger.warning(f"Failed to load prompt from {prompt_path}: {e}")

    return fallback


# ── State Extraction Pipeline ─────────────────────────────────

class ExtractionPipeline:
    """LangGraph pipeline for extracting world state changes from
    the RP LLM's narrative response.

    Flow:
        format_prompt → extract_state → validate_changes → update_database
    """

    def __init__(self, db: Database, instruct_client: LLMClient, prompts_dir: str):
        self.db = db
        self.instruct_client = instruct_client
        self.prompts_dir = prompts_dir
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        workflow = StateGraph(ExtractionState)

        workflow.add_node("format_prompt", self._format_prompt)
        workflow.add_node("extract_state", self._extract_state)
        workflow.add_node("validate_changes", self._validate_changes)
        workflow.add_node("update_database", self._update_database)

        workflow.set_entry_point("format_prompt")
        workflow.add_edge("format_prompt", "extract_state")
        workflow.add_edge("extract_state", "validate_changes")

        workflow.add_conditional_edges(
            "validate_changes",
            self._route_validation,
            {
                "valid": "update_database",
                "invalid": END,
            },
        )
        workflow.add_edge("update_database", END)

        return workflow.compile()

    def _select_prompt_file(self, mode: str) -> str:
        """Select the extraction prompt based on session mode.

        Priority:
          1. {prompts_dir}/instruct/extract_{mode}.yaml  (mode-specific)
          2. {prompts_dir}/instruct/default.yaml          (fallback)
          3. hardcoded default                          (last resort)
        """
        # Try mode-specific prompt first
        mode_path = f"{self.prompts_dir}/instruct/extract_{mode}.yaml"
        if Path(mode_path).exists():
            logger.info(f"Using mode-specific extraction prompt: extract_{mode}.yaml")
            return load_prompt_file(mode_path, self._default_instruct_prompt())

        # Fallback to default
        logger.info(f"No extract_{mode}.yaml found, falling back to default.yaml")
        return load_prompt_file(
            f"{self.prompts_dir}/instruct/default.yaml",
            self._default_instruct_prompt(),
        )

    def _format_prompt(self, state: ExtractionState) -> dict:
        """Build the extraction prompt for the Instruct LLM.

        Selects the correct prompt based on mode (character/scenario),
        injects current world state, and includes character context
        for multi-character and initial extractions.
        """
        # Load current world state from DB
        world_state = self.db.get_world_state(state["session_id"])
        state["world_state"] = world_state

        mode = state.get("mode", "character") or "character"

        # Load mode-appropriate prompt
        system_prompt = self._select_prompt_file(mode)

        # Inject current world state into the prompt
        state_json = json.dumps(world_state, indent=2, ensure_ascii=False) if world_state else "{}"

        # Build character context header
        char_context = ""
        character_name = state.get("character_name", "")
        persona_name = state.get("persona_name", "")
        tracked = state.get("tracked_characters", [])
        is_initial = state.get("is_initial", False)

        if mode == "scenario":
            char_context = "\nMODE: Scenario extraction. Track world-building, factions, plot, and characters dynamically."
        else:
            if tracked and len(tracked) > 1:
                char_context = f"\nMODE: Multi-character extraction. Track these characters: {', '.join(tracked)}"
            elif character_name:
                char_context = f"\nMODE: Single-character extraction. Primary character: {character_name}"

        if persona_name:
            char_context += f"\nPlayer persona: {persona_name}"

        if is_initial:
            char_context += "\nThis is an INITIAL extraction from the character/scenario card. Extract as much relevant state as possible."

        # Build the system message with world state injected
        system_content = f"{system_prompt}{char_context}\n\nCurrent world state:\n```json\n{state_json}\n```"

        # Build the user message
        if is_initial:
            # For initial extraction, the assistant_response contains card data
            char_desc = state.get("character_description", "")
            user_content = (
                f"This is an initial extraction from a character or scenario card. "
                f"Extract the starting world state from the following card data.\n\n"
                f"--- Card Data ---\n{state['assistant_response']}\n"
                f"--- End Card Data ---\n\n"
                f"{state.get('conversation_context', '')}"
            )
        else:
            user_content = (
                f"Analyze the following narrative response and extract world state changes.\n\n"
                f"--- Narrative Response ---\n{state['assistant_response']}\n"
                f"--- End Response ---\n\n"
                f"{state.get('conversation_context', '')}"
            )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        state["formatted_messages"] = messages
        return state

    def _extract_state(self, state: ExtractionState) -> dict:
        """Call the Instruct LLM to extract state changes."""
        import asyncio

        messages = state.get("formatted_messages", [])

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context — use create_task
                # This node is called from async code, so we need a workaround
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    raw = pool.submit(
                        asyncio.run,
                        self.instruct_client.chat_complete(
                            messages=messages,
                            temperature=0.1,
                            max_tokens=2048,
                        ),
                    ).result(timeout=120)
            else:
                raw = asyncio.run(
                    self.instruct_client.chat_complete(
                        messages=messages,
                        temperature=0.1,
                        max_tokens=2048,
                    )
                )

            state["raw_response"] = raw

            # Parse JSON from the response
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                # Strip markdown code fences
                lines = cleaned.split("\n")
                # Remove first line (```json or ```) and last line (```)
                if lines[0].strip().startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()

            # Try to find JSON object in the response
            changes = {}
            try:
                changes = json.loads(cleaned)
                if not isinstance(changes, dict):
                    logger.warning(f"Instruct LLM returned non-dict: {type(changes)}")
                    changes = {}
            except json.JSONDecodeError:
                # Try to find a JSON object within the text
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        changes = json.loads(cleaned[start : end + 1])
                        if not isinstance(changes, dict):
                            changes = {}
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Could not parse JSON from Instruct LLM response: "
                            f"{cleaned[:200]}"
                        )

            state["extracted_changes"] = changes
            state["success"] = bool(changes)

        except Exception as e:
            logger.error(f"State extraction failed: {e}")
            state["extracted_changes"] = {}
            state["success"] = False
            state["error"] = str(e)

        return state

    def _validate_changes(self, state: ExtractionState) -> dict:
        """Validate extracted changes before applying."""
        changes = state.get("extracted_changes", {})

        if not changes or not isinstance(changes, dict):
            state["success"] = False
            return state

        # Filter out nonsensical keys
        valid_keys = {}
        for k, v in changes.items():
            # Skip empty values and internal markers
            if v is None:
                continue
            if k.startswith("_"):
                continue
            # Validate value types (allow str, int, float, bool, list, dict)
            if isinstance(v, (str, int, float, bool, list, dict)):
                valid_keys[k] = v
            else:
                valid_keys[k] = str(v)

        state["extracted_changes"] = valid_keys
        state["success"] = bool(valid_keys)
        return state

    def _route_validation(self, state: ExtractionState) -> str:
        """Route to database update if changes are valid."""
        return "valid" if state.get("success") else "invalid"

    def _update_database(self, state: ExtractionState) -> dict:
        """Apply validated changes to the session database."""
        changes = state.get("extracted_changes", {})
        count = self.db.update_world_state(
            session_id=state["session_id"],
            changes=changes,
            message_id=state["message_id"],
            swipe_index=state["swipe_index"],
        )
        state["changes_applied"] = count
        logger.info(
            f"Extraction complete: {count} fields updated "
            f"(msg={state['message_id']}, swipe={state['swipe_index']})"
        )
        return state

    async def run(
        self,
        session_id: str,
        message_id: int,
        swipe_index: int,
        assistant_response: str,
        conversation_context: str = "",
        mode: str = "character",
        character_name: str = "",
        persona_name: str = "",
        tracked_characters: List[str] = None,
        character_description: str = "",
        is_initial: bool = False,
    ) -> Dict[str, Any]:
        """Execute the full extraction pipeline asynchronously.

        Args:
            session_id: Session UUID
            message_id: Message counter
            swipe_index: Current swipe position
            assistant_response: The RP LLM's narrative output (or card data for initial)
            conversation_context: Additional context (e.g. latest user message)
            mode: "character" or "scenario" — selects the extraction prompt
            character_name: Active character name from extension
            persona_name: User persona name from extension
            tracked_characters: List of character names to track (multi-char)
            character_description: Full character description (for initial extraction)
            is_initial: True if this is extracting from the card, False for ongoing
        """
        initial_state = ExtractionState(
            session_id=session_id,
            message_id=message_id,
            swipe_index=swipe_index,
            assistant_response=assistant_response,
            conversation_context=conversation_context,
            world_state={},
            formatted_messages=[],
            raw_response="",
            extracted_changes={},
            success=False,
            error=None,
            changes_applied=0,
            mode=mode,
            character_name=character_name,
            persona_name=persona_name,
            tracked_characters=tracked_characters or [],
            character_description=character_description,
            is_initial=is_initial,
        )

        try:
            result = await self.graph.ainvoke(initial_state)
            return dict(result)
        except Exception as e:
            logger.error(f"Extraction pipeline error: {e}")
            return {
                "success": False,
                "error": str(e),
                "changes_applied": 0,
            }

    @staticmethod
    def _default_instruct_prompt() -> str:
        return (
            "You are a state extraction engine for a roleplay system. "
            "Analyze the narrative response and extract any CHANGES to "
            "the world state as a JSON object.\n\n"
            "Rules:\n"
            "- Only extract FACTUAL changes: new locations, time passage, "
            "character states, acquired items, relationships, injuries, etc.\n"
            "- Do NOT extract dialogue, thoughts, or narrative descriptions.\n"
            "- Use descriptive lowercase_snake_case keys.\n"
            "- Values should be concise: short strings, numbers, or booleans.\n"
            "- If nothing meaningfully changed, return {}\n"
            "- Return ONLY valid JSON, no explanation."
        )


# ── World State Translation Pipeline ─────────────────────────

class TranslationPipeline:
    """Translates JSON world state into natural language context
    for injection into the RP LLM's prompt.

    This prevents the RP LLM from seeing raw JSON (which causes
    meta-commentary about data structures) and instead gives it
    a narrative summary of the current world state.
    """

    def __init__(self, db: Database, instruct_client: LLMClient, prompts_dir: str):
        self.db = db
        self.instruct_client = instruct_client
        self.prompts_dir = prompts_dir

    async def translate(self, session_id: str) -> str:
        """Translate the session's world state into a narrative summary.

        Returns an empty string if there's no world state to translate,
        which signals the caller to skip injection.
        """
        world_state = self.db.get_world_state(session_id)

        if not world_state:
            return ""

        session_meta = self.db.get_session(session_id)

        # Load translation prompt
        system_prompt = load_prompt_file(
            f"{self.prompts_dir}/world_state/translate.yaml",
            self._default_translation_prompt(),
        )

        state_json = json.dumps(world_state, indent=2, ensure_ascii=False)

        # Build context about the character for the translator
        char_info = ""
        if session_meta:
            name = session_meta.get("character_name", "")
            desc = session_meta.get("character_description", "")[:500]
            if name:
                char_info = f"\nCharacter: {name}\n{desc}" if desc else f"\nCharacter: {name}"

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Current world state:\n```json\n{state_json}\n```"
                    f"{char_info}\n\n"
                    f"Provide a brief narrative summary of the current state."
                ),
            },
        ]

        try:
            summary = await self.instruct_client.chat_complete(
                messages=messages,
                temperature=0.3,
                max_tokens=512,
            )
            return summary.strip()
        except Exception as e:
            logger.warning(f"World state translation failed: {e}")
            # Fallback: return a simple key-value summary instead of narrative
            return self._simple_fallback(world_state)

    @staticmethod
    def _simple_fallback(world_state: Dict[str, Any]) -> str:
        """Generate a simple key-value summary if the LLM translation fails."""
        lines = []
        for key, value in world_state.items():
            lines.append(f"- {key.replace('_', ' ').title()}: {value}")
        return "Current state:\n" + "\n".join(lines) if lines else ""

    @staticmethod
    def _default_translation_prompt() -> str:
        return (
            "You are a context summarizer for a creative roleplay system. "
            "Convert the JSON world state into a brief, natural language "
            "summary suitable for injecting into a writer's context.\n\n"
            "Guidelines:\n"
            "- Write in third person\n"
            "- Cover current location, time, key character states\n"
            "- Be concise (2-3 paragraphs maximum)\n"
            "- Read naturally, not like a data dump\n"
            "- Only include information relevant to the current scene\n"
            "- Do not add new information not present in the state"
        )


# ── Thinking / Refinement Helpers ────────────────────────────

async def run_thinking_passes(
    rp_client: LLMClient,
    messages: List[Dict[str, str]],
    num_passes: int,
) -> List[str]:
    """Run internal thinking/planning passes before generation.

    Each pass asks the RP LLM to plan its response based on context.
    The thinking notes are collected and can be prepended to the
    actual generation prompt.

    Returns a list of thinking note strings (one per pass).
    """
    if num_passes <= 0:
        return []

    notes = []
    for i in range(num_passes):
        thinking_prompt = [
            {
                "role": "system",
                "content": (
                    "You are planning a narrative response. Analyze the context "
                    "and outline your planned response. Consider:\n"
                    "- What just happened and what should happen next\n"
                    "- Character consistency and voice\n"
                    "- Scene pacing and sensory details to include\n"
                    "- Any world state elements that should be reflected\n\n"
                    "Keep your plan brief (3-5 sentences). Do not write the "
                    "actual response yet."
                ),
            },
            *messages[-4:],  # Last few messages for context
        ]

        try:
            note = await rp_client.chat_complete(
                messages=thinking_prompt,
                temperature=0.4,
                max_tokens=256,
            )
            notes.append(note.strip())
        except Exception as e:
            logger.warning(f"Thinking pass {i + 1} failed: {e}")
            break

    return notes


async def run_refinement_passes(
    rp_client: LLMClient,
    messages: List[Dict[str, str]],
    initial_response: str,
    num_passes: int,
) -> str:
    """Run post-generation refinement passes.

    Each pass asks the RP LLM to review and improve the generated response.
    The improved version replaces the original for the final output.

    Returns the refined response string.
    """
    if num_passes <= 0:
        return initial_response

    current = initial_response
    for i in range(num_passes):
        refine_prompt = [
            {
                "role": "system",
                "content": (
                    "You are refining a narrative response. Review the "
                    "following text and produce an improved version that:\n"
                    "- Has better prose quality and flow\n"
                    "- Maintains character voice consistently\n"
                    "- Includes more vivid sensory details\n"
                    "- Fixes any logical inconsistencies\n\n"
                    "Output ONLY the improved text. Do not add commentary."
                ),
            },
            {
                "role": "user",
                "content": f"Refine this response:\n\n{current}",
            },
        ]

        try:
            current = await rp_client.chat_complete(
                messages=refine_prompt,
                temperature=0.6,
                max_tokens=2048,
            )
            current = current.strip()
        except Exception as e:
            logger.warning(f"Refinement pass {i + 1} failed: {e}")
            break

    return current