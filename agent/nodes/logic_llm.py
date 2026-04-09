"""
Node: Call the Logic LLM for state extraction.

Sends the RP response to a smaller, cheaper model that extracts
structured state changes (location updates, new knowledge, etc.)
as JSON. This JSON is then used to update the database.
"""

import json
import httpx
from agent.config import (
    LOGIC_LLM_BASE_URL, LOGIC_LLM_API_KEY, LOGIC_LLM_MODEL, LOGIC_LLM_BACKEND
)

# The system prompt that makes the Logic LLM output structured JSON
LOGIC_SYSTEM_PROMPT = """You are a state extraction engine for a roleplaying game. Your ONLY job is to analyze the latest roleplay response and extract any changes to character state.

You MUST respond with ONLY a valid JSON object (no markdown, no explanation, no extra text).

The JSON must follow this exact structure:
{
  "changes": [
    {
      "character_id": "character_name_or_id",
      "category": "location|knowledge|relationship|inventory|status|personality",
      "action": "set|add|remove",
      "key": "short_identifier",
      "value": "the actual value",
      "reasoning": "brief explanation of why this change",
      "confidence": 0.9
    }
  ],
  "narrative_summary": "One sentence summary of what happened"
}

RULES:
- Only extract changes that are EXPLICITLY stated or clearly implied in the text
- Do NOT hallucinate changes
- If nothing changed, return {"changes": [], "narrative_summary": "No significant state changes"}
- character_id should match the character involved (use "unknown" if not specified)
- Categories: "location" (where they are), "knowledge" (what they learned), "relationship" (how they feel about others), "inventory" (items), "status" (health, mood, condition), "personality" (character traits shown)
- confidence: 1.0 if explicitly stated, 0.7-0.8 if implied, 0.5 if uncertain
- Keep the narrative_summary under 50 words"""


async def call_logic_llm(state: dict) -> dict:
    """
    Graph Node: Extract state changes from the RP response using the Logic LLM.
    
    This runs AFTER the RP LLM has generated a response. The Logic LLM
    analyzes the narrative and outputs structured JSON with state changes.
    
    Args:
        state: Current agent state (must have rp_response).
    
    Returns:
        Updated state with logic_output populated.
    """
    rp_response = state.get("rp_response", "")
    character_id = state.get("character_id", "unknown")
    debug = state.get("debug_info", {})
    
    if not rp_response or rp_response.startswith("[ERROR"):
        return {
            **state,
            "logic_output": {"changes": [], "narrative_summary": "No response to analyze"},
            "debug_info": {**debug, "logic_llm_skipped": True}
        }
    
    # Build the extraction prompt
    user_prompt = (
        f"Character ID: {character_id}\n\n"
        f"Latest roleplay response:\n"
        f"---\n{rp_response}\n---\n\n"
        f"Extract all state changes from this response as JSON."
    )
    
    try:
        raw_output = await _call_logic_model(
            system=LOGIC_SYSTEM_PROMPT,
            user=user_prompt
        )
        
        # Parse the JSON (handle markdown code blocks)
        logic_output = _parse_json_response(raw_output)
        
    except Exception as e:
        logic_output = {"changes": [], "narrative_summary": f"Failed to extract state: {str(e)}"}
        debug["logic_llm_error"] = str(e)
    
    return {
        **state,
        "logic_output": logic_output,
        "debug_info": {
            **debug,
            "logic_llm_backend": LOGIC_LLM_BACKEND,
            "logic_model": LOGIC_LLM_MODEL,
            "state_changes_found": len(logic_output.get("changes", []))
        }
    }


async def _call_logic_model(system: str, user: str) -> str:
    """Call the Logic LLM via OpenAI-compatible API."""
    headers = {}
    if LOGIC_LLM_API_KEY and LOGIC_LLM_API_KEY != "none":
        headers["Authorization"] = f"Bearer {LOGIC_LLM_API_KEY}"
    
    payload = {
        "model": LOGIC_LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "max_tokens": 2048,
        "temperature": 0.1,  # Low temperature for deterministic extraction
    }
    
    # Ollama uses /api/chat, others use /v1/chat/completions
    if LOGIC_LLM_BACKEND == "ollama":
        url = f"{LOGIC_LLM_BASE_URL.rstrip('/')}/api/chat"
        payload["stream"] = False
        payload["format"] = "json"
    else:
        url = f"{LOGIC_LLM_BASE_URL.rstrip('/')}/v1/chat/completions"
    
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
    
    if LOGIC_LLM_BACKEND == "ollama":
        return data.get("message", {}).get("content", "").strip()
    else:
        return data["choices"][0]["message"]["content"].strip()


def _parse_json_response(raw: str) -> dict:
    """
    Parse JSON from the Logic LLM output, handling:
    - Plain JSON
    - JSON wrapped in ```json ... ``` code blocks
    - JSON with leading/trailing text
    """
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from code block
    if "```json" in raw:
        start = raw.index("```json") + 7
        end = raw.index("```", start)
        return json.loads(raw[start:end].strip())
    
    if "```" in raw:
        start = raw.index("```") + 3
        end = raw.index("```", start)
        return json.loads(raw[start:end].strip())
    
    # Try finding JSON object in the text
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(raw[start:end])
    
    raise ValueError(f"Could not parse JSON from Logic LLM output: {raw[:200]}")
