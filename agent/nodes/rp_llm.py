"""
Node: Call the RP LLM (Roleplay Language Model).

Takes the enriched prompt (user message + DB context) and sends it
to the creative RP model running on KoboldCPP, Ollama, etc.
"""

import httpx
from agent.config import (
    RP_LLM_BASE_URL, RP_LLM_API_KEY, RP_LLM_MODEL, RP_LLM_BACKEND
)


async def call_rp_llm(state: dict) -> dict:
    """
    Graph Node: Call the RP LLM with the enriched prompt.
    
    The prompt includes:
    - System prompt (character card info from ST)
    - DB context (character state, past events)
    - Conversation history
    
    Supports backends:
    - "kobold": KoboldCPP native API
    - "openai_compatible": Ollama / OpenAI-compatible API
    
    Args:
        state: Current agent state (must have messages and db_context).
    
    Returns:
        Updated state with rp_response populated.
    """
    messages = state.get("messages", [])
    db_context = state.get("db_context")
    debug = state.get("debug_info", {})
    
    # Build the messages payload
    api_messages = []
    
    # System message with DB context injected
    system_content = ""
    if db_context and db_context.get("summary"):
        system_content = (
            "[WORLD STATE AND CHARACTER INFORMATION - use this to stay consistent]\n"
            f"{db_context['summary']}\n\n"
            "[END OF WORLD STATE]\n\n"
        )
    
    for msg in messages:
        role = msg.type if hasattr(msg, 'type') else "user"
        content = msg.content if hasattr(msg, 'content') else str(msg)
        
        if role == "system" and content:
            # Prepend DB context to the system message
            api_messages.append({
                "role": "system",
                "content": system_content + content
            })
        elif role in ("user", "assistant", "system"):
            api_messages.append({
                "role": role,
                "content": str(content)
            })
    
    # If no system message was found, add context as one
    if not any(m["role"] == "system" for m in api_messages) and system_content:
        api_messages.insert(0, {"role": "system", "content": system_content})
    
    # Call the appropriate backend
    try:
        if RP_LLM_BACKEND == "kobold":
            rp_response = await _call_kobold(api_messages)
        else:
            rp_response = await _call_openai_compatible(api_messages, RP_LLM_BASE_URL, RP_LLM_API_KEY, RP_LLM_MODEL)
        
    except Exception as e:
        rp_response = f"[ERROR calling RP LLM: {str(e)}]"
        debug["rp_llm_error"] = str(e)
    
    return {
        **state,
        "rp_response": rp_response,
        "debug_info": {**debug, "rp_llm_backend": RP_LLM_BACKEND, "rp_model": RP_LLM_MODEL}
    }


async def _call_kobold(messages: list) -> str:
    """Call KoboldCPP's /api/v1/generate endpoint."""
    # Convert messages to a single prompt string
    prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            prompt += f"{msg['content']}\n\n"
        elif msg["role"] == "user":
            prompt += f"### User:\n{msg['content']}\n\n"
        elif msg["role"] == "assistant":
            prompt += f"### Assistant:\n{msg['content']}\n\n"
    prompt += "### Assistant:\n"
    
    payload = {
        "prompt": prompt,
        "max_context_length": 4096,
        "max_length": 512,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
    }
    
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{RP_LLM_BASE_URL}/api/v1/generate",
            json=payload
        )
        response.raise_for_status()
        data = response.json()
    
    return data.get("results", [{}])[0].get("text", "").strip()


async def _call_openai_compatible(messages: list, base_url: str, api_key: str, model: str) -> str:
    """Call an OpenAI-compatible API (Ollama, vLLM, etc.)."""
    headers = {}
    if api_key and api_key != "none":
        headers["Authorization"] = f"Bearer {api_key}"
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.8,
    }
    
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
    
    return data["choices"][0]["message"]["content"].strip()
