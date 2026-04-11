#!/usr/bin/env python3
"""
templates.py — Message template formatters for Agent-StateSync.

Supports: ChatML, Llama 3, Alpaca, Mistral, Raw.
"""

import logging
from typing import List, Dict, Any, Callable

logger = logging.getLogger(__name__)


# ── Template Formatters ───────────────────────────────────────

def format_chatml(messages: List[Dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def format_llama3(messages: List[Dict[str, str]]) -> str:
    parts = ["<|begin_of_text|>"]
    system_parts = []
    conversation_parts = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_parts.append(content)
        elif role == "user":
            conversation_parts.append(
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{content}<|eot_id|>"
            )
        elif role == "assistant":
            conversation_parts.append(
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{content}<|eot_id|>"
            )

    if system_parts:
        sys_content = "\n\n".join(system_parts)
        parts.append(
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{sys_content}<|eot_id|>"
        )

    parts.extend(conversation_parts)
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "\n".join(parts)


def format_alpaca(messages: List[Dict[str, str]]) -> str:
    system_parts = []
    exchanges = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_parts.append(content)
        elif role == "user":
            exchanges.append(("user", content))
        elif role == "assistant":
            exchanges.append(("assistant", content))

    parts = []

    instruction = "\n\n".join(system_parts) if system_parts else ""
    if instruction:
        parts.append(f"### Instruction:\n{instruction}")

    user_content = []
    for role, content in exchanges:
        if role == "user":
            user_content.append(content)
        elif role == "assistant":
            if user_content:
                input_block = "\n\n".join(user_content)
                parts.append(f"### Input:\n{input_block}")
                parts.append(f"### Response:\n{content}")
                user_content = []

    if user_content:
        input_block = "\n\n".join(user_content)
        parts.append(f"### Input:\n{input_block}")

    parts.append("### Response:\n")
    return "\n\n".join(parts)


def format_mistral(messages: List[Dict[str, str]]) -> str:
    parts = ["<s>"]
    system_content = ""
    first_user = True

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_content = content
        elif role == "user":
            if first_user and system_content:
                parts.append(
                    f"[INST] {system_content}\n\n{content} [/INST]"
                )
                system_content = ""
                first_user = False
            else:
                parts.append(f"</s><s>[INST] {content} [/INST]")
                first_user = False
        elif role == "assistant":
            parts.append(content)

    return "".join(parts)


# ── Registry ──────────────────────────────────────────────────

TEMPLATE_REGISTRY: Dict[str, Callable[[List[Dict[str, str]]], str]] = {
    "chatml": format_chatml,
    "llama3": format_llama3,
    "alpaca": format_alpaca,
    "mistral": format_mistral,
}


def format_messages(
    messages: List[Dict[str, str]], template: str
) -> List[Dict[str, str]]:
    if not template or template.lower() == "raw":
        return messages

    formatter = TEMPLATE_REGISTRY.get(template.lower())
    if formatter:
        formatted = formatter(messages)
        logger.debug(
            f"Formatted {len(messages)} messages with '{template}' "
            f"-> {len(formatted)} chars"
        )
        return [{"role": "user", "content": formatted}]

    logger.warning(f"Unknown template '{template}', sending messages as-is")
    return messages


def inject_world_state_context(
    messages: List[Dict[str, str]],
    world_state_summary: str,
) -> List[Dict[str, str]]:
    if not world_state_summary or not world_state_summary.strip():
        return messages

    ws_msg = {
        "role": "system",
        "content": (
            f"[World State Context — do not reference this directly in your "
            f"response, use it for consistency]\n\n{world_state_summary}"
        ),
    }

    insert_idx = 0
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            insert_idx = i + 1
        else:
            break

    result = messages.copy()
    result.insert(insert_idx, ws_msg)
    return result