#!/usr/bin/env python3
"""
parser.py — [SYSTEM_META] tag extraction and validation.

The SillyTavern extension injects a metadata tag as messages[0]:
  [SYSTEM_META] session_id=abc-123 message_id=5 type=new swipe_index=0

This module parses it into a typed object and strips it from the
message array before forwarding to the RP LLM.
"""

import re
import logging
from typing import Optional, List, Tuple
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class SystemMeta(BaseModel):
    """Parsed [SYSTEM_META] tag contents."""

    session_id: str = Field(..., description="UUID identifying the chat session")
    message_id: int = Field(..., ge=0, description="Sequential message counter")
    type: str = Field(
        ...,
        description="Message type: new, continue, swipe, redo"
    )
    swipe_index: int = Field(0, ge=0, description="Current swipe position")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        valid = {"new", "continue", "swipe", "redo"}
        v = v.lower().strip()
        if v not in valid:
            logger.warning(f"Unknown message type '{v}', defaulting to 'new'")
            return "new"
        return v


# Regex pattern to match the [SYSTEM_META] tag
META_PATTERN = re.compile(
    r"\[SYSTEM_META\]\s*"
    r"session_id=(?P<session_id>\S+)\s+"
    r"message_id=(?P<message_id>\d+)\s+"
    r"type=(?P<type>\w+)\s+"
    r"swipe_index=(?P<swipe_index>\d+)"
)


def parse_system_meta(content: str) -> Optional[SystemMeta]:
    """Extract [SYSTEM_META] from a message string.

    Searches the entire content for the tag (it may appear at the
    beginning of a system message injected by the extension).

    Returns SystemMeta if found, None otherwise.
    """
    match = META_PATTERN.search(content)
    if not match:
        return None

    try:
        meta = SystemMeta(
            session_id=match.group("session_id"),
            message_id=int(match.group("message_id")),
            type=match.group("type"),
            swipe_index=int(match.group("swipe_index")),
        )
        logger.debug(
            f"Parsed meta: session={meta.session_id}, "
            f"msg={meta.message_id}, type={meta.type}, "
            f"swipe={meta.swipe_index}"
        )
        return meta
    except Exception as e:
        logger.error(f"Failed to parse [SYSTEM_META]: {e}")
        return None


def strip_meta_from_messages(
    messages: List[dict],
) -> Tuple[Optional[SystemMeta], List[dict]]:
    """Parse [SYSTEM_META] from the first message and remove it.

    The extension injects the meta tag as messages[0] (role: system).
    This function:
      1. Checks messages[0] for the [SYSTEM_META] tag
      2. Parses it into a SystemMeta object
      3. Removes the meta message from the array
      4. Returns (meta, cleaned_messages)

    If no meta tag is found, returns (None, messages unchanged).
    """
    if not messages:
        return None, messages

    first_msg = messages[0]

    # Only check the first message (that's where the extension injects it)
    if first_msg.get("role") != "system":
        return None, messages

    content = first_msg.get("content", "")
    meta = parse_system_meta(content)

    if meta:
        logger.info(f"Stripped [SYSTEM_META] from messages[0] → {meta.type}")
        cleaned = messages[1:]  # Remove the meta message entirely
        return meta, cleaned

    return None, messages