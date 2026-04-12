#!/usr/bin/env python3
"""
parser.py — [SYSTEM_META] tag extraction and validation.

The SillyTavern extension injects a metadata tag as messages[0]:
  [SYSTEM_META]
  session_id=abc-123
  message_id=5
  type=new
  swipe_index=0
  character_name=Lyra
  persona_name=Marcus
  mode=character
  tracked=Lyra,Kai
  [/SYSTEM_META]

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
    character_name: str = Field("", description="Active character name (from context.name2)")
    persona_name: str = Field("", description="User persona name (from context.name1)")
    mode: str = Field("character", description="Card mode: character or scenario")
    tracked: str = Field("", description="Comma-separated list of tracked character names")

    @property
    def tracked_list(self) -> List[str]:
        """Parse the tracked string into a list of names."""
        if not self.tracked or self.tracked.strip() == "":
            return []
        return [name.strip() for name in self.tracked.split(",") if name.strip()]

    @property
    def is_scenario(self) -> bool:
        """Whether this is a scenario card (vs character card)."""
        return self.mode == "scenario"

    @property
    def is_multi_character(self) -> bool:
        """Whether multiple characters are being tracked."""
        return len(self.tracked_list) > 1

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        valid = {"new", "continue", "swipe", "redo"}
        v = v.lower().strip()
        if v not in valid:
            logger.warning(f"Unknown message type '{v}', defaulting to 'new'")
            return "new"
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        valid = {"character", "scenario"}
        v = v.lower().strip()
        if v not in valid:
            logger.warning(f"Unknown mode '{v}', defaulting to 'character'")
            return "character"
        return v


# Regex pattern to match the [SYSTEM_META] tag (multi-line format)
# Supports both old single-line and new multi-line formats
META_PATTERN = re.compile(
    r"\[SYSTEM_META\]\s*"
    r"(?:\n|\s)*"
    r"session_id=(?P<session_id>\S+)\s*"
    r"(?:\n|\s)+"
    r"message_id=(?P<message_id>\d+)\s*"
    r"(?:\n|\s)+"
    r"type=(?P<type>\w+)\s*"
    r"(?:\n|\s)+"
    r"swipe_index=(?P<swipe_index>\d+)"
    r"(?:\s*\n\s*character_name=(?P<character_name>[^\n]+))?"
    r"(?:\s*\n\s*persona_name=(?P<persona_name>[^\n]+))?"
    r"(?:\s*\n\s*mode=(?P<mode>[^\n]+))?"
    r"(?:\s*\n\s*tracked=(?P<tracked>[^\n]+))?"
    r"\s*\n*\s*\[/SYSTEM_META\]",
    re.DOTALL,
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
            character_name=(match.group("character_name") or "").strip(),
            persona_name=(match.group("persona_name") or "").strip(),
            mode=(match.group("mode") or "character").strip(),
            tracked=(match.group("tracked") or "").strip(),
        )
        logger.debug(
            f"Parsed meta: session={meta.session_id}, "
            f"msg={meta.message_id}, type={meta.type}, "
            f"swipe={meta.swipe_index}, char={meta.character_name}, "
            f"persona={meta.persona_name}, mode={meta.mode}, "
            f"tracked={meta.tracked_list}"
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
        logger.info(
            f"Stripped [SYSTEM_META] from messages[0] → "
            f"{meta.type}, char={meta.character_name}, mode={meta.mode}"
        )
        cleaned = messages[1:]  # Remove the meta message entirely
        return meta, cleaned

    return None, messages