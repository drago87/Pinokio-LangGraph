#!/usr/bin/env python3
"""
config.py — Settings management for Agent-StateSync.

Loads configuration from (in priority order):
  1. SillyTavern extension runtime sync (POST /api/config)
  2. config.ini (created by Pinokio install.json)
  3. Built-in defaults

All settings live in config.ini with human-readable sections and keys.

Config priority for URLs:
  The extension sends URLs on every first request. Those overwrite whatever
  is in config.ini. If the extension never connects (or sends nothing),
  the config.ini values are used as the fallback.
"""

import configparser
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ── APP_DIR: always resolves to the app/ folder ──────────────
# server.py is at app/server.py, so __file__ gives us the app/ dir.
# All relative paths (config.ini, ./dbs, ./prompts) resolve from here.
APP_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = APP_DIR / "config.ini"

# ── Defaults ──────────────────────────────────────────────────

DEFAULTS: Dict[str, Dict[str, Any]] = {
    "agent": {
        "port": 8001,
    },
    "rp_llm": {
        "url": "http://localhost:5001",
        "backend": "kobold",
        "model": "",
        "api_key": "none",
        "disabled": "false",
    },
    "instruct_llm": {
        "url": "http://localhost:11434",
        "backend": "ollama",
        "model": "llama3",
        "api_key": "none",
        "disabled": "false",
    },
    "pipeline": {
        "thinking_steps": 0,
        "refinement_steps": 0,
        "history_count": 2,
    },
    "templates": {
        "rp": "raw",
        "instruct": "raw",
    },
    "paths": {
        "db_dir": "./dbs",
        "prompts_dir": "./prompts",
    },
    "agent_settings": {
        "context_window_messages": 20,
        "db_retrieval_count": 10,
    },
    "debug": {
        "debug_mode": "false",
        "dry_run": "false",
        "debug_stepping": "false",
    },
    "prompts": {
        "rp_prompt_file": "default.yaml",
        "instruct_prompt_file": "default.yaml",
        "world_state_prompt_file": "translate.yaml",
    },
}

_BOOL_KEYS = frozenset({
    "disabled", "debug_mode", "dry_run",
})


def _to_bool(v: str) -> bool:
    return v.strip().lower() in ("true", "1", "yes", "on")


class AgentConfig:
    """Flat config object with attribute access to all settings."""

    def __init__(self):
        self.port: int = DEFAULTS["agent"]["port"]

        # RP LLM
        self.rp_llm_url: str = DEFAULTS["rp_llm"]["url"]
        self.rp_llm_backend: str = DEFAULTS["rp_llm"]["backend"]
        self.rp_llm_model: str = DEFAULTS["rp_llm"]["model"]
        self.rp_llm_api_key: str = DEFAULTS["rp_llm"]["api_key"]
        self.rp_llm_disabled: bool = False

        # Instruct LLM
        self.instruct_llm_url: str = DEFAULTS["instruct_llm"]["url"]
        self.instruct_llm_backend: str = DEFAULTS["instruct_llm"]["backend"]
        self.instruct_llm_model: str = DEFAULTS["instruct_llm"]["model"]
        self.instruct_llm_api_key: str = DEFAULTS["instruct_llm"]["api_key"]
        self.instruct_llm_disabled: bool = False

        # Pipeline
        self.thinking_steps: int = DEFAULTS["pipeline"]["thinking_steps"]
        self.refinement_steps: int = DEFAULTS["pipeline"]["refinement_steps"]
        self.history_count: int = DEFAULTS["pipeline"]["history_count"]

        # Templates
        self.rp_template: str = DEFAULTS["templates"]["rp"]
        self.instruct_template: str = DEFAULTS["templates"]["instruct"]

        # Paths
        self.db_dir: str = DEFAULTS["paths"]["db_dir"]
        self.prompts_dir: str = DEFAULTS["paths"]["prompts_dir"]

        # Agent settings
        self.context_window_messages: int = DEFAULTS["agent_settings"]["context_window_messages"]
        self.db_retrieval_count: int = DEFAULTS["agent_settings"]["db_retrieval_count"]

        # Debug
        self.debug_mode: bool = False
        self.dry_run: bool = False
        self.debug_stepping: bool = False

        # Prompt files
        self.rp_prompt_file: str = DEFAULTS["prompts"]["rp_prompt_file"]
        self.instruct_prompt_file: str = DEFAULTS["prompts"]["instruct_prompt_file"]
        self.world_state_prompt_file: str = DEFAULTS["prompts"]["world_state_prompt_file"]

        # Internal
        self._parser: configparser.ConfigParser = None


class ConfigManager:
    """Manages agent configuration: loads from config.ini,
    syncs from extension at runtime."""

    def __init__(self):
        self.config = AgentConfig()
        self._load_config_ini()

    # ── config.ini loading ─────────────────────────────────────

    def _load_config_ini(self):
        """Read config.ini and apply values to self.config."""
        path = CONFIG_FILE  # Already a Path (APP_DIR / "config.ini")
        if not path.exists():
            logger.info(f"{path} not found — using defaults")
            return

        parser = configparser.ConfigParser(
            interpolation=configparser.BasicInterpolation(),
            default_section="agent",
        )
        parser.read(path, encoding="utf-8")
        self.config._parser = parser

        def get(section: str, key: str, default, coerce=None):
            if not parser.has_section(section):
                return default
            try:
                raw = parser.get(section, key)
                if coerce:
                    return coerce(raw)
                return raw
            except configparser.NoOptionError:
                return default

        def getbool(section: str, key: str) -> bool:
            return get(section, key, "false", _to_bool)

        # Agent
        self.config.port = get("agent", "port", self.config.port, int)

        # RP LLM
        self.config.rp_llm_url = get("rp_llm", "url", self.config.rp_llm_url)
        self.config.rp_llm_backend = get("rp_llm", "backend", self.config.rp_llm_backend)
        self.config.rp_llm_model = get("rp_llm", "model", self.config.rp_llm_model)
        self.config.rp_llm_api_key = get("rp_llm", "api_key", self.config.rp_llm_api_key)
        self.config.rp_llm_disabled = getbool("rp_llm", "disabled")

        # Instruct LLM
        self.config.instruct_llm_url = get("instruct_llm", "url", self.config.instruct_llm_url)
        self.config.instruct_llm_backend = get("instruct_llm", "backend", self.config.instruct_llm_backend)
        self.config.instruct_llm_model = get("instruct_llm", "model", self.config.instruct_llm_model)
        self.config.instruct_llm_api_key = get("instruct_llm", "api_key", self.config.instruct_llm_api_key)
        self.config.instruct_llm_disabled = getbool("instruct_llm", "disabled")

        # Pipeline
        self.config.thinking_steps = get("pipeline", "thinking_steps", self.config.thinking_steps, int)
        self.config.refinement_steps = get("pipeline", "refinement_steps", self.config.refinement_steps, int)
        self.config.history_count = get("pipeline", "history_count", self.config.history_count, int)

        # Templates
        self.config.rp_template = get("templates", "rp", self.config.rp_template)
        self.config.instruct_template = get("templates", "instruct", self.config.instruct_template)

        # Paths
        self.config.db_dir = get("paths", "db_dir", self.config.db_dir)
        self.config.prompts_dir = get("paths", "prompts_dir", self.config.prompts_dir)

        # Agent settings
        self.config.context_window_messages = get("agent_settings", "context_window_messages", self.config.context_window_messages, int)
        self.config.db_retrieval_count = get("agent_settings", "db_retrieval_count", self.config.db_retrieval_count, int)

        # Debug
        self.config.debug_mode = getbool("debug", "debug_mode")
        self.config.dry_run = getbool("debug", "dry_run")
        self.config.debug_stepping = getbool("debug", "debug_stepping")

        # Prompt files (optional section)
        self.config.rp_prompt_file = get("prompts", "rp_prompt_file", self.config.rp_prompt_file)
        self.config.instruct_prompt_file = get("prompts", "instruct_prompt_file", self.config.instruct_prompt_file)
        self.config.world_state_prompt_file = get("prompts", "world_state_prompt_file", self.config.world_state_prompt_file)

        logger.info(f"Loaded config from {CONFIG_FILE}")

    # ── Save ───────────────────────────────────────────────────

    def save(self):
        """Persist current config back to config.ini."""
        parser = configparser.ConfigParser(
            interpolation=configparser.BasicInterpolation(),
        )

        parser["agent"] = {
            "port": str(self.config.port),
        }

        parser["rp_llm"] = {
            "url": self.config.rp_llm_url,
            "backend": self.config.rp_llm_backend,
            "model": self.config.rp_llm_model,
            "api_key": self.config.rp_llm_api_key,
            "disabled": str(self.config.rp_llm_disabled).lower(),
        }

        parser["instruct_llm"] = {
            "url": self.config.instruct_llm_url,
            "backend": self.config.instruct_llm_backend,
            "model": self.config.instruct_llm_model,
            "api_key": self.config.instruct_llm_api_key,
            "disabled": str(self.config.instruct_llm_disabled).lower(),
        }

        parser["pipeline"] = {
            "thinking_steps": str(self.config.thinking_steps),
            "refinement_steps": str(self.config.refinement_steps),
            "history_count": str(self.config.history_count),
        }

        parser["templates"] = {
            "rp": self.config.rp_template,
            "instruct": self.config.instruct_template,
        }

        parser["paths"] = {
            "db_dir": self.config.db_dir,
            "prompts_dir": self.config.prompts_dir,
        }

        parser["agent_settings"] = {
            "context_window_messages": str(self.config.context_window_messages),
            "db_retrieval_count": str(self.config.db_retrieval_count),
        }

        parser["debug"] = {
            "debug_mode": str(self.config.debug_mode).lower(),
            "dry_run": str(self.config.dry_run).lower(),
            "debug_stepping": str(self.config.debug_stepping).lower(),
        }

        parser["prompts"] = {
            "rp_prompt_file": self.config.rp_prompt_file,
            "instruct_prompt_file": self.config.instruct_prompt_file,
            "world_state_prompt_file": self.config.world_state_prompt_file,
        }

        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                parser.write(f)
            logger.debug(f"Config saved to {CONFIG_FILE}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    # ── Extension sync ─────────────────────────────────────────

    # Keys that must never be overwritten with empty/invalid values.
    # If the extension sends an empty string for these, we keep the
    # existing config.ini / default value as a fallback.
    _PROTECTED_URL_KEYS = frozenset({
        "rp_llm_url",
        "instruct_llm_url",
    })

    def update_from_extension(self, data: dict) -> dict:
        """Apply config received from SillyTavern extension.

        Extension values OVERWRITE config.ini values — EXCEPT for URL
        fields.  If the extension sends an empty/whitespace URL, the
        existing config.ini (or built-in default) value is preserved.
        This prevents a misconfigured extension from nuking the
        fallback addresses that were set in config.ini.

        The extension sends lower_snake_case keys. Only known fields
        are applied.

        Returns a dict of applied changes for logging.
        """
        # Map: extension_key → (config_attribute, coerce)
        KEY_MAP = {
            "rp_llm_url": ("rp_llm_url", str),
            "rp_llm_backend": ("rp_llm_backend", str),
            "rp_llm_model": ("rp_llm_model", str),
            "rp_llm_api_key": ("rp_llm_api_key", str),
            "instruct_llm_url": ("instruct_llm_url", str),
            "instruct_llm_backend": ("instruct_llm_backend", str),
            "instruct_llm_model": ("instruct_llm_model", str),
            "instruct_llm_api_key": ("instruct_llm_api_key", str),
            "thinking_steps": ("thinking_steps", int),
            "refinement_steps": ("refinement_steps", int),
            "history_count": ("history_count", int),
            "rp_template": ("rp_template", str),
            "instruct_template": ("instruct_template", str),
            "context_window_messages": ("context_window_messages", int),
            "db_retrieval_count": ("db_retrieval_count", int),
            "debug_mode": ("debug_mode", _to_bool),
            "dry_run": ("dry_run", _to_bool),
            "server_port": ("port", int),
        }

        applied = {}

        for key, value in data.items():
            if key not in KEY_MAP:
                logger.debug(f"Ignored unknown config key: {key}")
                continue

            attr, coerce = KEY_MAP[key]
            old_value = getattr(self.config, attr)
            new_value = coerce(value) if coerce else value

            # --- URL protection: reject empty / whitespace values ---
            # If the extension sends an empty URL, keep whatever we
            # already have (config.ini or built-in default).
            if key in self._PROTECTED_URL_KEYS:
                if not new_value or not str(new_value).strip():
                    logger.info(
                        f"Config: ignoring empty {key} from extension, "
                        f"keeping existing value: {old_value}"
                    )
                    continue

            setattr(self.config, attr, new_value)
            applied[key] = {"old": old_value, "new": new_value}
            logger.debug(f"Config updated: {key} = {new_value}")

        if applied:
            self.save()
            logger.info(f"Config synced from extension: {list(applied.keys())}")

        return applied

    # ── Helpers ────────────────────────────────────────────────

    def get_effective_urls(self) -> dict:
        """Return fully-formed URLs for both LLM endpoints.

        Ensures URLs have a scheme prefix and strips trailing slashes.
        If a URL is empty or invalid, falls back to the built-in default
        so the agent never tries to connect to a broken address.
        """
        def normalize(url: str, default: str) -> str:
            url = (url or "").strip()
            if not url:
                url = default
            if not url.startswith("http://") and not url.startswith("https://"):
                url = f"http://{url}"
            return url.rstrip("/")

        return {
            "rp_llm_url": normalize(
                self.config.rp_llm_url, DEFAULTS["rp_llm"]["url"]
            ),
            "instruct_llm_url": normalize(
                self.config.instruct_llm_url, DEFAULTS["instruct_llm"]["url"]
            ),
        }

    def get_resolved_path(self, relative_path: str) -> Path:
        """Resolve a relative path from config.ini against APP_DIR.

        config.ini uses ./dbs and ./prompts which are relative to app/.
        This ensures they always resolve correctly regardless of CWD.
        """
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return APP_DIR / p

    @property
    def db_dir_path(self) -> Path:
        """Resolved absolute path to the databases directory."""
        return self.get_resolved_path(self.config.db_dir)

    @property
    def prompts_dir_path(self) -> Path:
        """Resolved absolute path to the prompts directory."""
        return self.get_resolved_path(self.config.prompts_dir)

    @property
    def host(self) -> str:
        return "0.0.0.0"

    @property
    def port(self) -> int:
        return self.config.port