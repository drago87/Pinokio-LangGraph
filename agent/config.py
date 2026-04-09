"""
Configuration management for the LangGraph ST Agent.

Loads settings from .env file and provides them to all modules.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the app root
_APP_ROOT = Path(__file__).parent.parent
load_dotenv(_APP_ROOT / ".env")
load_dotenv(_APP_ROOT / ".env.local")

# ============================================================
# LLM Endpoints
# ============================================================

# RP LLM — The creative roleplay model (KoboldCPP, Ollama, etc.)
RP_LLM_BASE_URL = os.getenv("RP_LLM_BASE_URL", "http://localhost:5001")
RP_LLM_API_KEY = os.getenv("RP_LLM_API_KEY", "none")
RP_LLM_MODEL = os.getenv("RP_LLM_MODEL", "")
RP_LLM_BACKEND = os.getenv("RP_LLM_BACKEND", "kobold")  # "kobold" or "openai_compatible"

# Logic LLM — The smaller model that extracts structured state data
LOGIC_LLM_BASE_URL = os.getenv("LOGIC_LLM_BASE_URL", "http://localhost:11434")
LOGIC_LLM_API_KEY = os.getenv("LOGIC_LLM_API_KEY", "none")
LOGIC_LLM_MODEL = os.getenv("LOGIC_LLM_MODEL", "llama3")
LOGIC_LLM_BACKEND = os.getenv("LOGIC_LLM_BACKEND", "openai_compatible")  # "ollama" or "openai_compatible"

# ============================================================
# Database
# ============================================================

DB_DIR = os.getenv("DB_DIR", str(_APP_ROOT / "data"))
CHROMA_DIR = os.path.join(DB_DIR, "chroma")
SQLITE_PATH = os.path.join(DB_DIR, "state.db")

# ============================================================
# Server
# ============================================================

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8001"))

# ============================================================
# Agent Behavior
# ============================================================

# System prompt injected before context retrieval
RP_SYSTEM_PROMPT = os.getenv("RP_SYSTEM_PROMPT", "")
# How many past messages to include in context window
CONTEXT_WINDOW_MESSAGES = int(os.getenv("CONTEXT_WINDOW_MESSAGES", "20"))
# How many DB entries to retrieve per query
DB_RETRIEVAL_COUNT = int(os.getenv("DB_RETRIEVAL_COUNT", "10"))
# Whether to include tool calls/DB info in the ST response (debug mode)
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


def save_config(config: dict):
    """Save configuration values to .env file."""
    env_path = _APP_ROOT / ".env"
    existing = {}
    if env_path.exists():
        load_dotenv(env_path, override=True)
        # Read existing values to preserve comments
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    existing[key.strip()] = value.strip()

    # Update with new values
    existing.update(config)

    # Write back
    with open(env_path, "w") as f:
        f.write("# LangGraph ST Agent Configuration\n")
        f.write("# ====================================\n\n")
        f.write("# RP LLM (Roleplay Model — KoboldCPP, Ollama, etc.)\n")
        f.write(f"RP_LLM_BASE_URL={existing.get('RP_LLM_BASE_URL', 'http://localhost:5001')}\n")
        f.write(f"RP_LLM_API_KEY={existing.get('RP_LLM_API_KEY', 'none')}\n")
        f.write(f"RP_LLM_MODEL={existing.get('RP_LLM_MODEL', '')}\n")
        f.write(f"RP_LLM_BACKEND={existing.get('RP_LLM_BACKEND', 'kobold')}\n")
        f.write("\n")
        f.write("# Logic LLM (Small model for state extraction)\n")
        f.write(f"LOGIC_LLM_BASE_URL={existing.get('LOGIC_LLM_BASE_URL', 'http://localhost:11434')}\n")
        f.write(f"LOGIC_LLM_API_KEY={existing.get('LOGIC_LLM_API_KEY', 'none')}\n")
        f.write(f"LOGIC_LLM_MODEL={existing.get('LOGIC_LLM_MODEL', 'llama3')}\n")
        f.write(f"LOGIC_LLM_BACKEND={existing.get('LOGIC_LLM_BACKEND', 'openai_compatible')}\n")
        f.write("\n")
        f.write("# Database\n")
        f.write(f"# DB_DIR={existing.get('DB_DIR', str(_APP_ROOT / 'data'))}\n")
        f.write("\n")
        f.write("# Server\n")
        f.write(f"SERVER_PORT={existing.get('SERVER_PORT', '8001')}\n")
        f.write("\n")
        f.write("# Agent Behavior\n")
        f.write(f"CONTEXT_WINDOW_MESSAGES={existing.get('CONTEXT_WINDOW_MESSAGES', '20')}\n")
        f.write(f"DB_RETRIEVAL_COUNT={existing.get('DB_RETRIEVAL_COUNT', '10')}\n")
        f.write(f"DEBUG_MODE={existing.get('DEBUG_MODE', 'false')}\n")
