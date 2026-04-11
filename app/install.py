#!/usr/bin/env python3
"""
PinokioLangGraph — Install Script

Called by install.json after pip dependencies are installed.
Creates the dbs/ directory and config.ini with default values.
If config.ini already exists, it is left untouched (user settings preserved).
"""

from pathlib import Path

APP_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = APP_DIR / "config.ini"
DB_DIR = APP_DIR / "dbs"

DEFAULT_CONFIG = """\
# PinokioLangGraph — Agent Configuration
# =======================================
# Created by install.py. Edit this file anytime to change settings.
# To regenerate defaults, delete this file and re-run Install.

[agent]
# Port the agent listens on
port = 8001

[rp_llm]
# Creative roleplay model (writes narrative prose)
url = http://localhost:5001
backend = kobold
model =
api_key = none
# Set to true to skip all calls to the RP LLM (useful for debugging)
disabled = false

[instruct_llm]
# State extraction model (pulls JSON from narrative)
url = http://localhost:11434
backend = ollama
model = llama3
api_key = none
# Set to true to skip all calls to the Instruct LLM (useful for debugging)
disabled = false

[pipeline]
# Internal planning passes before generating (0 = off)
thinking_steps = 0
# Post-generation review passes (0 = off)
refinement_steps = 0
# User/assistant pairs sent to RP LLM (0 = all)
history_count = 2

[templates]
# Message format for each LLM: chatml, llama3, alpaca, mistral, raw
rp = raw
instruct = raw

[paths]
# Where session databases are stored (relative to app/)
db_dir = ./dbs
# Where prompt YAML files live (relative to app/)
prompts_dir = ./prompts

[agent_settings]
# Max messages in the context window
context_window_messages = 20
# State entries to retrieve from DB
db_retrieval_count = 10

[debug]
# Set to true to enable verbose logging (shows everything in the Pinokio terminal)
debug_mode = false
# Set to true to run the full pipeline WITHOUT sending anything to either LLM
# The agent will log exactly what it received from ST and what it WOULD send
dry_run = false
"""


def main():
    print("=" * 50)
    print("PinokioLangGraph — Setup")
    print("=" * 50)

    # Create databases directory
    DB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Directory ready: {DB_DIR}")

    # Create config.ini only if it doesn't exist
    if CONFIG_PATH.exists():
        print(f"[OK] Config already exists: {CONFIG_PATH}")
        print("     (not overwritten — edit manually to change settings)")
    else:
        CONFIG_PATH.write_text(DEFAULT_CONFIG, encoding="utf-8")
        print(f"[OK] Config created: {CONFIG_PATH}")

    print()
    print("Installation complete!")
    print(f"Edit {CONFIG_PATH.name} to configure your LLM URLs and settings.")
    print("Then click Start in Pinokio to launch the Agent.")
    print()


if __name__ == "__main__":
    main()