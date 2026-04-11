# PinokioLangGraph

FastAPI + LangGraph agent for SillyTavern RP state management. Routes chat between a creative narrative LLM and a state-extraction LLM, maintaining per-session world state in SQLite databases with full swipe/redo support.

This is the **Agent backend** — the server that sits between SillyTavern and your LLMs. It requires the [SillyTavern Extension](https://github.com/drago87/Agent-StateSync) to function.

> **Need the extension?** [github.com/drago87/Agent-StateSync](https://github.com/drago87/Agent-StateSync)

---

## What It Does

The Agent sits between SillyTavern and your LLMs as an intelligent proxy. Every chat completion passes through a pipeline that:

1. **Parses session metadata** from a `[SYSTEM_META]` tag injected by the extension
2. **Handles swipe/redo** by reverting previous state changes in the database
3. **Translates world state** from JSON to natural language (so the RP LLM doesn't break immersion)
4. **Injects world state context** as a hidden system message for the RP LLM
5. **Streams the RP LLM response** back to SillyTavern via SSE
6. **Extracts state changes** from the response using the Instruct LLM (background task)
7. **Updates the session database** with the new world state

The result: your RP LLM writes creative prose while a second, smaller model silently tracks what changed in the world — location, time, character states, items, relationships, injuries, etc. — and feeds that context back on every turn for perfect continuity.

---

## Architecture

```
SillyTavern
    │
    ├── [ST Extension](https://github.com/drago87/Agent-StateSync) (index.js)
    │     ├── Injects [SYSTEM_META] tag
    │     ├── Trims history
    │     ├── Detects message type (new/continue/swipe/redo)
    │     └── Manages session lifecycle
    │
    ▼ POST /v1/chat/completions
Agent (This Repo — FastAPI + LangGraph)  ←── port 8001
    │
    ├── parser.py         → Extracts [SYSTEM_META], strips it from messages
    ├── db.py             → Per-session SQLite with swipe-aware state tracking
    ├── templates.py      → Formats messages (ChatML, Llama3, Alpaca, Mistral, Raw)
    ├── agent.py          → LangGraph extraction + translation pipelines
    │
    ├──▶ RP LLM (Creative)     → Streams narrative via SSE to SillyTavern
    │     Koboldcpp / Ollama / any OpenAI-compatible endpoint
    │
    └──▶ Instruct LLM (Data)   → Extracts JSON state changes (background)
          Ollama / Koboldcpp / any OpenAI-compatible endpoint
```

### Data Flow Per Turn

```
User sends message in SillyTavern
        │
        ▼
Extension injects [SYSTEM_META] and trims history
        │
        ▼ POST /v1/chat/completions
Agent receives request
        │
        ├─ Parse [SYSTEM_META] (session_id, message_id, type, swipe_index)
        ├─ If swipe/redo → revert previous DB changes
        ├─ Load world state from SQLite
        ├─ Translate JSON → natural language (Instruct LLM)
        ├─ Inject world state as hidden system message
        ├─ Format messages with selected template
        ├─ (Optional) Thinking passes before generation
        │
        ├─▶ Stream RP LLM response → SSE → SillyTavern
        │
        ├─ (Optional) Refinement passes after generation
        │
        └─▶ Background: Instruct LLM extracts state changes
              │
              └─ Update session SQLite database
```

---

## Components

| File | Purpose |
|------|---------|
| `server.py` | FastAPI app with all HTTP endpoints and the main SSE pipeline |
| `config.py` | Settings management — loads settings.json, receives config from extension |
| `parser.py` | Regex extraction of `[SYSTEM_META]` tag from messages |
| `db.py` | Per-session SQLite manager with swipe/redo reversion logic |
| `templates.py` | Message formatters: ChatML, Llama3, Alpaca, Mistral, Raw |
| `llm_client.py` | Async HTTP clients for both LLMs (streaming + non-streaming) |
| `agent.py` | LangGraph pipelines: state extraction, world state translation, thinking/refinement |
| `settings.json` | Default configuration file |
| `requirements.txt` | Python dependencies |

### Pinokio Manifests

| File | Purpose |
|------|---------|
| `pinokio.json` | Main Pinokio plugin manifest (install + run) |
| `install.json` | Install/uninstall steps |
| `start.json` | Startup configuration |
| `update.json` | Update procedure |

### Prompts (`prompts/`)

| Directory | Purpose |
|-----------|---------|
| `prompts/rp/` | System prompts for the RP LLM (narrative writing style) |
| `prompts/instruct/` | System prompts for the Instruct LLM (state extraction format) |
| `prompts/world_state/` | System prompts for JSON → natural language translation |

Each YAML file contains a `system_prompt` field and optional `thinking_passes`/`refinement_passes` overrides. Copy a file to create custom prompts per character or scenario.

### Databases (`dbs/`)

Auto-created per session. Each chat gets its own `dbs/{session_id}.db` SQLite file with tables for session metadata, world state, state changes (with swipe tracking), and message logging.

---

## Installation

### Option 1: Pinokio (Recommended)

1. Open [Pinokio](https://pinokio.computer/)
2. Install this plugin
3. Click **Install** — Pinokio will prompt you for:
   - RP LLM URL (e.g. `http://localhost:5001` for KoboldCPP)
   - RP LLM backend and model name
   - Instruct LLM URL (e.g. `http://localhost:11434` for Ollama)
   - Instruct LLM backend and model name
4. Click **Start** — the Agent starts and waits for connections

### Option 2: Manual

```bash
git clone https://github.com/drago87/PinokioLangGraph.git
cd PinokioLangGraph
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python server.py
```

Create a `.env` file in the project root (or copy from `.env.example`):

```env
RP_LLM_BASE_URL=http://localhost:5001
RP_LLM_API_KEY=none
RP_LLM_BACKEND=kobold
RP_LLM_MODEL=

INSTRUCT_LLM_BASE_URL=http://localhost:11434
INSTRUCT_LLM_API_KEY=none
INSTRUCT_LLM_BACKEND=ollama
INSTRUCT_LLM_MODEL=llama3

SERVER_PORT=8001
```

### SillyTavern Extension

This repo does not include the SillyTavern extension — it lives in its own repository.

Install the extension from [github.com/drago87/Agent-StateSync](https://github.com/drago87/Agent-StateSync), then in SillyTavern:

1. Open Extensions settings → Agent-StateSync
2. Enter the Agent IP:Port (e.g. `192.168.0.1:8001`)
3. Enter the RP LLM and Instruct LLM endpoints
4. Enable the extension

In SillyTavern's API settings, set the Chat Completion API URL to point to the Agent (e.g. `http://192.168.0.1:8001/v1/chat/completions`).

---

## Configuration

### Extension Settings (set in SillyTavern UI)

| Setting | Description | Default |
|---------|-------------|---------|
| Enable State Sync | Master toggle | Off |
| Agent IP:Port | The Agent server address | (blank = auto-detect) |
| RP LLM IP:Port | Creative narrative endpoint | `192.168.0.1:5001` |
| RP LLM Template | Message format (ChatML/Llama3/Alpaca/Mistral/Raw) | Raw |
| Instruct LLM IP:Port | State extraction endpoint | `192.168.0.1:11434` |
| Instruct LLM Template | Message format for Instruct LLM | Raw |
| Thinking Steps | Internal planning passes before generation (0-2) | 0 |
| Refinement Steps | Post-generation review passes (0-1) | 0 |
| History Messages | User/assistant pairs sent to RP LLM (0=all, 2-8) | 2 |

### Agent Settings (.env)

Created automatically by Pinokio install, or manually as `.env` in the project root:

| Key | Description | Default |
|-----|-------------|---------|
| `RP_LLM_BASE_URL` | RP LLM base URL | `http://localhost:5001` |
| `RP_LLM_API_KEY` | RP LLM API key (if required) | `none` |
| `RP_LLM_BACKEND` | RP LLM backend type | `kobold` |
| `RP_LLM_MODEL` | Override RP model name (blank = auto) | (blank) |
| `INSTRUCT_LLM_BASE_URL` | Instruct LLM base URL | `http://localhost:11434` |
| `INSTRUCT_LLM_API_KEY` | Instruct LLM API key (if required) | `none` |
| `INSTRUCT_LLM_BACKEND` | Instruct LLM backend type | `ollama` |
| `INSTRUCT_LLM_MODEL` | Override Instruct model name | (blank) |
| `SERVER_PORT` | Listen port | `8001` |
| `CONTEXT_WINDOW_MESSAGES` | Max messages for context window | `20` |
| `DB_RETRIEVAL_COUNT` | State entries to retrieve | `10` |
| `DEBUG_MODE` | Enable verbose logging | `false` |

The extension pushes its settings to the Agent via `POST /api/config` on first request. Values set in the extension UI take priority for templates and pipeline settings (thinking, refinement, history).

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/chat/completions` | Main pipeline — receives from SillyTavern, streams RP LLM response |
| GET | `/v1/models` | Lists available models from both LLMs |
| POST | `/api/sessions` | Create a new session (returns session_id UUID) |
| GET | `/api/sessions` | List all sessions with stats |
| GET | `/api/sessions/{id}` | Get session metadata, stats, and world state |
| POST | `/api/sessions/{id}/init` | Initialize session with character card data |
| POST | `/api/config` | Receive config from extension |
| GET | `/health` | Health check |

---

## [SYSTEM_META] Protocol

The extension injects a metadata tag as `messages[0]` on every request:

```
[SYSTEM_META] session_id=a1b2c3d4-... message_id=5 type=new swipe_index=0
```

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | UUID | Identifies the chat session (maps to a SQLite DB file) |
| `message_id` | int | Sequential counter per chat (incremented on new turns) |
| `type` | string | `new`, `continue`, `swipe`, or `redo` |
| `swipe_index` | int | Current swipe position (increments on swipe, resets on new/redo) |

### Message Type Detection

The extension detects the type by comparing the current request against the previous one using content hashing:

- **new** — Different user message than last request (normal turn)
- **continue** — Identical messages, same length (SillyTavern continuation)
- **swipe** — Same user message, different/missing assistant response (regenerate)
- **redo** — Shorter conversation + changed user message (edited a previous message)

### Swipe/Redo Database Reversion

- **Swipe**: When the user regenerates a response, state changes from the previous swipe_index for that message_id are reverted (set `applied=FALSE` in `state_changes`, restore old values in `world_state`).
- **Redo**: When the user edits a previous message, all state changes from that message_id onwards are reverted, since the conversation context has fundamentally changed.

---

## Database Schema

Each session gets its own `dbs/{session_id}.db` file:

```sql
-- Session metadata (character card, persona, init status)
sessions (
    session_id, character_name, character_description,
    character_personality, character_scenario, character_first_mes,
    character_mes_example, persona_name, persona_description,
    created_at, initialized
)

-- Current world state (single source of truth)
world_state (
    key PRIMARY KEY, value, category, source, updated_at
)

-- Audit log of every state change (swipe-aware)
state_changes (
    id, message_id, swipe_index, field,
    old_value, new_value, applied,
    created_at,
    UNIQUE(message_id, swipe_index, field)
)

-- Optional message log for debugging/replay
message_log (id, message_id, swipe_index, role, content, created_at)
```

---

## Prompt Customization

Create new YAML files in `prompts/rp/`, `prompts/instruct/`, or `prompts/world_state/` to customize behavior. Each file follows this format:

```yaml
system_prompt: |
  Your custom system prompt here...

thinking_passes: 0    # Optional: override extension setting
refinement_passes: 0  # Optional: override extension setting
```

The pipeline loads `default.yaml` from each directory. To use a custom prompt, update the corresponding `*_prompt_file` setting in `settings.json`.

---

## Hardware Setup (Reference)

This was designed for a dual-GPU setup but works with any configuration where the two LLMs are accessible via HTTP:

| Component | GPU | Software | Typical Models |
|-----------|-----|----------|----------------|
| RP LLM (Creative) | RTX 4060 Ti (16GB) | Koboldcpp or Ollama | 12B-27B parameter |
| Instruct LLM (Data) | RX 5700 XT (8GB) | Ollama or Koboldcpp | 7B-8B parameter |
| Agent | CPU only | Python/FastAPI | N/A |

---

## Dependencies

```
fastapi>=0.115.0,<1.0.0
uvicorn[standard]>=0.32.0
httpx>=0.27.0
pydantic>=2.0.0
langgraph>=0.2.0
langchain-core>=0.3.0
pyyaml>=6.0
```

---

## Troubleshooting

**"Cannot connect to LLM"** — Verify the LLM is running and the URL is correct. Test with `curl http://your-llm:port/v1/models`.

**"No [SYSTEM_META] found"** — The extension is either disabled or the request didn't go through the fetch interceptor. Check that the extension is enabled in SillyTavern and the API URL is pointed at this Agent's `/v1/chat/completions`. See the [extension repo](https://github.com/drago87/Agent-StateSync) for setup instructions.

**State changes not being extracted** — Check the Instruct LLM endpoint. If it's returning errors, check the Agent's console logs for details. The Instruct LLM needs to return valid JSON.

**Swipe not reverting state** — Ensure the message type is being detected as `swipe`. Check the extension console logs in SillyTavern (F12 → Console) for the detected type.

**Session not initializing** — The extension calls `/api/sessions/{id}/init` on first load. If the Agent wasn't running, it retries on the next request. Make sure the Agent is started before opening the chat in SillyTavern.
