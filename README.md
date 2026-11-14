# personal-assistant

A conversational personal assistant built with the [Mistral Agents SDK](https://docs.mistral.ai/studio/agents/getting-started/quickstart) (`from mistralai import agents`). It chats, persists **memories** and **files**, and **recalls knowledge**.

## How it works

| Capability | Mechanism |
| --- | --- |
| Conversation | Agent sessions; once deployed (`agents deploy`), Mistral stores each session, so chats persist across reconnects. Long sessions auto-compact. |
| Long-term memory | `remember` / `recall` / `list_memories` / `forget` tools backed by a Studio **library** (`personal-assistant-memory`): one markdown document per memory. |
| Files | `save_file` / `list_saved_files` / `read_saved_file` store text files as documents in the same library. In-session scratch files use the sandbox file tools (`read_file`, `write_file`, `edit`, `bash`). |
| Knowledge recall | `recall` runs RAG over the library via the `document_library` conversation tool; the `web_search` connector covers current facts; `all_connectors()` mounts the user's connected sources (Notion, Gmail, Slack, GitHub, calendars, ...). |

The memory protocol is in the agent's instructions (`main.py`): recall before answering anything personal, remember what is worth keeping.

## Run locally

```bash
uv sync
cp .env.example .env   # put your MISTRAL_API_KEY in .env
uv run --env-file .env agents chat                 # interactive chat
uv run --env-file .env agents chat -m "Hi! Remember that I live in Paris."   # one message
```

The memory library is found by name (`personal-assistant-memory`) and created on first use, or set `PERSONAL_ASSISTANT_LIBRARY_ID` to an existing library. In this workspace the library `01a1da07-3d1b-72cc-a9e5-9af23fe24341` already exists and is seeded with a `README.md`.

## Deploy (Mistral-managed, durable sessions)

1. Push this project to a GitHub repository (the GitHub connector must be connected in [Mistral Studio](https://console.mistral.ai/build/connectors)).
2. `uv run --env-file .env agents deploy` — builds the `Dockerfile` from the pushed commit and registers the agent as `personal-assistant`.
3. Chat with the deployed agent: `uv run --env-file .env agents chat personal-assistant`, or from code with `agents.Agent.load_from(agent_name="personal-assistant")`.

A `MistralSandbox` needs the `cloud_user` workspace role; without it, run locally with `PERSONAL_ASSISTANT_SANDBOX=bashkit` (in-process shell/file tools, no cloud role) or `none`.

## Configuration (environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `MISTRAL_API_KEY` | required locally | API key. On a managed deployment the worker's service account is used instead. |
| `PERSONAL_ASSISTANT_LIBRARY_ID` | — | Memory library id; skips name lookup. |
| `PERSONAL_ASSISTANT_LIBRARY_NAME` | `personal-assistant-memory` | Memory library name (created on first use if missing). |
| `PERSONAL_ASSISTANT_MODEL` | `mistral-large-4` | The agent's model. |
| `PERSONAL_ASSISTANT_RECALL_MODEL` | `mistral-medium-latest` | Model used for `recall` RAG queries. |
| `PERSONAL_ASSISTANT_SANDBOX` | `mistral` | `mistral` (MistralSandbox), `bashkit`, or `none`. |

## Tests

Offline, no API key — a `ScriptedModel` drives the agent while a fake store records the memory calls:

```bash
uv run pytest
uv run agents info   # validates the entrypoint and capabilities
```

## Files

- `main.py` — the `agent` (model, instructions, tools, connectors, sandbox).
- `memory_tools.py` — the `remember` / `recall` / file tools (`@agents.tool` functions).
- `memory_store.py` — the library-backed store (upload, RAG recall, list, read, delete).
- `test_agent.py` — scripted-model tests.
- `Dockerfile`, `.dockerignore` — what `agents deploy` builds.
