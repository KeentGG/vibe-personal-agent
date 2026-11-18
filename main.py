"""Conversational personal assistant with persistent memory, files and knowledge recall.

Run it:   uv run --env-file .env agents chat
Deploy:  push to GitHub, then  uv run --env-file .env agents deploy
"""

import os

from mistralai import agents
from mistralai.client import Mistral
from mistralai.client.models import MessageOutputEntry, TextChunk, ToolReferenceChunk

from memory_tools import (
    forget,
    list_memories,
    list_saved_files,
    read_saved_file,
    recall,
    remember,
    save_file,
)

INSTRUCTIONS = """You are the user's personal assistant. You chat, remember, and act.

## Conversation
This chat is a session: everything said here stays in its history, and once the agent is
deployed, sessions are stored by Mistral and persist across reconnects. Use that context;
do not ask the user to repeat themselves.

## Long-term memory
Memory lives in your memory library and survives sessions, restarts and deployments.
- Before answering anything personal about the user (their facts, preferences, past
  decisions, projects, things they asked you to remember), call `recall` with a focused query.
- When the user tells you something worth keeping, call `remember` with a short category
  (profile, preference, project, decision, task, note, ...) and self-contained content.
  Do not remember trivial chit-chat. When in doubt, save it under "note".
- Use `list_memories` to browse what you remember and `forget` (by document id) to delete one.

## Files
- `save_file` persists a text file long-term; `list_saved_files` and `read_saved_file`
  (by document id or file name) get files back. Persist anything the user may want later.
- For scratch work during a session, use the sandbox file tools (read_file, write_file,
  edit, bash); those files live only in the session's sandbox.

## Knowledge
- `recall` searches everything ever remembered or saved.
- Use `web_search` for current events and facts not in your memory; cite the sources it returns.
- Use the user's connected sources (Notion, Gmail, Slack, GitHub, calendars, ...) through
  their connectors when the question touches them.

## Style
Be concise, warm and proactive. Ask one clarifying question when a request is ambiguous.
When you remember or save something, say so in one line.
"""


def _environment() -> agents.Sandbox:
    kind = os.environ.get("PERSONAL_ASSISTANT_SANDBOX", "mistral").strip().lower()
    if kind == "none":
        return agents.NoSandbox()
    if kind == "bashkit":
        return agents.BashkitSandbox()
    return agents.MistralSandbox()


async def web_search(query: str) -> str:
    """Search the web for current information. Returns an answer with its source titles and URLs."""
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    response = await client.beta.conversations.start_async(
        model=os.environ.get("PERSONAL_ASSISTANT_MODEL", "mistral-large-4"),
        instructions="Always use the web_search tool to answer queries. Never answer from memory alone.",
        tools=[{"type": "web_search"}],
        inputs=query,
        store=False,
    )
    lines = []
    for entry in response.outputs:
        if not isinstance(entry, MessageOutputEntry):
            continue
        for chunk in entry.content:
            if isinstance(chunk, TextChunk) and chunk.text.strip():
                lines.append(chunk.text.strip())
            elif isinstance(chunk, ToolReferenceChunk) and chunk.url:
                lines.append(f"- {chunk.title or ''} {chunk.url}".strip())
    return "\n".join(lines) or "No results."


agent = agents.Agent(
    model=os.environ.get("PERSONAL_ASSISTANT_MODEL", "mistral-large-4"),
    name="personal-assistant",
    description=(
        "Conversational personal assistant with persistent memory and files "
        "(Studio library) and knowledge recall (library RAG + web search + the user's connectors)."
    ),
    instructions=INSTRUCTIONS,
    tools=[remember, recall, list_memories, save_file, read_saved_file, list_saved_files, forget, web_search],
    connectors=[agents.all_connectors()],
    environment=_environment(),
    max_iterations=30,
)
