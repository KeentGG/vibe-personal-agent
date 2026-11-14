"""Offline tests: a ScriptedModel drives the agent, a fake store records the memory calls.

Run with: uv run pytest
"""

from typing import Any

import pytest
from mistralai import agents

import main
import memory_store


class FakeStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def remember(self, category: str, content: str) -> dict[str, Any]:
        self.calls.append(("remember", (category, content)))
        return {"document_id": "doc-mem-1", "file_name": f"memory-{category}-20260101-000000.md", "processing_status": "done"}

    def recall(self, query: str) -> str:
        self.calls.append(("recall", (query,)))
        return "The user lives in Paris."

    def list_documents(self, prefix: str, limit: int) -> list[dict[str, Any]]:
        self.calls.append(("list_documents", (prefix, limit)))
        return [{"document_id": "doc-1", "file_name": f"{prefix}x.md", "created_at": "2026-01-01", "processing_status": "done"}]

    def read_document(self, document_id: str | None = None, file_name: str | None = None) -> str:
        self.calls.append(("read_document", (document_id, file_name)))
        return "# Saved file\n\ncontent\n"

    def save_file(self, file_name: str, content: str) -> dict[str, Any]:
        self.calls.append(("save_file", (file_name, content)))
        return {"document_id": "doc-file-1", "file_name": f"file-{file_name}", "processing_status": "done"}

    def delete_document(self, document_id: str) -> str:
        self.calls.append(("delete_document", (document_id,)))
        return "memory-preference-20260101-000000.md"


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test that forgets to fake the store fails instead of calling the Mistral API."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_CLIENT_API_KEY", raising=False)


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    store = FakeStore()
    monkeypatch.setattr(memory_store, "get_store", lambda: store)
    return store


def _test_agent() -> agents.Agent:
    return agents.Agent(
        model="mistral-large-4",
        name="personal-assistant-test",
        instructions=main.INSTRUCTIONS,
        tools=[main.remember, main.recall, main.list_memories, main.save_file, main.read_saved_file, main.list_saved_files, main.forget],
        environment=agents.NoSandbox(),
    )


async def test_remember_then_recall(fake_store: FakeStore) -> None:
    model = agents.testing.ScriptedModel(
        [
            agents.testing.run_typescript(
                'return await tools.client.remember({category: "preference", content: "The user lives in Paris."});'
            ),
            agents.testing.run_typescript(
                'return await tools.client.recall({query: "Where does the user live?"});'
            ),
            agents.testing.reply("You live in Paris."),
        ]
    )
    state = await _test_agent().override(model=model).run("Remember that I live in Paris, then tell me where I live.")

    assert isinstance(state, agents.Completed), state
    assert state.text == "You live in Paris."
    assert [call.name for call in state.tool_calls] == ["client.remember", "client.recall"]
    assert fake_store.calls == [
        ("remember", ("preference", "The user lives in Paris.")),
        ("recall", ("Where does the user live?",)),
    ]
    model.assert_consumed()


async def test_save_then_read_file(fake_store: FakeStore) -> None:
    model = agents.testing.ScriptedModel(
        [
            agents.testing.run_typescript(
                'return await tools.client.save_file({file_name: "ideas.md", content: "# Ideas\\n\\n- one\\n"});'
            ),
            agents.testing.run_typescript(
                'return await tools.client.read_saved_file({file_name: "file-ideas.md"});'
            ),
            agents.testing.reply("Saved and read back ideas.md."),
        ]
    )
    state = await _test_agent().override(model=model).run("Save my ideas file, then read it back.")

    assert isinstance(state, agents.Completed), state
    assert [call.name for call in state.tool_calls] == ["client.save_file", "client.read_saved_file"]
    assert fake_store.calls[0] == ("save_file", ("ideas.md", "# Ideas\n\n- one\n"))
    assert fake_store.calls[1] == ("read_document", (None, "file-ideas.md"))
    model.assert_consumed()


async def test_forget(fake_store: FakeStore) -> None:
    model = agents.testing.ScriptedModel(
        [
            agents.testing.run_typescript(
                'return await tools.client.forget({document_id: "doc-mem-1"});'
            ),
            agents.testing.reply("Forgot it."),
        ]
    )
    state = await _test_agent().override(model=model).run("Forget memory doc-mem-1.")

    assert isinstance(state, agents.Completed), state
    assert [call.name for call in state.tool_calls] == ["client.forget"]
    assert fake_store.calls == [("delete_document", ("doc-mem-1",))]
    model.assert_consumed()
