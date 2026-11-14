"""Tools the personal-assistant agent calls to persist and recall knowledge.

Every tool is a thin wrapper over the library store; failures the model should
read as written are raised as agents.ToolError.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field

from mistralai import agents

import memory_store


def _store() -> memory_store.LibraryStore:
    return memory_store.get_store()


@agents.tool
def remember(
    category: Annotated[str, Field(description="Short memory category: profile, preference, project, decision, task, note, ...")],
    content: Annotated[str, Field(description="The fact to remember, written so it can be understood without the surrounding conversation.")],
) -> dict[str, Any]:
    """Remember something about the user long-term, so it can be recalled in any future session. Returns the saved document id and name."""
    try:
        return _store().remember(category, content)
    except Exception as exc:
        raise agents.ToolError(f"Could not save the memory: {exc}") from exc


@agents.tool
def recall(
    query: Annotated[str, Field(description="A focused natural-language query over everything ever remembered or saved, e.g. 'Where does the user live?'")],
) -> str:
    """Search the long-term memory (memories and saved files) and return the grounded answer. Call this before answering anything personal about the user."""
    try:
        return _store().recall(query)
    except Exception as exc:
        raise agents.ToolError(f"Memory recall failed: {exc}") from exc


@agents.tool
def list_memories(
    limit: Annotated[int, Field(description="How many recent memories to list.")] = 20,
) -> list[dict[str, Any]]:
    """List the most recent memories, newest first. Each entry has document_id, file_name, created_at, processing_status."""
    try:
        return _store().list_documents(memory_store.MEMORY_PREFIX, limit)
    except Exception as exc:
        raise agents.ToolError(f"Could not list memories: {exc}") from exc


@agents.tool
def save_file(
    file_name: Annotated[str, Field(description="File name with a text extension, e.g. 'notes.md', 'script.py', 'config.json'.")],
    content: Annotated[str, Field(description="The full text content of the file.")],
) -> dict[str, Any]:
    """Persist a text file long-term so it survives sessions. Returns the saved document id and name. For binary files, use the sandbox file tools instead."""
    try:
        return _store().save_file(file_name, content)
    except Exception as exc:
        raise agents.ToolError(f"Could not save the file: {exc}") from exc


@agents.tool
def list_saved_files(
    limit: Annotated[int, Field(description="How many recent files to list.")] = 20,
) -> list[dict[str, Any]]:
    """List the most recently saved files, newest first."""
    try:
        return _store().list_documents(memory_store.FILE_PREFIX, limit)
    except Exception as exc:
        raise agents.ToolError(f"Could not list saved files: {exc}") from exc


@agents.tool
def read_saved_file(
    document_id: Annotated[str | None, Field(description="Document id from list_saved_files / list_memories. Leave empty when passing file_name.")] = None,
    file_name: Annotated[str | None, Field(description="Exact stored file name, e.g. 'file-notes.md'. Leave empty when passing document_id.")] = None,
) -> str:
    """Read the full text content of a saved file or memory document."""
    try:
        return _store().read_document(document_id=document_id, file_name=file_name)
    except Exception as exc:
        raise agents.ToolError(f"Could not read the document: {exc}") from exc


@agents.tool
def forget(
    document_id: Annotated[str, Field(description="Document id of the memory or file to delete, from list_memories / list_saved_files.")],
) -> str:
    """Permanently delete one memory or saved file by its document id. Returns the deleted document's name."""
    try:
        return _store().delete_document(document_id)
    except Exception as exc:
        raise agents.ToolError(f"Could not delete the document: {exc}") from exc
