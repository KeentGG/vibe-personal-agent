"""Persistent memory and file store for the personal-assistant agent.

Backed by a Mistral Studio library: every memory and every saved file is one
document in the library, so they survive sessions, processes and deployments.
Recall is retrieval-augmented generation over the same library through the
`document_library` conversation tool.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

DEFAULT_LIBRARY_NAME = "personal-assistant-memory"
MEMORY_PREFIX = "memory-"
FILE_PREFIX = "file-"

_PENDING_STATUSES = {"todo", "in_progress", "waiting_for_capacity", "running"}
_TEXT_EXTENSIONS = {
    ".md", ".markdown", ".txt", ".rst", ".json", ".csv", ".tsv", ".yaml", ".yml",
    ".toml", ".ini", ".xml", ".html", ".css", ".js", ".ts", ".py", ".sh", ".sql",
    ".log", ".env",
}
_PROCESSING_TIMEOUT_SECONDS = 30


class StoreError(RuntimeError):
    """A failure the agent should read as written."""


def get_client() -> Any:
    """A Mistral client for the process: the API key locally, the deployment's
    service account on a Mistral-managed worker."""
    api_key = os.environ.get("MISTRAL_API_KEY") or os.environ.get("MISTRAL_CLIENT_API_KEY")
    if api_key:
        from mistralai.client import Mistral

        return Mistral(api_key=api_key)
    from mistralai.workflows.client import get_mistral_client
    from mistralai.workflows.core.auth import RunAs

    return get_mistral_client(run_as=RunAs.DEPLOYMENT)


class LibraryStore:
    def __init__(self) -> None:
        self._client: Any = None
        self._library_id: str | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = get_client()
        return self._client

    def library_id(self) -> str:
        env_id = os.environ.get("PERSONAL_ASSISTANT_LIBRARY_ID")
        if env_id:
            return env_id
        if self._library_id is not None:
            return self._library_id
        name = os.environ.get("PERSONAL_ASSISTANT_LIBRARY_NAME", DEFAULT_LIBRARY_NAME)
        for library in self.client.beta.libraries.list(search=name).data:
            if library.name == name:
                self._library_id = library.id
                return self._library_id
        created = self.client.beta.libraries.create(
            name=name,
            description="Long-term memory and persisted files of the personal-assistant agent.",
        )
        self._library_id = created.id
        return self._library_id

    def _upload(self, file_name: str, content: str) -> dict[str, Any]:
        document = self.client.beta.libraries.documents.upload(
            library_id=self.library_id(),
            file={"file_name": file_name, "content": content.encode("utf-8")},
        )
        status = self.client.beta.libraries.documents.status(
            library_id=self.library_id(), document_id=document.id
        )
        deadline = time.monotonic() + _PROCESSING_TIMEOUT_SECONDS
        while str(getattr(status, "process_status", "")).lower() in _PENDING_STATUSES:
            if time.monotonic() >= deadline:
                break
            time.sleep(2)
            status = self.client.beta.libraries.documents.status(
                library_id=self.library_id(), document_id=document.id
            )
        return {
            "document_id": document.id,
            "file_name": document.name,
            "processing_status": getattr(status, "process_status", None),
        }

    def remember(self, category: str, content: str) -> dict[str, Any]:
        slug = re.sub(r"[^a-z0-9]+", "-", category.lower()).strip("-") or "note"
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        body = f"# Memory: {category}\n\n{content}\n"
        return self._upload(f"{MEMORY_PREFIX}{slug}-{stamp}.md", body)

    def recall(self, query: str) -> str:
        response = self.client.beta.conversations.start(
            model=os.environ.get("PERSONAL_ASSISTANT_RECALL_MODEL", "mistral-medium-latest"),
            inputs=query,
            tools=[{"type": "document_library", "library_ids": [self.library_id()]}],
            instructions=(
                "Answer only from the documents in the library, and cite the document "
                "you used. If nothing in the library is relevant, say so in one sentence."
            ),
            store=False,
        )
        texts: list[str] = []
        for entry in response.outputs:
            content = getattr(entry, "content", None)
            if isinstance(content, str):
                texts.append(content)
            elif content:
                for chunk in content:
                    if getattr(chunk, "type", None) == "text":
                        texts.append(chunk.text)
        answer = "\n".join(texts).strip()
        if not answer:
            raise StoreError("The memory library returned no answer for that query.")
        return answer

    def list_documents(self, prefix: str, limit: int) -> list[dict[str, Any]]:
        documents = self.client.beta.libraries.documents.list(
            library_id=self.library_id(), page_size=100, sort_by="created_at", sort_order="desc"
        ).data
        out: list[dict[str, Any]] = []
        for document in documents:
            if not document.name.startswith(prefix):
                continue
            out.append(
                {
                    "document_id": document.id,
                    "file_name": document.name,
                    "created_at": getattr(document, "created_at", None),
                    "processing_status": getattr(document, "processing_status", None),
                }
            )
            if len(out) >= limit:
                break
        return out

    def _find_by_name(self, file_name: str) -> str:
        documents = self.client.beta.libraries.documents.list(
            library_id=self.library_id(), page_size=100
        ).data
        for document in documents:
            if document.name == file_name:
                return document.id
        raise StoreError(f"No document named {file_name!r} in the memory library.")

    def read_document(self, document_id: str | None = None, file_name: str | None = None) -> str:
        if document_id is None:
            if file_name is None:
                raise StoreError("Pass document_id or file_name.")
            document_id = self._find_by_name(file_name)
        extracted = self.client.beta.libraries.documents.text_content(
            library_id=self.library_id(), document_id=document_id
        )
        text = getattr(extracted, "text", None)
        return text if isinstance(text, str) else str(extracted)

    def save_file(self, file_name: str, content: str) -> dict[str, Any]:
        extension = os.path.splitext(file_name)[1].lower()
        if extension not in _TEXT_EXTENSIONS:
            raise StoreError(
                f"save_file only stores text files (one of {sorted(_TEXT_EXTENSIONS)}); "
                f"{file_name!r} is not one. Use the sandbox file tools for binary files."
            )
        return self._upload(f"{FILE_PREFIX}{file_name}", content)

    def delete_document(self, document_id: str) -> str:
        deleted = self.client.beta.libraries.documents.delete(
            library_id=self.library_id(), document_id=document_id
        )
        return getattr(deleted, "name", document_id)


_STORE: LibraryStore | None = None


def get_store() -> LibraryStore:
    global _STORE
    if _STORE is None:
        _STORE = LibraryStore()
    return _STORE
