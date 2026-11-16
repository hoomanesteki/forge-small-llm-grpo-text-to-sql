"""Stand-ins for what tests can't use for real: libraries that would reach the network,
and worker processes that a test double in this process never reaches."""

from __future__ import annotations

from collections.abc import Callable
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import httpx2
from huggingface_hub.errors import HfHubHTTPError, RemoteEntryNotFoundError

from forge.sql.executor import SqliteExecutor


class WordTokenizer:
    """A Hugging Face tokenizer, as far as Forge uses one: one token per word of a chat
    template that tags each message with its role.

    ``chat_template`` is only what ``get_chat_template`` reports; the rendering is fixed.
    """

    def __init__(self, chat_template: str = "{{ role tag }} {{ content }}, per message") -> None:
        self.chat_template = chat_template
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def get_chat_template(self) -> str:
        return self.chat_template

    def apply_chat_template(
        self, conversation: list[dict[str, str]], **kwargs: Any
    ) -> str | dict[str, list[int]]:
        self.calls.append((conversation, kwargs))
        text = " ".join(f"<|{m['role']}|> {m['content']}" for m in conversation)
        if kwargs.get("add_generation_prompt"):
            text += " <|assistant|>"
        # transformers tokenizes unless told not to.
        if not kwargs.get("tokenize", True):
            return text
        return {"input_ids": list(range(len(text.split())))}


class InlineSandbox:
    """A ``SandboxPool`` that runs each job in this process, so monkeypatched code runs.

    It has no deadline: tests that need the kill use a real pool.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.resolve()
        self.executor = SqliteExecutor(self.data_dir)

    def call(self, fn: Callable[..., Any], *args: Any, timeout_s: float) -> Any:
        return fn(self.executor, *args)


def hub_error(status: int, kind: type[HfHubHTTPError] = HfHubHTTPError) -> HfHubHTTPError:
    """The error huggingface_hub raises when the Hub answers ``status``."""
    request = httpx2.Request("POST", "https://huggingface.co/api/models/owner/forge-runs")
    return kind(f"{status} from the Hub", response=httpx2.Response(status, request=request))


class FakeHub:
    """``huggingface_hub.HfApi`` as the Hub run store calls it: one repo whose files are
    held in memory, by path. Each call is recorded as (method, keyword arguments), and
    raises the first error ``fail`` lists for its method, if any, before it does anything."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail: dict[str, list[BaseException]] = {}
        self.tokens: list[str | None] = []

    def client(self, token: str | None = None) -> FakeHub:
        """Stands in for ``HfApi(token=...)``, keeping the token it was built with."""
        self.tokens.append(token)
        return self

    def _record(self, method: str, kwargs: dict[str, Any]) -> None:
        self.calls.append((method, kwargs))
        if self.fail.get(method):
            raise self.fail[method].pop(0)

    def upload_file(self, **kwargs: Any) -> None:
        self._record("upload_file", kwargs)
        data = kwargs["path_or_fileobj"]
        self.files[kwargs["path_in_repo"]] = (
            data if isinstance(data, bytes) else Path(data).read_bytes()
        )

    def upload_folder(self, **kwargs: Any) -> None:
        self._record("upload_folder", kwargs)
        root, prefix = Path(kwargs["folder_path"]), kwargs["path_in_repo"]
        added = {}
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root).as_posix()
            # huggingface_hub matches patterns with fnmatch, whose * crosses folders.
            if path.is_file() and not any(fnmatchcase(rel, p) for p in kwargs["ignore_patterns"]):
                added[f"{prefix}/{rel}"] = path.read_bytes()
        # Remote files under the folder that match, unless the same call uploads them.
        for name in list(self.files):
            rel = name.removeprefix(f"{prefix}/")
            doomed = any(fnmatchcase(rel, p) for p in kwargs.get("delete_patterns") or [])
            if name.startswith(f"{prefix}/") and doomed and name not in added:
                del self.files[name]
        self.files.update(added)

    def list_repo_files(self, **kwargs: Any) -> list[str]:
        self._record("list_repo_files", kwargs)
        return sorted(self.files)

    def delete_files(self, **kwargs: Any) -> None:
        self._record("delete_files", kwargs)
        for name in [
            n for n in self.files if any(fnmatchcase(n, p) for p in kwargs["delete_patterns"])
        ]:
            del self.files[name]

    def hf_hub_download(self, **kwargs: Any) -> str:
        self._record("hf_hub_download", kwargs)
        name = kwargs["filename"]
        if name not in self.files:
            raise hub_error(404, RemoteEntryNotFoundError)
        target = Path(kwargs["local_dir"]) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[name])
        return str(target)

    def snapshot_download(self, **kwargs: Any) -> str:
        self._record("snapshot_download", kwargs)
        root = Path(kwargs["local_dir"])
        # Like the real one, it keeps its download metadata inside the folder.
        (root / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        for name, data in self.files.items():
            if any(fnmatchcase(name, p) for p in kwargs["allow_patterns"]):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_bytes(data)
        return str(root)
