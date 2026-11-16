"""Stand-ins for what tests can't use for real: libraries that would reach the network
or need a GPU, and worker processes that a test double in this process never reaches."""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fnmatch import fnmatchcase
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx2
import pytest
from huggingface_hub.errors import (
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from forge.core.types import Example, RunManifest
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


class VocabTokenizer(WordTokenizer):
    """A ``WordTokenizer`` that also encodes and decodes, one id a distinct word, as vLLM's
    engine turns a prompt into ids and back. ``encode`` puts ``<s>`` first unless told not
    to, as a tokenizer with a BOS token does."""

    def __init__(self, chat_template: str = "{{ role tag }} {{ content }}, per message") -> None:
        super().__init__(chat_template)
        self.vocab: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        words = (["<s>"] if add_special_tokens else []) + text.split()
        return [self.vocab.setdefault(word, len(self.vocab)) for word in words]

    def decode(self, ids: Sequence[int]) -> str:
        words = {i: word for word, i in self.vocab.items()}
        return " ".join(words[i] for i in ids)


@dataclass
class FakeCompletion:
    """``vllm.CompletionOutput``: one sample of a prompt."""

    index: int
    text: str
    token_ids: list[int]


@dataclass
class FakeRequestOutput:
    """``vllm.RequestOutput``: a prompt's ids and its samples."""

    prompt_token_ids: list[int]
    outputs: list[FakeCompletion]


class FakeSamplingParams:
    """``vllm.SamplingParams``, keeping the arguments it was made with."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


@dataclass(frozen=True)
class FakeLoRARequest:
    """``vllm.lora.request.LoRARequest``."""

    lora_name: str
    lora_int_id: int
    lora_path: str


# A fake model: (the prompt as the engine decodes it, sample index) -> reply text.
VllmReply = Callable[[str, int], str]


def gold_reply(examples: Sequence[Example]) -> VllmReply:
    """A ``VllmReply`` that answers each question with its gold query in a sql block, by
    the question the prompt ends with."""
    gold = {" ".join(example.question.split()): example.gold_sql for example in examples}

    def reply(text: str, sample_idx: int) -> str:
        question = text.rsplit("Question:", 1)[-1].rsplit("<|assistant|>", 1)[0].strip()
        return f"```sql\n{gold[question]}\n```"

    return reply


@dataclass
class FakeVllm:
    """vLLM as ``VllmBackend`` imports it, once ``install`` puts it in ``sys.modules``.

    ``engines`` holds the keyword arguments of each ``LLM`` made, ``calls`` each request as
    (method, keyword arguments), ``prompt_ids`` each prompt's token ids and ``prompts``
    each prompt as the engine decoded them. ``reply`` answers a prompt; ``reverse`` returns
    each prompt's samples last first. ``tokenizer`` is what the engine's ``get_tokenizer``
    returns: a ``VocabTokenizer``, or a real one.
    """

    reply: VllmReply
    tokenizer: Any = field(default_factory=VocabTokenizer)
    reverse: bool = False
    engines: list[dict[str, Any]] = field(default_factory=list)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    prompt_ids: list[list[int]] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeVllm:
        vllm, lora, request = (ModuleType(n) for n in ("vllm", "vllm.lora", "vllm.lora.request"))
        vllm.LLM = functools.partial(FakeLLM, self)  # type: ignore[attr-defined]
        vllm.SamplingParams = FakeSamplingParams  # type: ignore[attr-defined]
        request.LoRARequest = FakeLoRARequest  # type: ignore[attr-defined]
        for module in (vllm, lora, request):
            monkeypatch.setitem(sys.modules, module.__name__, module)
        return self


class FakeLLM:
    """``vllm.LLM`` for prompts of token ids, which it decodes and asks its ``FakeVllm`` to
    answer. It has no ``chat``: Forge renders every prompt itself."""

    def __init__(self, vllm: FakeVllm, **kwargs: Any) -> None:
        vllm.engines.append(kwargs)
        self.vllm = vllm

    def get_tokenizer(self) -> Any:
        return self.vllm.tokenizer

    def generate(
        self, prompts: list[dict[str, list[int]]], sampling_params: list[Any], **kwargs: Any
    ) -> list[FakeRequestOutput]:
        self.vllm.calls.append(("generate", {"sampling_params": sampling_params, **kwargs}))
        tokenizer = self.vllm.tokenizer
        outputs = []
        for prompt, params in zip(prompts, sampling_params, strict=True):
            prompt_ids = prompt["prompt_token_ids"]
            text = tokenizer.decode(prompt_ids)
            self.vllm.prompt_ids.append(prompt_ids)
            self.vllm.prompts.append(text)
            samples = []
            for i in range(params.kwargs["n"]):
                reply = self.vllm.reply(text, i)
                samples.append(
                    FakeCompletion(i, reply, tokenizer.encode(reply, add_special_tokens=False))
                )
            if self.vllm.reverse:
                samples.reverse()
            outputs.append(FakeRequestOutput(prompt_ids, samples))
        return outputs


class InlineSandbox:
    """A ``SandboxPool`` that runs each job in this process, so monkeypatched code runs.

    It has no deadline: tests that need the kill use a real pool.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.resolve()
        self.executor = SqliteExecutor(self.data_dir)

    def call(self, fn: Callable[..., Any], *args: Any, timeout_s: float) -> Any:
        return fn(self.executor, *args)


# The repo lookup, a request huggingface_hub sends once, so the store retries its failures.
REPO_URL = "https://huggingface.co/api/models/owner/forge-runs"
# The file listing, a request huggingface_hub retries itself before it raises.
TREE_URL = f"{REPO_URL}/tree/main"
# A file of the repo: a download sends a HEAD here for its metadata, then a GET.
RESOLVE_URL = "https://huggingface.co/owner/forge-runs/resolve/main/runs/x/manifest.json"


def hub_error(
    status: int,
    kind: type[HfHubHTTPError] = HfHubHTTPError,
    *,
    url: str = REPO_URL,
    method: str = "POST",
) -> HfHubHTTPError:
    """The error huggingface_hub raises when the Hub answers ``status`` to ``url``."""
    request = httpx2.Request(method, url)
    return kind(f"{status} from the Hub", response=httpx2.Response(status, request=request))


def connect_error(url: str, method: str = "GET") -> httpx2.ConnectError:
    """The error huggingface_hub raises when the connection for ``url`` is refused."""
    return httpx2.ConnectError("connection refused", request=httpx2.Request(method, url))


def dropped_error(url: str, method: str = "GET") -> httpx2.ReadError:
    """The error httpx2 raises when the Hub drops the connection for ``url`` mid-answer."""
    return httpx2.ReadError("connection reset by peer", request=httpx2.Request(method, url))


def _unreachable(error: BaseException) -> bool:
    """Whether huggingface_hub 2.4's downloads report ``error`` as a Hub they can't reach:
    a refused connection, a timeout, offline mode, and any HTTP error but a missing repo,
    revision or file, a gated repo and a 401, which they raise as they are."""
    if isinstance(error, httpx2.ConnectError | httpx2.TimeoutException | OfflineModeIsEnabled):
        return True
    passed_on = RepositoryNotFoundError | RevisionNotFoundError | RemoteEntryNotFoundError
    return (
        isinstance(error, HfHubHTTPError)
        and not isinstance(error, passed_on)
        and error.response.status_code != 401
    )


class FakeHub:
    """``huggingface_hub.HfApi`` as the Hub run store calls it: one repo whose files are
    held in memory, by path. Each call is recorded as (method, keyword arguments), and
    raises the first error ``fail`` lists for its method, if any, before it does anything.
    As the real client does, ``hf_hub_download`` and ``snapshot_download`` raise an error
    that says the Hub can't be reached as a ``LocalEntryNotFoundError`` caused by it."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail: dict[str, list[BaseException]] = {}
        self.tokens: list[str | None] = []

    def client(self, token: str | None = None) -> FakeHub:
        """Stands in for ``HfApi(token=...)``, keeping the token it was built with."""
        self.tokens.append(token)
        return self

    def _record(self, method: str, kwargs: dict[str, Any], *, download: bool = False) -> None:
        self.calls.append((method, kwargs))
        if self.fail.get(method):
            error = self.fail[method].pop(0)
            if download and _unreachable(error):
                raise LocalEntryNotFoundError(
                    "An error happened while trying to locate the file on the Hub. Please "
                    "check your connection and try again."
                ) from error
            raise error

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
        self._record("hf_hub_download", kwargs, download=True)
        name = kwargs["filename"]
        if name not in self.files:
            raise hub_error(404, RemoteEntryNotFoundError)
        target = Path(kwargs["local_dir"]) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[name])
        return str(target)

    def snapshot_download(self, **kwargs: Any) -> str:
        self._record("snapshot_download", kwargs, download=True)
        root = Path(kwargs["local_dir"])
        # Like the real one, it keeps its download metadata inside the folder.
        (root / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        for name, data in self.files.items():
            if any(fnmatchcase(name, p) for p in kwargs["allow_patterns"]):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_bytes(data)
        return str(root)


def sft_run_folder(
    runs: Path, run_id: str, model_ref: str, checkpoints: Sequence[str], **changes: Any
) -> RunManifest:
    """A finished SFT run in ``runs/<run_id>/`` as a GPU job leaves one, without training:
    its manifest, and an adapter in each of ``checkpoints`` whose files name the folder."""
    fields: dict[str, Any] = {
        "run_id": run_id,
        "kind": "sft",
        "git_sha": "c" * 40,
        "config": {"sft": {"num_train_epochs": 1.0}},
        "config_hash": "4" * 64,
        "data_manifest_hash": "5" * 64,
        "model_ref": model_ref,
        "platform": "kaggle",
        "gpu": "Tesla T4",
        "seed": 42,
        "started_at": datetime(2026, 11, 20, 8, 0, tzinfo=UTC),
        "ended_at": datetime(2026, 11, 20, 10, 30, tzinfo=UTC),
        "gpu_hours": 2.5,
        "status": "finished",
        "outputs": {"config": "config.yaml", "adapter": "checkpoints/final"},
        **changes,
    }
    manifest = RunManifest(**fields)
    folder = runs / run_id
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    for rel in checkpoints:
        (folder / rel).mkdir(parents=True)
        (folder / rel / "adapter_config.json").write_text(f'{{"from": "{rel}"}}\n')
        (folder / rel / "adapter_model.safetensors").write_text(f"weights of {rel}")
    return manifest
