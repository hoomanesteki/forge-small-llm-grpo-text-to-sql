"""Training configs: one YAML file a run, in plan section 3.10's shape.

    run:  {kind: grpo, model_ref: <repo>@<revision>, seed: 42, backend: auto, device: auto}
    data: {prompt_version: p1, split: train}
    lora: {r: 16, alpha: 32, dropout: 0.0, target: [q_proj, k_proj, ...]}
    grpo: {num_generations: 6, max_prompt_length: 1536, max_completion_length: 256, ...}
    ops:  {checkpoint_every: 50, time_guard_minutes: 675, est_gpu_hours: 2.5}

A file holds the ``sft`` section or the ``grpo`` one, whichever ``run.kind`` names. The
other sections may be left out, and a field left out takes the starting value of
ADR-0010 (LoRA and SFT) or ADR-0011 (GRPO). Every check runs when the file loads, before
any model does:

- A key no section has is refused, so a misspelt one can't fall back to its default, and
  so is a key set twice, which YAML would quietly take the last of. A section with
  nothing under it, which YAML reads as null, is refused as empty rather than read as
  left out; only the trainer ``run.kind`` doesn't name may be null, as ``model_dump``
  writes it.
- Numbers are strict: YAML's ``true`` is not 1, and ``2.5`` steps are not 2. PyYAML reads
  ``5e-6`` as text (YAML 1.1 wants ``5.0e-6``), so this loader reads it as the number it
  means.
- ``max_completion_length`` is at most 512 tokens: a SQL reply needs far fewer, and each
  token of it costs T4 memory in every one of a group's completions.
- ``num_generations`` divides the generation batch, ``per_device_train_batch_size`` times
  ``gradient_accumulation_steps``, so every prompt's group fits in one batch whole.
- ``max_prompt_length`` is Forge's own prompt filter (``forge.train.backends``): TRL 1.18
  has no such setting, so it is never passed to TRL.
- An SFT row longer than ``max_length`` is left out rather than cut, and ``subset_rows``
  trains on a fixed subset of the rest (``forge.train.sft``); neither goes to TRL.
- ``keep_checkpoints_at`` takes shares of the run's steps above 0 and below 1, each once:
  the checkpoint at 1, the run's end, is ``checkpoints/final``.
- ``stop_after_step`` has the session that reaches that step save and stop there, as the
  time guard does, so a resumed session carries the run on: a pilot rehearses the end of
  a session without waiting out the guard (``forge.train.callbacks``).
- ``data.split`` is ``train``: training never reads val or dev (ADR-0004).
- Unsloth and 4-bit loading need CUDA, so a run on the CPU or MPS asks for neither.

The run store is the one ``FORGE_RUN_STORE`` names (``forge.ops.hub``), so the plan's
``ops.hub_run_repo`` is not a setting here: one place names it. The plan's
``run.init_adapter``, ``data.curriculum``, ``data.filter_by_sft_pass_rate`` and
``rewards`` come with the stories that use them (FG-504, FG-505 and FG-501).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    Strict,
    StrictBool,
    StrictStr,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from forge.data.prompts import MAX_PROMPT_TOKENS
from forge.data.schema import check_version

BackendName = Literal["auto", "trl", "unsloth"]
TrainDevice = Literal["auto", "cpu", "mps", "cuda"]
RunKind = Literal["sft", "grpo"]
LIBRARY_DEFAULT: Literal["library_default"] = "library_default"
MAX_COMPLETION_TOKENS = 512
# Every linear layer of a Qwen block (ADR-0010).
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

Count = Annotated[int, Strict(), Field(ge=1)]
Seed = Annotated[int, Strict(), Field(ge=0)]
Positive = Annotated[float, Strict(), Field(gt=0, allow_inf_nan=False)]
Fraction = Annotated[float, Strict(), Field(ge=0, lt=1)]
# A share of a run's steps short of its end, which checkpoints/final is.
Share = Annotated[float, Strict(), Field(gt=0, lt=1)]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(_Section):
    """What is trained, from what, and where."""

    kind: RunKind
    model_ref: Annotated[StrictStr, Field(min_length=1)]  # a Hub repo, best as <repo>@<sha>
    seed: Seed = 0
    backend: BackendName = "auto"
    device: TrainDevice = "auto"
    load_in_4bit: StrictBool = False

    @model_validator(mode="after")
    def _cuda_only(self) -> RunConfig:
        if self.device in ("cpu", "mps"):
            if self.backend == "unsloth":
                raise ValueError(
                    f"backend unsloth trains on CUDA only, not on device {self.device}"
                )
            if self.load_in_4bit:
                raise ValueError(
                    f"load_in_4bit needs bitsandbytes on CUDA, not device {self.device}"
                )
        return self


class DataConfig(_Section):
    prompt_version: StrictStr = "p1"
    split: StrictStr = "train"

    @field_validator("prompt_version")
    @classmethod
    def _known_version(cls, value: str) -> str:
        return check_version(value)

    @field_validator("split")
    @classmethod
    def _train_only(cls, value: str) -> str:
        if value != "train":
            raise ValueError(
                f"training reads the train split only, never {value!r}: val picks "
                "checkpoints and dev is for reported numbers (ADR-0004)"
            )
        return value


class LoraConfig(_Section):
    r: Count = 16
    alpha: Count = 32
    dropout: Fraction = 0.0
    target: tuple[StrictStr, ...] = LORA_TARGETS  # module names, as PEFT matches them

    @field_validator("target")
    @classmethod
    def _distinct_targets(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("LoRA needs at least one target module")
        repeated = sorted({name for name in value if value.count(name) > 1})
        if repeated:
            raise ValueError(f"target names {', '.join(repeated)} more than once")
        return value


class SftConfig(_Section):
    learning_rate: Positive = 2e-4
    num_train_epochs: Positive = 1.0
    max_steps: Count | None = None  # when set, the run stops there, whatever the epochs
    lr_scheduler_type: Literal["linear", "cosine", "constant", "constant_with_warmup"] = "cosine"
    warmup_ratio: Fraction = 0.03
    per_device_train_batch_size: Count = 4
    gradient_accumulation_steps: Count = 4
    max_length: Count = 2048  # prompt and reply, in tokens; a longer row is left out
    logging_steps: Count = 10
    # TRL's default, set here so the manifest says it: the backward pass recomputes each
    # layer's activations rather than keep them, which long rows need on a laptop.
    gradient_checkpointing: StrictBool = True
    # Pad each micro-batch to a multiple of this many tokens; None pads to its longest row.
    # PyTorch's MPS backend builds and keeps a graph for each new sequence length, so rows
    # of every length grow a run's memory; a multiple bounds how many lengths there are.
    pad_to_multiple_of: Count | None = None
    # Train on this many of the rows that fit, a subset forge.train.sft draws by hash,
    # rather than on all of them; for a run with less time than an epoch takes.
    subset_rows: Count | None = None


class GrpoConfig(_Section):
    num_generations: Annotated[int, Strict(), Field(ge=2)] = 6
    max_prompt_length: Count = MAX_PROMPT_TOKENS
    max_completion_length: Annotated[int, Strict(), Field(ge=1, le=MAX_COMPLETION_TOKENS)] = 256
    learning_rate: Positive = 5e-6
    per_device_train_batch_size: Count = 6  # completions, not prompts
    gradient_accumulation_steps: Count = 4
    max_steps: Count = 600
    # library_default leaves beta to TRL; the manifest records the value it used.
    beta: (
        Annotated[float, Strict(), Field(ge=0, allow_inf_nan=False)] | Literal["library_default"]
    ) = LIBRARY_DEFAULT
    # A GRPO step takes a minute or more on a T4, so every one is worth a metrics line.
    logging_steps: Count = 1
    # The share of GPU memory Unsloth's colocated vLLM takes for rollouts; the TRL backend
    # generates with transformers and has no use for it. 0.5 is Unsloth's own default.
    gpu_memory_utilization: Annotated[float, Strict(), Field(gt=0, le=1)] = 0.5

    @property
    def generation_batch(self) -> int:
        """Completions generated at once: TRL's generation_batch_size on one GPU."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @model_validator(mode="after")
    def _whole_groups(self) -> GrpoConfig:
        if self.generation_batch % self.num_generations:
            raise ValueError(
                f"num_generations {self.num_generations} must divide the generation batch, "
                f"per_device_train_batch_size x gradient_accumulation_steps = "
                f"{self.per_device_train_batch_size} x {self.gradient_accumulation_steps} = "
                f"{self.generation_batch}, so each prompt's group fits in it whole"
            )
        return self


class OpsConfig(_Section):
    """How a job runs, which no result depends on."""

    checkpoint_every: Count = 50  # steps between checkpoints pushed to the run store
    # Shares of the run's steps whose checkpoints retention keeps, beside the latest, for
    # val to choose from (FG-404); the run's end is checkpoints/final, always kept.
    keep_checkpoints_at: tuple[Share, ...] = ()
    # A Kaggle session ends at 12 h; 11 h 15 min leaves time for the last save and push.
    time_guard_minutes: Positive = 675.0
    # The step after which the session that trains it saves and stops, as at the time
    # guard; a session resumed from that step trains on. None stops only at the guard.
    stop_after_step: Count | None = None
    est_gpu_hours: Positive | None = None  # what forge kaggle run budgets for the job

    @field_validator("keep_checkpoints_at")
    @classmethod
    def _distinct_shares(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if len(set(value)) < len(value):
            raise ValueError(f"each share is kept once, got {list(value)}")
        return tuple(sorted(value))


class TrainConfig(_Section):
    run: RunConfig
    data: DataConfig = DataConfig()
    lora: LoraConfig = LoraConfig()
    sft: SftConfig | None = None
    grpo: GrpoConfig | None = None
    ops: OpsConfig = OpsConfig()

    # YAML reads a section with nothing under it, such as `grpo:` alone, as null. Only the
    # trainer run.kind doesn't name may be null: a config's model_dump writes it so.
    @field_validator("run", "data", "lora", "ops", mode="before")
    @classmethod
    def _not_empty(cls, value: Any, info: ValidationInfo) -> Any:
        if value is None:
            name = str(info.field_name)
            required = cls.model_fields[name].is_required()
            raise ValueError(_empty("the section", name, required=required))
        return value

    @model_validator(mode="after")
    def _one_trainer(self) -> TrainConfig:
        kind = self.run.kind
        other = "grpo" if kind == "sft" else "sft"
        if getattr(self, kind) is None:
            if kind in self.model_fields_set:
                raise ValueError(_empty(f"the {kind} section", kind, required=False))
            raise ValueError(f"run.kind is {kind}, so the file needs the {kind} section")
        if getattr(self, other) is not None:
            raise ValueError(f"run.kind is {kind}, so the file can't have the {other} section")
        return self

    @property
    def max_seq_length(self) -> int:
        """The longest sequence the trainer sees, prompt and reply, in tokens."""
        if self.sft is not None:
            return self.sft.max_length
        if self.grpo is not None:
            return self.grpo.max_prompt_length + self.grpo.max_completion_length
        raise AssertionError("a validated TrainConfig has an sft or a grpo section")


def _empty(what: str, name: str, *, required: bool) -> str:
    """The error for section ``name``, ``what`` the error calls it, written with nothing
    under it."""
    fix = "write its settings under it"
    if not required:
        fix += f", or `{name}: {{}}` for the starting values"
    return f"{what} is empty; {fix}"


class _Loader(yaml.SafeLoader):
    """PyYAML's safe loader, reading ``5e-6`` as a float as YAML 1.2 does, and refusing a
    mapping that sets a key twice."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        for index, key in enumerate(keys):
            if key in keys[:index]:
                raise yaml.constructor.ConstructorError(
                    None, None, f"{key!r} is set twice", node.value[index][0].start_mark
                )
        return super().construct_mapping(node, deep=deep)


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+$"),
    list("-+0123456789"),
)


def load_train_config(path: Path) -> TrainConfig:
    """The training config in the YAML file at ``path``, checked as the module says.

    Raises ``OSError`` when the file can't be read, and ``ValueError`` naming each wrong
    field when it isn't a valid config.
    """
    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=_Loader)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must be a mapping of sections, such as 'run:' and 'lora:'")
    return parse_train_config(loaded, where=str(path))


def parse_train_config(data: dict[str, Any], *, where: str = "the config") -> TrainConfig:
    """A training config from its sections as a mapping; raises ``ValueError`` as
    ``load_train_config`` does."""
    try:
        return TrainConfig.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'top level'}: "
            f"{error['msg'].removeprefix('Value error, ')}"
            for error in exc.errors()
        )
        raise ValueError(f"{where}: {problems}") from exc
