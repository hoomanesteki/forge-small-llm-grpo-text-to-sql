from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from forge.ops.kaggle import (
    ARGV_LINE,
    GPU_MACHINE_SHAPE,
    METADATA_FILE,
    bake_argv,
    check_kernel_metadata,
    load_template,
    render_kernel_metadata,
    title_slug,
    write_kernel_dir,
)

ROOT = Path(__file__).resolve().parents[2]
VALUES = {
    "owner": "forge-owner",
    "slug": "forge-throughput-spike",
    "title": "Forge throughput spike",
    "code_file": "throughput.py",
}


def _rendered(**overrides: Any) -> dict[str, Any]:
    meta = render_kernel_metadata(load_template(ROOT), VALUES)
    meta.update(overrides)
    return meta


@pytest.mark.phase(3)
def test_metadata_template_valid() -> None:
    template = load_template(ROOT)
    # The template only has placeholders where a push differs, and nothing else to fill.
    assert template["id"] == "{owner}/{slug}"
    assert template["title"] == "{title}"
    assert template["code_file"] == "{code_file}"

    meta = render_kernel_metadata(template, VALUES)
    assert check_kernel_metadata(meta) == []
    assert meta["id"] == "forge-owner/forge-throughput-spike"
    assert title_slug(meta["title"]) == "forge-throughput-spike"
    assert len(meta["title"]) >= 5
    assert meta["code_file"] == "throughput.py"
    assert (meta["language"], meta["kernel_type"]) == ("python", "script")
    # Real booleans, each set explicitly: the CLI's default for internet contradicts its docs.
    assert meta["is_private"] is True
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is True
    assert meta["machine_shape"] == GPU_MACHINE_SHAPE == "NvidiaTeslaT4"
    for key in ("dataset_sources", "kernel_sources", "competition_sources", "model_sources"):
        assert meta[key] == [], key


@pytest.mark.phase(3)
def test_cpu_kernel_and_dataset_sources() -> None:
    meta = render_kernel_metadata(
        load_template(ROOT), VALUES, gpu=False, dataset_sources=["forge-owner/forge-code"]
    )
    assert meta["enable_gpu"] is False
    assert "machine_shape" not in meta
    assert meta["dataset_sources"] == ["forge-owner/forge-code"]
    assert check_kernel_metadata(meta) == []


@pytest.mark.phase(3)
@pytest.mark.parametrize(
    ("overrides", "problem"),
    [
        ({"id": "forge-owner/forge-throughput-spike/3"}, "id must be 'owner/slug'"),
        ({"id": "forge-owner/Forge_Spike"}, "bad slug"),
        ({"title": "Forge"}, "slugifies to 'forge'"),
        ({"title": "Spk"}, "at least 5 characters"),
        ({"code_file": "throughput.ipynb"}, "code_file must name a .py script"),
        ({"code_file": "../throughput.py"}, "relative to the kernel folder"),
        ({"language": "r"}, "language must be 'python'"),
        ({"kernel_type": "notebook"}, "kernel_type must be 'script'"),
        ({"is_private": False}, "never public"),
        ({"is_private": "true"}, "is_private must be set to true or false"),
        ({"enable_tpu": True}, "enable_tpu must be false"),
        ({"machine_shape": "NvidiaTeslaP100"}, "needs machine_shape 'NvidiaTeslaT4'"),
        ({"enable_gpu": False}, "a CPU kernel takes no machine_shape"),
        ({"accelerator": "NvidiaTeslaT4"}, "unknown key 'accelerator'"),
        ({"dataset_sources": "forge-owner/forge-code"}, "dataset_sources must be a list"),
        ({"dataset_sources": ["forge-code"]}, "must have 2 '/'-separated parts"),
        ({"model_sources": ["qwen-lm/qwen3/transformers/1.7b"]}, "must have 5"),
        ({"code_file": "{code_file}"}, "still holds a template placeholder"),
    ],
)
def test_metadata_check_rejects(overrides: dict[str, Any], problem: str) -> None:
    problems = check_kernel_metadata(_rendered(**overrides))
    assert any(problem in p for p in problems), problems


@pytest.mark.phase(3)
def test_missing_internet_flag_is_an_error() -> None:
    meta = _rendered()
    del meta["enable_internet"]
    assert check_kernel_metadata(meta) == ["enable_internet must be set to true or false, got None"]


@pytest.mark.phase(3)
def test_render_refuses_a_missing_value() -> None:
    values = {k: v for k, v in VALUES.items() if k != "owner"}
    with pytest.raises(ValueError, match=r"needs a value for \{owner\}"):
        render_kernel_metadata(load_template(ROOT), values)
    with pytest.raises(ValueError, match="title 'Forge spike' slugifies to 'forge-spike'"):
        render_kernel_metadata(load_template(ROOT), {**VALUES, "title": "Forge spike"})


@pytest.mark.phase(3)
def test_bake_argv_writes_the_arguments_into_the_script() -> None:
    source = f'"""A kernel."""\n\n{ARGV_LINE}\n\nprint(KERNEL_ARGV)\n'
    argv = ["limit-probe", "--interval-seconds", "30", 'quote " and \\ back']
    baked = bake_argv(source, argv)
    namespace: dict[str, Any] = {}
    exec(compile(baked, "kernel.py", "exec"), namespace)  # the baked line is plain Python
    assert namespace["KERNEL_ARGV"] == argv
    assert baked.count("KERNEL_ARGV: list[str] = [") == 1

    with pytest.raises(ValueError, match="exactly once"):
        bake_argv("print('no argv line')\n", argv)
    with pytest.raises(ValueError, match="exactly once"):
        bake_argv(source + ARGV_LINE + "\n", argv)


@pytest.mark.phase(3)
def test_write_kernel_dir(tmp_path: Path) -> None:
    meta = _rendered()
    out = write_kernel_dir(tmp_path / "kernel", meta, "print('hi')\n")
    assert json.loads((out / METADATA_FILE).read_text(encoding="utf-8")) == meta
    assert (out / "throughput.py").read_text(encoding="utf-8") == "print('hi')\n"

    with pytest.raises(ValueError, match="never public"):
        write_kernel_dir(tmp_path / "bad", {**meta, "is_private": False}, "")
    assert not (tmp_path / "bad").exists()
