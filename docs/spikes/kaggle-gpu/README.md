# FG-301 reports

What the first Kaggle sessions measured on 2026-11-16 (UTC), each file as the kernel wrote
it, byte for byte. [`../kaggle-gpu.md`](../kaggle-gpu.md) quotes them, question by
question, with the field each number comes from. None of them holds a secret: the
`secrets` stage and the secret probe record names, error text and whether a read worked,
never a value.

| File | What it is | Kernel and version | Code | Fetched with |
|---|---|---|---|---|
| `spike-report.json` | The GPU spike's report: every stage of `kaggle/spike/throughput.py run`, schema `forge-kaggle-spike/1` | `hoomanesteki/forge-throughput-spike` version 2, a T4 x2 session, 17:54 to 18:29 | `c8f4b41`, pushed by `forge kaggle run --job spike --config configs/spike/throughput.yaml` as run `2026-11-16-throughput-spike-c8f4b41` | `forge kaggle output 2026-11-16-throughput-spike-c8f4b41`, into `runs/<run_id>/kernel/` |
| `kernels.json` | The push's record, which `forge kaggle run` wrote beside the run: version, push time, estimate and timeout | the same | the same | written on the laptop at the push |
| `limit-probe.json` | The session-limit probe's heartbeat, rewritten every 30 s until the kill | `hoomanesteki/forge-session-limit-probe`, a CPU session pushed with `kaggle kernels push -p build/kaggle-spike/limit -t 900`, 15:56 to 16:11 | `c8f4b41`'s `kaggle/spike/throughput.py limit-probe`, bundled by `scripts/kaggle_spike.py`: the pushed `build/kaggle-spike/limit/throughput.py` differs from that commit's file only in its baked `KERNEL_ARGV` | `kaggle kernels output hoomanesteki/forge-session-limit-probe` |
| `limit-spike-report.json` | That kernel's `spike-report.json`, renamed here: the `launch` record only, as every launch writes one | the same | the same | the same |
| `secret-probe.json` | Why `HF_TOKEN` couldn't be read: the names of the Kaggle variables in a pushed run and the error each label gave | `hoomanesteki/forge-throughput-spike` version 3, a CPU session, after version 2 | an uncommitted script, pushed by hand; `kaggle/spike/secret_probe.py` is its typed copy, and its docstring says how it was pushed and what changed since | `kaggle kernels output hoomanesteki/forge-throughput-spike` |

The kernels' logs stay out of git (`*.log` is ignored): the GPU spike's is 540 KB, mostly
install and download progress. The spike doc quotes the lines it needs from it.

## FG-495's mount probe

Where Kaggle puts a kernel's sources, which an eval job's checkpoint adapter comes from: a
kernel source's whole output at `/kaggle/input/notebooks/<owner>/<slug>/`, a dataset at
`/kaggle/input/datasets/<owner>/<slug>/`.

| File | What it is | Kernel and version | Code | Fetched with |
|---|---|---|---|---|
| `mount-probe.json` | Every folder under `/kaggle/input` to 7 levels down, and up to 40 files in each, with their sizes, in a kernel with an earlier kernel and the code dataset as its sources | a free CPU kernel run on 2026-11-16, its sources `hoomanesteki/forge-2026-11-16-smoke-kaggle-50-val-c8f4b41` and `hoomanesteki/forge-code`, as the file names them | [probe.py](probe.py), which writes it to `/kaggle/working` | the kernel's output |
