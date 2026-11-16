# ADR-0005: Kaggle T4 script kernels for GPU work, the Hub as run store

- Status: accepted
- Date: 2026-11-15

## Context

Forge's cash budget is $0, so GPU hours are the scarce resource: about 43 of them in
plan section 3.8. What the free platforms offer, from the research in
[the FG-301 spike](../spikes/kaggle-gpu.md) (nothing here is measured yet):

- Kaggle gives a phone-verified account T4 GPUs with internet, in sessions of at most
  12 hours, against a weekly quota users report at about 30 hours. `kaggle quota -v`
  shows the live figure. The P100 was retired in September 2026; a kernel that asks for
  one runs on a T4 after a warning.
- The Kaggle CLI (kaggle 2.2.5) pushes a script kernel, polls its status and fetches its
  output: `kaggle kernels push -p DIR --accelerator NvidiaTeslaT4 -t SECONDS`, then
  `kaggle kernels status` and `kaggle kernels output`. It cannot attach a Kaggle Secret:
  that only happens in the web editor, and users report that the binding does not
  survive a later CLI push.
- A kernel keeps only what it wrote to `/kaggle/working`. Whether a session killed at its
  limit keeps that output is not documented.
- Colab's free tier usually gives a T4, publishes no limits, can interrupt a session, and
  can't be driven by a script on the free tier.

## Decision

1. GPU jobs run as Kaggle script kernels on a T4. Every kernel is private, a Python
   script, with GPU and internet set explicitly and `machine_shape` `NvidiaTeslaT4`.
   `forge.ops.kaggle` checks the metadata (`kaggle/kernel-metadata.template.json`) before
   every push, and the push names the accelerator and a timeout.
2. A pushed kernel uploads one script, so the job's script is self-contained and carries
   its arguments written in. Code and data reach it as private Kaggle datasets (FG-302).
3. The run store is the private Hugging Face repo of plan section 3.9. How a job gets
   its results there, in this order:
   1. A Kaggle Secret `HF_TOKEN`, if FG-301 shows that pushed runs can read it.
   2. Otherwise no token on Kaggle at all. The job writes everything to
      `/kaggle/working`, and `forge runs pull` fetches it with `kaggle kernels output` and
      uploads it to the Hub from the laptop. The time guard (FG-305) ends each job before
      the 12-hour limit, so the session ends normally instead of being killed.
   3. Only if a job must save checkpoints off the machine during the run and the second
      way loses them: a private Kaggle dataset holding a fine-grained Hugging Face token
      that can write to the run-store repo and nothing else. Its risk: the token sits in
      Kaggle's storage in every version of that dataset, and anyone who gets the dataset
      can write to the run store, though not to the public model repos. Rotate the token
      at each phase gate and delete the old dataset versions.
4. Colab is the manual fallback. `forge kaggle run --emit-notebook` (FG-303) writes an
   equivalent notebook; nothing drives Colab automatically.
5. `kaggle quota -v` is checked before every GPU push, and every Kaggle GPU session gets a
   ledger row. A CPU kernel spends no GPU quota and gets none.

## Consequences

- One GPU at a time. A "GPU T4 x2" session has two T4s, but the jobs use the first.
- Without a readable secret, a job's results reach the Hub only after `forge runs pull`,
  and a session that dies without a clean exit may lose them. FG-301's session-limit
  probe shows how much survives a kill; until then, plan for nothing.
- The kernel image changes without notice (it moved to CUDA 13 in late October 2026), so
  each job logs the image build and its package versions.
- The fp16-only T4 shapes the training stack: see ADR-0006.

## Revisit if

Kaggle changes its quota, accelerators or policy (Google Cloud plans to retire the T4 on
2027-08-01), or FG-301 finds that pushed runs can read Secrets or that a killed session
keeps its output.
