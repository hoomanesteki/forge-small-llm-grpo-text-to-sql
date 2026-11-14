# ADR-0001: Record architecture decisions

- Status: accepted
- Date: 2026-11-14

## Context

Forge makes choices that are easy to forget and expensive to revisit: the dataset, the
split policy, the compute platform, the model, the reward. A reviewer reading results
months later needs to know why each was made and when it should be reopened.

## Decision

Every significant decision gets a short record in `docs/adr/NNNN-title.md` with the
context, the decision, its consequences and a "revisit if" condition. Records are
append-only; a changed decision gets a new record that supersedes the old one.

## Consequences

- Plan section 3.13 lists the expected ADRs; each story that settles one writes it.
- Contract changes in `src/forge/core` require an ADR in the same PR.
