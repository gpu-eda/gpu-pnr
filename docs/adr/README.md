# Architecture Decision Records

Numbered records of non-obvious design choices, kept forever, amended in place
when scope expands, never deleted.

For the full convention, see
[Claude Project Discipline — ADRs](https://robtaylor.github.io/claude-project-discipline/adr.html).

## Format

See [`0000-template.md`](0000-template.md). Status lifecycle:

```
Proposed → Accepted (YYYY-MM-DD) → [Superseded by ADR MMMM | Withdrawn]
```

When scope expands, amend the Status line in place rather than rewriting:

```
**Status:** Accepted (2026-05-11). Scope expanded 2026-08-01 — see Decision §3.
```

When superseded, update Status on the old ADR and add a new ADR that references
it. Never delete the old one.

## Index

| ADR  | Title | Status |
|------|-------|--------|
| 0001 | [PyTorch MPS as host platform](0001-pytorch-mps-host.md) | Accepted 2026-05-09 |
| 0002 | [Scan-based axis sweeps over Python-loop sweeps](0002-scan-based-sweeps.md) | Accepted 2026-05-09 |
| 0003 | [Async convergence check via `check_every`](0003-async-convergence-check.md) | Accepted 2026-05-09 |
| 0004 | [CPU-side backtrace on unified memory](0004-cpu-backtrace.md) | Accepted 2026-05-09 |
| 0005 | [Mask-based segmented scan over INF_PROXY](0005-mask-based-segmented-scan.md) | Accepted 2026-05-10. Amended 2026-05-10 (autotune); amended 2026-05-10 (loop-invariant hoist). |
| 0006 | [Sequential per-layer via relax over parallel scan along layer axis](0006-sequential-via-relax.md) | Accepted 2026-05-10 |
| 0007 | [HPWL-ascending net ordering as default](0007-hpwl-ascending-net-ordering.md) | Accepted 2026-05-10 |
| 0008 | [Defer `route_nets_batched`; sweep-sharing only via tile decomposition](0008-defer-route-nets-batched.md) | Accepted 2026-05-10 |
| 0009 | [Per-net independent grids for the Hazard3 spike](0009-per-net-grids.md) | Accepted 2026-05-11 |
| 0010 | [Per-axis cost tensors (w, w_v) for preferred-direction routing](0010-per-axis-cost-tensors.md) | Accepted 2026-05-11. Amended 2026-05-11 (modeling-level correction). |
| 0011 | [PDK technology rules encoded as structural constraints, not cost weights](0011-pdk-rules-as-structural-constraints.md) | Accepted 2026-05-11 |
| 0012 | [Tile decomposition for chip-scale 3D routing](0012-tile-decomposition.md) | Accepted 2026-05-14 (substrate validated by tile prototype) |
