# ADR 0007 — HPWL-ascending net ordering as default

**Status:** Accepted (2026-05-10).

## Context

Sequential multi-net routing routes nets one at a time, marking each
successful path's cells as obstacles for subsequent nets. The input order
matters: an early greedy choice can block later nets from any feasible path.

Phase 2.1 endpoint reservation produced a useful negative result: reserving all
pin cells up-front (then temporarily un-reserving the active net's pins)
preserved correctness (no shared wires) but did **not** improve success rate
on random workloads — see [`../results.md`](../results.md). Mechanism:
reserving every pin forces early nets onto longer detours, those longer paths
create more barriers, later nets are blocked equally.

**HPWL (Half-Perimeter Wire Length):** the half-perimeter of the
axis-aligned bounding box around a net's pins — `(rmax − rmin) +
(cmax − cmin)`. Standard EDA proxy for net complexity: cheap to
compute (no routing required), lower-bounds the actual routed wire
length, monotonic in pin spread. For 2-pin nets HPWL collapses to
the Manhattan distance between the two pins. Implementation:
`src/gpu_pnr/ordering.py:_hpwl`.

Three orderings tested on the same 256² seed-42 workload with reservation on:

| nets | identity | hpwl_asc   | hpwl_desc |
|------|----------|------------|-----------|
| 10   | 9/10     | **10/10**  | 9/10      |
| 20   | 11/20    | **18/20**  | 12/20     |
| 30   | 15/30    | **21/30**  | 10/30     |
| 50   | 23/50    | **32/50**  | 15/50     |
| 80   | 26/80    | **41/80**  | 25/80     |

HPWL-ascending (route short nets first) wins consistently, +58% routed at 80
nets. Per-routed-net wirelength also improves (198 → 168 at 50 nets).

HPWL-descending is a clean negative: long nets routed first dominate the grid
and choke the short ones.

## Decision

Make **HPWL-ascending** the default net ordering for `route_nets`, with the
ordering exposed as a pluggable strategy (`gpu_pnr.ordering`). ~30 lines, zero
algorithmic complexity, applied as a pre-step before the sequential routing
loop.

## Consequences

- Success rate up ~50–60% on the random-net benchmark with no kernel change.
- Per-net wirelength down ~15%.
- Establishes ordering as a **first-class** lever; ripup-and-reroute (a
  potential later addition) operates on the same ordering abstraction.
- Settles the "is ordering or sweep-sharing more impactful?" question
  experimentally: ordering is. See [ADR 0008](0008-defer-route-nets-batched.md).

## Walk-back options

- **If a future workload has highly variable net sizes where HPWL is
  uninformative** (e.g., almost all nets the same size) — try fanout-based
  ordering or a learned strategy (E4-adjacent).
- **If ripup-and-reroute lands** — the order may need to update on each ripup
  pass; the strategy abstraction supports re-ordering between passes.

## Links

- [`../results.md`](../results.md) — Phase 2.2 measurements.
- [ADR 0008](0008-defer-route-nets-batched.md) — the related deferral
  decision about sweep-sharing.
