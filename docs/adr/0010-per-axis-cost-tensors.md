# ADR 0010 — Per-axis cost tensors (w, w_v) for preferred-direction routing

**Status:** Accepted (2026-05-11).

## Context

Real PDKs assign a preferred wiring direction to each metal layer
(gf180mcuD: M1 horizontal, M2 vertical, M3 horizontal, ...). The
[Phase 3.2 spike](../spikes/phase32-hazard3-real-fixture.md) showed
that ignoring this is the load-bearing reason our TritonRoute
comparison is misleading: our router routes vertical wire on M1 even
though M1-V is illegal in real DR. The spike's `m1_cost` knob — a
post-`build_grid` multiplier on Metal1 cells only — was a stop-gap
that approximated pin-access-only behaviour and closed the via gap
from 0.15× to 0.78× of TritonRoute. It is layer-specific and
direction-blind; the natural generalisation is **per-axis cost
tensors**.

`sweep_sssp` and `sweep_sssp_3d` already take a single cost tensor
`w` interpreted as "cost to enter each cell." `_precompute_axis(w,
mask, axis, seg_barrier)` happens to be parameterised over axis but
the same `w` was passed for both axes. Two design candidates were
flagged in the preferred-direction handoff:

- **Option A** — per-layer (h_mult, v_mult); kernel exposes a
  factored input.
- **Option B** — fully general per-cell `w_h[l,r,c]`, `w_v[l,r,c]`.

The handoff's critical-context analysis observed these are *the same
kernel surgery*: both pass separate `(w_h, w_v)` to
`_precompute_axis`. The only difference is how the application builds
the inputs. So picking is really a choice about API surface, not
kernel architecture.

## Decision

1. **Kernel takes two cost tensors at the surface:** `sweep_sssp_3d(w,
   source, ..., w_v=None)`. When `w_v is None`, `w` is used for both
   axes (isotropic, back-compat with all existing tests). When `w_v`
   is provided, axis=2 ("H") sweeps use `w`, axis=1 ("V") sweeps use
   `w_v`. This is the **option B** general form.

2. **`axis_costs(w, h_mult, v_mult)` is the option-A builder helper.**
   It multiplies a base `w` by per-layer scalar multipliers,
   preserving inf cells, and returns the `(w_h, w_v)` pair the kernel
   wants. Option A use cases (preferred direction from a PDK alternation
   table) go through this helper; option B use cases (e.g., per-cell
   anisotropy from DRC rules or differential routing) build the
   tensors directly.

3. **The via-landing mask is `mask_h & mask_v`**, not the union or
   either-singleton. A cell with `w_h = inf, w_v = finite` is
   approachable vertically and is a legal via-landing site (a via
   doesn't "enter" via H or V — it lands and the wire continues from
   there). Only cells blocked in *both* directions are via-unreachable.

4. **`backtrace_3d` charges `w_h[cur]` for column-changing predecessors
   and `w_v[cur]` for row-changing predecessors.** Diagonal moves are
   not possible in this kernel (4-connected), so the axis is always
   determined by which coordinate changed.

5. **`route_nets_3d` propagates `w_v` and maintains two parallel
   working tensors** (`w_cur`, `w_v_cur`) that get marked inf in
   lockstep when reserving pins or committing wires. Pin reservation
   blocks both axes at a cell; restoring a pin restores both.

6. **`dijkstra_grid_3d`** (the ground-truth reference) takes the same
   `w_v` argument with the same semantics, so anisotropic kernel
   distances have a Dijkstra baseline to compare against.

## Consequences

- **Correctness verified.** 8 new tests (sweep level and router level)
  cover: isotropic w_v=None equivalence, factored anisotropic vs
  Dijkstra (the kernel agrees with the per-axis Dijkstra reference
  inside the 5e-2 atol the existing 3D tests use), preferred-direction
  forcing layer-hops, axis-only obstacles, axis-aware backtrace
  cost-summing, and anisotropic pin reservation under two-net
  collision. Full suite 51/51.
- **The `m1_cost` knob is gone** from `scripts/spike_route_many_nets.py`,
  replaced by an `off_mult` arg that drives the gf180mcuD per-layer
  preferred-direction pattern through `axis_costs`. Per-net topology
  inspection confirms the kernel now via-stacks M1→M2 for V wire on
  M1-pref layers, instead of running illegal V on M1.
- **Hot path is unchanged for the isotropic case.** When `w_v is None`,
  `mask_v` and `mask_via` alias `mask_h`; the per-iteration `step`
  closure performs the same kernel ops as before. The new branches
  (autotune second-axis max, via-mask intersection) only execute when
  `w_v is not None`.
- **Memory doubles on the anisotropic path.** The grid carries two
  cost tensors (`w` and `w_v`) and `route_nets_3d` clones each
  separately. Within the budget of per-net mini-grids in the current
  spike — measured at ~50 ms/net — this is invisible.
- **The TR-comparison hypothesis was not borne out.** The handoff
  predicted preferred-direction would close the via gap from 0.78×
  toward 1.0× and flatten the wire-ratio drift across sample sizes.
  Measured numbers (50/200/500 nets, off_mult ∈ [2, 100]) tie exactly
  with `m1_cost=10`: 1.08×/0.76× → 1.26×/0.78× → 1.36×/0.80×. The
  residual gap is now attributable to per-via-pair cost asymmetry
  (Phase 3 plan deliverable 3) or multi-pin Steiner topology
  (deliverable 2), not to preferred direction. See
  [`../results.md`](../results.md) for the comparison table.

## Walk-back options

- **If anisotropy needs to extend to vias** (per-via-pair cost array,
  not a scalar `via_cost`) — the kernel surgery is parallel to this
  ADR: replace the scalar with a length-`(L-1)` 1-D tensor, applied
  per layer-pair in the sequential relax loop. ADR 0006 already
  anticipates this; the walk-back is just to extend `sweep_sssp_3d`'s
  signature.
- **If 4-connected grids stop being enough** (e.g., diagonal moves
  needed for high-density routing) — the axis-aware backtrace logic
  becomes ambiguous (a move from (cur±1, cur±1) doesn't have a unique
  axis). Adding diagonal moves would require either reverting to
  isotropic costs or carrying a per-edge-direction cost tensor
  proper.
- **If the per-axis `seg_barrier` autotune becomes inadequate** — the
  current implementation picks one `seg_barrier` shared across both
  axes, using the larger of the two max-cost estimates. If `w_h` and
  `w_v` have wildly different scales (say 100×), the polluted-mask
  threshold becomes loose on the cheap axis. The walk-back is to let
  each axis carry its own `seg_barrier` in its `_ScanState`; the data
  structure already supports this — `sweep_sssp_3d` just needs to
  call `_autotune_seg_barrier` twice and pass distinct barriers.

## Links

- [`../plans/phase3-detailed-routing.md`](../plans/phase3-detailed-routing.md)
  — WS3.2 deliverable 1; this ADR ships it.
- [`../spikes/phase32-hazard3-real-fixture.md`](../spikes/phase32-hazard3-real-fixture.md)
  — spike that motivated this work; M1-cost knob's saturation hinted
  at the preferred-direction generalisation.
- [`../results.md`](../results.md) — Phase 3.2 preferred-direction
  comparison numbers.
- [ADR 0005](0005-mask-based-segmented-scan.md) — the mask trick
  this generalises (each axis carries its own mask).
- [ADR 0006](0006-sequential-via-relax.md) — the 3D-via decision
  this extends; the walk-back "if `via_cost` becomes a per-via-pair
  array" is the next slice after this one.
- [ADR 0002](0002-scan-based-sweeps.md) — the scan trick the per-axis
  form preserves.
