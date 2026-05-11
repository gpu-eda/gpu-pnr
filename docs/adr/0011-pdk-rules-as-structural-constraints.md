# ADR 0011 — PDK technology rules encoded as structural constraints, not cost weights

**Status:** Accepted (2026-05-11).

## Context

After the per-axis cost-tensor work landed ([ADR 0010](0010-per-axis-cost-tensors.md))
the full-chip visualization (`scripts/render_routes.py`) showed our
router placing ~100K cells of illegal Metal1 wire under the
preferred-direction (PD) cost model alone. gf180mcuD's PDK forbids
wire on M1 — only pin landings and via anchors are allowed there.
We introduced two short-term knobs to recover the constraint:
`m1_penalty` (soft: multiply M1 cell costs by a factor) and
`m1_pin_only` (hard: mask M1 cells to `inf` except pin coords).
Both worked, but as *cost-tuning knobs* on a flag-controlled basis,
not as part of the cost-grid construction.

This is the wrong layering. The PDK rule "M1 carries no wire" is
*structural* — it's a DRC constraint enforced by the technology,
not a heuristic the router weighs against other options. Real DR
tools (TritonRoute, OpenROAD-drt) keep technology rules and cost
weights distinct: rules are constraints applied by the framework;
costs are knobs the search algorithm balances.

The longer-term consequence of mixing them is that every new PDK
constraint (per-via-pair DRC rules, layer-pair forbidden patterns,
antenna rules) would arrive as a new cost knob, and the cost-tuning
surface would compound. Separating constraints from costs early
keeps the model clean as constraints accumulate.

## Decision

1. **PDK descriptor as a structured type.** `Pdk` is a frozen
   dataclass in `scripts/_hazard3_io.py` holding the layer stack,
   per-layer preferred direction, indices of pin-access-only layers,
   and pitch. Future fields will hold per-via-pair cost arrays and
   per-pair DRC rules. `GF180MCUD` is the canonical instance for
   our current target. A second PDK would be a second instance.

2. **Constraints applied by rule-application functions, not cost
   multipliers.** `apply_pin_access_rules(w, pdk, pin_cells)`
   in-place modifies the cost tensor to encode the no-wire-on-M1
   rule: every M1 cell becomes `inf` except a small landing-pad
   neighbourhood around each pin coordinate. Future PDK rules
   (per-via-pair cost arrays, layer-pair DRC patterns) will arrive
   as additional `apply_<rule>_rules()` functions or via fields
   directly consumed by `build_grid`.

3. **Costs become heuristics tuned on top.** `axis_costs` (the
   preferred-direction multiplier helper) is now strictly a
   heuristic-weight builder. `preferred_direction_multipliers(pdk,
   off_mult, m1_penalty)` reads the PD table from the PDK and
   produces (h_mult, v_mult). The optional `m1_penalty` ablation
   parameter is preserved as a debug knob for studying the soft-vs-
   structural-constraint comparison; it is *not* part of the PDK's
   constraint surface.

4. **Default-on; explicit bypass for ablation.** The two scripts
   (`spike_route_many_nets.py` and `render_routes.py`) apply PDK
   rules by default. A `--no-pdk-rules` flag (or `NO_PDK_RULES=1`
   positional in the spike) disables rule application for legacy
   comparisons. This makes "doing the right thing" the obvious
   path and forces the experimenter to opt into a non-DRC-compliant
   run.

## Consequences

- **Numbers preserved.** N=500 spike numbers under the new default
  match the prior `m1_pin_only=True` run exactly (49,552 wirelength
  cells / 1000 vias / 1.36× wire / 0.80× via vs TR). 51 tests still
  pass.
- **The script `m1_pin_only` flag is gone.** It became implicit in
  the PDK rule. The `m1_penalty` knob remains but is documented as
  an ablation knob; under the default behaviour the M1 cells are
  already `inf` so the multiplier has no observable effect.
- **`apply_pin_access_rules` is the seam for future PDK rules.**
  When per-via-pair DRC patterns or layer-pair constraints land,
  they extend the `Pdk` dataclass and gain their own `apply_*_rules`
  function (or a unified `apply_pdk_rules` that calls all of them).
  The scripts call `apply_pdk_rules(...)` once after `build_grid`;
  the rule surface grows by adding fields and rule applicators,
  not by adding cost knobs.
- **The "PD = m1_cost generalization" framing in
  [ADR 0010](0010-per-axis-cost-tensors.md) was a modeling-level
  error** (now amended in ADR 0010 Consequences §6). The kernel-
  level equivalence still holds (axis-aware costs are the general
  primitive). This ADR records the higher-level correction: the
  M1 constraint is structural, not a cost tuning matter, and
  belongs to the PDK descriptor rather than the cost-weight surface.

## Walk-back options

- **If multi-PDK support is needed before tile decomposition** —
  the `Pdk` dataclass is already in `scripts/_hazard3_io.py`, but
  the file is named for the Hazard3 fixture specifically. Move
  `Pdk` + the `apply_*_rules` functions to a fresh module
  (`scripts/pdk.py` or `src/gpu_pnr/pdk.py`) and add a second PDK
  instance there. Probably worth doing alongside the chip-scale
  routing refactor (WS3.3) rather than now.

- **If a rule needs to interact with the kernel itself** (e.g.,
  forbidden-pattern DRC that can't be expressed as an `inf` mask) —
  the kernel API may need to grow. Defer the API decision until a
  concrete rule of that shape arrives; until then, structural rules
  expressible as `inf` masks cover gf180mcuD's needs and the
  cost-grid layer alone is sufficient.

- **If the soft-cost knob (m1_penalty) turns out to be useful for
  non-DRC purposes** (e.g., congestion-driven discouragement of a
  layer that's still legally routable) — `m1_penalty` stays put,
  but should be renamed and lifted from "M1-specific" to a more
  general "layer-penalty" form. Not blocking.

## Links

- [`../plans/phase3-detailed-routing.md`](../plans/phase3-detailed-routing.md)
  — WS3.2 deliverable 3; this ADR ships it.
- [`../results.md`](../results.md) — Phase 3.2 investigation section,
  the data that motivated the structural-vs-cost split.
- [ADR 0010](0010-per-axis-cost-tensors.md) — per-axis cost tensors,
  amended (Consequences §6) to record the modeling-level correction
  this ADR formalises.
- [`../spikes/phase32-hazard3-real-fixture.md`](../spikes/phase32-hazard3-real-fixture.md)
  — original spike where the M1-cost knob first appeared.
