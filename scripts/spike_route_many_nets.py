#!/usr/bin/env python3
"""Phase 3.2 multi-net spike: route N Hazard3 2-pin nets independently.

Each net gets its own (L, H, W) cost grid built from its own guide rectangles.
This isolates per-net behavior and avoids the cross-net interference the full
chip-scale router will eventually need to handle. The autotune picks a
SEG_BARRIER appropriate to each net's geometry.

Reports aggregate stats including a comparison to TritonRoute's wire and
via counts (parsed from final/def/...).

PDK rules (M1-as-pin-access-only for gf180mcuD) are enforced by default
on the cost grid -- the wire cannot legally land on M1 except at pin
landing pads. The optional `off_mult` argument layers preferred-direction
heuristics on top: each layer's non-preferred axis is multiplied by
`off_mult` (M1=H, M2=V, M3=H, M4=V, M5=H). With off_mult >> 1 the router
via-stacks between layers so each wire segment travels along its layer's
cheap axis.

Run: uv run python scripts/spike_route_many_nets.py [N] [SEED] [OFF_MULT] [M1_PENALTY] [NO_PDK_RULES] [MULTIPIN]
  N defaults to 50, SEED defaults to 0, OFF_MULT defaults to 1.0 (isotropic),
  M1_PENALTY defaults to 1.0 (a debug knob -- redundant under PDK rules
  but kept for ablation studies).
  NO_PDK_RULES (0/1, default 0): when 1, the M1-as-pin-only PDK rule
  is NOT applied; this is the legacy "M1 is freely routable" mode
  retained for comparison with the pre-PDK-rule cost model.
  MULTIPIN (0/1, default 0): when 1, sample 3+-pin nets only and route
  them via route_multipin_nets_3d (incremental tree growth). When 0
  the spike samples 2-pin nets only and uses route_nets_3d, preserving
  historical TR-comparison numbers.
"""

from __future__ import annotations

import random
import sys
import time

from _hazard3_io import (
    FINAL_DEF,
    GF180MCUD,
    GUIDE,
    LAYER_ORDER,
    PITCH_DBU,
    apply_pin_access_rules,
    build_grid,
    parse_def_nets,
    parse_guides,
    preferred_direction_multipliers,
    rect_center_to_grid,
)
from gpu_pnr.router import route_multipin_nets_3d, route_nets_3d
from gpu_pnr.sweep import axis_costs

PDK = GF180MCUD


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    off_mult = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    m1_penalty = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    no_pdk_rules = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False
    multipin = bool(int(sys.argv[6])) if len(sys.argv) > 6 else False
    random.seed(seed)
    apply_rules = not no_pdk_rules
    if no_pdk_rules:
        print("PDK rules: DISABLED (legacy mode) -- M1 is freely routable")
    else:
        print(f"PDK rules ({PDK.name}): pin-access-only layers = "
              f"{[PDK.layer_order[i] for i in PDK.pin_access_only_layers]}")
    h_mult: list[float] | None
    v_mult: list[float] | None
    if off_mult != 1.0 or m1_penalty != 1.0:
        h_mult, v_mult = preferred_direction_multipliers(PDK, off_mult, m1_penalty)
        print(
            f"Preferred-direction off-axis multiplier: {off_mult}x\n"
            f"  M1-penalty (ablation knob): {m1_penalty}x\n"
            f"  h_mult per layer: {h_mult}\n"
            f"  v_mult per layer: {v_mult}"
        )
    else:
        h_mult = None
        v_mult = None

    print(f"Loading guides from {GUIDE.name}...")
    all_nets = parse_guides(GUIDE)
    # In multi-pin mode the upper bound is enforced to skip clock /
    # power-distribution nets whose guide bboxes (and 100+ pins) make
    # each sweep prohibitively slow on a per-net mini-grid; those nets
    # need a different routing strategy and are deferred.
    max_pins_multipin = 20
    if multipin:
        pin_count_target = f"3..{max_pins_multipin} (multi-pin)"
        candidates = [
            (name, rects) for name, rects in all_nets.items()
            if 3 <= sum(1 for r in rects if r[4] == "Metal1") <= max_pins_multipin
        ]
    else:
        pin_count_target = "exactly 2 (point-to-point)"
        candidates = [
            (name, rects) for name, rects in all_nets.items()
            if sum(1 for r in rects if r[4] == "Metal1") == 2
        ]
    print(f"  {len(all_nets)} total nets, {len(candidates)} match (pins {pin_count_target})")

    print(f"Loading TritonRoute output from {FINAL_DEF.name}...")
    triton = parse_def_nets(FINAL_DEF)
    print(f"  {len(triton)} TritonRoute-routed nets")

    # Sort by total guide-rectangle count and pick the smallest N -- keeps the
    # spike fast and avoids degenerate cases where a single net's guide spans
    # most of the chip.
    candidates.sort(key=lambda nr: len(nr[1]))
    sample = candidates[:n]
    print(f"  picking smallest {len(sample)} nets for the spike\n")

    via_cost = 5.0
    total_routed = 0
    total_wl_cells = 0
    total_vias = 0
    total_time_ms = 0.0
    route_counts_by_layer = {layer: 0 for layer in LAYER_ORDER}
    failures: list[tuple[str, str]] = []

    # TritonRoute aggregate over the same nets we routed (cells, not DBU).
    triton_total_wl_cells = 0
    triton_total_vias = 0
    triton_missing = 0

    progress_every = 100 if len(sample) >= 500 else max(1, len(sample) // 10)
    batch_start = time.perf_counter()

    for i, (net_name, rects) in enumerate(sample):
        if i > 0 and i % progress_every == 0:
            elapsed = time.perf_counter() - batch_start
            print(
                f"  [{i}/{len(sample)}] routed={total_routed} "
                f"failed={len(failures)} elapsed={elapsed:.1f}s "
                f"avg={1000*elapsed/i:.0f}ms/net",
                flush=True,
            )
        # Pre-routing setup can fail on malformed guides; the router itself
        # shouldn't ever raise for routable inputs (a None path is the
        # expected "failed" signal), so we deliberately don't wrap the
        # routing call in the same except -- a kernel exception during
        # routing is a real bug we want to see.
        try:
            w, origin = build_grid(rects)
            metal1 = [r for r in rects if r[4] == "Metal1"]
            pins = [rect_center_to_grid(r, origin) for r in metal1]
            if apply_rules:
                apply_pin_access_rules(w, PDK, pins)
            if h_mult is not None and v_mult is not None:
                w_h, w_v = axis_costs(w, h_mult, v_mult)
            else:
                w_h, w_v = w, None
        except (ValueError, IndexError) as e:
            failures.append((net_name, f"setup: {type(e).__name__}: {e}"))
            continue
        t0 = time.perf_counter()
        if multipin:
            mp_res = route_multipin_nets_3d(
                w_h, [pins], via_cost=via_cost, w_v=w_v, net_timeout_s=60.0,
            )[0]
            t1 = time.perf_counter()
            total_time_ms += (t1 - t0) * 1000
            if not mp_res.routed or mp_res.paths is None:
                failures.append((net_name, "router returned None"))
                continue
            cells = mp_res.cells
            wirelength = max(0, len(cells) - len(pins))
            # Via count: across all attachment edges, count layer transitions.
            via_count = 0
            for p in mp_res.paths:
                via_count += sum(
                    1 for (la, _, _), (lb, _, _) in zip(p, p[1:]) if la != lb
                )
            total_routed += 1
            total_wl_cells += wirelength
            total_vias += via_count
            for lyr_idx in {c[0] for c in cells}:
                route_counts_by_layer[LAYER_ORDER[lyr_idx]] += 1
        else:
            results = route_nets_3d(
                w_h, [(pins[0], pins[1])], via_cost=via_cost, w_v=w_v
            )
            t1 = time.perf_counter()
            total_time_ms += (t1 - t0) * 1000
            res = results[0]
            if res.path is None:
                failures.append((net_name, "router returned None"))
                continue
            total_routed += 1
            total_wl_cells += res.length
            via_count = sum(
                1 for (la, _, _), (lb, _, _) in zip(res.path, res.path[1:]) if la != lb
            )
            total_vias += via_count
            for lyr_idx in {p[0] for p in res.path}:
                route_counts_by_layer[LAYER_ORDER[lyr_idx]] += 1
        if net_name in triton:
            triton_wl_dbu, triton_vc = triton[net_name]
            # Assumes uniform 200nm pitch across all layers (gf180mcuD); a
            # per-layer pitch table would be needed for non-isotropic PDKs.
            triton_total_wl_cells += triton_wl_dbu // PITCH_DBU
            triton_total_vias += triton_vc
        else:
            triton_missing += 1

    print(f"=== Aggregate over {len(sample)} nets ===")
    print(f"  routed: {total_routed} / {len(sample)} ({100 * total_routed / len(sample):.1f}%)")
    print(f"  total wirelength: {total_wl_cells} cells")
    print(f"  total via transitions: {total_vias}")
    print(f"  avg per-net time: {total_time_ms / len(sample):.1f} ms")
    print(f"  total elapsed routing time: {total_time_ms / 1000:.2f} s")
    print()
    print("Layer occupancy (number of routed nets that used the layer):")
    for layer in LAYER_ORDER:
        print(f"  {layer}: {route_counts_by_layer[layer]}")

    # Restrict comparison to nets we actually routed AND TritonRoute also has.
    if total_routed > 0:
        matched = total_routed - triton_missing
        print()
        print(f"=== TritonRoute comparison (over {matched} matched nets) ===")
        print(f"  TritonRoute total wirelength: {triton_total_wl_cells} cells")
        print(f"  TritonRoute total vias:       {triton_total_vias}")
        wl_ratio = (
            f"{total_wl_cells / triton_total_wl_cells:.2f}x"
            if triton_total_wl_cells else "n/a"
        )
        via_ratio = (
            f"{total_vias / triton_total_vias:.2f}x"
            if triton_total_vias else "n/a"
        )
        print(f"  ours / TritonRoute wirelength: {wl_ratio}")
        print(f"  ours / TritonRoute vias:       {via_ratio}")
        if triton_missing:
            print(f"  ({triton_missing} nets we routed had no entry in TritonRoute output -- skipped)")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for name, reason in failures[:10]:
            print(f"  {name}: {reason}")
        if len(failures) > 10:
            print(f"  ... ({len(failures) - 10} more)")


if __name__ == "__main__":
    main()
