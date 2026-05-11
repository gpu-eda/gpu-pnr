#!/usr/bin/env python3
"""Phase 3.2 multi-net spike: route N Hazard3 2-pin nets independently.

Each net gets its own (L, H, W) cost grid built from its own guide rectangles.
This isolates per-net behavior and avoids the cross-net interference the full
chip-scale router will eventually need to handle. The autotune picks a
SEG_BARRIER appropriate to each net's geometry.

Reports aggregate stats including a comparison to TritonRoute's wire and
via counts (parsed from final/def/...).

The optional `off_mult` argument enables per-layer preferred-direction
modelling: each layer's non-preferred axis is multiplied by `off_mult`,
the preferred axis stays at 1.0. gf180mcuD's alternation is
M1=H, M2=V, M3=H, M4=V, M5=H. With off_mult >> 1 the router via-stacks
between layers so each wire segment travels along its layer's cheap axis,
approximating real ASIC preferred-direction routing.

Run: uv run python scripts/spike_route_many_nets.py [N] [SEED] [OFF_MULT] [M1_PENALTY] [M1_PIN_ONLY]
  N defaults to 50, SEED defaults to 0, OFF_MULT defaults to 1.0 (isotropic),
  M1_PENALTY defaults to 1.0. When M1_PENALTY > 1.0 it overrides M1's
  preferred-direction multipliers so BOTH axes on M1 cost that much, which
  approximates gf180mcuD's pin-access-only convention for M1.

  M1_PIN_ONLY (0/1, default 0): when set, all M1 cells are marked
  unroutable (inf) except the per-net source/sink coords. The router
  must via-stack off M1 immediately and route the wire body on M2+.
  This is the strictest approximation of gf180mcuD's "no M1 wire" DRC
  rule (closer than M1_PENALTY, which still permits M1 routing if
  the cost balance allows). When set, M1_PENALTY is moot.
"""

from __future__ import annotations

import random
import sys
import time

from _hazard3_io import (
    FINAL_DEF,
    GUIDE,
    LAYER_ORDER,
    PITCH_DBU,
    build_grid,
    parse_def_nets,
    parse_guides,
    rect_center_to_grid,
)
from gpu_pnr.router import route_nets_3d
from gpu_pnr.sweep import axis_costs

# gf180mcuD preferred direction per metal layer, aligned with LAYER_ORDER.
PREFERRED_DIRECTION = ["H", "V", "H", "V", "H"]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    off_mult = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    m1_penalty = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    m1_pin_only = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False
    random.seed(seed)
    if m1_pin_only:
        m1_penalty = 1.0  # superseded by the pin-only mask
        print("M1 mode: pin-cell only (non-pin M1 cells set to inf)")
    h_mult: list[float] | None
    v_mult: list[float] | None
    if off_mult != 1.0 or m1_penalty != 1.0:
        assert len(PREFERRED_DIRECTION) == len(LAYER_ORDER)
        h_mult = [1.0 if d == "H" else off_mult for d in PREFERRED_DIRECTION]
        v_mult = [1.0 if d == "V" else off_mult for d in PREFERRED_DIRECTION]
        if m1_penalty != 1.0:
            h_mult[0] = m1_penalty
            v_mult[0] = m1_penalty
        print(
            f"Preferred-direction off-axis multiplier: {off_mult}x\n"
            f"  M1 pin-only penalty: {m1_penalty}x (overrides M1 entries)\n"
            f"  h_mult per layer: {h_mult}\n"
            f"  v_mult per layer: {v_mult}"
        )
    else:
        h_mult = None
        v_mult = None

    print(f"Loading guides from {GUIDE.name}...")
    all_nets = parse_guides(GUIDE)
    two_pin = [
        (name, rects) for name, rects in all_nets.items()
        if sum(1 for r in rects if r[4] == "Metal1") == 2
    ]
    print(f"  {len(all_nets)} total nets, {len(two_pin)} are 2-pin")

    print(f"Loading TritonRoute output from {FINAL_DEF.name}...")
    triton = parse_def_nets(FINAL_DEF)
    print(f"  {len(triton)} TritonRoute-routed nets")

    # Sort by total guide-rectangle count and pick the smallest N -- keeps the
    # spike fast and avoids degenerate cases where a single net's guide spans
    # most of the chip.
    two_pin.sort(key=lambda nr: len(nr[1]))
    sample = two_pin[:n]
    print(f"  picking smallest {len(sample)} 2-pin nets for the spike\n")

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

    for net_name, rects in sample:
        # Pre-routing setup can fail on malformed guides; the router itself
        # shouldn't ever raise for routable inputs (a None path is the
        # expected "failed" signal), so we deliberately don't wrap the
        # routing call in the same except -- a kernel exception during
        # routing is a real bug we want to see.
        try:
            w, origin = build_grid(rects)
            metal1 = [r for r in rects if r[4] == "Metal1"]
            source = rect_center_to_grid(metal1[0], origin)
            sink = rect_center_to_grid(metal1[1], origin)
            if m1_pin_only:
                # Strict pin-only model: every M1 cell becomes an obstacle
                # except the source/sink coords (and their layer-0 neighbours
                # within a single cell radius, to absorb minor center-of-rect
                # rounding when the pin's true via-anchor lands one cell off
                # the rect center). All wire body must traverse on M2+.
                w[0] = float("inf")
                for pin in (source, sink):
                    pl, pr, pc = pin
                    if pl != 0:
                        continue
                    rlo, rhi = max(0, pr - 1), min(w.shape[1], pr + 2)
                    clo, chi = max(0, pc - 1), min(w.shape[2], pc + 2)
                    w[0, rlo:rhi, clo:chi] = 1.0
            if h_mult is not None and v_mult is not None:
                w_h, w_v = axis_costs(w, h_mult, v_mult)
            else:
                w_h, w_v = w, None
        except (ValueError, IndexError) as e:
            failures.append((net_name, f"setup: {type(e).__name__}: {e}"))
            continue
        t0 = time.perf_counter()
        results = route_nets_3d(
            w_h, [(source, sink)], via_cost=via_cost, w_v=w_v
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
