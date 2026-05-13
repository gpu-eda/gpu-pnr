#!/usr/bin/env python3
"""Minimal end-to-end chip-scale routing prototype (pre-WS3.3).

Replaces the per-net mini-grids the spike has been using with a single
chip-scale (L, H, W) cost tensor for the whole Hazard3 die. Routes a
sample of N nets sequentially via `route_multipin_nets_3d` -- the
existing multi-pin router already commits each net's cells as inf to a
shared `w_cur`, so under chip-scale routing it correctly prevents
inter-net cell conflicts that the per-net mini-grid architecture
allowed.

Validates two things:
  1. Architecturally: routes share a single grid; cross-net cell
     conflicts are 0 (not just rare).
  2. Connectivity: each routed net's cells form a single connected
     component (4-connected in-layer plus via cross-layer) containing
     all of the net's pins.

This prototype is intentionally crude:
  - One single chip-scale grid (no tile decomposition).
  - Sequential per-net routing (no sweep-sharing across nets).
  - Hard guide constraint (off-guide cells = inf), like the per-net spike.
  - Sample bounded by N to keep wall-clock tractable; full-chip
    routing on this layout is gated on tile decomposition + sweep-
    sharing per-tile (WS3.3 proper).

Run: uv run python scripts/chip_scale_prototype.py [N] [SEED] [OFF_MULT]
  N defaults to 20 multi-pin nets, SEED defaults to 0,
  OFF_MULT defaults to 10.0.
"""

from __future__ import annotations

import sys
import time

from _hazard3_io import (
    FINAL_DEF,
    GF180MCUD,
    GUIDE,
    LAYER_ORDER,
    PITCH_DBU,
    apply_pin_access_rules,
    build_chip_grid,
    parse_def_diearea,
    parse_guides,
    preferred_direction_multipliers,
)
from gpu_pnr.router import MultiPin3DResult, route_multipin_nets_3d
from gpu_pnr.sweep import axis_costs

PDK = GF180MCUD


def _rect_center_to_chip_cell(
    rect: tuple[int, int, int, int, str],
    chip_origin: tuple[int, int],
) -> tuple[int, int, int]:
    """Map a guide rect's center to a chip-scale (layer, row, col) cell."""
    x0, y0, x1, y1, layer = rect
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    return (
        LAYER_ORDER.index(layer),
        (cy - chip_origin[1]) // PITCH_DBU,
        (cx - chip_origin[0]) // PITCH_DBU,
    )


def _connected_component_size(
    start: tuple[int, int, int],
    cells: set[tuple[int, int, int]],
) -> set[tuple[int, int, int]]:
    """BFS over `cells` using 4-connected in-layer + via cross-layer edges."""
    if start not in cells:
        return set()
    seen = {start}
    frontier = [start]
    while frontier:
        nxt = []
        for cl, ci, cj in frontier:
            neighbours = (
                (cl, ci - 1, cj), (cl, ci + 1, cj),
                (cl, ci, cj - 1), (cl, ci, cj + 1),
                (cl - 1, ci, cj), (cl + 1, ci, cj),
            )
            for n in neighbours:
                if n in cells and n not in seen:
                    seen.add(n)
                    nxt.append(n)
        frontier = nxt
    return seen


def _verify_connectivity(
    results: list[MultiPin3DResult],
) -> tuple[int, int, int]:
    """For each routed result, verify all pins are in one connected
    component. Returns (passed, failed_disconnected, failed_missing_pin).
    """
    passed = failed_disconnected = failed_missing_pin = 0
    for res in results:
        if not res.routed or res.paths is None:
            continue
        cells = res.cells
        missing = [p for p in res.pins if p not in cells]
        if missing:
            failed_missing_pin += 1
            continue
        seed_pin = res.pins[0]
        component = _connected_component_size(seed_pin, cells)
        all_pins_in_component = all(p in component for p in res.pins)
        if not all_pins_in_component:
            failed_disconnected += 1
        else:
            passed += 1
    return passed, failed_disconnected, failed_missing_pin


def _check_cross_net_conflicts(
    results: list[MultiPin3DResult],
) -> int:
    """Count cells claimed by more than one net. With route_multipin_nets_3d
    on a shared grid this must be 0; any non-zero count is a real bug.
    """
    cell_owners: dict[tuple[int, int, int], int] = {}
    conflicts = 0
    for net_idx, res in enumerate(results):
        if not res.routed:
            continue
        for c in res.cells:
            prev = cell_owners.get(c)
            if prev is None:
                cell_owners[c] = net_idx
            elif prev != net_idx:
                conflicts += 1
    return conflicts


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    _seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    off_mult = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0

    print(f"Loading guides from {GUIDE.name}...", flush=True)
    all_nets = parse_guides(GUIDE)
    print(f"  {len(all_nets)} total nets", flush=True)

    print(f"Parsing DIEAREA from {FINAL_DEF.name}...", flush=True)
    xlo, ylo, xhi, yhi = parse_def_diearea(FINAL_DEF)
    print(f"  DIEAREA: ({xlo},{ylo}) to ({xhi},{yhi})", flush=True)
    H = (yhi - ylo) // PITCH_DBU + 1
    W = (xhi - xlo) // PITCH_DBU + 1
    print(f"  chip grid: {len(PDK.layer_order)}x{H}x{W} = "
          f"{len(PDK.layer_order) * H * W * 4 / 1e9:.2f} GB float32",
          flush=True)

    t0 = time.perf_counter()
    print("Building chip-scale cost grid...", flush=True)
    w_chip = build_chip_grid(all_nets, xlo, ylo, xhi, yhi)
    print(f"  built in {time.perf_counter() - t0:.1f}s; "
          f"shape {tuple(w_chip.shape)}", flush=True)

    # Sample N nets capped at 20 pins to avoid clock/power-distribution.
    candidates = [
        (name, rects) for name, rects in all_nets.items()
        if 2 <= sum(1 for r in rects if r[4] == "Metal1") <= 20
    ]
    candidates.sort(key=lambda nr: len(nr[1]))
    sample = candidates[:n]
    print(f"\nSampled {len(sample)} nets (smallest by guide-rect count, 2-20 pins).",
          flush=True)

    chip_origin = (xlo, ylo)
    nets_pins: list[list[tuple[int, int, int]]] = []
    for _, rects in sample:
        m1 = [r for r in rects if r[4] == "Metal1"]
        pins = [_rect_center_to_chip_cell(r, chip_origin) for r in m1]
        nets_pins.append(pins)
    total_pins = sum(len(p) for p in nets_pins)
    print(f"  {total_pins} total pins across {len(nets_pins)} nets",
          flush=True)

    # Apply PDK rules to the chip-scale grid using ALL pins at once.
    all_pins = [p for pins in nets_pins for p in pins]
    apply_pin_access_rules(w_chip, PDK, all_pins)
    print(f"\nApplied PDK pin-access rules at {len(all_pins)} pin cells.",
          flush=True)

    h_mult, v_mult = preferred_direction_multipliers(PDK, off_mult)
    print(f"Preferred-direction off_mult={off_mult}", flush=True)
    print(f"  h_mult={h_mult}; v_mult={v_mult}", flush=True)
    print("Building axis_costs...", flush=True)
    t0 = time.perf_counter()
    w_h, w_v = axis_costs(w_chip, h_mult, v_mult)
    print(f"  built in {time.perf_counter() - t0:.1f}s", flush=True)

    # Route everything on the shared chip-scale grid.
    print(f"\nRouting {len(nets_pins)} nets on the shared chip-scale grid...",
          flush=True)

    def on_net_done(idx: int, res: MultiPin3DResult, dt: float) -> None:
        status = "routed" if res.routed else "FAILED"
        ncells = len(res.cells) if res.routed else 0
        print(
            f"  [{idx+1}/{len(nets_pins)}] net={sample[idx][0]:<10s} "
            f"pins={len(res.pins)} {status} cells={ncells} dt={dt:.2f}s",
            flush=True,
        )

    t0 = time.perf_counter()
    results = route_multipin_nets_3d(
        w_h, nets_pins, via_cost=5.0, w_v=w_v, net_timeout_s=60.0,
        progress_callback=on_net_done,
    )
    elapsed = time.perf_counter() - t0
    print(f"  routed {sum(1 for r in results if r.routed)}/{len(results)} "
          f"in {elapsed:.1f}s ({1000*elapsed/len(results):.0f}ms/net)",
          flush=True)

    # Verification.
    print("\n=== Verification ===", flush=True)
    conflicts = _check_cross_net_conflicts(results)
    passed, failed_disc, failed_missing = _verify_connectivity(results)
    print(f"  cross-net cell conflicts: {conflicts}", flush=True)
    print(f"  connectivity: {passed} ok, "
          f"{failed_disc} disconnected, {failed_missing} missing-pin",
          flush=True)

    if conflicts == 0 and failed_disc == 0 and failed_missing == 0:
        print("\nAll routed nets verify correctly on chip-scale grid.",
              flush=True)
    else:
        print("\nFAILURES present -- inspect results for the offending nets.",
              flush=True)


if __name__ == "__main__":
    main()
