#!/usr/bin/env python3
"""WS3.3 tile-decomposition prototype: route one 256² × 5 Hazard3 tile.

Validates the ADR 0012 design on a single tile before building the full
chip-scale tile manager (`src/gpu_pnr/tile_router.py`):

  1. Tile substrate works: nets sharing one cost grid produce 0 cross-net
     cell conflicts.
  2. Halo region serves its purpose: count how many committed cells fall
     in the outer 32-cell ring vs the inner 256² owned region. High halo
     occupancy → either halo too narrow or net assignment too greedy.
  3. Throughput vs per-net mini-grids: compare wall-clock per net on the
     tile-shared grid to the per-net spike's 41–50 ms/net baseline.
     Tier A predicts ~31 ms/source in the K=100 batched regime; this
     prototype runs sequentially first, so expect closer to the
     per-net baseline plus a constant from the larger grid.

This prototype intentionally:
  - Uses sequential routing (route_multipin_nets_3d) on the tile-shared
    grid, not K=100 batched. Batched routing requires conflict detection
    + ripup which belongs in the full tile_router module.
  - Picks a single tile location (not the full chip). The full router
    will iterate over all tile positions; here we pick the densest tile
    to maximise the net count we can validate.
  - Applies PDK pin-access rules but skips preferred-direction off_mult
    by default. Add off_mult > 1 to enable.

Run: uv run python scripts/tile_decomp_prototype.py [TILE_SIZE] [HALO] [OFF_MULT]
  TILE_SIZE defaults to 256 (the ADR 0012 locked value),
  HALO defaults to 32 (the ADR 0012 initial value),
  OFF_MULT defaults to 1.0 (isotropic).
"""

from __future__ import annotations

import sys
import time

import torch

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


def _rect_to_chip_cell(
    rect: tuple[int, int, int, int, str],
    chip_origin: tuple[int, int],
) -> tuple[int, int, int]:
    """Map a guide rect's center to (layer, row, col) on the chip-scale grid."""
    x0, y0, x1, y1, layer = rect
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    return (
        LAYER_ORDER.index(layer),
        (cy - chip_origin[1]) // PITCH_DBU,
        (cx - chip_origin[0]) // PITCH_DBU,
    )


def _net_chip_pins(
    rects: list[tuple[int, int, int, int, str]],
    chip_origin: tuple[int, int],
) -> list[tuple[int, int, int]]:
    """Extract Metal1 pin cells of a net on the chip-scale grid."""
    return [
        _rect_to_chip_cell(r, chip_origin)
        for r in rects
        if r[4] == "Metal1"
    ]


def _net_bbox(pins: list[tuple[int, int, int]]) -> tuple[int, int, int, int]:
    """Tight (row_min, col_min, row_max, col_max) over pin cells."""
    rmin = min(p[1] for p in pins)
    rmax = max(p[1] for p in pins)
    cmin = min(p[2] for p in pins)
    cmax = max(p[2] for p in pins)
    return rmin, cmin, rmax, cmax


def _pick_densest_tile(
    nets_pins: dict[str, list[tuple[int, int, int]]],
    chip_h: int,
    chip_w: int,
    tile_size: int,
    stride: int = 128,
) -> tuple[int, int, list[str]]:
    """Scan tile positions on `stride`-cell grid; return (row, col, owned_nets)
    for the position with the most nets whose pin bbox fits inside the
    owned tile region (caller's responsibility to extend with halo)."""
    best_count = 0
    best_pos = (0, 0)
    best_owned: list[str] = []
    for r0 in range(0, chip_h - tile_size, stride):
        for c0 in range(0, chip_w - tile_size, stride):
            r1 = r0 + tile_size
            c1 = c0 + tile_size
            owned = []
            for name, pins in nets_pins.items():
                if not pins:
                    continue
                rmin, cmin, rmax, cmax = _net_bbox(pins)
                if r0 <= rmin and rmax < r1 and c0 <= cmin and cmax < c1:
                    owned.append(name)
            if len(owned) > best_count:
                best_count = len(owned)
                best_pos = (r0, c0)
                best_owned = owned
    return best_pos[0], best_pos[1], best_owned


def _check_conflicts(results: list[MultiPin3DResult]) -> int:
    cell_owners: dict[tuple[int, int, int], int] = {}
    conflicts = 0
    for idx, res in enumerate(results):
        if not res.routed:
            continue
        for c in res.cells:
            prev = cell_owners.get(c)
            if prev is None:
                cell_owners[c] = idx
            elif prev != idx:
                conflicts += 1
    return conflicts


def _halo_cell_count(
    results: list[MultiPin3DResult],
    tile_size: int,
    halo: int,
) -> tuple[int, int]:
    """Count committed cells inside the inner 256² owned region vs in the
    halo ring. Returns (owned, halo_cells). Coordinates here are local to
    the (256 + 2*halo)² tile sub-grid: owned = [halo, halo+tile_size)."""
    owned = 0
    halo_cells = 0
    for res in results:
        if not res.routed:
            continue
        for (_, r, c) in res.cells:
            if halo <= r < halo + tile_size and halo <= c < halo + tile_size:
                owned += 1
            else:
                halo_cells += 1
    return owned, halo_cells


def main() -> None:
    tile_size = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    halo = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    off_mult = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    full_h = tile_size + 2 * halo
    full_w = tile_size + 2 * halo
    print(
        f"Tile: owned {tile_size}² + halo {halo} = routable {full_h}² × "
        f"{len(LAYER_ORDER)} layers", flush=True,
    )

    print(f"Loading guides from {GUIDE.name}...", flush=True)
    all_nets = parse_guides(GUIDE)
    print(f"  {len(all_nets)} total nets", flush=True)

    print(f"Parsing DIEAREA from {FINAL_DEF.name}...", flush=True)
    xlo, ylo, xhi, yhi = parse_def_diearea(FINAL_DEF)
    chip_h = (yhi - ylo) // PITCH_DBU + 1
    chip_w = (xhi - xlo) // PITCH_DBU + 1
    print(f"  DIEAREA ({xlo},{ylo})..({xhi},{yhi}) → chip {chip_h}×{chip_w} cells",
          flush=True)

    # Extract per-net pin cells on the chip-scale grid.
    chip_origin = (xlo, ylo)
    nets_pins: dict[str, list[tuple[int, int, int]]] = {}
    for name, rects in all_nets.items():
        pins = _net_chip_pins(rects, chip_origin)
        pin_count = sum(1 for r in rects if r[4] == "Metal1")
        # Cap pin count to skip clock/power nets.
        if 2 <= pin_count <= 20:
            nets_pins[name] = pins
    print(f"  {len(nets_pins)} nets after pin-count filter (2..20)", flush=True)

    t0 = time.perf_counter()
    print(f"Scanning for densest {tile_size}² owned tile (stride 128)...",
          flush=True)
    r0, c0, owned_names = _pick_densest_tile(
        nets_pins, chip_h, chip_w, tile_size,
    )
    print(
        f"  best tile: owned rows [{r0}:{r0+tile_size}) "
        f"cols [{c0}:{c0+tile_size}) with {len(owned_names)} nets "
        f"({time.perf_counter() - t0:.1f}s scan)", flush=True,
    )
    if not owned_names:
        print("\nNo nets fit any 256² owned tile; aborting prototype.",
              flush=True)
        return

    # Build the full chip-scale grid, then slice out the tile sub-grid
    # extended by `halo` cells in each direction. Cells outside the chip
    # die area are inf (off-die obstacle).
    print("\nBuilding chip-scale cost grid (one-time setup)...", flush=True)
    t0 = time.perf_counter()
    w_chip = build_chip_grid(all_nets, xlo, ylo, xhi, yhi)
    print(f"  built in {time.perf_counter() - t0:.1f}s; "
          f"shape {tuple(w_chip.shape)}", flush=True)

    # Slice the (halo-extended) tile out of the chip grid. Pad with inf if
    # the slice reaches off-chip (negative or beyond chip_h/chip_w).
    tile_r0 = r0 - halo
    tile_c0 = c0 - halo
    print(f"\nSlicing tile sub-grid: rows [{tile_r0}:{tile_r0+full_h}) "
          f"cols [{tile_c0}:{tile_c0+full_w})", flush=True)
    L = w_chip.shape[0]
    w_tile = torch.full((L, full_h, full_w), float("inf"))
    src_r0 = max(0, tile_r0)
    src_c0 = max(0, tile_c0)
    src_r1 = min(chip_h, tile_r0 + full_h)
    src_c1 = min(chip_w, tile_c0 + full_w)
    dst_r0 = src_r0 - tile_r0
    dst_c0 = src_c0 - tile_c0
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    dst_c1 = dst_c0 + (src_c1 - src_c0)
    w_tile[:, dst_r0:dst_r1, dst_c0:dst_c1] = w_chip[
        :, src_r0:src_r1, src_c0:src_c1
    ]
    del w_chip  # free the chip-scale tensor; the prototype only needs the tile

    # Re-base pin coordinates from chip-scale to tile-local.
    nets_pins_local: list[list[tuple[int, int, int]]] = []
    net_names_used: list[str] = []
    for name in owned_names:
        chip_pins = nets_pins[name]
        local_pins = [(ll, r - tile_r0, c - tile_c0) for (ll, r, c) in chip_pins]
        # Confirm all local pin cells land inside the tile and at finite cost.
        ok = all(
            0 <= r < full_h and 0 <= c < full_w
            and torch.isfinite(w_tile[ll, r, c])
            for (ll, r, c) in local_pins
        )
        if ok:
            nets_pins_local.append(local_pins)
            net_names_used.append(name)
    print(f"  {len(nets_pins_local)} nets with all pins on-tile & finite",
          flush=True)
    if not nets_pins_local:
        print("\nAll tile-fitting nets have pins on inf cells; aborting.",
              flush=True)
        return

    # Apply PDK pin-access rules (M1-as-pin-only) using every net's pins.
    all_pins = [p for pins in nets_pins_local for p in pins]
    apply_pin_access_rules(w_tile, PDK, all_pins)
    print(f"Applied PDK pin-access rules at {len(all_pins)} pin cells.",
          flush=True)

    if off_mult != 1.0:
        h_mult, v_mult = preferred_direction_multipliers(PDK, off_mult)
        print(f"Preferred-direction off_mult={off_mult}", flush=True)
        w_h, w_v = axis_costs(w_tile, h_mult, v_mult)
    else:
        w_h, w_v = w_tile, None

    # Route on the tile-shared grid.
    print(f"\nRouting {len(nets_pins_local)} nets sequentially on the "
          f"tile-shared grid...", flush=True)

    def on_done(idx: int, res: MultiPin3DResult, dt: float) -> None:
        if (idx + 1) % 10 == 0 or idx + 1 == len(nets_pins_local):
            print(
                f"  [{idx+1}/{len(nets_pins_local)}] last net "
                f"{net_names_used[idx]} pins={len(res.pins)} "
                f"{'routed' if res.routed else 'FAILED'} dt={dt*1000:.0f}ms",
                flush=True,
            )

    t0 = time.perf_counter()
    results = route_multipin_nets_3d(
        w_h, nets_pins_local, via_cost=5.0, w_v=w_v, net_timeout_s=60.0,
        progress_callback=on_done,
    )
    elapsed = time.perf_counter() - t0

    routed = sum(1 for r in results if r.routed)
    total_cells = sum(len(r.cells) for r in results if r.routed)
    total_pins = sum(len(r.pins) for r in results if r.routed)
    wirelength = max(0, total_cells - total_pins)

    # Classify failures: pin collision (own pin already in routed_cells from a
    # prior net), pin on inf (pin became inf after PDK rules), or routing
    # failure (router couldn't find a path). Repeats the router's gating
    # logic on the un-mutated w_h to attribute each failure mode.
    pin_coords_seen: set[tuple[int, int, int]] = set()
    pin_collision = 0
    pin_on_inf = 0
    route_fail = 0
    for res in results:
        if res.routed:
            for c in res.cells:
                pin_coords_seen.add(c)
            continue
        pins = res.pins
        # Was any of this net's pins committed by a prior net?
        if any(p in pin_coords_seen for p in pins):
            pin_collision += 1
            continue
        # Was any of this net's pins on an inf cell post-PDK?
        if not all(torch.isfinite(w_h[ll, r, c]) for (ll, r, c) in pins):
            pin_on_inf += 1
            continue
        route_fail += 1
    print(
        f"\nFailure classification: pin_collision={pin_collision} "
        f"pin_on_inf={pin_on_inf} route_fail={route_fail}",
        flush=True,
    )
    via_count = sum(
        sum(1 for (la, _, _), (lb, _, _) in zip(p, p[1:]) if la != lb)
        for r in results if r.routed and r.paths is not None
        for p in r.paths
    )
    owned_count, halo_count = _halo_cell_count(results, tile_size, halo)
    conflicts = _check_conflicts(results)

    print("\n=== Tile prototype results ===", flush=True)
    print(f"  tile config: owned {tile_size}² halo {halo} layers {L}",
          flush=True)
    print(f"  routed: {routed} / {len(nets_pins_local)} "
          f"({100*routed/len(nets_pins_local):.1f}%)", flush=True)
    print(f"  total wirelength: {wirelength} cells", flush=True)
    print(f"  total vias: {via_count}", flush=True)
    print(f"  cross-net cell conflicts: {conflicts} "
          f"({'PASS' if conflicts == 0 else 'FAIL'})", flush=True)
    print(f"  total elapsed: {elapsed:.1f}s "
          f"({1000*elapsed/len(nets_pins_local):.0f}ms/net)", flush=True)
    print()
    print("Halo cell occupancy:", flush=True)
    total_committed = owned_count + halo_count
    if total_committed > 0:
        print(f"  inner {tile_size}² (owned): {owned_count} cells "
              f"({100*owned_count/total_committed:.1f}%)", flush=True)
        print(f"  halo ring (width {halo}): {halo_count} cells "
              f"({100*halo_count/total_committed:.1f}%)", flush=True)
        if halo_count == 0:
            print("  → no routes used the halo; halo width may be over-provisioned.",
                  flush=True)
        elif halo_count / total_committed > 0.10:
            print("  → halo carries >10% of committed cells; widen halo or revisit "
                  "net-assignment policy.", flush=True)
    else:
        print("  (no committed cells)", flush=True)

    print("\nLayer occupancy (cells per metal layer):", flush=True)
    layer_cells = {ll: 0 for ll in range(L)}
    for r in results:
        if r.routed:
            for ll, _, _ in r.cells:
                layer_cells[ll] += 1
    for ll in range(L):
        print(f"  {LAYER_ORDER[ll]}: {layer_cells[ll]} cells", flush=True)


if __name__ == "__main__":
    main()
