#!/usr/bin/env python3
"""Rough head-to-head: GuideRouter vs OpenROAD drt on *matched* nets (Hazard3).

Not a Slice deliverable — an exploratory quality probe. The real ≤1.2×
wire/via gate is WS3.3 Slice 6 (whole-chip, post rip-up + tail). This compares
only the nets we *both* route, so the in-cap-coverage gap (no Slice 4/5 yet)
doesn't bias the per-net wire/via ratio.

Our wire is on the coarse 1120-DBU track grid; drt's is fine geometry. So a
ratio > 1 is partly the track-grid quantisation, partly genuine routing-quality
gap — read it as an upper-bound on our excess, not a verdict.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

import torch

from _hazard3_io import (
    FINAL_DEF,
    GF180MCUD,
    GUIDE,
    LAYER_ORDER,
    apply_pin_access_rules,
    build_chip_grid,
    parse_def_diearea,
    parse_def_nets,
    parse_guides,
)

from gpu_pnr.guide_router import GuideRouter
from track_pitch_sweep_prototype import (
    MAX_PINS,
    MIN_PINS,
    TRACK_PITCH_DBU,
    _net_pins,
    _pct,
)

PDK = GF180MCUD
VIA_COST = 5.0
DBU_PER_UM = 2000  # DEF: UNITS DISTANCE MICRONS 2000
PITCH = TRACK_PITCH_DBU  # 1120 DBU per track-grid step


def _hms_to_s(hms: str) -> float:
    """`HH:MM:SS[.mmm]` → seconds."""
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _drt_initial_route_s(drt_dir) -> float | None:
    """Elapsed seconds of drt's 0th optimization iteration (the initial detailed
    route, still DRC-dirty) — the like-for-like stage to compare our single
    pass against, NOT the full DRC-clean run. Parsed from the routing log:
    the `DRT-0267` timing line following `Start 0th optimization iteration`.
    Returns None if the log shape isn't recognised."""
    log = drt_dir / "openroad-detailedrouting.log"
    if not log.exists():
        return None
    seen_0th = False
    for line in log.read_text().splitlines():
        if "Start 0th optimization iteration" in line:
            seen_0th = True
        elif seen_0th and "DRT-0267" in line and "elapsed time =" in line:
            # "... elapsed time = 00:01:14, memory = ..."
            after = line.split("elapsed time =", 1)[1].strip()
            return _hms_to_s(after.split(",", 1)[0].strip())
    return None


def drt_reference() -> dict[str, float | None]:
    """OpenROAD drt reference for the fixture's run, from the detailed-routing
    step beside FINAL_DEF (`44-openroad-detailedrouting/`). `runtime_s` is the
    *full* DRC-clean run (rip-up + DRC convergence); `initial_route_s` is just
    the 0th iteration (initial dirty route) — the honest like-for-like stage
    against our single pass. Parsed, not re-run."""
    run_dir = FINAL_DEF.parents[2]  # .../RUN_*/final/def/x.def → .../RUN_*
    drt_dir = run_dir / "44-openroad-detailedrouting"
    metrics = json.loads((drt_dir / "or_metrics_out.json").read_text())
    return {
        "nets": metrics["route__net"],
        "wire_um": metrics["route__wirelength"],
        "vias": metrics["route__vias"],
        "runtime_s": _hms_to_s((drt_dir / "runtime.txt").read_text().strip()),
        "initial_route_s": _drt_initial_route_s(drt_dir),
    }


def our_wire_vias(paths: list[list[tuple[int, int, int]]]) -> tuple[int, int]:
    """(wire_dbu, vias) from a routed tree's paths: in-layer edges are wire
    (one track pitch each), layer-change edges are vias. Edges deduped across
    the tree's overlapping seed→pin paths."""
    edges: set[frozenset[tuple[int, int, int]]] = set()
    for path in paths:
        for a, b in zip(path, path[1:]):
            edges.add(frozenset((a, b)))
    wire = 0
    vias = 0
    for e in edges:
        a, b = tuple(e)
        if a[0] == b[0]:
            wire += PITCH
        else:
            vias += 1
    return wire, vias


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto", help="auto | mps | cpu")
    p.add_argument("--sample", type=int, default=300)
    p.add_argument("--margin", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    device = ("mps" if torch.backends.mps.is_available() else "cpu") \
        if args.device == "auto" else args.device

    print("parsing drt routed DEF (reference)...", flush=True)
    drt = parse_def_nets(FINAL_DEF)
    print(f"  drt nets parsed: {len(drt)}", flush=True)

    all_nets = parse_guides(GUIDE)
    xlo, ylo, xhi, yhi = parse_def_diearea(FINAL_DEF)
    chip_origin = (xlo, ylo)
    chip_h = (yhi - ylo) // PITCH + 1
    chip_w = (xhi - xlo) // PITCH + 1
    chip_shape = (len(LAYER_ORDER), chip_h, chip_w)
    w_chip = build_chip_grid(all_nets, xlo, ylo, xhi, yhi, pitch_dbu=PITCH).to(device)

    # Deterministic sample keeping NAMES, matched against the drt DEF.
    names = list(all_nets.keys())
    random.Random(args.seed).shuffle(names)
    samp_names: list[str] = []
    nets: list[list[tuple[int, int, int]]] = []
    guides: list[list[tuple[int, int, int, int, str]]] = []
    for nm in names:
        if len(nets) >= args.sample:
            break
        rects = all_nets[nm]
        pins = _net_pins(rects, chip_origin, PITCH)
        if not (MIN_PINS <= len(pins) <= MAX_PINS) or nm not in drt:
            continue
        samp_names.append(nm)
        nets.append(pins)
        guides.append(rects)

    def prep(w_sub, local_pins):
        apply_pin_access_rules(w_sub, PDK, local_pins)

    router = GuideRouter(
        w_chip, chip_origin=chip_origin, layer_order=LAYER_ORDER,
        pitch_dbu=PITCH, margin=args.margin, chip_shape=chip_shape,
        prep_subgrid=prep,
    )
    t0 = time.perf_counter()
    results = router.route(nets, guides, via_cost=VIA_COST)
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    wl_ratios: list[float] = []
    via_ratios: list[float] = []
    our_wl_tot = drt_wl_tot = 0
    our_via_tot = drt_via_tot = 0
    matched = 0
    for nm, res in zip(samp_names, results):
        if not res.routed or res.paths is None:
            continue
        our_wl, our_v = our_wire_vias(res.paths)
        drt_wl, drt_v = drt[nm]
        if drt_wl == 0:
            continue
        matched += 1
        our_wl_tot += our_wl
        drt_wl_tot += drt_wl
        our_via_tot += our_v
        drt_via_tot += drt_v
        wl_ratios.append(our_wl / drt_wl)
        if drt_v > 0:
            via_ratios.append(our_v / drt_v)

    wl_sorted = sorted(wl_ratios)
    via_sorted = sorted(via_ratios)
    print(f"\nsample {len(nets)} matched nets; routed+matched {matched} "
          f"in {elapsed:.1f}s ({device})", flush=True)
    print("\n  WIRELENGTH (our coarse 1120-DBU grid vs drt fine geometry):")
    print(f"    aggregate ratio (Σours/Σdrt): "
          f"{our_wl_tot / max(drt_wl_tot,1):.3f}×")
    print(f"    per-net ratio: median={_pct(wl_sorted,0.5):.3f}× "
          f"mean={sum(wl_ratios)/max(len(wl_ratios),1):.3f}× "
          f"p90={_pct(wl_sorted,0.9):.3f}×")
    print(f"    totals: ours={our_wl_tot/DBU_PER_UM:.0f}µm "
          f"drt={drt_wl_tot/DBU_PER_UM:.0f}µm")
    print("\n  VIAS:")
    print(f"    aggregate ratio (Σours/Σdrt): "
          f"{our_via_tot / max(drt_via_tot,1):.3f}×")
    print(f"    per-net ratio: median={_pct(via_sorted,0.5):.3f}× "
          f"mean={sum(via_ratios)/max(len(via_ratios),1):.3f}×")
    print(f"    totals: ours={our_via_tot} drt={drt_via_tot}")

    # Execution time. The honest like-for-like is OUR single dirty pass vs drt's
    # INITIAL route (0th iter, also dirty) — NOT drt's full DRC-clean run, which
    # includes rip-up + DRC convergence we don't do. drt is multi-threaded; both
    # are wall-clock and NOT hardware-matched, so read per-net as orientation.
    ref = drt_reference()
    nets_n = int(ref["nets"] or 0)
    runtime_s = float(ref["runtime_s"] or 0.0)
    init_s = ref["initial_route_s"]
    our_ms_net = elapsed * 1000.0 / max(len(nets), 1)
    print("\n  EXECUTION TIME (like-for-like = our pass vs drt's initial route):")
    print(f"    ours:           {our_ms_net:.1f} ms/net  ({len(nets)} in-cap "
          f"nets, {device}, single dirty pass)")
    if init_s is not None:
        drt_init_ms = init_s * 1000.0 / max(nets_n, 1)
        print(f"    drt initial:    {drt_init_ms:.1f} ms/net  ({nets_n} nets, "
              f"{init_s:.0f}s 0th iter, multi-thread, dirty)")
        print(f"    → ours/drt-initial: {our_ms_net/max(drt_init_ms,1e-9):.1f}× "
              f"(>1 = we are slower at comparable work)")
    drt_full_ms = runtime_s * 1000.0 / max(nets_n, 1)
    print(f"    drt full run:   {drt_full_ms:.1f} ms/net  ({runtime_s/60:.1f} min, "
          f"rip-up + DRC-clean — NOT comparable to our dirty pass)")
    print("\n  Gate (Slice 6, full-chip): ≤1.2× wire, ≤1.2× vias. "
          "This is in-cap-only, coarse-grid, no rip-up — orientation only.")


if __name__ == "__main__":
    main()
