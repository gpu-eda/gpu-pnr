#!/usr/bin/env python3
"""Pin-count distribution on Hazard3 — settles the multi-pin batching open Q.

ADR 0012 Amendment 4's batched small-grid sweep is single-source. Multi-pin
nets need `pin_count - 1` attachment sweeps via the existing incremental
tree-growth (`route_multipin_nets_3d`). The Slice 3 strategy choice in
`docs/plans/ws33-tile-router-implementation.md` turns on how much of the
routing work lives in 2-pin nets vs ≥3-pin nets:

- **(c)** batch 2-pin nets, route ≥3-pin sequentially — works iff 2-pin
  dominates both *net count* AND *sweep-cell-work* (cells × sweeps per net).
- **(a)/(b)** batch round-r attachment sweeps — worth the bookkeeping iff
  ≥3-pin nets carry a meaningful share of total work.

Same filter as `track_pitch_sweep_prototype`: 2..20 M1 pins, has
`guide_region`, fits the 256-axis sub-grid cap, all pins land inside the
region. No sweeping — pure data measurement.

Run:
  uv run python scripts/measure_pin_count_distribution.py
  uv run python scripts/measure_pin_count_distribution.py --pitch 200   # A/B
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict

from _hazard3_io import (
    FINAL_DEF,
    GUIDE,
    LAYER_ORDER,
    parse_def_diearea,
    parse_guides,
)

from track_pitch_sweep_prototype import (
    MAX_PINS,
    MIN_PINS,
    SUBGRID_AXIS_CAP,
    TRACK_PITCH_DBU,
    _net_pins,
)

from gpu_pnr.guides import guide_region


def measure(pitch: int) -> dict[str, object]:
    all_nets = parse_guides(GUIDE)
    xlo, ylo, xhi, yhi = parse_def_diearea(FINAL_DEF)
    chip_origin = (xlo, ylo)
    chip_h = (yhi - ylo) // pitch + 1
    chip_w = (xhi - xlo) // pitch + 1
    chip_shape = (len(LAYER_ORDER), chip_h, chip_w)

    n_total = len(all_nets)
    n_routable = 0  # 2..20 M1 pins
    n_in_cap = 0  # routable + has guide + ≤256 axis + pins in region
    by_pincount_routable: Counter[int] = Counter()
    by_pincount_in_cap: Counter[int] = Counter()
    cells_by_pincount: dict[int, int] = defaultdict(int)  # in-cap only
    work_by_pincount: dict[int, int] = defaultdict(int)  # cells × (pin-1)

    for _, rects in all_nets.items():
        pins = _net_pins(rects, chip_origin, pitch)
        npins = len(pins)
        if not (MIN_PINS <= npins <= MAX_PINS):
            continue
        n_routable += 1
        by_pincount_routable[npins] += 1

        reg = guide_region(rects, chip_origin, LAYER_ORDER, pitch,
                           margin=4, chip_shape=chip_shape)
        if reg is None:
            continue
        _, nh, nw = reg.shape
        if nh > SUBGRID_AXIS_CAP or nw > SUBGRID_AXIS_CAP:
            continue
        if not all(reg.contains(p) for p in pins):
            continue
        n_in_cap += 1
        by_pincount_in_cap[npins] += 1
        cells_by_pincount[npins] += reg.cell_count
        work_by_pincount[npins] += reg.cell_count * (npins - 1)

    return {
        "pitch": pitch,
        "chip_shape": chip_shape,
        "n_total": n_total,
        "n_routable": n_routable,
        "n_in_cap": n_in_cap,
        "by_pincount_routable": dict(by_pincount_routable),
        "by_pincount_in_cap": dict(by_pincount_in_cap),
        "cells_by_pincount": dict(cells_by_pincount),
        "work_by_pincount": dict(work_by_pincount),
    }


def report(
    pitch: int,
    chip_shape: tuple[int, int, int],
    n_total: int,
    n_routable: int,
    n_in_cap: int,
    by_pincount_in_cap: dict[int, int],
    cells_by_pincount: dict[int, int],
    work_by_pincount: dict[int, int],
) -> None:
    print(f"\n=== pitch {pitch} DBU, chip {chip_shape} ===")
    print(f"  total nets: {n_total:>6d}")
    print(f"  routable (2..20 M1 pins): {n_routable:>6d} "
          f"({100*n_routable/n_total:.1f}%)")
    print(f"  in-cap (routable + has-guide + ≤256² + pins-fit): "
          f"{n_in_cap:>6d} ({100*n_in_cap/max(n_routable,1):.1f}% of routable)")

    nets_total = sum(by_pincount_in_cap.values()) or 1
    cells_total = sum(cells_by_pincount.values()) or 1
    work_total = sum(work_by_pincount.values()) or 1

    print("\n  in-cap distribution (Slice 3 batching population):")
    print(f"    {'pins':>4} {'nets':>7} {'%nets':>6} {'cum%':>6}"
          f"  {'cells':>13} {'%cells':>6} {'cum%':>6}"
          f"  {'work':>14} {'%work':>6} {'cum%':>6}")
    cum_n = cum_c = cum_w = 0
    for pc in sorted(by_pincount_in_cap):
        n = by_pincount_in_cap[pc]
        c = cells_by_pincount.get(pc, 0)
        w = work_by_pincount.get(pc, 0)
        cum_n += n
        cum_c += c
        cum_w += w
        print(f"    {pc:>4d} {n:>7d} {100*n/nets_total:>5.1f}% {100*cum_n/nets_total:>5.1f}%"
              f"  {c:>13d} {100*c/cells_total:>5.1f}% {100*cum_c/cells_total:>5.1f}%"
              f"  {w:>14d} {100*w/work_total:>5.1f}% {100*cum_w/work_total:>5.1f}%")

    pc2_nets = by_pincount_in_cap.get(2, 0)
    pc2_work = work_by_pincount.get(2, 0)
    print("\n  Headlines for Slice 3 strategy:")
    print(f"    2-pin share of in-cap nets:  {100*pc2_nets/max(n_in_cap,1):.1f}%")
    print(f"    2-pin share of sweep-work:   {100*pc2_work/work_total:.1f}%")
    print(f"    ≥3-pin share of in-cap nets: "
          f"{100*(n_in_cap - pc2_nets)/max(n_in_cap,1):.1f}%")
    print(f"    ≥3-pin share of sweep-work:  "
          f"{100*(work_total - pc2_work)/work_total:.1f}%")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pitch", type=int, default=TRACK_PITCH_DBU,
                   help="grid pitch in DBU (default 1120 = track pitch)")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    d = measure(args.pitch)
    report(
        pitch=d["pitch"],  # type: ignore[arg-type]
        chip_shape=d["chip_shape"],  # type: ignore[arg-type]
        n_total=d["n_total"],  # type: ignore[arg-type]
        n_routable=d["n_routable"],  # type: ignore[arg-type]
        n_in_cap=d["n_in_cap"],  # type: ignore[arg-type]
        by_pincount_in_cap=d["by_pincount_in_cap"],  # type: ignore[arg-type]
        cells_by_pincount=d["cells_by_pincount"],  # type: ignore[arg-type]
        work_by_pincount=d["work_by_pincount"],  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    main()
