#!/usr/bin/env python3
"""WS3.3 Slice 2 — Hazard3 tile partition measurement (classify-only).

Loads the Hazard3 LibreLane fixture and reports, for each halo value, the
multi-tile-spanning fraction and per-tile net-count distribution under
`partition_chip` + `classify_nets` (ADR 0012 §3, §6). No routing.

This script gates Slice 6 (the coarsened multi-tile-spanning pass): if
the spanning fraction at halo=32 exceeds 25%, ADR 0012 walk-back §3
(split-across-tiles handshake instead of coarsened pass) is the design
the user must approve before Slice 6 starts. See the plan's Slice 2
section in `docs/plans/ws33-tile-router-implementation.md`.

Run: uv run python scripts/measure_tile_partition.py [--tile-size 256] [--halos 16,32,64]
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time

# Reuse the prototype's I/O helpers.
from _hazard3_io import (
    FINAL_DEF,
    GUIDE,
    PITCH_DBU,
    parse_def_diearea,
    parse_guides,
)
from tile_decomp_prototype import _net_chip_pins

from gpu_pnr.tile_router import classify_nets, net_bbox, partition_chip

logger = logging.getLogger(__name__)


# Same filter the prototype uses to drop clock/power nets; the orchestrator
# wants the spanning fraction of the *routable* net population, not the raw
# guide list (which contains a handful of huge nets that would dominate any
# spanning count). See `scripts/tile_decomp_prototype.py:191`.
MIN_PINS = 2
MAX_PINS = 20


def load_hazard3() -> tuple[int, int, list[list[tuple[int, int, int]]], dict[str, int]]:
    """Load Hazard3 fixture; return (chip_h, chip_w, routable_nets, stats).

    `routable_nets` is the prototype-filtered population (2..20 M1 pin
    cells per net). `stats` reports the filter breakdown so the headline
    table can surface what was excluded.
    """
    logger.info("Loading guides from %s", GUIDE.name)
    all_nets = parse_guides(GUIDE)
    logger.info("  %d total nets in guide file", len(all_nets))

    logger.info("Parsing DIEAREA from %s", FINAL_DEF.name)
    xlo, ylo, xhi, yhi = parse_def_diearea(FINAL_DEF)
    chip_h = (yhi - ylo) // PITCH_DBU + 1
    chip_w = (xhi - xlo) // PITCH_DBU + 1
    logger.info("  chip dimensions: %d × %d cells", chip_h, chip_w)

    chip_origin = (xlo, ylo)
    routable: list[list[tuple[int, int, int]]] = []
    n_empty = 0
    n_too_few = 0
    n_too_many = 0
    for rects in all_nets.values():
        pins = _net_chip_pins(rects, chip_origin)
        pin_count = len(pins)
        if pin_count == 0:
            n_empty += 1
        elif pin_count < MIN_PINS:
            n_too_few += 1
        elif pin_count > MAX_PINS:
            n_too_many += 1
        else:
            routable.append(pins)

    stats = {
        "total_in_guide": len(all_nets),
        "routable": len(routable),
        "excluded_no_m1_pins": n_empty,
        "excluded_single_pin": n_too_few,
        "excluded_over_20_pins": n_too_many,
    }
    logger.info(
        "  %d routable nets after pin-count filter [%d..%d]",
        len(routable), MIN_PINS, MAX_PINS,
    )
    logger.info(
        "  excluded: %d empty, %d single-pin, %d >20-pin (clock/power)",
        n_empty, n_too_few, n_too_many,
    )
    return chip_h, chip_w, routable, stats


def measure_halo(
    chip_h: int,
    chip_w: int,
    nets: list[list[tuple[int, int, int]]],
    tile_size: int,
    halo: int,
) -> dict[str, float | int]:
    """Run partition + classify for one halo value; return measurement dict."""
    tiles = partition_chip(chip_h, chip_w, tile_size=tile_size, halo=halo)
    per_tile, spanning = classify_nets(nets, tiles, halo)

    counts = [len(per_tile.get(t, [])) for t in tiles]
    # Non-empty tile counts give a more useful distribution shape; many
    # tiles on a chip are empty (cell-free regions, etc.).
    nonempty = [c for c in counts if c > 0]

    n_total = len(nets)
    n_span = len(spanning)
    frac = (n_span / n_total) if n_total else 0.0

    if counts:
        cmin = min(counts)
        cmax = max(counts)
        cmean = statistics.mean(counts)
        cmedian = int(statistics.median(counts))
        # p90 via the simple "9th decile of the sorted list" definition.
        sorted_counts = sorted(counts)
        idx = max(0, min(len(sorted_counts) - 1, int(0.9 * len(sorted_counts))))
        p90 = sorted_counts[idx]
    else:
        cmin = cmax = cmedian = p90 = 0
        cmean = 0.0

    return {
        "halo": halo,
        "n_tiles": len(tiles),
        "n_nets": n_total,
        "n_spanning": n_span,
        "frac_spanning": frac,
        "tile_count_min": cmin,
        "tile_count_median": cmedian,
        "tile_count_p90": p90,
        "tile_count_max": cmax,
        "tile_count_mean": cmean,
        "n_nonempty_tiles": len(nonempty),
    }


def print_headline_table(rows: list[dict[str, float | int]], tile_size: int) -> None:
    """Print the headline halo-comparison table to stdout."""
    print(f"\n=== Hazard3 tile partition measurement (tile_size={tile_size}) ===")
    print(
        f"{'halo':>5} {'tiles':>6} {'nets':>6} {'spanning':>9} "
        f"{'%':>6} {'min':>4} {'med':>4} {'p90':>4} {'max':>5} {'mean':>6} {'nonempty':>9}"
    )
    print("-" * 80)
    for r in rows:
        print(
            f"{r['halo']:>5} "
            f"{r['n_tiles']:>6} "
            f"{r['n_nets']:>6} "
            f"{r['n_spanning']:>9} "
            f"{r['frac_spanning'] * 100:>5.1f}% "
            f"{r['tile_count_min']:>4} "
            f"{r['tile_count_median']:>4} "
            f"{r['tile_count_p90']:>4} "
            f"{r['tile_count_max']:>5} "
            f"{r['tile_count_mean']:>6.1f} "
            f"{r['n_nonempty_tiles']:>9}"
        )


_HIST_BUCKETS = [(0, 0, "0"), (1, 5, "1-5"), (6, 10, "6-10"), (11, 20, "11-20"),
                 (21, 50, "21-50"), (51, 100, "51-100"), (101, 10**9, "100+")]

_SPAN_EXTENT_BUCKETS = [(2, 4, "2-4"), (5, 9, "5-9"), (10, 25, "10-25"),
                        (26, 10**9, "26+")]


def _bbox_tiles_touched(bbox: tuple[int, int, int, int], tile_size: int) -> int:
    """Count tile owned regions whose interior intersects the bbox.

    Independent of halo: tells us how many tiles a hypothetical net-split
    walk-back (ADR 0012 walk-back §3) would need to coordinate across.
    """
    rmin, cmin, rmax, cmax = bbox
    n_rows = rmax // tile_size - rmin // tile_size + 1
    n_cols = cmax // tile_size - cmin // tile_size + 1
    return n_rows * n_cols


def print_span_extent_histogram(
    nets: list[list[tuple[int, int, int]]],
    chip_h: int,
    chip_w: int,
    tile_size: int,
    halo: int,
) -> None:
    """For spanning nets at the given halo, histogram bbox-tiles-touched.

    Resolves which ADR 0012 walk-back fits: if most spans touch 2-4
    tiles → net-split (walk-back §3) is tractable; if most touch many
    tiles → coarsened-pass is structurally necessary.
    """
    tiles = partition_chip(chip_h, chip_w, tile_size=tile_size, halo=halo)
    _, spanning = classify_nets(nets, tiles, halo)

    touched = [_bbox_tiles_touched(net_bbox(nets[i]), tile_size) for i in spanning]
    hist = [sum(1 for t in touched if lo <= t <= hi) for lo, hi, _ in _SPAN_EXTENT_BUCKETS]
    n = len(touched)

    print(f"\n=== Spanning-net bbox extent (halo={halo}, tiles touched by bbox) ===")
    print(f"{'tiles touched':>14} {'nets':>6} {'%':>6}")
    print("-" * 32)
    for (_, _, lab), h in zip(_SPAN_EXTENT_BUCKETS, hist):
        pct = (100.0 * h / n) if n else 0.0
        print(f"{lab:>14} {h:>6} {pct:>5.1f}%")
    if touched:
        sorted_touched = sorted(touched)
        median = sorted_touched[len(sorted_touched) // 2]
        p90 = sorted_touched[max(0, min(len(sorted_touched) - 1, int(0.9 * len(sorted_touched))))]
        print(f"\nspanning median={median}, p90={p90}, max={max(touched)}, n={n}")


def print_histogram(
    nets: list[list[tuple[int, int, int]]],
    chip_h: int,
    chip_w: int,
    tile_size: int,
    halo: int,
) -> None:
    """Print a coarse net-count-per-tile histogram for the design-point halo."""
    tiles = partition_chip(chip_h, chip_w, tile_size=tile_size, halo=halo)
    per_tile, _ = classify_nets(nets, tiles, halo)
    counts = [len(per_tile.get(t, [])) for t in tiles]
    hist = [sum(1 for c in counts if lo <= c <= hi) for lo, hi, _ in _HIST_BUCKETS]

    print(f"\n=== Per-tile net-count histogram (halo={halo}) ===")
    print(f"{'bucket':>8} {'tiles':>6}")
    print("-" * 18)
    for (_, _, lab), h in zip(_HIST_BUCKETS, hist):
        print(f"{lab:>8} {h:>6}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tile-size", type=int, default=256,
        help="Owned tile side in cells (ADR 0012 §1 locked at 256).",
    )
    p.add_argument(
        "--halos", type=str, default="16,32,64",
        help="Comma-separated halo widths in cells (default: 16,32,64).",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable INFO-level logging of load/parse progress.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    halos = [int(h) for h in args.halos.split(",") if h.strip()]
    tile_size = args.tile_size

    t0 = time.perf_counter()
    chip_h, chip_w, nets, stats = load_hazard3()
    t_load = time.perf_counter() - t0

    print(
        f"Loaded Hazard3 fixture in {t_load:.2f}s: "
        f"chip {chip_h}×{chip_w}, {stats['routable']} routable nets "
        f"(from {stats['total_in_guide']} total; "
        f"excluded {stats['excluded_no_m1_pins']} empty + "
        f"{stats['excluded_single_pin']} single-pin + "
        f"{stats['excluded_over_20_pins']} >20-pin)."
    )

    t0 = time.perf_counter()
    rows = [measure_halo(chip_h, chip_w, nets, tile_size, h) for h in halos]
    t_measure = time.perf_counter() - t0

    print_headline_table(rows, tile_size)
    # The design-point halo is 32 (ADR 0012 §3).
    if 32 in halos:
        print_histogram(nets, chip_h, chip_w, tile_size, halo=32)
        print_span_extent_histogram(nets, chip_h, chip_w, tile_size, halo=32)

    print(f"\nPartition + classify wall-clock: {t_measure:.2f}s for {len(halos)} halo value(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
