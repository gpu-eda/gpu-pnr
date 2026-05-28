#!/usr/bin/env python3
"""WS3.3 guide-constrained sweep — Hazard3 sub-grid size measurement.

Loads the Hazard3 LibreLane fixture and reports, per net, the size of the
guide-constrained sweep sub-grid produced by `gpu_pnr.guides.guide_region`
(ADR 0012 Amendment 1). No routing — this is the size-distribution
measurement that validates (or refutes) the Amendment's "~3,000 cells,
~0.16 ms/net" throughput estimate before the sweep prototype is built.

For each net we map its GRT guide rectangles to a `(L,H,W)` sub-grid bbox
(union of guides + margin, contiguous layer span) and record the cell
count. We then compare against:

  - The Amendment's 3,000-cell / 0.16 ms-net estimate.
  - The full 256²×5 = 327,680-cell grid the un-constrained sweep visits
    (18 ms/net on M4 Pro MPS, 31 ms/net on M2 CI — see the GPU-vs-DRT
    spike).
  - The 256² per-axis cap: nets whose H or W exceeds 256 would need the
    coarsened-pass fallback (Amendment §7), not a direct sub-grid sweep.

The per-net ms estimate is a *linear* extrapolation (ms ∝ cell_count)
from the 256²×5 baseline. Real sweep cost has fixed per-call overhead and
iteration-count effects, so treat this as an optimistic lower bound — the
actual figure is what the sweep prototype (follow-up 2) measures.

Run: uv run python scripts/measure_guide_regions.py [--margin 4] [-v]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

# Reuse the fixture I/O helpers (same pattern as measure_tile_partition.py).
from _hazard3_io import (
    FINAL_DEF,
    GUIDE,
    LAYER_ORDER,
    PITCH_DBU,
    parse_def_diearea,
    parse_guides,
)
from tile_decomp_prototype import _net_chip_pins

from gpu_pnr.guides import guide_region

logger = logging.getLogger(__name__)

# Same routable filter the prototype / Slice-2 measurement use: drop nets
# with no M1 pins, single-pin nets, and >20-pin clock/power nets. See
# scripts/tile_decomp_prototype.py:191.
MIN_PINS = 2
MAX_PINS = 20

# Throughput baselines from docs/spikes/gpu-vs-drt-throughput.md.
FULL_GRID_CELLS = 256 * 256 * 5  # 327,680 — un-constrained sweep footprint
MS_PER_NET_FULL_M4 = 18.0  # M4 Pro MPS, full 256²×5 grid
MS_PER_NET_FULL_M2 = 31.0  # M2 Mac Mini CI golden, same grid
# ADR 0012 Amendment 1 estimate, for comparison.
ADR_ESTIMATE_CELLS = 50 * 30 * 2  # 3,000
ADR_ESTIMATE_MS = 0.16
# Per-axis cap above which a net routes via the coarsened-pass fallback.
SUBGRID_AXIS_CAP = 256


@dataclass
class RegionStats:
    """Guide-region size distribution for the routable Hazard3 net population."""

    chip_shape: tuple[int, int, int]
    n_total: int
    n_excluded: int
    n_none: int
    n_over_cap: int = 0
    cell_counts: list[int] = field(default_factory=list)
    heights: list[int] = field(default_factory=list)
    widths: list[int] = field(default_factory=list)
    layer_spans: list[int] = field(default_factory=list)

    @property
    def n_routable(self) -> int:
        return len(self.cell_counts)


def _pct(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile `q` (0..1) of an already-sorted list."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def measure(margin: int, pitch: int = PITCH_DBU) -> RegionStats:
    """Compute the guide-region size distribution for routable Hazard3 nets.

    `pitch` is the cost-tensor grid pitch in DBU. The default (200) is the
    over-sampled grid the router currently uses; pass the true track pitch
    (1120 for gf180mcuD M1-M4) to see the track-resolution distribution.
    """
    logger.info("Loading guides from %s", GUIDE.name)
    all_nets = parse_guides(GUIDE)
    logger.info("  %d total nets in guide file", len(all_nets))

    logger.info("Parsing DIEAREA from %s", FINAL_DEF.name)
    xlo, ylo, xhi, yhi = parse_def_diearea(FINAL_DEF)
    chip_h = (yhi - ylo) // pitch + 1
    chip_w = (xhi - xlo) // pitch + 1
    chip_shape = (len(LAYER_ORDER), chip_h, chip_w)
    chip_origin = (xlo, ylo)
    logger.info("  chip dimensions: %d × %d × %d cells (pitch %d DBU)",
                *chip_shape, pitch)

    stats = RegionStats(
        chip_shape=chip_shape, n_total=len(all_nets), n_excluded=0, n_none=0,
    )
    for rects in all_nets.values():
        pins = _net_chip_pins(rects, chip_origin)
        if not (MIN_PINS <= len(pins) <= MAX_PINS):
            stats.n_excluded += 1
            continue
        reg = guide_region(
            rects, chip_origin, LAYER_ORDER, pitch,
            margin=margin, chip_shape=chip_shape,
        )
        if reg is None:
            stats.n_none += 1
            continue
        nl, nh, nw = reg.shape
        stats.cell_counts.append(reg.cell_count)
        stats.heights.append(nh)
        stats.widths.append(nw)
        stats.layer_spans.append(nl)
        if nh > SUBGRID_AXIS_CAP or nw > SUBGRID_AXIS_CAP:
            stats.n_over_cap += 1

    return stats


def _fmt_dist(name: str, vals: list[int], unit: str = "") -> str:
    if not vals:
        return f"  {name:<18} (no data)"
    s = sorted(vals)
    return (
        f"  {name:<18} "
        f"min={s[0]:>8}{unit}  "
        f"median={_pct(s, 0.5):>10.0f}{unit}  "
        f"p90={_pct(s, 0.9):>10.0f}{unit}  "
        f"p99={_pct(s, 0.99):>11.0f}{unit}  "
        f"max={s[-1]:>10}{unit}"
    )


def print_report(res: RegionStats, margin: int) -> None:
    cell_counts = res.cell_counts
    n_routable = res.n_routable
    cl, ch, cw = res.chip_shape

    print(
        f"\nHazard3 guide-region sizes (margin={margin} cells, "
        f"chip {cl}×{ch}×{cw}):"
    )
    print(
        f"  {res.n_total} nets total; "
        f"{res.n_excluded} excluded (pin filter [{MIN_PINS}..{MAX_PINS}]); "
        f"{res.n_none} with no routable guide; "
        f"{n_routable} measured."
    )

    print("\nPer-net sub-grid dimensions:")
    print(_fmt_dist("rows (H)", res.heights, " c"))
    print(_fmt_dist("cols (W)", res.widths, " c"))
    print(_fmt_dist("layers (L)", res.layer_spans, ""))
    print(_fmt_dist("cells (L·H·W)", cell_counts))

    if not cell_counts:
        return

    s = sorted(cell_counts)
    median_cells = _pct(s, 0.5)
    p90_cells = _pct(s, 0.9)
    total_cells = sum(cell_counts)
    mean_cells = total_cells / n_routable

    print("\nThroughput (LINEAR ms ∝ cells extrapolation — optimistic):")
    print(
        f"  baseline: full 256²×5 = {FULL_GRID_CELLS:,} cells "
        f"@ {MS_PER_NET_FULL_M4:.0f} ms/net (M4 Pro MPS), "
        f"{MS_PER_NET_FULL_M2:.0f} ms/net (M2 CI)."
    )
    for label, base_ms in (("M4 Pro", MS_PER_NET_FULL_M4), ("M2 CI", MS_PER_NET_FULL_M2)):
        med_ms = base_ms * median_cells / FULL_GRID_CELLS
        p90_ms = base_ms * p90_cells / FULL_GRID_CELLS
        total_s = base_ms * total_cells / FULL_GRID_CELLS / 1000.0
        print(
            f"  {label:<7} median {med_ms:6.2f} ms/net  "
            f"p90 {p90_ms:6.2f} ms/net  "
            f"total {total_s:6.1f} s for {n_routable} nets"
        )

    print("\nVs ADR 0012 Amendment 1 estimate:")
    print(
        f"  ADR assumed ~{ADR_ESTIMATE_CELLS:,} cells "
        f"({ADR_ESTIMATE_MS} ms/net). "
        f"Measured median {median_cells:,.0f} cells "
        f"({median_cells / ADR_ESTIMATE_CELLS:.1f}× the estimate), "
        f"mean {mean_cells:,.0f}."
    )
    n_over = res.n_over_cap
    print(
        f"  {n_over}/{n_routable} ({100 * n_over / n_routable:.1f}%) exceed the "
        f"{SUBGRID_AXIS_CAP}² per-axis cap → coarsened-pass fallback."
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--margin", type=int, default=4,
        help="Guide-bbox margin in grid cells (gpu_pnr.guides default: 4).",
    )
    p.add_argument(
        "--pitch", type=int, default=PITCH_DBU,
        help=(
            "Cost-tensor grid pitch in DBU (default: %(default)s, the "
            "over-sampled router grid). True gf180mcuD track pitch is 1120."
        ),
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

    t0 = time.perf_counter()
    res = measure(args.margin, args.pitch)
    elapsed = time.perf_counter() - t0

    print_report(res, args.margin)
    print(f"\nMeasurement wall-clock: {elapsed:.2f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
