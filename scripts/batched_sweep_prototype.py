#!/usr/bin/env python3
"""WS3.3 batched small-grid sweep prototype — ADR 0012 Amendment 1 §3 /
slot-scale-parallelism spike validation.

The track-pitch prototype (docs/results.md Phase 3.3) measured single-stream
per-net routing as *overhead-bound*: MPS 16.8 ms/net median vs CPU 2.57,
because a ~4k-cell sub-grid starves the GPU — fixed kernel-launch +
convergence-sync cost swamps the tiny compute. The slot-scale spike
(docs/spikes/slot-scale-parallelism.md) named the fix and flagged it as the
load-bearing unproven hypothesis: batch K *independent* small sub-grids into
one kernel call so the per-net launch+sync overhead is amortised instead of
paid K times.

This prototype measures that hypothesis. For a sample of routable, in-cap
Hazard3 nets at the track pitch, it builds each net's guide-constrained
sub-grid, then times the per-net *single-source distance sweep* two ways:

  SEQUENTIAL — K separate `sweep_sssp_3d` calls (one grid each).
  BATCHED    — pad the K sub-grids to a common shape (inf pad = wall), stack to
               (K, L, H, W), one `sweep_sssp_3d_batched` call.

Reports ms/net for each and the batched speedup, plus the padding waste
(padded cells / real cells) so we can judge whether option-A padded-stack
batching already wins or whether size bucketing (option B) is needed to claw
the win back from padding overhead. `--sort-by-size` batches similar-sized nets
together (low padding waste — the best case for option A); the default
(shuffled) order is the realistic-variance case.

The unit is the single-source sweep, matching the slot-scale spike's framing
("ms/net for K independent sub-grids vs K sequential single-grid sweeps"). It
deliberately excludes backtrace and multi-pin tree growth — those add CPU work
that would dilute the kernel-overhead signal this prototype isolates. Device
transfer (H2D) is excluded from the timed window; padding/stack host cost is
reported separately.

Run:
  uv run python scripts/batched_sweep_prototype.py                       # auto device
  uv run python scripts/batched_sweep_prototype.py --device cpu --sample 128 --batch 32
  uv run python scripts/batched_sweep_prototype.py --sort-by-size        # low-variance batches
  uv run python scripts/batched_sweep_prototype.py --pitch 200           # A/B the over-sampled grid
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time
from dataclasses import dataclass

import torch

from _hazard3_io import (
    FINAL_DEF,
    GUIDE,
    LAYER_ORDER,
    apply_pin_access_rules,
    build_chip_grid,
    parse_def_diearea,
    parse_guides,
)

# Reuse the track-pitch prototype's net filter, pin mapper, and percentile
# helper so the two prototypes sample an identical net population.
from track_pitch_sweep_prototype import (
    HAZARD3_ROUTABLE_NETS,
    MAX_PINS,
    MIN_PINS,
    PDK,
    SUBGRID_AXIS_CAP,
    TRACK_PITCH_DBU,
    _net_pins,
    _pct,
)

from gpu_pnr.guides import guide_region
from gpu_pnr.sweep import sweep_sssp_3d, sweep_sssp_3d_batched

logger = logging.getLogger(__name__)

# Match the routed track-pitch prototype's via cost so the sweep cost model is
# identical; the throughput question is insensitive to the exact value.
VIA_COST = 5.0


@dataclass
class SubGrid:
    """One net's guide-constrained sub-grid, ready to sweep (on CPU)."""

    w: torch.Tensor  # (L, H, W), pin-access rules applied, inf for obstacles
    source: tuple[int, int, int]  # first pin, in sub-grid coords
    cells: int  # L*H*W of the region bbox (the real, unpadded cell count)


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def collect_subgrids(
    all_nets: dict[str, list[tuple[int, int, int, int, str]]],
    chip_origin: tuple[int, int],
    chip_shape: tuple[int, int, int],
    w_chip: torch.Tensor,
    pitch: int,
    margin: int,
    sample: int,
    seed: int,
) -> tuple[list[SubGrid], dict[str, int]]:
    """Slice `sample` in-cap routable nets into independent CPU sub-grids."""
    names = list(all_nets.keys())
    random.Random(seed).shuffle(names)
    out: list[SubGrid] = []
    skip = {"not_routable": 0, "no_guide": 0, "over_cap": 0, "off_region": 0,
            "pin_on_inf": 0}
    for name in names:
        if len(out) >= sample:
            break
        rects = all_nets[name]
        pins = _net_pins(rects, chip_origin, pitch)
        if not (MIN_PINS <= len(pins) <= MAX_PINS):
            skip["not_routable"] += 1
            continue
        reg = guide_region(
            rects, chip_origin, LAYER_ORDER, pitch,
            margin=margin, chip_shape=chip_shape,
        )
        if reg is None:
            skip["no_guide"] += 1
            continue
        _, nh, nw = reg.shape
        if nh > SUBGRID_AXIS_CAP or nw > SUBGRID_AXIS_CAP:
            skip["over_cap"] += 1
            continue
        if not all(reg.contains(p) for p in pins):
            skip["off_region"] += 1
            continue
        assert reg.l0 == 0, f"expected M1-anchored region, got l0={reg.l0}"

        w_sub = w_chip[reg.l0:reg.l1, reg.r0:reg.r1, reg.c0:reg.c1].clone()
        local = [reg.rebase(p) for p in pins]
        apply_pin_access_rules(w_sub, PDK, local)
        source = local[0]
        if not torch.isfinite(w_sub[source]):
            skip["pin_on_inf"] += 1
            continue
        out.append(SubGrid(w_sub, source, reg.cell_count))
    return out, skip


def pad_stack(
    batch: list[SubGrid], device: str
) -> tuple[torch.Tensor, list[tuple[int, int, int]], tuple[int, int, int], float]:
    """Pad K sub-grids to a common shape (inf = wall), stack to (K,L,H,W).

    Returns (batched_tensor_on_device, sources, padded_shape, host_ms)."""
    t0 = time.perf_counter()
    lmax = max(sg.w.shape[0] for sg in batch)
    hmax = max(sg.w.shape[1] for sg in batch)
    wmax = max(sg.w.shape[2] for sg in batch)
    k = len(batch)
    t = torch.full((k, lmax, hmax, wmax), math.inf)
    for i, sg in enumerate(batch):
        gl, gh, gw = sg.w.shape
        t[i, :gl, :gh, :gw] = sg.w
    sources = [sg.source for sg in batch]
    host_ms = (time.perf_counter() - t0) * 1000.0
    return t.to(device), sources, (lmax, hmax, wmax), host_ms


def time_sequential(batch: list[SubGrid], device: str) -> tuple[float, list[int]]:
    """K separate single-grid sweeps. H2D excluded from the timed window."""
    devs = [(sg.w.to(device), sg.source) for sg in batch]
    _sync(device)
    t0 = time.perf_counter()
    iters = [sweep_sssp_3d(w, s, via_cost=VIA_COST)[1] for w, s in devs]
    _sync(device)
    return (time.perf_counter() - t0) * 1000.0, iters


def time_batched(
    batch: list[SubGrid], device: str
) -> tuple[float, int, tuple[int, int, int], float, torch.Tensor]:
    """One batched sweep over the padded stack. H2D + pad/stack excluded."""
    batch_dev, sources, shape, host_ms = pad_stack(batch, device)
    _sync(device)
    t0 = time.perf_counter()
    d, iters = sweep_sssp_3d_batched(batch_dev, sources, via_cost=VIA_COST)
    _sync(device)
    return (time.perf_counter() - t0) * 1000.0, iters, shape, host_ms, d


def _warmup(device: str) -> None:
    """One throwaway call to each kernel so first-timed run doesn't eat MPS
    shader compilation / allocator setup."""
    w = torch.full((2, 8, 8), 1.0).to(device)
    sweep_sssp_3d(w, (0, 0, 0), via_cost=VIA_COST)
    wb = torch.full((2, 2, 8, 8), 1.0).to(device)
    sweep_sssp_3d_batched(wb, [(0, 0, 0), (0, 7, 7)], via_cost=VIA_COST)
    _sync(device)


def _check_equivalence(batch: list[SubGrid], d_batched: torch.Tensor,
                       device: str) -> float:
    """Max abs diff (finite cells) between batched and sequential for the first
    net — a cheap correctness sanity check on real data."""
    sg = batch[0]
    d_single, _ = sweep_sssp_3d(sg.w.to(device), sg.source, via_cost=VIA_COST)
    gl, gh, gw = sg.w.shape
    got = d_batched[0, :gl, :gh, :gw].cpu()
    ref = d_single.cpu()
    finite = torch.isfinite(ref)
    if not finite.any():
        return 0.0
    return float((got[finite] - ref[finite]).abs().max().item())


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pitch", type=int, default=TRACK_PITCH_DBU,
                   help="grid pitch in DBU (default 1120 = gf180mcuD track pitch)")
    p.add_argument("--device", type=str, default="auto",
                   help="auto | mps | cpu (default auto: mps if available)")
    p.add_argument("--sample", type=int, default=128,
                   help="number of in-cap nets to sweep for timing")
    p.add_argument("--batch", type=int, default=16,
                   help="K: nets per batched kernel call (memory grows with "
                        "K × padded sub-grid; 16 is safe for 256-cap nets)")
    p.add_argument("--sort-by-size", action="store_true",
                   help="batch similar-sized nets together (low padding waste — "
                        "best case for option A). Default: shuffled order.")
    p.add_argument("--margin", type=int, default=4, help="guide_region margin (cells)")
    p.add_argument("--seed", type=int, default=0, help="sample shuffle seed")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )
    device = ("mps" if torch.backends.mps.is_available() else "cpu") \
        if args.device == "auto" else args.device

    all_nets = parse_guides(GUIDE)
    xlo, ylo, xhi, yhi = parse_def_diearea(FINAL_DEF)
    chip_origin = (xlo, ylo)
    chip_h = (yhi - ylo) // args.pitch + 1
    chip_w = (xhi - xlo) // args.pitch + 1
    chip_shape = (len(LAYER_ORDER), chip_h, chip_w)

    print(f"Batched small-grid sweep prototype — pitch {args.pitch} DBU, "
          f"device {device}", flush=True)
    print(f"  batch K={args.batch}, sample={args.sample}, "
          f"{'size-sorted' if args.sort_by_size else 'shuffled'} batches",
          flush=True)

    t0 = time.perf_counter()
    print("  building chip-scale cost grid...", flush=True)
    w_chip = build_chip_grid(all_nets, xlo, ylo, xhi, yhi, pitch_dbu=args.pitch)
    print(f"    shape {tuple(w_chip.shape)} in {time.perf_counter() - t0:.1f}s",
          flush=True)

    subgrids, skip = collect_subgrids(
        all_nets, chip_origin, chip_shape, w_chip, args.pitch,
        args.margin, args.sample, args.seed,
    )
    if not subgrids:
        print("  no sub-grids collected; nothing to measure.", flush=True)
        return
    if args.sort_by_size:
        subgrids.sort(key=lambda sg: sg.cells)

    cells = sorted(sg.cells for sg in subgrids)
    print(f"\n  collected {len(subgrids)} in-cap nets "
          f"(skipped: {skip['over_cap']} over-cap, {skip['no_guide']} no-guide, "
          f"{skip['off_region']} off-region, {skip['pin_on_inf']} pin-on-inf)",
          flush=True)
    print(f"  sub-grid cells: median={_pct(cells, 0.5):.0f} "
          f"p90={_pct(cells, 0.9):.0f} max={cells[-1]}", flush=True)

    _warmup(device)

    batches = [subgrids[i:i + args.batch]
               for i in range(0, len(subgrids), args.batch)]
    seq_per_net: list[float] = []
    bat_per_net: list[float] = []
    waste: list[float] = []
    pad_host_ms: list[float] = []
    bat_iters_all: list[int] = []
    tot_seq = tot_bat = 0.0
    checked = None
    for batch in batches:
        seq_ms, _ = time_sequential(batch, device)
        bat_ms, bat_iters, shape, host_ms, d = time_batched(batch, device)
        # `d` is only consumed for a one-off correctness check on the first
        # batch; later batches run the sweep purely for timing.
        if checked is None:
            checked = _check_equivalence(batch, d, device)
        k = len(batch)
        real = sum(sg.cells for sg in batch)
        padded = shape[0] * shape[1] * shape[2] * k
        seq_per_net.append(seq_ms / k)
        bat_per_net.append(bat_ms / k)
        waste.append(padded / real)
        pad_host_ms.append(host_ms / k)
        bat_iters_all.append(bat_iters)
        tot_seq += seq_ms
        tot_bat += bat_ms

    sq = sorted(seq_per_net)
    bq = sorted(bat_per_net)
    wq = sorted(waste)
    speedup = tot_seq / tot_bat if tot_bat else float("nan")

    print(f"\n=== batched vs sequential single-source sweep ({device}) ===",
          flush=True)
    print(f"  {len(batches)} batches of up to K={args.batch}; correctness "
          f"check (net 0, max |Δdist|): {checked:.3g}", flush=True)
    print(f"  {'':<22}{'median':>10}{'mean':>10}{'p90':>10}", flush=True)
    print(f"  {'sequential ms/net':<22}{_pct(sq, 0.5):>10.2f}"
          f"{sum(sq)/len(sq):>10.2f}{_pct(sq, 0.9):>10.2f}", flush=True)
    print(f"  {'batched ms/net':<22}{_pct(bq, 0.5):>10.2f}"
          f"{sum(bq)/len(bq):>10.2f}{_pct(bq, 0.9):>10.2f}", flush=True)
    print(f"  {'padding waste (×)':<22}{_pct(wq, 0.5):>10.2f}"
          f"{sum(wq)/len(wq):>10.2f}{_pct(wq, 0.9):>10.2f}", flush=True)
    print(f"\n  overall: sequential {tot_seq/len(subgrids):.2f} ms/net, "
          f"batched {tot_bat/len(subgrids):.2f} ms/net "
          f"→ {speedup:.2f}× {'faster' if speedup > 1 else 'SLOWER'} batched",
          flush=True)
    print(f"  pad/stack host overhead (excluded above): "
          f"{sum(pad_host_ms)/len(pad_host_ms):.3f} ms/net", flush=True)
    print(f"  batched sweep iters/call (slowest net bounds the batch): "
          f"mean {sum(bat_iters_all)/len(bat_iters_all):.0f}, "
          f"max {max(bat_iters_all)}", flush=True)
    seq_total = tot_seq / len(subgrids) * HAZARD3_ROUTABLE_NETS / 1000.0
    bat_total = tot_bat / len(subgrids) * HAZARD3_ROUTABLE_NETS / 1000.0
    print(f"  whole-chip extrapolation ({HAZARD3_ROUTABLE_NETS} nets): "
          f"sequential {seq_total:.0f}s → batched {bat_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
