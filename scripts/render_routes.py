#!/usr/bin/env python3
"""Per-layer visual diff of our routes vs TritonRoute.

Two modes:

* `chip` (default): take the same smallest-N two-pin sample as
  spike_route_many_nets.py, route them, and overlay our + TR's wire
  footprints on a chip-scale canvas. One PNG per metal layer.

* `net <name>`: render a single named net's per-layer paths from
  our router + TR side-by-side on the net's own bounding-box, with
  no downsample. One panel per layer the net touches.

In both modes the overlay convention is:
  - blue (B)    -> only our wire
  - orange (O)  -> only TR's wire
  - purple (P)  -> both routers placed wire here
  - grey        -> guide region for this net (chip mode: omitted; net mode: shown)

Run:
  uv run python scripts/render_routes.py [--mode chip|net] [--n N] [--net NAME]
                                         [--scale K] [--out DIR] [--off-mult X]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from _hazard3_io import (
    FINAL_DEF,
    GUIDE,
    LAYER_ORDER,
    PITCH_DBU,
    build_grid,
    parse_def_diearea,
    parse_def_net_geometry,
    parse_guides,
    rect_center_to_grid,
)
from gpu_pnr.router import route_nets_3d
from gpu_pnr.sweep import axis_costs

# gf180mcuD preferred direction per metal layer, aligned with LAYER_ORDER.
PREFERRED_DIRECTION = ["H", "V", "H", "V", "H"]


def _preferred_direction_multipliers(
    off_mult: float, m1_penalty: float = 1.0
) -> tuple[list[float], list[float]]:
    """Return (h_mult, v_mult) for gf180mcuD's M1=H, M2=V, ... alternation.

    When `m1_penalty` > 1.0, M1 is treated as pin-access-only: both H and
    V on M1 are made expensive (h_mult[0] = v_mult[0] = m1_penalty),
    overriding the preferred-direction off_mult assignment for that
    layer. Real gf180mcuD DR forbids M1 wire entirely; the
    preferred-direction model alone leaves M1-H cheap which the router
    happily exploits, so this knob recovers TR's pin-only convention.
    """
    h = [1.0 if d == "H" else off_mult for d in PREFERRED_DIRECTION]
    v = [1.0 if d == "V" else off_mult for d in PREFERRED_DIRECTION]
    if m1_penalty != 1.0:
        h[0] = m1_penalty
        v[0] = m1_penalty
    return h, v


def _stamp_segment(
    canvas: np.ndarray, layer_idx: int, x0: int, y0: int, x1: int, y1: int
) -> None:
    """Mark every cell along an axis-aligned segment (DEF DBU coords) on canvas.

    Segments in the DEF are always horizontal or vertical -- diagonals don't
    occur. If a segment is diagonal it indicates a parser bug; we stamp only
    the endpoints in that case to avoid silently dropping data.
    """
    H, W = canvas.shape[-2:]
    c0 = x0 // PITCH_DBU
    c1 = x1 // PITCH_DBU
    r0 = y0 // PITCH_DBU
    r1 = y1 // PITCH_DBU
    if r0 == r1:
        ca, cb = sorted((c0, c1))
        if 0 <= r0 < H:
            canvas[layer_idx, r0, max(0, ca) : min(W, cb + 1)] = True
    elif c0 == c1:
        ra, rb = sorted((r0, r1))
        if 0 <= c0 < W:
            canvas[layer_idx, max(0, ra) : min(H, rb + 1), c0] = True
    else:
        if 0 <= r0 < H and 0 <= c0 < W:
            canvas[layer_idx, r0, c0] = True
        if 0 <= r1 < H and 0 <= c1 < W:
            canvas[layer_idx, r1, c1] = True


def _stamp_via(canvas: np.ndarray, via_name: str, x: int, y: int) -> None:
    """Stamp a via's anchor cell on both layers it connects.

    DEF via naming convention: `Via<N>_<dir>` connects Metal<N> and Metal<N+1>
    (N is 1-indexed in the name, 0-indexed in LAYER_ORDER).
    """
    if not via_name.startswith("Via"):
        return
    digits = ""
    for ch in via_name[3:]:
        if ch.isdigit():
            digits += ch
        else:
            break
    if not digits:
        return
    n = int(digits)
    lower_idx = n - 1
    upper_idx = n
    H, W = canvas.shape[-2:]
    r = y // PITCH_DBU
    c = x // PITCH_DBU
    if not (0 <= r < H and 0 <= c < W):
        return
    if 0 <= lower_idx < canvas.shape[0]:
        canvas[lower_idx, r, c] = True
    if 0 <= upper_idx < canvas.shape[0]:
        canvas[upper_idx, r, c] = True


def _overlay_rgb(
    ours: np.ndarray, theirs: np.ndarray, only_ours_rgb=(60, 120, 220),
    only_theirs_rgb=(230, 130, 30), both_rgb=(130, 70, 180),
    bg_rgb=(245, 245, 245),
) -> np.ndarray:
    """Compose an (H, W, 3) uint8 image from two (H, W) bool masks."""
    H, W = ours.shape
    img = np.empty((H, W, 3), dtype=np.uint8)
    img[:] = bg_rgb
    only_o = ours & ~theirs
    only_t = theirs & ~ours
    both = ours & theirs
    img[only_o] = only_ours_rgb
    img[only_t] = only_theirs_rgb
    img[both] = both_rgb
    return img


def _downsample_or_canvas(canvas: np.ndarray, scale: int) -> np.ndarray:
    """Bin `scale x scale` cells into one pixel; any True in the bin -> set.

    Done via slice-OR over `scale*scale` shifted views so the type checker
    stays happy; equivalent to reshape-and-any.
    """
    if scale <= 1:
        return canvas
    L, H, W = canvas.shape
    H_out = H // scale
    W_out = W // scale
    if H_out == 0 or W_out == 0:
        return canvas
    out = np.zeros((L, H_out, W_out), dtype=bool)
    for dr in range(scale):
        for dc in range(scale):
            out |= canvas[:, dr:H_out * scale:scale, dc:W_out * scale:scale]
    return out


def _sample_two_pin(n: int) -> list[tuple[str, list]]:
    all_nets = parse_guides(GUIDE)
    two_pin = [
        (name, rects) for name, rects in all_nets.items()
        if sum(1 for r in rects if r[4] == "Metal1") == 2
    ]
    two_pin.sort(key=lambda nr: len(nr[1]))
    return two_pin[:n]


def _apply_m1_pin_only(w: torch.Tensor, pins: list[tuple[int, int, int]]) -> None:
    """In-place: mark all M1 cells as obstacles, then carve a 1-cell-radius
    landing pad around each pin coord. Forces wire body onto M2+ regardless
    of cost balance -- the strictest approximation of gf180mcuD's no-M1-wire
    DRC convention. Pin reservation in route_nets_3d still works because
    it operates on specific coords (which we restore here as finite).
    """
    H, W = w.shape[1], w.shape[2]
    w[0] = float("inf")
    for pl, pr, pc in pins:
        if pl != 0:
            continue
        rlo, rhi = max(0, pr - 1), min(H, pr + 2)
        clo, chi = max(0, pc - 1), min(W, pc + 2)
        w[0, rlo:rhi, clo:chi] = 1.0


def render_chip(
    n: int,
    scale: int,
    out_dir: Path,
    off_mult: float,
    m1_penalty: float,
    m1_pin_only: bool,
) -> None:
    """Chip-scale per-layer overlay over the N smallest two-pin nets."""
    xlo, ylo, xhi, yhi = parse_def_diearea(FINAL_DEF)
    H_chip = (yhi - ylo) // PITCH_DBU + 1
    W_chip = (xhi - xlo) // PITCH_DBU + 1
    L = len(LAYER_ORDER)
    ours = np.zeros((L, H_chip, W_chip), dtype=bool)
    theirs = np.zeros((L, H_chip, W_chip), dtype=bool)

    sample = _sample_two_pin(n)
    geo = parse_def_net_geometry(FINAL_DEF)
    print(f"chip canvas: {H_chip}x{W_chip} cells, scale={scale} -> "
          f"{H_chip // scale}x{W_chip // scale} px")
    print(f"routing {len(sample)} nets...")

    h_mult: list[float] | None
    v_mult: list[float] | None
    if m1_pin_only:
        m1_penalty = 1.0  # superseded by the pin-only mask
        print("  M1 mode: pin-cell only (non-pin M1 cells set to inf)")
    if off_mult != 1.0 or m1_penalty != 1.0:
        h_mult, v_mult = _preferred_direction_multipliers(off_mult, m1_penalty)
        print(f"  off_mult={off_mult}; m1_penalty={m1_penalty};")
        print(f"  h_mult={h_mult}; v_mult={v_mult}")
    else:
        h_mult = v_mult = None

    t0 = time.perf_counter()
    routed = missing_in_tr = 0
    for net_name, rects in sample:
        try:
            w, origin = build_grid(rects)
            m1 = [r for r in rects if r[4] == "Metal1"]
            src = rect_center_to_grid(m1[0], origin)
            snk = rect_center_to_grid(m1[1], origin)
            if m1_pin_only:
                _apply_m1_pin_only(w, [src, snk])
            if h_mult is not None and v_mult is not None:
                w_h, w_v = axis_costs(w, h_mult, v_mult)
            else:
                w_h, w_v = w, None
        except (ValueError, IndexError):
            continue
        res = route_nets_3d(w_h, [(src, snk)], via_cost=5.0, w_v=w_v)
        path = res[0].path
        if path is None:
            continue
        routed += 1
        # Map our (layer, row, col) cells back to chip cells.
        ox = origin[0] // PITCH_DBU
        oy = origin[1] // PITCH_DBU
        for lyr, r, c in path:
            chip_r = oy + r
            chip_c = ox + c
            if 0 <= chip_r < H_chip and 0 <= chip_c < W_chip:
                ours[lyr, chip_r, chip_c] = True
        # TR geometry, if present.
        if net_name not in geo:
            missing_in_tr += 1
            continue
        segs, tr_vias = geo[net_name]
        for layer_name, x0, y0, x1, y1 in segs:
            if layer_name not in LAYER_ORDER:
                continue
            lyr = LAYER_ORDER.index(layer_name)
            _stamp_segment(theirs, lyr, x0 - xlo, y0 - ylo, x1 - xlo, y1 - ylo)
        for _, vx, vy, via_name in tr_vias:
            _stamp_via(theirs, via_name, vx - xlo, vy - ylo)
    print(f"  routed {routed}/{len(sample)} ({missing_in_tr} missing TR geometry)"
          f" in {time.perf_counter() - t0:.1f}s")

    ours_ds = _downsample_or_canvas(ours, scale)
    theirs_ds = _downsample_or_canvas(theirs, scale)
    out_dir.mkdir(parents=True, exist_ok=True)
    for lyr_idx, layer_name in enumerate(LAYER_ORDER):
        img = _overlay_rgb(ours_ds[lyr_idx], theirs_ds[lyr_idx])
        # Flip Y so the chip looks like a layout (origin at bottom-left).
        img = np.flipud(img)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, interpolation="nearest")
        only_o = int(np.sum(ours_ds[lyr_idx] & ~theirs_ds[lyr_idx]))
        only_t = int(np.sum(theirs_ds[lyr_idx] & ~ours_ds[lyr_idx]))
        both = int(np.sum(ours_ds[lyr_idx] & theirs_ds[lyr_idx]))
        ax.set_title(
            f"{layer_name} (px @ {scale}x bin): "
            f"only-ours={only_o}  only-TR={only_t}  both={both}  "
            f"pref={PREFERRED_DIRECTION[lyr_idx]}"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        # Inline legend strip in figure caption space.
        ax.text(0.01, -0.04, "blue=only-ours  orange=only-TR  purple=both",
                transform=ax.transAxes, fontsize=9, color="dimgray")
        out_path = out_dir / f"chip_{lyr_idx}_{layer_name}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out_path}")


def render_net(
    net_name: str,
    out_dir: Path,
    off_mult: float,
    m1_penalty: float,
    m1_pin_only: bool,
) -> None:
    """Single-net per-layer overlay on the net's own bbox."""
    all_nets = parse_guides(GUIDE)
    if net_name not in all_nets:
        raise SystemExit(f"net {net_name!r} not in guides")
    rects = all_nets[net_name]
    w, origin = build_grid(rects)
    L, H, W = w.shape
    print(f"net {net_name}: per-net grid {L}x{H}x{W}; origin={origin}")

    m1 = [r for r in rects if r[4] == "Metal1"]
    if len(m1) < 2:
        raise SystemExit(f"net {net_name} has fewer than 2 Metal1 pins; net mode "
                         f"currently expects 2-pin nets")
    src = rect_center_to_grid(m1[0], origin)
    snk = rect_center_to_grid(m1[1], origin)
    if m1_pin_only:
        _apply_m1_pin_only(w, [src, snk])
        m1_penalty = 1.0

    h_mult: list[float] | None
    v_mult: list[float] | None
    if off_mult != 1.0 or m1_penalty != 1.0:
        h_mult, v_mult = _preferred_direction_multipliers(off_mult, m1_penalty)
    else:
        h_mult = v_mult = None
    if h_mult is not None and v_mult is not None:
        w_h, w_v = axis_costs(w, h_mult, v_mult)
    else:
        w_h, w_v = w, None
    res = route_nets_3d(w_h, [(src, snk)], via_cost=5.0, w_v=w_v)
    path = res[0].path

    ours = np.zeros((L, H, W), dtype=bool)
    if path is None:
        print("  our router returned None for this net")
    else:
        for lyr, r, c in path:
            ours[lyr, r, c] = True

    theirs = np.zeros((L, H, W), dtype=bool)
    geo = parse_def_net_geometry(FINAL_DEF)
    segs, vias = geo.get(net_name, ([], []))
    for layer_name, x0, y0, x1, y1 in segs:
        if layer_name not in LAYER_ORDER:
            continue
        lyr = LAYER_ORDER.index(layer_name)
        _stamp_segment(theirs, lyr, x0 - origin[0], y0 - origin[1],
                       x1 - origin[0], y1 - origin[1])
    for _, vx, vy, via_name in vias:
        _stamp_via(theirs, via_name, vx - origin[0], vy - origin[1])

    out_dir.mkdir(parents=True, exist_ok=True)
    guide_mask = torch.isfinite(w).cpu().numpy()
    for lyr_idx, layer_name in enumerate(LAYER_ORDER):
        ours_l = ours[lyr_idx]
        theirs_l = theirs[lyr_idx]
        guide_l = guide_mask[lyr_idx]
        if not (ours_l.any() or theirs_l.any() or guide_l.any()):
            continue
        # Build a 3-channel image: grey background, dim grey for guide,
        # then colored overlay from ours/theirs.
        img = np.full((H, W, 3), 245, dtype=np.uint8)
        img[guide_l] = (215, 215, 215)
        img[ours_l & ~theirs_l] = (60, 120, 220)
        img[theirs_l & ~ours_l] = (230, 130, 30)
        img[ours_l & theirs_l] = (130, 70, 180)
        img = np.flipud(img)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img, interpolation="nearest")
        ax.set_title(
            f"{net_name} -- {layer_name} (pref={PREFERRED_DIRECTION[lyr_idx]}); "
            f"ours={int(ours_l.sum())} cells, TR={int(theirs_l.sum())} cells"
        )
        ax.set_xlabel(f"src={src} snk={snk}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        out_path = out_dir / f"net_{net_name.strip('_')}_{lyr_idx}_{layer_name}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("chip", "net"), default="chip")
    p.add_argument("--n", type=int, default=500,
                   help="chip mode: number of smallest 2-pin nets")
    p.add_argument("--net", type=str, default=None,
                   help="net mode: net name (required for --mode net)")
    p.add_argument("--scale", type=int, default=4,
                   help="chip mode: NxN grid-cell bin per output pixel")
    p.add_argument("--out", type=Path, default=Path("viz_output"))
    p.add_argument("--off-mult", type=float, default=10.0,
                   help="preferred-direction off-axis multiplier; 1.0 = isotropic")
    p.add_argument("--m1-penalty", type=float, default=1.0,
                   help="multiplier on BOTH axes of M1 wire cost; 1.0 = no "
                        "extra penalty; >1.0 approximates gf180mcuD's "
                        "pin-access-only convention for M1")
    p.add_argument("--m1-pin-only", action="store_true",
                   help="strict: M1 cells become inf except the pin coords + "
                        "1-cell-radius landing pad. Overrides --m1-penalty. "
                        "Mimics DRC: no M1 wire allowed")
    args = p.parse_args()

    if args.mode == "chip":
        render_chip(
            args.n, args.scale, args.out, args.off_mult, args.m1_penalty,
            args.m1_pin_only,
        )
    else:
        if args.net is None:
            print("--net NAME is required for --mode net", file=sys.stderr)
            sys.exit(2)
        render_net(
            args.net, args.out, args.off_mult, args.m1_penalty, args.m1_pin_only,
        )


if __name__ == "__main__":
    main()
