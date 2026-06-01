"""Guide-constrained chip-scale router for WS3.3 detailed routing.

Implements the guide-constrained sweep model of
[ADR 0012](../../docs/adr/0012-tile-decomposition.md) as amended
(Amendments 1–4): instead of partitioning the chip into fixed 256² tiles,
each net routes on a sub-grid sized to its **global-routing guides**
(`guide_region`, `gpu_pnr.guides`), sampled at the routing-track pitch,
indexing into one shared chip-scale cost tensor. The batched small-grid
sweep (`sweep_sssp_3d_batched`) parallelises across many such independent
sub-grids.

This module supersedes the fixed-tile `tile_router.py` (deleted): the tile
machinery — `Tile`, `partition_chip`, halo assignment — died with
Amendment 1. `net_bbox` survives as the no-guide fallback region builder.

Slice 1 (this commit) ships per-net region construction + in-cap/tail
classification only. Routing (single-stream → batched → conflict/ripup →
coarsened tail) lands in Slices 2–6 per
`docs/plans/ws33-tile-router-implementation.md`.

Conventions:
  - Net pins are `(layer, row, col)` grid cells.
  - A net's search space is a `GuideRegion` (half-open `[l0,l1)×[r0,r1)×
    [c0,c1)`), built from its guides, or — when it has no usable guide —
    from its pin bounding box plus a margin (the fallback, destined for the
    coarsened tail in Slice 5).
  - **In-cap** nets have a guide region whose row and col extents both fit
    the `axis_cap` (256 per ADR 0012 §1, as a *max* not a default). These
    route directly via the batched sweep. **Tail** nets — over-cap or
    no-guide — route on the coarsened grid (Slice 5).
  - The layer extent is never capped (≤ a handful of metals).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gpu_pnr.guides import (
    GuideRect,
    GuideRegion,
    clamp_region_bounds,
    guide_region,
)

if TYPE_CHECKING:
    import torch

    from gpu_pnr.router import MultiPin3DResult


# A net's pin list: (layer, row, col) cells.
Net3D = list[tuple[int, int, int]]

# Max sub-grid axis extent for a directly-swept (in-cap) net (ADR 0012 §1).
DEFAULT_AXIS_CAP = 256


def net_bbox(pins: Net3D) -> tuple[int, int, int, int]:
    """Return closed `(rmin, cmin, rmax, cmax)` bbox over a net's pin cells.

    The layer dimension is ignored; the bbox is purely 2D. Survives from
    the retired tile router as the no-guide fallback region builder and the
    HPWL source.
    """
    if not pins:
        raise ValueError("net_bbox requires at least one pin")
    rmin = rmax = pins[0][1]
    cmin = cmax = pins[0][2]
    for _, r, c in pins[1:]:
        if r < rmin:
            rmin = r
        elif r > rmax:
            rmax = r
        if c < cmin:
            cmin = c
        elif c > cmax:
            cmax = c
    return (rmin, cmin, rmax, cmax)


def net_hpwl(pins: Net3D) -> int:
    """Half-perimeter wirelength of a net's pin bbox (ADR 0007 ordering key)."""
    rmin, cmin, rmax, cmax = net_bbox(pins)
    return (rmax - rmin) + (cmax - cmin)


@dataclass
class NetPlan:
    """A net's routing plan after classification — the unit slices route from.

    `index` is the net's position in the caller's input list, so results can
    be reassembled in input order. `region` is always present: the guide
    region for guided nets, or the pin-bbox fallback otherwise. `has_guide`
    is False when the net had no usable guide (fell back). `in_cap` is True
    iff the net is guided *and* its region fits the axis cap — only those
    route via the batched sweep; the rest go to the coarsened tail.
    """

    index: int
    pins: Net3D
    region: GuideRegion
    has_guide: bool
    in_cap: bool


def _pin_bbox_region(
    pins: Net3D,
    margin: int,
    chip_shape: tuple[int, int, int] | None,
) -> GuideRegion:
    """Build a fallback region from a net's pin bbox + margin (no-guide path).

    Row/col span is the pin bbox expanded by `margin`; the layer span is the
    pins' own contiguous layer range `[min_l, max_l + 1)` (vias relax through
    adjacent layers, ADR 0006). Always contains every pin. Minimal by design
    — no-guide nets route on the coarsened grid (Slice 5), which defines its
    own search space; this region is just the bbox descriptor for that pass.
    """
    rmin, cmin, rmax, cmax = net_bbox(pins)
    layers = [p[0] for p in pins]
    l0, l1 = min(layers), max(layers) + 1
    r0, r1 = rmin - margin, rmax + 1 + margin
    c0, c1 = cmin - margin, cmax + 1 + margin
    if chip_shape is not None:
        l0, l1, r0, r1, c0, c1 = clamp_region_bounds(
            l0, l1, r0, r1, c0, c1, chip_shape
        )
    return GuideRegion(l0=l0, l1=l1, r0=r0, r1=r1, c0=c0, c1=c1)


def _net_region(
    pins: Net3D,
    rects: Sequence[GuideRect],
    chip_origin: tuple[int, int],
    layer_order: Sequence[str],
    pitch_dbu: int,
    margin: int,
    chip_shape: tuple[int, int, int] | None,
) -> tuple[GuideRegion, bool]:
    """Return `(region, has_guide)` for one net.

    Uses `guide_region` when the net has a guide on a routable layer; falls
    back to the pin bbox otherwise. `has_guide` reflects which path ran.
    """
    region = guide_region(
        rects, chip_origin, layer_order, pitch_dbu,
        margin=margin, chip_shape=chip_shape,
    )
    if region is not None:
        return region, True
    return _pin_bbox_region(pins, margin, chip_shape), False


def classify_nets(
    nets: list[Net3D],
    guides: Sequence[Sequence[GuideRect]],
    chip_origin: tuple[int, int],
    layer_order: Sequence[str],
    pitch_dbu: int,
    *,
    margin: int = 4,
    chip_shape: tuple[int, int, int] | None = None,
    axis_cap: int = DEFAULT_AXIS_CAP,
) -> tuple[list[NetPlan], list[NetPlan]]:
    """Partition nets into `(in_cap, tail)` plans.

    `guides[i]` is net `i`'s guide rectangles (possibly empty). A net is
    **in-cap** iff it has a usable guide and its region's row and col extents
    both fit `axis_cap`; otherwise it's **tail** (over-cap or no-guide). Every
    input net lands in exactly one bucket. The in-cap list is returned
    HPWL-ascending ([ADR 0007](../../docs/adr/0007-hpwl-ascending-net-ordering.md));
    the tail keeps input order (Slice 5 owns tail ordering).

    Pin-in-region containment (the prototype's `off_region` check) is a
    routing-time concern deferred to Slice 2, not a classification criterion.
    """
    if len(guides) != len(nets):
        raise ValueError(
            f"guides must be parallel to nets: {len(guides)} guides, "
            f"{len(nets)} nets"
        )
    in_cap: list[NetPlan] = []
    tail: list[NetPlan] = []
    for idx, pins in enumerate(nets):
        region, has_guide = _net_region(
            pins, guides[idx], chip_origin, layer_order, pitch_dbu,
            margin, chip_shape,
        )
        _, nh, nw = region.shape
        is_in_cap = has_guide and nh <= axis_cap and nw <= axis_cap
        plan = NetPlan(
            index=idx, pins=pins, region=region,
            has_guide=has_guide, in_cap=is_in_cap,
        )
        (in_cap if is_in_cap else tail).append(plan)
    in_cap.sort(key=lambda p: net_hpwl(p.pins))
    return in_cap, tail


class GuideRouter:
    """Chip-scale guide-constrained router (ADR 0012 as amended).

    Slice 1: only the classification surface is implemented. `route` is a
    stub; the pipeline (single-stream → batched sweep → conflict/ripup →
    coarsened tail) lands in Slices 2–6 per
    `docs/plans/ws33-tile-router-implementation.md`.

    `chip_shape` is taken from `w_chip` when a tensor is given, else from the
    explicit `chip_shape` argument (so the classifier is testable without a
    cost tensor).
    """

    def __init__(
        self,
        w_chip: torch.Tensor | None,
        w_v_chip: torch.Tensor | None = None,
        *,
        chip_origin: tuple[int, int],
        layer_order: Sequence[str],
        pitch_dbu: int,
        margin: int = 4,
        chip_shape: tuple[int, int, int] | None = None,
        axis_cap: int = DEFAULT_AXIS_CAP,
    ) -> None:
        if pitch_dbu <= 0:
            raise ValueError(f"pitch_dbu must be positive; got {pitch_dbu}")
        if axis_cap <= 0:
            raise ValueError(f"axis_cap must be positive; got {axis_cap}")
        self.w_chip = w_chip
        self.w_v_chip = w_v_chip
        self.chip_origin = chip_origin
        self.layer_order = layer_order
        self.pitch_dbu = pitch_dbu
        self.margin = margin
        self.axis_cap = axis_cap
        if chip_shape is not None:
            self.chip_shape: tuple[int, int, int] | None = chip_shape
        elif w_chip is not None:
            assert w_chip.ndim == 3, f"w_chip must be (L,H,W); got {tuple(w_chip.shape)}"
            self.chip_shape = tuple(w_chip.shape)  # type: ignore[assignment]
        else:
            self.chip_shape = None

    def classify(
        self,
        nets: list[Net3D],
        guides: Sequence[Sequence[GuideRect]],
    ) -> tuple[list[NetPlan], list[NetPlan]]:
        """Partition nets into `(in_cap, tail)` plans — see `classify_nets`."""
        return classify_nets(
            nets, guides, self.chip_origin, self.layer_order, self.pitch_dbu,
            margin=self.margin, chip_shape=self.chip_shape,
            axis_cap=self.axis_cap,
        )

    def route(
        self,
        nets: list[Net3D],
        guides: Sequence[Sequence[GuideRect]],
    ) -> list[MultiPin3DResult]:
        """Route nets; API mirrors `route_multipin_nets_3d` plus per-net guides.

        Slice 1 stub; routing lands in Slice 2+.
        """
        del nets, guides
        raise NotImplementedError("Slice 1 stub; routing lands in Slice 2+")
