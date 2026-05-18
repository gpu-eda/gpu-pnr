"""Chip-scale tile manager for WS3.3 detailed routing.

Implements the tile decomposition substrate described in
[ADR 0012](../../docs/adr/0012-tile-decomposition.md): the chip is
partitioned into non-overlapping owned regions of `tile_size`² cells;
each tile additionally routes within a `halo`-cell envelope that
overlaps with adjacent tiles to absorb short detours without
cross-tile coordination per step.

Slice 1 (this commit) ships pure data structures and the net-to-tile
assignment algorithm only — the actual routing pipeline (per-tile
sweeps, halo reconciliation, coarsened multi-tile-spanning pass) lands
in Slice 3+ per `docs/plans/ws33-tile-router-implementation.md`.

Conventions:
  - Owned regions use **half-open intervals** `[r0, r1)` × `[c0, c1)`,
    matching Python slicing.
  - `net_bbox` returns **closed** bounds `(rmin, cmin, rmax, cmax)`
    (inclusive on both ends).
  - A tile's halo envelope is the owned region expanded by `halo` cells
    in each direction (also half-open). The envelope is purely geometric;
    `Tile` does not clamp to chip bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from gpu_pnr.router import MultiPin3DResult


# A net's pin list: (layer, row, col) cells.
Net3D = list[tuple[int, int, int]]


@dataclass(frozen=True)
class Tile:
    """Owned region of a chip partition, plus a halo width (ADR 0012 §3).

    Owned region uses half-open intervals: cell `(r, c)` is owned iff
    `r0 <= r < r1` and `c0 <= c < c1`. The halo envelope (owned region
    expanded by `halo` cells in each direction) is exposed via
    properties; it is *not* clamped to chip bounds — clamping, where
    needed, is the caller's responsibility.
    """

    r0: int
    c0: int
    r1: int
    c1: int
    halo: int

    @property
    def envelope_r0(self) -> int:
        """Top edge of the routable (owned + halo) envelope, half-open."""
        return self.r0 - self.halo

    @property
    def envelope_c0(self) -> int:
        """Left edge of the routable (owned + halo) envelope, half-open."""
        return self.c0 - self.halo

    @property
    def envelope_r1(self) -> int:
        """Bottom edge of the routable envelope (exclusive)."""
        return self.r1 + self.halo

    @property
    def envelope_c1(self) -> int:
        """Right edge of the routable envelope (exclusive)."""
        return self.c1 + self.halo

    def owns(self, r: int, c: int) -> bool:
        """True iff `(r, c)` lies in this tile's owned (inner) region."""
        return self.r0 <= r < self.r1 and self.c0 <= c < self.c1


def partition_chip(
    chip_h: int,
    chip_w: int,
    tile_size: int = 256,
    halo: int = 32,
) -> list[Tile]:
    """Tile a `chip_h × chip_w` chip with non-overlapping `tile_size`² owned regions.

    Right/bottom-edge tiles may be partial when `chip_h` / `chip_w` is
    not a multiple of `tile_size`. Owned regions tile the chip exactly
    (no overlap, no gap); halo envelopes do overlap by design
    (ADR 0012 §3).
    """
    if chip_h <= 0 or chip_w <= 0:
        raise ValueError(f"chip dims must be positive; got {chip_h}x{chip_w}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive; got {tile_size}")
    if halo < 0:
        raise ValueError(f"halo must be non-negative; got {halo}")

    tiles: list[Tile] = []
    for r0 in range(0, chip_h, tile_size):
        r1 = min(r0 + tile_size, chip_h)
        for c0 in range(0, chip_w, tile_size):
            c1 = min(c0 + tile_size, chip_w)
            tiles.append(Tile(r0=r0, c0=c0, r1=r1, c1=c1, halo=halo))
    return tiles


def net_bbox(pins: list[tuple[int, int, int]]) -> tuple[int, int, int, int]:
    """Return closed `(rmin, cmin, rmax, cmax)` bbox over a net's pin cells.

    The layer dimension is ignored; partitioning is purely 2D per ADR 0012 §1.
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


def _envelope_contains_bbox(
    tile: Tile, bbox: tuple[int, int, int, int], halo: int,
) -> bool:
    """True iff the entire (closed) bbox lies in the tile's owned+halo envelope.

    Envelope is half-open `[r0-halo, r1+halo) × [c0-halo, c1+halo)`; a
    closed bbox `(rmin, cmin, rmax, cmax)` fits iff every cell is in
    the envelope, i.e. `rmin >= r0-halo` and `rmax <= r1+halo-1`, etc.
    """
    rmin, cmin, rmax, cmax = bbox
    return (
        rmin >= tile.r0 - halo
        and rmax <= tile.r1 + halo - 1
        and cmin >= tile.c0 - halo
        and cmax <= tile.c1 + halo - 1
    )


def assign_net_to_tile(
    bbox: tuple[int, int, int, int],
    tiles: list[Tile],
    halo: int,
) -> Tile | None:
    """Return the tile whose owned+halo envelope contains `bbox`, or `None`.

    Implements the ADR 0012 §6 net-assignment rule:
      - If exactly one tile's envelope contains the bbox → that tile.
      - If multiple tiles' envelopes contain it (halos overlap, so a
        small bbox near a tile boundary can fit several) → tiebreak
        picks the tile whose **owned region** contains the bbox center
        `((rmin+rmax)//2, (cmin+cmax)//2)`.
      - If even that's ambiguous (center exactly on a tile boundary so
        no owned region contains it among the candidates) → pick the
        lexicographically-smallest `(r0, c0)`.
      - If no tile's envelope contains the bbox → `None` (the net is
        multi-tile-spanning, handled by the coarsened pass in Slice 6).
    """
    candidates = [t for t in tiles if _envelope_contains_bbox(t, bbox, halo)]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    rmin, cmin, rmax, cmax = bbox
    cr = (rmin + rmax) // 2
    cc = (cmin + cmax) // 2

    by_center = [t for t in candidates if t.owns(cr, cc)]
    if len(by_center) == 1:
        return by_center[0]
    pool = by_center or candidates
    # Lexicographic tiebreak on (r0, c0).
    return min(pool, key=lambda t: (t.r0, t.c0))


def classify_nets(
    nets: list[Net3D],
    tiles: list[Tile],
    halo: int,
) -> tuple[dict[Tile, list[int]], list[int]]:
    """Bucket each net by assigned tile, or into the multi-tile-spanning list.

    Returns `(per_tile_indices, spanning_indices)`. Every input net's
    index appears in exactly one bucket. Per ADR 0012 §6.
    """
    per_tile: dict[Tile, list[int]] = {}
    spanning: list[int] = []
    for idx, pins in enumerate(nets):
        if not pins:
            # Degenerate net with no pins: nowhere to route, flag as spanning
            # so the caller sees it (matches Slice 6 expectation that
            # unassignable nets surface up).
            spanning.append(idx)
            continue
        bbox = net_bbox(pins)
        tile = assign_net_to_tile(bbox, tiles, halo)
        if tile is None:
            spanning.append(idx)
        else:
            per_tile.setdefault(tile, []).append(idx)
    return per_tile, spanning


class TileRouter:
    """Chip-scale router built on the tile decomposition of ADR 0012.

    Slice 1: only the geometry/assignment surface is implemented.
    `route` is a stub; the full pipeline (per-tile K=100 batched sweeps,
    halo reconciliation, coarsened multi-tile-spanning pass) lands in
    Slices 3-7 per `docs/plans/ws33-tile-router-implementation.md`.
    """

    def __init__(
        self,
        w_chip: torch.Tensor | None,
        w_v_chip: torch.Tensor | None = None,
        tile_size: int = 256,
        halo: int = 32,
    ) -> None:
        if tile_size <= 0:
            raise ValueError(f"tile_size must be positive; got {tile_size}")
        if halo < 0:
            raise ValueError(f"halo must be non-negative; got {halo}")
        self.w_chip = w_chip
        self.w_v_chip = w_v_chip
        self.tile_size = tile_size
        self.halo = halo

    def route(self, nets: list[Net3D]) -> list[MultiPin3DResult]:
        """Route N-pin nets across the chip; API-compatible with `route_multipin_nets_3d`.

        Slice 1 stub; routing lands in Slice 3+.
        """
        del nets
        raise NotImplementedError("Slice 1 stub; routing lands in Slice 3+")
