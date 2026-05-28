"""Guide ingestion for guide-constrained detailed routing.

[ADR 0012 Amendment 1](../../docs/adr/0012-tile-decomposition.md#amendment-1-2026-05-28-guide-constrained-sweep-replaces-fixed-tile-k-batching)
pivots WS3.3 from fixed 256² tiles to **per-net adaptive sub-grids**
defined by each net's global-routing guides. This module is the
ingestion path: it maps a net's GRT guide rectangles to a sub-grid
bounding box in our `(layer, row, col)` grid coordinates, so the sweep
can index a small slice of the chip-scale cost tensor instead of a full
tile.

The raw `*.guide` text parser lives in `scripts/_hazard3_io.parse_guides`
(fixture I/O). This module takes already-parsed rectangles and is pure
integer geometry — no torch, no PDK coupling beyond a `layer_order`
lookup the caller supplies.

Coordinate conventions (consistent with `tile_router` and
`scripts/_hazard3_io`):
  - Guide rects are `(xlo, ylo, xhi, yhi, layer_name)` in DEF DBU,
    axis-aligned, `xlo < xhi` and `ylo < yhi`.
  - Grid mapping is `col = (x - origin_x) // pitch`,
    `row = (y - origin_y) // pitch` — the same integer-floor mapping
    `build_chip_grid` uses. Because GRT guides are GCell-aligned (every
    edge is a multiple of the GCell pitch, itself a multiple of the grid
    pitch), the half-open upper bound here coincides with
    `build_chip_grid`'s `(x1 - origin) // pitch`, so a region sliced from
    its output lines up cell-for-cell. For non-aligned rects the two can
    differ by one cell (this region is the conservative superset — it
    covers every cell a rect touches).
  - A `GuideRegion` uses **half-open** intervals `[l0, l1) × [r0, r1) ×
    [c0, c1)`, matching Python/torch slicing: `w_chip[l0:l1, r0:r1,
    c0:c1]` is exactly the region.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# A guide rectangle: (xlo, ylo, xhi, yhi, layer_name) in DEF DBU.
GuideRect = tuple[int, int, int, int, str]


@dataclass(frozen=True)
class GuideRegion:
    """A net's guide-constrained sweep sub-grid, in grid coordinates.

    Half-open bounds: the region is `w_chip[l0:l1, r0:r1, c0:c1]`.

    The layer range is **contiguous** — `[min_guide_layer, max_guide_layer]`
    inclusive — even if an intermediate layer carries no guide for this
    net. Via transitions relax through adjacent layers
    ([ADR 0006](../../docs/adr/0006-sequential-via-relax.md)), so a
    sub-grid that skipped a middle layer would sever the via stack. The
    row/col span is the union bounding box of the net's guides, expanded
    by a margin (see `guide_region`).
    """

    l0: int
    l1: int
    r0: int
    r1: int
    c0: int
    c1: int

    @property
    def shape(self) -> tuple[int, int, int]:
        """`(L, H, W)` of the sub-grid — the shape of the sliced tensor."""
        return (self.l1 - self.l0, self.r1 - self.r0, self.c1 - self.c0)

    @property
    def cell_count(self) -> int:
        """Total cells in the sub-grid (`L * H * W`)."""
        nl, nh, nw = self.shape
        return nl * nh * nw

    def contains(self, cell: tuple[int, int, int]) -> bool:
        """True iff `(layer, row, col)` lies inside this region."""
        lyr, r, c = cell
        return (
            self.l0 <= lyr < self.l1
            and self.r0 <= r < self.r1
            and self.c0 <= c < self.c1
        )

    def rebase(self, cell: tuple[int, int, int]) -> tuple[int, int, int]:
        """Translate a chip-global `(layer, row, col)` into region-local coords.

        Inverse of adding `(l0, r0, c0)`. Does not check membership; call
        `contains` first if the cell's origin is uncertain.
        """
        lyr, r, c = cell
        return (lyr - self.l0, r - self.r0, c - self.c0)


def guide_region(
    rects: Sequence[GuideRect],
    chip_origin: tuple[int, int],
    layer_order: Sequence[str],
    pitch_dbu: int,
    *,
    margin: int = 4,
    chip_shape: tuple[int, int, int] | None = None,
) -> GuideRegion | None:
    """Map a net's guide rectangles to a grid-space sub-grid bounding box.

    Args:
        rects: the net's guide rectangles, `(xlo, ylo, xhi, yhi, layer)`
            in DBU. Rects whose layer is not in `layer_order` (e.g. a
            metal above the routed stack) are ignored.
        chip_origin: `(origin_x, origin_y)` in DBU — the chip-scale grid
            origin, i.e. DIEAREA lower-left, matching `build_chip_grid`.
        layer_order: layer names indexed as in the cost tensor, e.g.
            `("Metal1", ..., "Metal5")`.
        pitch_dbu: grid pitch in DBU (one cell per pitch).
        margin: rows/cols of slack added on every side of the row/col
            union bbox (grid cells, not GCells). Gives the sweep room to
            reach pins or detour just outside the guides, mirroring DRT's
            "small expansion margin". Does **not** expand the layer
            range. Default 4 (≈0.8 µm at 200 DBU pitch). See ADR 0012
            Amendment 1.
        chip_shape: optional `(L, H, W)` of the chip grid. When given,
            the returned region is clamped to `[0, L) × [0, H) × [0, W)`
            so it is always a valid slice. The layer range is **not**
            clamped by margin (margin never touches layers) but is
            clamped to `[0, L)` defensively.

    Returns:
        A `GuideRegion`, or `None` if `rects` contains no rectangle on a
        layer in `layer_order` (nothing routable to bound).

    The row/col bbox is computed in DBU first, then mapped to grid cells,
    so the half-open grid span covers every cell any guide rectangle
    touches. `xhi`/`yhi` are exclusive DEF coordinates; the last covered
    cell is `(yhi - 1 - origin_y) // pitch`, so the half-open upper bound
    is that plus one.
    """
    if pitch_dbu <= 0:
        raise ValueError(f"pitch_dbu must be positive; got {pitch_dbu}")
    if margin < 0:
        raise ValueError(f"margin must be non-negative; got {margin}")

    layer_index = {name: i for i, name in enumerate(layer_order)}
    origin_x, origin_y = chip_origin

    x_lo = y_lo = None
    x_hi = y_hi = None
    l_min = l_max = None
    for x0, y0, x1, y1, layer in rects:
        lyr = layer_index.get(layer)
        if lyr is None:
            continue
        if x_lo is None or x0 < x_lo:
            x_lo = x0
        if y_lo is None or y0 < y_lo:
            y_lo = y0
        if x_hi is None or x1 > x_hi:
            x_hi = x1
        if y_hi is None or y1 > y_hi:
            y_hi = y1
        if l_min is None or lyr < l_min:
            l_min = lyr
        if l_max is None or lyr > l_max:
            l_max = lyr

    if l_min is None:
        return None
    assert x_lo is not None and y_lo is not None
    assert x_hi is not None and y_hi is not None
    assert l_max is not None

    # DBU bbox -> half-open grid span. Lower bound floors; upper bound is
    # one past the last cell the rect's exclusive `xhi`/`yhi` touches.
    c0 = (x_lo - origin_x) // pitch_dbu - margin
    r0 = (y_lo - origin_y) // pitch_dbu - margin
    c1 = (x_hi - 1 - origin_x) // pitch_dbu + 1 + margin
    r1 = (y_hi - 1 - origin_y) // pitch_dbu + 1 + margin
    l0 = l_min
    l1 = l_max + 1

    if chip_shape is not None:
        chip_l, chip_h, chip_w = chip_shape
        l0 = max(0, min(l0, chip_l))
        l1 = max(l0, min(l1, chip_l))
        r0 = max(0, min(r0, chip_h))
        r1 = max(r0, min(r1, chip_h))
        c0 = max(0, min(c0, chip_w))
        c1 = max(c0, min(c1, chip_w))

    return GuideRegion(l0=l0, l1=l1, r0=r0, r1=r1, c0=c0, c1=c1)
