"""Tests for the guide-region mapper (guide-constrained sweep ingestion).

Covers `guide_region` and the `GuideRegion` helpers. See
`docs/adr/0012-tile-decomposition.md` Amendment 1 (§ guide-constrained
sweep) for the design and `src/gpu_pnr/guides.py` for conventions.

Layer order and pitch mirror the gf180mcuD fixture
(`scripts/_hazard3_io.GF180MCUD`): five Metal layers at 200 DBU pitch,
GCell pitch 16800 DBU = 84 grid cells.
"""

from __future__ import annotations

import pytest

from gpu_pnr.guides import GuideRegion, guide_region

LAYERS = ("Metal1", "Metal2", "Metal3", "Metal4", "Metal5")
PITCH = 200
GCELL = 16800  # one GCell side in DBU; = 84 grid cells at PITCH=200


def test_single_gcell_region():
    """One Metal1 GCell at the origin maps to an 84×84 single-layer region."""
    rects = [(0, 0, GCELL, GCELL, "Metal1")]
    reg = guide_region(rects, (0, 0), LAYERS, PITCH, margin=0)
    assert reg is not None
    assert reg == GuideRegion(l0=0, l1=1, r0=0, r1=84, c0=0, c1=84)
    assert reg.shape == (1, 84, 84)
    assert reg.cell_count == 84 * 84


def test_union_bbox_across_gcells():
    """Region row/col span is the union bbox of all guide rects."""
    rects = [
        (0, 0, GCELL, GCELL, "Metal1"),  # bottom-left GCell
        (GCELL, 0, 2 * GCELL, GCELL, "Metal2"),  # one GCell to the right
        (0, GCELL, GCELL, 2 * GCELL, "Metal3"),  # one GCell up
    ]
    reg = guide_region(rects, (0, 0), LAYERS, PITCH, margin=0)
    assert reg is not None
    # Union spans 2 GCells in each axis -> 168 cells.
    assert reg.r0 == 0 and reg.r1 == 168
    assert reg.c0 == 0 and reg.c1 == 168
    assert reg.l0 == 0 and reg.l1 == 3


def test_layer_range_is_contiguous():
    """Guides on Metal1 + Metal3 yield layers [0:3] — Metal2 included.

    Via stacks relax through adjacent layers (ADR 0006); a sub-grid that
    skipped the empty middle layer would sever the via path.
    """
    rects = [
        (0, 0, GCELL, GCELL, "Metal1"),  # layer 0
        (0, 0, GCELL, GCELL, "Metal3"),  # layer 2 — skips Metal2
    ]
    reg = guide_region(rects, (0, 0), LAYERS, PITCH, margin=0)
    assert reg is not None
    assert reg.l0 == 0
    assert reg.l1 == 3  # half-open: layers 0, 1, 2 — Metal2 (1) bridged


def test_margin_expands_rows_and_cols_not_layers():
    """Margin pads row/col span on every side; layer range is untouched."""
    rects = [(0, 0, GCELL, GCELL, "Metal2")]
    reg = guide_region(rects, (0, 0), LAYERS, PITCH, margin=4)
    assert reg is not None
    assert reg.r0 == -4 and reg.r1 == 88
    assert reg.c0 == -4 and reg.c1 == 88
    # Single layer (Metal2) — margin must not widen the layer range.
    assert reg.l0 == 1 and reg.l1 == 2


def test_chip_shape_clamps_to_valid_slice():
    """With chip_shape, a margin that runs off-grid is clamped to bounds."""
    rects = [(0, 0, GCELL, GCELL, "Metal1")]
    reg = guide_region(
        rects, (0, 0), LAYERS, PITCH, margin=4, chip_shape=(5, 100, 100)
    )
    assert reg is not None
    # Lower bound clamps -4 -> 0; upper bound 88 stays (< 100).
    assert reg.r0 == 0 and reg.r1 == 88
    assert reg.c0 == 0 and reg.c1 == 88


def test_chip_shape_clamps_upper_bound():
    """A guide near the chip edge clamps the region's far corner to the grid."""
    # Rect's far corner extends past a 90×90 chip once margin is added.
    rects = [(16000, 16000, 16000 + GCELL, 16000 + GCELL, "Metal1")]
    reg = guide_region(
        rects, (0, 0), LAYERS, PITCH, margin=4, chip_shape=(5, 90, 90)
    )
    assert reg is not None
    assert reg.r1 == 90 and reg.c1 == 90  # clamped to chip extent
    assert reg.r0 == 80 - 4 and reg.c0 == 80 - 4  # 16000/200 - margin


def test_nonzero_origin():
    """A real-fixture-style rect maps to the origin cell with a nonzero origin."""
    origin = (1948800, 3712800)
    rects = [(1948800, 3712800, 1948800 + GCELL, 3712800 + GCELL, "Metal1")]
    reg = guide_region(rects, origin, LAYERS, PITCH, margin=0)
    assert reg == GuideRegion(l0=0, l1=1, r0=0, r1=84, c0=0, c1=84)


def test_unknown_layer_rects_ignored():
    """Rects on layers outside layer_order don't widen the region."""
    rects = [
        (0, 0, GCELL, GCELL, "Metal1"),
        (0, 0, 10 * GCELL, 10 * GCELL, "Metal9"),  # not in stack — ignored
    ]
    reg = guide_region(rects, (0, 0), LAYERS, PITCH, margin=0)
    assert reg is not None
    assert reg.l0 == 0 and reg.l1 == 1  # Metal9 didn't bump l_max
    assert reg.r1 == 84 and reg.c1 == 84  # Metal9's huge bbox ignored


def test_no_routable_rects_returns_none():
    """All-unknown-layer rects (or empty input) yield None."""
    assert guide_region([], (0, 0), LAYERS, PITCH) is None
    assert guide_region(
        [(0, 0, GCELL, GCELL, "Metal9")], (0, 0), LAYERS, PITCH
    ) is None


def test_region_contains_and_rebase():
    """contains() respects half-open bounds; rebase() subtracts the origin."""
    reg = GuideRegion(l0=1, l1=3, r0=10, r1=20, c0=5, c1=15)
    assert reg.shape == (2, 10, 10)
    assert reg.cell_count == 200
    assert reg.contains((1, 10, 5))
    assert reg.contains((2, 19, 14))
    assert not reg.contains((0, 10, 5))  # layer below l0
    assert not reg.contains((1, 20, 5))  # row == r1 (exclusive)
    assert reg.rebase((1, 10, 5)) == (0, 0, 0)
    assert reg.rebase((2, 15, 10)) == (1, 5, 5)


def test_invalid_args_raise():
    rects = [(0, 0, GCELL, GCELL, "Metal1")]
    with pytest.raises(ValueError):
        guide_region(rects, (0, 0), LAYERS, 0)  # non-positive pitch
    with pytest.raises(ValueError):
        guide_region(rects, (0, 0), LAYERS, PITCH, margin=-1)
