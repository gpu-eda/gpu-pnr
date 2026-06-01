"""Tests for the guide-constrained router geometry + classification module.

Covers Slice 1 of the WS3.3 guide router
(`docs/plans/ws33-tile-router-implementation.md`): per-net region
construction (guide bbox or pin-bbox fallback) and in-cap vs tail
classification. Routing lands in later slices. Design source of truth is
`docs/adr/0012-tile-decomposition.md` as amended (the guide-constrained
pivot, Amendments 1–4).
"""

from __future__ import annotations

import pytest

from gpu_pnr.guide_router import (
    GuideRouter,
    NetPlan,
    classify_nets,
    net_bbox,
)

# Shared synthetic geometry: origin at 0, 1000 DBU/cell, three layers.
ORIGIN = (0, 0)
LAYERS = ("M1", "M2", "M3")
PITCH = 1000
CHIP_SHAPE = (3, 512, 512)


def _rect(x0, y0, x1, y1, layer="M1"):
    return (x0, y0, x1, y1, layer)


def test_net_bbox_closed_bounds():
    """net_bbox returns closed (rmin, cmin, rmax, cmax) over pin cells; layer ignored."""
    pins = [(0, 10, 20), (2, 30, 5), (1, 15, 40)]
    assert net_bbox(pins) == (10, 5, 30, 40)


def test_net_bbox_requires_pins():
    with pytest.raises(ValueError, match="at least one pin"):
        net_bbox([])


def test_classify_in_cap_vs_over_cap():
    """A small-guide net is in-cap; a guide spanning > axis_cap cells is tail."""
    # Small guide: 5 cells per axis (well under the cap).
    small = [_rect(0, 0, 5000, 5000, "M1")]
    # Huge guide: 300 cells per axis (> 256 axis cap).
    huge = [_rect(0, 0, 300_000, 300_000, "M1")]
    nets = [
        [(0, 1, 1), (0, 3, 3)],      # net 0: small → in-cap
        [(0, 1, 1), (0, 250, 250)],  # net 1: huge → over-cap tail
    ]
    in_cap, tail = classify_nets(
        nets, [small, huge], ORIGIN, LAYERS, PITCH,
        chip_shape=CHIP_SHAPE, axis_cap=256,
    )
    assert [p.index for p in in_cap] == [0]
    assert [p.index for p in tail] == [1]
    assert in_cap[0].in_cap is True and in_cap[0].has_guide is True
    assert tail[0].in_cap is False and tail[0].has_guide is True  # over-cap, not no-guide
    # in-cap region must respect the axis cap on both spatial axes.
    _, nh, nw = in_cap[0].region.shape
    assert nh <= 256 and nw <= 256


def test_no_guide_falls_back_to_pin_bbox():
    """A net with no usable guide falls back to a pin-bbox region that contains
    every pin, is flagged has_guide=False, and lands in the tail."""
    pins = [(0, 100, 100), (1, 140, 130), (2, 110, 160)]
    # Guides only on a layer NOT in layer_order → guide_region returns None.
    off_layer = [_rect(0, 0, 5000, 5000, "M6")]
    in_cap, tail = classify_nets(
        [pins], [off_layer], ORIGIN, LAYERS, PITCH, chip_shape=CHIP_SHAPE,
    )
    assert in_cap == []
    assert len(tail) == 1
    plan = tail[0]
    assert plan.has_guide is False
    assert plan.in_cap is False
    # Fallback region contains every pin.
    assert all(plan.region.contains(p) for p in pins)
    # Layer span covers the pins' layer range [0, 2] → [0, 3).
    assert plan.region.l0 == 0 and plan.region.l1 == 3


def test_empty_guides_list_is_no_guide():
    """An empty guide list (not just off-layer) also triggers the fallback."""
    pins = [(0, 50, 50), (0, 60, 70)]
    in_cap, tail = classify_nets(
        [pins], [[]], ORIGIN, LAYERS, PITCH, chip_shape=CHIP_SHAPE,
    )
    assert in_cap == []
    assert tail[0].has_guide is False
    assert all(tail[0].region.contains(p) for p in pins)


def test_classify_partitions_all_nets():
    """Every input net index appears in exactly one of {in_cap, tail}."""
    small = [_rect(0, 0, 5000, 5000, "M1")]
    huge = [_rect(0, 0, 300_000, 300_000, "M1")]
    nets = [
        [(0, 1, 1), (0, 3, 3)],          # 0: small guide → in-cap
        [(0, 1, 1), (0, 250, 250)],      # 1: huge guide → tail (over-cap)
        [(0, 200, 200), (1, 205, 210)],  # 2: no guide → tail (fallback)
        [(0, 4, 4), (0, 6, 9)],          # 3: small guide → in-cap
    ]
    guides = [small, huge, [], small]
    in_cap, tail = classify_nets(
        nets, guides, ORIGIN, LAYERS, PITCH, chip_shape=CHIP_SHAPE, axis_cap=256,
    )
    seen = sorted([p.index for p in in_cap] + [p.index for p in tail])
    assert seen == list(range(len(nets)))
    assert len(seen) == len(nets)  # no duplicates
    assert {p.index for p in in_cap} == {0, 3}
    assert {p.index for p in tail} == {1, 2}


def test_in_cap_hpwl_ascending():
    """The in-cap bucket is ordered shortest-HPWL-first (ADR 0007)."""
    small = [_rect(0, 0, 60_000, 60_000, "M1")]  # fits 256 cap, holds all pins
    # Three in-cap nets with increasing half-perimeter wirelength.
    nets = [
        [(0, 0, 0), (0, 40, 40)],   # idx 0: HPWL 80
        [(0, 0, 0), (0, 5, 5)],     # idx 1: HPWL 10  ← smallest
        [(0, 0, 0), (0, 20, 10)],   # idx 2: HPWL 30
    ]
    guides = [small, small, small]
    in_cap, tail = classify_nets(
        nets, guides, ORIGIN, LAYERS, PITCH, chip_shape=CHIP_SHAPE, axis_cap=256,
    )
    assert tail == []
    assert [p.index for p in in_cap] == [1, 2, 0]  # HPWL 10, 30, 80


def test_classify_rejects_length_mismatch():
    """guides must be parallel to nets."""
    with pytest.raises(ValueError, match="parallel"):
        classify_nets(
            [[(0, 0, 0), (0, 1, 1)]], [], ORIGIN, LAYERS, PITCH,
            chip_shape=CHIP_SHAPE,
        )


def test_guide_router_classify_matches_free_function():
    """GuideRouter.classify is a thin wrapper over the free classify_nets."""
    router = GuideRouter(
        w_chip=None, w_v_chip=None, chip_origin=ORIGIN, layer_order=LAYERS,
        pitch_dbu=PITCH, chip_shape=CHIP_SHAPE, axis_cap=256,
    )
    small = [_rect(0, 0, 5000, 5000, "M1")]
    nets = [[(0, 1, 1), (0, 3, 3)]]
    in_cap, tail = router.classify(nets, [small])
    assert [p.index for p in in_cap] == [0]
    assert tail == []


def test_guide_router_route_is_stub():
    """Slice 1: GuideRouter.route is a placeholder; routing lands in Slice 2+."""
    router = GuideRouter(
        w_chip=None, w_v_chip=None, chip_origin=ORIGIN, layer_order=LAYERS,
        pitch_dbu=PITCH, chip_shape=CHIP_SHAPE,
    )
    with pytest.raises(NotImplementedError):
        router.route([[(0, 0, 0), (1, 1, 1)]], [[]])


def test_netplan_carries_index_pins_region():
    """NetPlan exposes the fields downstream slices route from."""
    small = [_rect(0, 0, 5000, 5000, "M1")]
    pins = [(0, 1, 1), (0, 3, 3)]
    in_cap, _ = classify_nets(
        [pins], [small], ORIGIN, LAYERS, PITCH, chip_shape=CHIP_SHAPE,
    )
    plan = in_cap[0]
    assert isinstance(plan, NetPlan)
    assert plan.index == 0
    assert plan.pins == pins
    assert plan.region.cell_count > 0
