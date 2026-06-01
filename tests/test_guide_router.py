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
import torch

from gpu_pnr.guide_router import (
    GuideRouter,
    NetPlan,
    classify_nets,
    net_bbox,
)
from gpu_pnr.guides import guide_region
from gpu_pnr.router import route_multipin_nets_3d

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


def test_route_requires_a_grid():
    """route needs a cost tensor; classification alone doesn't."""
    router = GuideRouter(
        w_chip=None, chip_origin=ORIGIN, layer_order=LAYERS,
        pitch_dbu=PITCH, chip_shape=CHIP_SHAPE,
    )
    with pytest.raises(ValueError, match="w_chip"):
        router.route([[(0, 0, 0), (0, 1, 1)]], [[]])


def _full_guide(h_cells: int, w_cells: int, layer: str = "M1"):
    """A guide rect covering the whole `h_cells × w_cells` grid on `layer`."""
    return [_rect(0, 0, w_cells * PITCH, h_cells * PITCH, layer)]


def test_route_single_net_matches_direct():
    """One in-cap net routed by GuideRouter equals route_multipin_nets_3d on the
    same guide sub-grid, with paths translated back to chip-global coords."""
    chip = torch.full((1, 10, 10), 1.0)
    pins = [(0, 5, 5), (0, 7, 7)]
    guide = [_rect(5000, 5000, 8000, 8000, "M1")]
    router = GuideRouter(
        chip, chip_origin=ORIGIN, layer_order=LAYERS, pitch_dbu=PITCH, margin=4,
    )
    [res] = router.route([pins], [guide])
    assert res.routed
    assert res.pins == pins  # result carries the chip-global pins

    # Reference: slice the original chip by the same region, route, translate.
    reg = guide_region(
        guide, ORIGIN, LAYERS, PITCH, margin=4, chip_shape=(1, 10, 10),
    )
    assert reg is not None
    w_sub = chip[reg.l0:reg.l1, reg.r0:reg.r1, reg.c0:reg.c1].clone()
    local = [reg.rebase(p) for p in pins]
    [ref] = route_multipin_nets_3d(w_sub, [local])
    ref_global = {
        (lyr + reg.l0, r + reg.r0, c + reg.c0) for (lyr, r, c) in ref.cells
    }
    assert res.cells == ref_global


def test_two_nets_detour_no_conflict():
    """A second net detours around the first's committed cells via the shared
    w_cur — both route, zero cross-net cell conflicts."""
    chip = torch.full((1, 5, 5), 1.0)
    guide = _full_guide(5, 5)
    net_a = [(0, 2, 0), (0, 2, 2)]   # HPWL 2 → routes first, claims row-2 cols 0-2
    net_b = [(0, 0, 1), (0, 4, 1)]   # HPWL 4 → must cross row 2; detours via open col
    router = GuideRouter(
        chip, chip_origin=ORIGIN, layer_order=LAYERS, pitch_dbu=PITCH, margin=4,
    )
    res_a, res_b = router.route([net_a, net_b], [guide, guide])
    assert res_a.routed and res_b.routed
    assert res_a.cells.isdisjoint(res_b.cells)  # 0 cross-net conflicts


def test_hpwl_order_decides_contention():
    """Routing order is HPWL-ascending, not input order: on a 1-row corridor the
    shorter net routes first and claims the shared cells, starving the longer
    one. Input order is [long, short] to prove HPWL order wins."""
    chip = torch.full((1, 1, 5), 1.0)
    guide = _full_guide(1, 5)
    long_net = [(0, 0, 0), (0, 0, 4)]   # HPWL 4; only path crosses cols 1-3
    short_net = [(0, 0, 1), (0, 0, 2)]  # HPWL 1; claims cols 1-2
    router = GuideRouter(
        chip, chip_origin=ORIGIN, layer_order=LAYERS, pitch_dbu=PITCH, margin=4,
    )
    res_long, res_short = router.route(
        [long_net, short_net], [guide, guide],  # input order: long first
    )
    assert res_short.routed       # shorter HPWL → routed first
    assert not res_long.routed    # corridor taken, no detour on one row
    # Results are returned in input order regardless of routing order.
    assert res_long.pins == long_net and res_short.pins == short_net


def test_tail_net_returns_unrouted():
    """No-guide (tail) nets are not routed in Slice 2 — placeholder result."""
    chip = torch.full((1, 5, 5), 1.0)
    net = [(0, 1, 1), (0, 3, 3)]
    router = GuideRouter(
        chip, chip_origin=ORIGIN, layer_order=LAYERS, pitch_dbu=PITCH, margin=4,
    )
    [res] = router.route([net], [[]])  # empty guide → tail
    assert not res.routed
    assert res.pins == net


def test_off_region_net_unrouted():
    """An in-cap net whose guide region doesn't contain every pin can't be swept
    on that sub-grid → unrouted (off-region handling, deferred from Slice 1)."""
    chip = torch.full((1, 10, 10), 1.0)
    tiny_guide = [_rect(0, 0, 1000, 1000, "M1")]  # region ~rows/cols [0,5)
    net = [(0, 0, 0), (0, 8, 8)]  # second pin outside the region
    router = GuideRouter(
        chip, chip_origin=ORIGIN, layer_order=LAYERS, pitch_dbu=PITCH, margin=4,
    )
    [res] = router.route([net], [tiny_guide])
    assert not res.routed


def test_prep_subgrid_applied_without_corrupting_shared_grid():
    """prep_subgrid runs once per in-cap net on a clone; its mutations must not
    leak into the shared w_cur (the router clones before prepping)."""
    chip = torch.full((1, 5, 5), 1.0)
    guide = _full_guide(5, 5)
    calls: list[list[tuple[int, int, int]]] = []

    def prep(w_sub: torch.Tensor, local_pins: list[tuple[int, int, int]]) -> None:
        calls.append(list(local_pins))
        w_sub[0, 0, 0] = float("inf")  # mutate the clone — must not persist

    net_a = [(0, 1, 0), (0, 1, 2)]
    net_b = [(0, 3, 0), (0, 3, 2)]
    router = GuideRouter(
        chip, chip_origin=ORIGIN, layer_order=LAYERS, pitch_dbu=PITCH,
        margin=4, prep_subgrid=prep,
    )
    res_a, res_b = router.route([net_a, net_b], [guide, guide])
    assert len(calls) == 2  # once per in-cap net
    assert res_a.routed and res_b.routed
    assert torch.isfinite(chip[0, 0, 0])  # original tensor untouched by prep


def test_route_preserves_input_order():
    """Output list aligns with input order across a mix of in-cap and tail nets."""
    chip = torch.full((1, 8, 8), 1.0)
    guide = _full_guide(8, 8)
    nets = [
        [(0, 1, 1), (0, 2, 2)],   # 0: in-cap, routes
        [(0, 5, 5), (0, 6, 6)],   # 1: no guide → tail
        [(0, 0, 0), (0, 3, 3)],   # 2: in-cap, routes
    ]
    guides = [guide, [], guide]
    router = GuideRouter(
        chip, chip_origin=ORIGIN, layer_order=LAYERS, pitch_dbu=PITCH, margin=4,
    )
    results = router.route(nets, guides)
    assert len(results) == 3
    assert [r.pins for r in results] == nets
    assert results[0].routed and results[2].routed
    assert not results[1].routed  # tail


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
