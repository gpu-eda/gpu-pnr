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


def test_same_round_nets_share_snapshot_resolved_by_ripup():
    """Slice 3 routes same-round nets against the SAME w_cur snapshot, so two
    nets can claim the same cell. Slice 4 now RESOLVES that: the lower-HPWL net
    keeps the contested cell, the loser is ripped up and reroutes around it.
    Both end routed with zero shared cells (was: an unresolved conflict)."""
    chip = torch.full((1, 5, 5), 1.0)
    guide = _full_guide(5, 5)
    net_a = [(0, 2, 0), (0, 2, 2)]   # HPWL 2 (winner): row-2 cols 0-2
    net_b = [(0, 0, 1), (0, 4, 1)]   # HPWL 4 (loser): col-1 crosses row 2 at (0,2,1)
    router = GuideRouter(
        chip, chip_origin=ORIGIN, layer_order=LAYERS, pitch_dbu=PITCH, margin=4,
    )
    res_a, res_b = router.route([net_a, net_b], [guide, guide])
    # Both route, and the conflict is resolved — no shared cells remain.
    assert res_a.routed and res_b.routed
    assert res_a.cells.isdisjoint(res_b.cells)
    # The lower-HPWL net (a) keeps the contested cell (0, 2, 1).
    assert (0, 2, 1) in res_a.cells
    assert (0, 2, 1) not in res_b.cells


def test_same_round_overlap_resolved_loser_reroutes():
    """Two overlapping same-round nets with room to detour both route after
    rip-up: the lower-HPWL (shorter) net keeps the contested corridor cells,
    the longer loser reroutes around them. Zero shared cells; both routed.
    (Was: Slice 3 left the overlap unresolved for Slice 4.)"""
    chip = torch.full((1, 3, 5), 1.0)  # 3 rows give the loser room to detour
    guide = _full_guide(3, 5)
    long_net = [(0, 0, 0), (0, 0, 4)]   # HPWL 4 (loser): spans cols 0-4 on row 0
    short_net = [(0, 0, 1), (0, 0, 2)]  # HPWL 1 (winner): claims row-0 cols 1-2
    router = GuideRouter(
        chip, chip_origin=ORIGIN, layer_order=LAYERS, pitch_dbu=PITCH, margin=4,
    )
    res_long, res_short = router.route(
        [long_net, short_net], [guide, guide],  # input order: long first
    )
    # Both route; the conflict is resolved (no shared cells).
    assert res_short.routed and res_long.routed
    assert res_long.cells.isdisjoint(res_short.cells)
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


def test_route_disjoint_nets_match_per_net_sequential():
    """Slice 3 round-batching correctness gate: for spatially DISJOINT nets
    (no shared sub-grid cells, no interaction), the batched route must produce
    exactly the same committed cell set as routing each net independently on
    its own sub-grid sliced from the same w_cur snapshot. Disjoint nets never
    see each other's commits regardless of batching, so the cell sets must be
    bit-identical."""
    chip = torch.full((1, 12, 12), 1.0)
    # Three spatially-separated nets, each with a guide bounding a small corner.
    net_a = [(0, 0, 0), (0, 1, 2)]
    net_b = [(0, 8, 8), (0, 10, 10)]
    net_c = [(0, 4, 9), (0, 5, 11)]
    nets = [net_a, net_b, net_c]
    guides = [
        [_rect(0, 0, 3000, 3000, "M1")],
        [_rect(8000, 8000, 12000, 12000, "M1")],
        [_rect(9000, 4000, 12000, 6000, "M1")],
    ]
    router = GuideRouter(
        chip, chip_origin=ORIGIN, layer_order=LAYERS, pitch_dbu=PITCH, margin=4,
    )
    results = router.route(nets, guides)
    assert all(r.routed for r in results)

    # Reference: route each net independently on its own sub-grid from the
    # SAME original snapshot (disjoint → no cross-net commit interaction).
    for plan_idx, (net, guide) in enumerate(zip(nets, guides)):
        reg = guide_region(
            guide, ORIGIN, LAYERS, PITCH, margin=4, chip_shape=(1, 12, 12),
        )
        assert reg is not None
        w_sub = chip[reg.l0:reg.l1, reg.r0:reg.r1, reg.c0:reg.c1].clone()
        local = [reg.rebase(p) for p in net]
        [ref] = route_multipin_nets_3d(w_sub, [local])
        ref_global = {
            (lyr + reg.l0, r + reg.r0, c + reg.c0) for (lyr, r, c) in ref.cells
        }
        assert results[plan_idx].cells == ref_global, (
            f"net {plan_idx}: batched cell set != per-net sequential"
        )

    # Sanity: the three committed cell sets are mutually disjoint.
    all_cells = [r.cells for r in results]
    for i in range(len(all_cells)):
        for j in range(i + 1, len(all_cells)):
            assert all_cells[i].isdisjoint(all_cells[j])


def test_route_two_pin_and_multipin_split_both_route():
    """Round-batching handles a mix of 2-pin and >=3-pin nets in the same
    call: both buckets route correctly. The 2-pin net finishes in round 1;
    the multi-pin net needs >=2 attachment rounds. Both are disjoint so they
    must fully route with no shared cells."""
    chip = torch.full((1, 16, 16), 1.0)
    two_pin = [(0, 0, 0), (0, 2, 2)]
    multi_pin = [(0, 10, 10), (0, 10, 13), (0, 13, 10), (0, 13, 13)]
    nets = [two_pin, multi_pin]
    guides = [
        [_rect(0, 0, 4000, 4000, "M1")],
        [_rect(9000, 9000, 15000, 15000, "M1")],
    ]
    router = GuideRouter(
        chip, chip_origin=ORIGIN, layer_order=LAYERS, pitch_dbu=PITCH, margin=4,
    )
    res_two, res_multi = router.route(nets, guides)
    assert res_two.routed and res_multi.routed
    # The 4-pin net's tree must connect all four pins.
    assert all(p in res_multi.cells for p in multi_pin)
    # Disjoint nets: no shared cells.
    assert res_two.cells.isdisjoint(res_multi.cells)

    # Each must equal its own per-net sequential reference.
    for net, guide, res in (
        (two_pin, guides[0], res_two),
        (multi_pin, guides[1], res_multi),
    ):
        reg = guide_region(
            guide, ORIGIN, LAYERS, PITCH, margin=4, chip_shape=(1, 16, 16),
        )
        assert reg is not None
        w_sub = chip[reg.l0:reg.l1, reg.r0:reg.r1, reg.c0:reg.c1].clone()
        local = [reg.rebase(p) for p in net]
        [ref] = route_multipin_nets_3d(w_sub, [local])
        ref_global = {
            (lyr + reg.l0, r + reg.r0, c + reg.c0) for (lyr, r, c) in ref.cells
        }
        assert res.cells == ref_global


def test_bucketed_route_identical_to_single_batch():
    """Size-bucketing is a pure throughput optimisation: grouping a round's
    nets into size-sorted buckets (vs one giant padded batch) must produce
    byte-identical routes. Every bucket shares the round's w_cur snapshot and
    commit stays HPWL-ordered, so bucket_size changes only padding/grouping,
    never which cells a net claims."""
    chip = torch.full((1, 16, 16), 1.0)
    # Differently-sized nets so bucket_size=2 yields >1 bucket of unequal grids.
    nets = [
        [(0, 0, 0), (0, 1, 1)],                       # tiny
        [(0, 8, 8), (0, 11, 11), (0, 9, 13)],         # multi-pin, larger
        [(0, 4, 0), (0, 6, 2)],                       # small
        [(0, 13, 1), (0, 15, 4), (0, 14, 6)],         # multi-pin
    ]
    guides = [
        [_rect(0, 0, 2000, 2000, "M1")],
        [_rect(8000, 8000, 14000, 14000, "M1")],
        [_rect(0, 4000, 3000, 7000, "M1")],
        [_rect(1000, 13000, 7000, 16000, "M1")],
    ]
    single = GuideRouter(
        chip, chip_origin=ORIGIN, layer_order=LAYERS, pitch_dbu=PITCH, margin=4,
        bucket_size=None,
    ).route(nets, guides)
    bucketed = GuideRouter(
        chip, chip_origin=ORIGIN, layer_order=LAYERS, pitch_dbu=PITCH, margin=4,
        bucket_size=2,
    ).route(nets, guides)

    assert [r.routed for r in single] == [r.routed for r in bucketed]
    for i, (s, b) in enumerate(zip(single, bucketed)):
        assert s.cells == b.cells, f"net {i}: bucketed route diverges from single batch"


def test_bucket_size_must_be_positive():
    """bucket_size, when given, must be a positive count."""
    chip = torch.full((1, 8, 8), 1.0)
    with pytest.raises(ValueError, match="bucket_size"):
        GuideRouter(
            chip, chip_origin=ORIGIN, layer_order=LAYERS, pitch_dbu=PITCH,
            bucket_size=0,
        )


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


# --- Slice 4: cross-net conflict detect + rip-up / reroute -------------------


def test_ripup_lower_hpwl_wins_loser_reroutes():
    """Two nets conflict on a shared cell; the lower-HPWL net keeps it and the
    loser reroutes around. Both end routed with zero shared cells (the core
    Slice 4 conflict-resolution guarantee, ADR 0007 + ADR 0008)."""
    chip = torch.full((1, 5, 5), 1.0)
    guide = _full_guide(5, 5)
    winner = [(0, 2, 0), (0, 2, 2)]   # HPWL 2: row-2 corridor
    loser = [(0, 0, 2), (0, 4, 2)]    # HPWL 4: col-2 crosses (0, 2, 2)
    router = GuideRouter(
        chip, chip_origin=ORIGIN, layer_order=LAYERS, pitch_dbu=PITCH, margin=4,
    )
    res_w, res_l = router.route([winner, loser], [guide, guide])
    assert res_w.routed and res_l.routed
    assert res_w.cells.isdisjoint(res_l.cells)  # conflict resolved
    # The winner keeps the contested cell; the loser detoured off it.
    assert (0, 2, 2) in res_w.cells
    assert (0, 2, 2) not in res_l.cells


def test_ripup_unroutable_loser_fails_winner_survives():
    """When a ripped-up loser cannot reroute (no room to detour after the
    winner commits), it ends routed=False (paths None) while the winner stays
    routed=True — bounded rip-up gives up cleanly, no crash."""
    # 1-row corridor: once the winner takes cols 1-2 there is no detour.
    chip = torch.full((1, 1, 5), 1.0)
    guide = _full_guide(1, 5)
    winner = [(0, 0, 1), (0, 0, 2)]   # HPWL 1: claims cols 1-2
    loser = [(0, 0, 0), (0, 0, 4)]    # HPWL 4: must cross cols 1-2, no detour
    router = GuideRouter(
        chip, chip_origin=ORIGIN, layer_order=LAYERS, pitch_dbu=PITCH, margin=4,
    )
    res_w, res_l = router.route([winner, loser], [guide, guide])
    assert res_w.routed                 # winner keeps its route
    assert not res_l.routed             # loser gives up after the rip-up cap
    assert res_l.paths is None          # unrouted sentinel preserved
    assert res_l.pins == loser          # result still carries the loser's pins


def test_no_conflict_fires_no_extra_reroute_batch(monkeypatch):
    """When no nets conflict, no rip-up reroute happens: the sweep is invoked
    only for the initial routing rounds, never an extra reroute pass. Asserted
    via a call counter wrapped around the batched sweep helper."""
    import gpu_pnr.guide_router as gr

    chip = torch.full((1, 12, 12), 1.0)
    # Spatially disjoint nets — guaranteed zero cross-net conflicts.
    net_a = [(0, 0, 0), (0, 1, 2)]
    net_b = [(0, 8, 8), (0, 10, 10)]
    nets = [net_a, net_b]
    guides = [
        [_rect(0, 0, 3000, 3000, "M1")],
        [_rect(8000, 8000, 12000, 12000, "M1")],
    ]

    real_attach = gr.GuideRouter._attach_batch
    calls = {"n": 0}

    def counting_attach(self, *args, **kwargs):
        calls["n"] += 1
        return real_attach(self, *args, **kwargs)

    monkeypatch.setattr(gr.GuideRouter, "_attach_batch", counting_attach)

    router = gr.GuideRouter(
        chip, chip_origin=ORIGIN, layer_order=LAYERS, pitch_dbu=PITCH, margin=4,
    )
    results = router.route(nets, guides)
    assert all(r.routed for r in results)
    # Both 2-pin disjoint nets finish in a single attachment round → exactly
    # one batched sweep. A rip-up reroute would add at least one more.
    assert calls["n"] == 1
