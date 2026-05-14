"""Tests for the 3D multi-net router (route_nets_3d)."""

from __future__ import annotations

import math

import torch

from gpu_pnr.router import route_multipin_nets_3d, route_nets_3d
from gpu_pnr.sweep import axis_costs


def _path_is_valid(path, w):
    if path is None:
        return False
    for (l1, i1, j1), (l2, i2, j2) in zip(path, path[1:]):
        in_layer = l1 == l2 and abs(i1 - i2) + abs(j1 - j2) == 1
        via = abs(l1 - l2) == 1 and i1 == i2 and j1 == j2
        if not (in_layer or via):
            return False
    for lyr, i, j in path:
        if math.isinf(float(w[lyr, i, j])):
            return False
    return True


def test_single_net_open_3d_grid():
    w = torch.ones(2, 8, 8)
    nets = [((0, 0, 0), (1, 7, 7))]
    results = route_nets_3d(w, nets, via_cost=1.0)
    assert len(results) == 1
    path = results[0].path
    assert path is not None
    assert path[0] == (0, 0, 0)
    assert path[-1] == (1, 7, 7)
    assert _path_is_valid(path, w)


def test_two_nets_disjoint_routes():
    w = torch.ones(2, 8, 8)
    nets = [
        ((0, 0, 0), (0, 0, 7)),
        ((0, 7, 0), (0, 7, 7)),
    ]
    results = route_nets_3d(w, nets, via_cost=10.0)
    p0, p1 = results[0].path, results[1].path
    assert p0 is not None and p1 is not None
    assert set(p0).isdisjoint(set(p1))


def test_second_net_uses_layer_above_to_bypass_first():
    """First net occupies a horizontal stripe on layer 0; second net's pins are
    on the opposite sides of that stripe and on layer 0. With via_cost low,
    the second net should detour up to layer 1 and back."""
    L, H, W = 2, 5, 5
    w = torch.ones(L, H, W)
    nets = [
        ((0, 2, 0), (0, 2, 4)),
        ((0, 0, 2), (0, 4, 2)),
    ]
    results = route_nets_3d(w, nets, via_cost=1.0)
    p0, p1 = results[0].path, results[1].path
    assert p0 is not None
    assert p1 is not None
    assert set(p0).isdisjoint(set(p1))
    layers_used_by_p1 = {p[0] for p in p1}
    assert 1 in layers_used_by_p1, "second net should detour through layer 1"


def test_blocked_endpoint_returns_none():
    w = torch.ones(2, 5, 5)
    w[0, 2, 2] = math.inf
    nets = [((0, 2, 2), (1, 4, 4))]
    results = route_nets_3d(w, nets, via_cost=1.0)
    assert results[0].path is None


def test_endpoint_collision_blocks_second():
    w = torch.ones(2, 5, 5)
    nets = [((0, 0, 0), (1, 4, 4)), ((1, 4, 4), (0, 0, 0))]
    results = route_nets_3d(w, nets, via_cost=1.0)
    assert results[0].routed
    assert results[1].path is None


def test_pin_reservation_protects_other_nets_pins_3d():
    """Same-layer-collision case: net A's pin sits on net B's natural path."""
    w = torch.ones(2, 5, 5)
    nets = [
        ((0, 0, 2), (0, 4, 2)),
        ((0, 2, 2), (0, 0, 4)),
    ]
    no_res = route_nets_3d(w, nets, via_cost=10.0, reserve_pins=False)
    with_res = route_nets_3d(w, nets, via_cost=10.0, reserve_pins=True)
    assert no_res[0].routed
    assert no_res[1].path is None, (
        "without reservation, first net consumes (0,2,2) which is second's source"
    )
    assert with_res[0].routed
    assert with_res[1].routed


def test_obstacle_layer_forces_via_detour():
    """Layer 0 is fully walled at row=2; pins are on layer 0 either side. Path
    must hop to layer 1 to cross."""
    H, W = 5, 5
    w0 = torch.ones(H, W)
    w0[2, :] = math.inf
    w1 = torch.ones(H, W)
    w = torch.stack([w0, w1], dim=0)
    nets = [((0, 0, 0), (0, 4, 4))]
    results = route_nets_3d(w, nets, via_cost=1.0)
    p0 = results[0].path
    assert p0 is not None
    assert _path_is_valid(p0, w)
    layers_used = {p[0] for p in p0}
    assert 1 in layers_used


def test_anisotropic_route_uses_layer_with_cheap_axis():
    """Layer 0 cheap for H, expensive for V; layer 1 cheap for V, expensive
    for H. A diagonal route should hop layers to keep each segment on its
    cheap axis."""
    H, W = 8, 8
    L = 2
    w = torch.ones(L, H, W)
    w_h, w_v = axis_costs(w, [1.0, 30.0], [30.0, 1.0])
    nets = [((0, 0, 0), (0, H - 1, W - 1))]
    results = route_nets_3d(w_h, nets, via_cost=0.5, w_v=w_v)
    path = results[0].path
    assert path is not None
    layers_used = {p[0] for p in path}
    assert 1 in layers_used, "anisotropy should drive the router off layer 0 for V"


def _tree_is_connected(
    paths: list[list[tuple[int, int, int]]],
    pins: list[tuple[int, int, int]],
) -> bool:
    """Check that all pins land in the union of paths and that path cells
    form a single connected component."""
    if not paths:
        return False
    cells = {c for p in paths for c in p}
    if not all(pin in cells for pin in pins):
        return False
    # BFS from any starting cell, walking 4-connected in-layer + via cross-layer.
    start = next(iter(cells))
    seen = {start}
    frontier = [start]
    while frontier:
        nxt = []
        for cl, ci, cj in frontier:
            neighbours = [
                (cl, ci - 1, cj), (cl, ci + 1, cj),
                (cl, ci, cj - 1), (cl, ci, cj + 1),
                (cl - 1, ci, cj), (cl + 1, ci, cj),
            ]
            for n in neighbours:
                if n in cells and n not in seen:
                    seen.add(n)
                    nxt.append(n)
        frontier = nxt
    return seen == cells


def test_multipin_three_pins_open_grid():
    """3-pin net on an open grid; all pins reach a single connected tree."""
    L, H, W = 2, 8, 8
    w = torch.ones(L, H, W)
    pins = [(0, 0, 0), (0, 7, 7), (1, 3, 3)]
    results = route_multipin_nets_3d(w, [pins], via_cost=1.0)
    assert len(results) == 1
    res = results[0]
    assert res.routed
    assert res.paths is not None
    assert len(res.paths) == len(pins), "expect one path per attachment edge"
    assert _tree_is_connected(res.paths, pins)


def test_multipin_four_pins_obstacle_detour():
    """4-pin net where layer 0 has a wall; tree must use layer 1 to bridge."""
    L, H, W = 2, 6, 6
    w = torch.ones(L, H, W)
    w[0, 3, :] = math.inf
    pins = [(0, 0, 0), (0, 5, 0), (0, 0, 5), (0, 5, 5)]
    results = route_multipin_nets_3d(w, [pins], via_cost=1.0)
    res = results[0]
    assert res.routed
    assert res.paths is not None
    assert _tree_is_connected(res.paths, pins)
    layers_used = {c[0] for c in res.cells}
    assert 1 in layers_used, "tree must hop to layer 1 to cross the wall"


def test_multipin_two_pin_matches_2pin_router():
    """2-pin nets should produce equivalent topology under both routers
    (same set of visited cells, possibly with different orderings)."""
    L, H, W = 2, 6, 6
    w = torch.ones(L, H, W)
    pins = [(0, 0, 0), (0, 5, 5)]
    res_two = route_nets_3d(w, [(pins[0], pins[1])], via_cost=1.0)
    res_multi = route_multipin_nets_3d(w, [pins], via_cost=1.0)
    assert res_two[0].path is not None
    assert res_multi[0].routed
    assert set(res_two[0].path) == res_multi[0].cells


def test_multipin_pin_reservation_blocks_other_net():
    """Two 3-pin nets sharing no pin cells; reservation keeps trees disjoint."""
    L, H, W = 2, 8, 8
    w = torch.ones(L, H, W)
    nets = [
        [(0, 0, 0), (0, 0, 7), (0, 4, 3)],
        [(0, 7, 0), (0, 7, 7), (0, 4, 5)],
    ]
    results = route_multipin_nets_3d(w, nets, via_cost=1.0)
    assert all(r.routed for r in results)
    cells_a = results[0].cells
    cells_b = results[1].cells
    assert cells_a.isdisjoint(cells_b)


def test_multipin_blocked_pin_returns_unrouted():
    """A pin on an inf cell should produce an unrouted result."""
    L, H, W = 2, 5, 5
    w = torch.ones(L, H, W)
    w[0, 2, 2] = math.inf
    pins = [(0, 0, 0), (0, 2, 2), (0, 4, 4)]
    results = route_multipin_nets_3d(w, [pins], via_cost=1.0)
    assert not results[0].routed


def test_anisotropic_pin_reservation_blocks_both_axes():
    """Reserving pins under anisotropy must obstruct both w_h and w_v at pin
    cells so the second net can't run a wire through the first net's pin."""
    H, W = 5, 5
    L = 2
    w = torch.ones(L, H, W)
    # Layer 0 prefers H; layer 1 prefers V.
    w_h, w_v = axis_costs(w, [1.0, 10.0], [10.0, 1.0])
    # Net 0's natural H route on layer 0 runs through (0,2,2), which is
    # Net 1's source. Picked the geometry deliberately so the anisotropy
    # picks layer-0 H over a layer-1 via-stack (vias=10, 4 H cells=4 << 20).
    nets = [
        ((0, 2, 0), (0, 2, 4)),
        ((0, 2, 2), (0, 0, 4)),
    ]
    no_res = route_nets_3d(w_h, nets, via_cost=10.0, reserve_pins=False, w_v=w_v)
    with_res = route_nets_3d(w_h, nets, via_cost=10.0, reserve_pins=True, w_v=w_v)
    assert no_res[0].routed
    assert no_res[1].path is None, (
        "without reservation, first net consumes (0,2,2) which is second's source"
    )
    assert with_res[0].routed
    assert with_res[1].routed


def test_route_nets_3d_per_pair_via_cost():
    """route_nets_3d plumbs per-pair via_cost into the kernel. A net spanning
    M1->M3 traverses two distinct via pairs; the path must complete with
    asymmetric per-pair via_cost."""
    L, H, W = 3, 4, 4
    w = torch.ones(L, H, W)
    nets = [((0, 0, 0), (2, 0, 0))]
    via_costs = [50.0, 1.0]
    results = route_nets_3d(w, nets, via_cost=via_costs)
    path = results[0].path
    assert path is not None
    # Path: (0,0,0) -> (1,0,0) -> (2,0,0). Two via edges (one per pair).
    assert len(path) == 3


def test_route_multipin_nets_3d_per_pair_via_cost():
    """route_multipin_nets_3d plumbs per-pair via_cost into the kernel."""
    L, H, W = 3, 6, 6
    w = torch.ones(L, H, W)
    nets = [[(0, 0, 0), (2, 5, 5), (1, 2, 2)]]
    via_costs = [3.0, 1.0]
    results = route_multipin_nets_3d(w, nets, via_cost=via_costs)
    assert results[0].routed
