"""Correctness tests for the 3D (multi-layer + via) sweep SSSP."""

from __future__ import annotations

import math

import torch

from gpu_pnr.baseline import dijkstra_grid, dijkstra_grid_3d
from gpu_pnr.sweep import (
    axis_costs,
    backtrace_3d,
    sweep_sssp,
    sweep_sssp_3d,
    sweep_sssp_3d_multi,
)


def _assert_distances_match(
    d_sweep: torch.Tensor, d_ref: torch.Tensor, atol: float = 1e-2
) -> None:
    """Compare sweep result against reference. Default atol of 1e-2 absorbs
    float32 drift from two sources: cumsum-based scans accumulate roundoff
    differently from Dijkstra's edge-by-edge sums, and the segmented-scan
    SEG_BARRIER bookkeeping adds/subtracts O(seg_id * 2e4) at intermediate
    steps, whose float32 ULP is ~0.024. Empirically the combined drift is
    ~5e-3 on the 3D random test at 14x14x3."""
    d_sweep_cpu = d_sweep.detach().cpu()
    d_ref_cpu = d_ref.detach().cpu()
    finite_mask = torch.isfinite(d_ref_cpu)
    inf_mask = ~finite_mask
    assert torch.allclose(
        d_sweep_cpu[finite_mask], d_ref_cpu[finite_mask], atol=atol
    ), f"finite mismatch:\nsweep={d_sweep_cpu}\nref={d_ref_cpu}"
    assert torch.all(torch.isinf(d_sweep_cpu[inf_mask])), "sweep finite where ref infinite"


def test_single_layer_3d_matches_2d():
    """L=1 with any via_cost must equal the existing 2D sweep result."""
    torch.manual_seed(0)
    H, W = 16, 16
    w2d = torch.rand(H, W) + 0.1
    w2d[5, 3:13] = math.inf
    w3d = w2d.unsqueeze(0)
    source_2d = (0, 0)
    source_3d = (0, 0, 0)
    d_2d, _ = sweep_sssp(w2d, source_2d)
    d_3d, _ = sweep_sssp_3d(w3d, source_3d, via_cost=5.0)
    _assert_distances_match(d_3d[0], d_2d)


def test_two_layers_zero_via_collapses_to_2d_min():
    """via_cost=0 means any layer's d at (r,c) >= the per-(r,c) min over both layers' 2D solutions."""
    torch.manual_seed(1)
    H, W = 12, 12
    w_layer0 = torch.rand(H, W) + 0.1
    w_layer1 = torch.rand(H, W) + 0.1
    w3d = torch.stack([w_layer0, w_layer1], dim=0)
    d3d, _ = sweep_sssp_3d(w3d, (0, 0, 0), via_cost=0.0)
    d_ref = dijkstra_grid_3d(w3d, (0, 0, 0), via_cost=0.0)
    _assert_distances_match(d3d, d_ref)
    finite = torch.isfinite(d3d.cpu())
    assert torch.allclose(d3d.cpu()[0][finite[0] & finite[1]], d3d.cpu()[1][finite[0] & finite[1]], atol=1e-4)


def test_high_via_keeps_path_on_source_layer():
    """When via_cost is huge and source layer has a clear path, distances on
    the other layer should be (source_layer_distance + 2*via_cost) since the
    only way to reach them is one via down and one via back up... actually no,
    with one via you stay on the other layer. So d[other] = d[same] + via_cost
    at minimum (single via from any reachable source-layer cell)."""
    H, W = 8, 8
    w3d = torch.ones(2, H, W)
    via_cost = 100.0
    d, _ = sweep_sssp_3d(w3d, (0, 0, 0), via_cost=via_cost)
    d_layer0 = d[0].cpu()
    d_layer1 = d[1].cpu()
    diff = d_layer1 - d_layer0
    finite = torch.isfinite(d_layer0) & torch.isfinite(d_layer1)
    assert (diff[finite] >= via_cost - 1e-4).all(), (
        "layer-1 distances must each cost at least one via more than layer-0"
    )
    assert (diff[finite] <= via_cost + 1e-4).all(), (
        "with via_cost dominating, the cheapest layer-1 path is exactly one via"
    )


def test_obstacle_detour_via_other_layer():
    """Layer 0 has an obstacle wall; layer 1 is open; via_cost is small.
    A sink past the wall on layer 0 must route via layer 1."""
    H, W = 10, 10
    w0 = torch.ones(H, W)
    w0[5, :] = math.inf
    w1 = torch.ones(H, W)
    w3d = torch.stack([w0, w1], dim=0)
    source = (0, 0, 0)
    sink = (0, H - 1, W - 1)
    d, _ = sweep_sssp_3d(w3d, source, via_cost=1.0)
    d_ref = dijkstra_grid_3d(w3d, source, via_cost=1.0)
    _assert_distances_match(d, d_ref)
    path = backtrace_3d(d.cpu(), w3d.cpu(), source, sink, via_cost=1.0)
    assert path is not None
    assert path[0] == source
    assert path[-1] == sink
    layers_used = {p[0] for p in path}
    assert 1 in layers_used, "path must use layer 1 to get past the wall"


def test_random_3d_matches_dijkstra():
    torch.manual_seed(2)
    L, H, W = 3, 14, 14
    w = torch.rand(L, H, W) + 0.1
    w[0, 6, 2:12] = math.inf
    w[2, 3:11, 7] = math.inf
    source = (0, 0, 0)
    via_cost = 0.7
    d_sweep, _ = sweep_sssp_3d(w, source, via_cost=via_cost)
    d_ref = dijkstra_grid_3d(w, source, via_cost=via_cost)
    _assert_distances_match(d_sweep, d_ref)


def test_backtrace_path_validity_with_vias():
    torch.manual_seed(3)
    L, H, W = 3, 10, 10
    w = torch.rand(L, H, W) + 0.1
    w[1, 4, 1:9] = math.inf
    source = (0, 0, 0)
    sink = (2, 9, 9)
    via_cost = 0.5
    d, _ = sweep_sssp_3d(w, source, via_cost=via_cost)
    path = backtrace_3d(d.cpu(), w.cpu(), source, sink, via_cost=via_cost)
    assert path is not None
    assert path[0] == source
    assert path[-1] == sink
    for (l1, i1, j1), (l2, i2, j2) in zip(path, path[1:]):
        in_layer = l1 == l2 and abs(i1 - i2) + abs(j1 - j2) == 1
        via = abs(l1 - l2) == 1 and i1 == i2 and j1 == j2
        assert in_layer or via, f"non-adjacent step {(l1, i1, j1)} -> {(l2, i2, j2)}"
    for lyr, i, j in path:
        assert not math.isinf(float(w[lyr, i, j])), "path goes through obstacle"


def test_unreachable_isolated_layer():
    """Source on layer 0 is fully walled off in layer 0; via_cost is finite but
    layer-0 obstacles force any path off layer 0 immediately."""
    H, W = 6, 6
    w0 = torch.ones(H, W)
    w0[1, :] = math.inf
    w1 = torch.ones(H, W)
    w3d = torch.stack([w0, w1], dim=0)
    source = (0, 0, 0)
    sink = (0, H - 1, W - 1)
    d, _ = sweep_sssp_3d(w3d, source, via_cost=1.0)
    assert torch.isfinite(d[sink])
    d_ref = dijkstra_grid_3d(w3d, source, via_cost=1.0)
    _assert_distances_match(d, d_ref)


def test_mps_matches_cpu_3d():
    if not torch.backends.mps.is_available():
        return
    torch.manual_seed(4)
    L, H, W = 3, 32, 32
    w_cpu = torch.rand(L, H, W) + 0.1
    w_cpu[1, 10, 5:25] = math.inf
    source = (0, 0, 0)
    via_cost = 0.8
    d_cpu, _ = sweep_sssp_3d(w_cpu, source, via_cost=via_cost)
    d_mps, _ = sweep_sssp_3d(w_cpu.to("mps"), source, via_cost=via_cost)
    finite = torch.isfinite(d_cpu)
    inf = ~finite
    d_mps_cpu = d_mps.cpu()
    assert torch.allclose(d_mps_cpu[finite], d_cpu[finite], atol=5e-2), (
        "MPS and CPU 3D sweep disagree beyond float32 sum-order drift"
    )
    assert torch.all(torch.isinf(d_mps_cpu[inf]))


def test_dijkstra_3d_collapses_to_2d_when_one_layer():
    """Sanity: dijkstra_grid_3d on (1, H, W) must equal dijkstra_grid on (H, W)."""
    torch.manual_seed(5)
    H, W = 10, 10
    w2d = torch.rand(H, W) + 0.1
    w2d[3, 1:8] = math.inf
    d_2d = dijkstra_grid(w2d, (0, 0))
    d_3d = dijkstra_grid_3d(w2d.unsqueeze(0), (0, 0, 0), via_cost=99.0)
    _assert_distances_match(d_3d[0], d_2d)


def test_anisotropic_w_v_none_equals_isotropic():
    """sweep_sssp_3d(w_v=None) must equal sweep_sssp_3d(w_v=w) -- back-compat."""
    torch.manual_seed(10)
    L, H, W = 3, 12, 12
    w = torch.rand(L, H, W) + 0.1
    w[1, 5, 2:10] = math.inf
    source = (0, 0, 0)
    via_cost = 0.7
    d_iso, _ = sweep_sssp_3d(w, source, via_cost=via_cost)
    d_explicit, _ = sweep_sssp_3d(w, source, via_cost=via_cost, w_v=w.clone())
    _assert_distances_match(d_explicit, d_iso)


def test_anisotropic_factored_matches_dijkstra():
    """Factored per-layer (h_mult, v_mult) costs match the dijkstra reference."""
    torch.manual_seed(11)
    L, H, W = 3, 14, 14
    w = torch.rand(L, H, W) + 0.1
    w[0, 6, 2:12] = math.inf
    w[2, 3:11, 7] = math.inf
    # Mimic gf180mcuD on 3 layers: M1=H-pref, M2=V-pref, M3=H-pref.
    h_mult = [1.0, 8.0, 1.0]
    v_mult = [8.0, 1.0, 8.0]
    w_h, w_v = axis_costs(w, h_mult, v_mult)
    source = (0, 0, 0)
    via_cost = 0.7
    d_sweep, _ = sweep_sssp_3d(w_h, source, via_cost=via_cost, w_v=w_v)
    d_ref = dijkstra_grid_3d(w_h, source, via_cost=via_cost, w_v=w_v)
    _assert_distances_match(d_sweep, d_ref, atol=5e-2)


def test_anisotropic_prefers_via_to_cheap_axis():
    """Two layers: layer 0 only allows cheap horizontal moves, layer 1 only
    allows cheap vertical moves. Routing diagonally must use both layers."""
    H, W = 8, 8
    L = 2
    w = torch.ones(L, H, W)
    # Layer 0: H cheap (1), V expensive (50).
    # Layer 1: H expensive (50), V cheap (1).
    h_mult = [1.0, 50.0]
    v_mult = [50.0, 1.0]
    w_h, w_v = axis_costs(w, h_mult, v_mult)
    source = (0, 0, 0)
    sink = (0, H - 1, W - 1)
    via_cost = 0.5
    d, _ = sweep_sssp_3d(w_h, source, via_cost=via_cost, w_v=w_v)
    path = backtrace_3d(
        d.cpu(), w_h.cpu(), source, sink, via_cost=via_cost, w_v=w_v.cpu()
    )
    assert path is not None
    assert path[0] == source
    assert path[-1] == sink
    layers_used = {p[0] for p in path}
    assert 1 in layers_used, "anisotropy must drive the router off-layer for V moves"


def test_anisotropic_obstacle_only_blocks_its_axis():
    """A cell with w_h=inf, w_v=finite should be enterable vertically and
    exit-able as a via-landing site, but not enterable horizontally."""
    H, W = 6, 6
    L = 2
    w_h = torch.ones(L, H, W)
    w_v = torch.ones(L, H, W)
    # Pillar of horizontal-blocked cells at column 3 of layer 0 -- you can
    # still traverse it vertically.
    w_h[0, :, 3] = math.inf
    source = (0, 0, 0)
    via_cost = 0.5
    d, _ = sweep_sssp_3d(w_h, source, via_cost=via_cost, w_v=w_v)
    d_ref = dijkstra_grid_3d(w_h, source, via_cost=via_cost, w_v=w_v)
    _assert_distances_match(d, d_ref, atol=5e-2)
    # The cell (0, 3, 3) is H-blocked. Approaching from (0, 2, 3) or (0, 4, 3)
    # is a V move (di != 0), which is allowed -- so it must be finite-reachable.
    assert torch.isfinite(d[0, 3, 3])


def test_anisotropic_backtrace_path_validity():
    """Backtrace under anisotropic costs returns a path whose summed
    per-edge axis-aware cost matches d[sink]."""
    torch.manual_seed(12)
    L, H, W = 3, 10, 10
    w = torch.rand(L, H, W) + 0.1
    h_mult = [1.0, 5.0, 1.0]
    v_mult = [5.0, 1.0, 5.0]
    w_h, w_v = axis_costs(w, h_mult, v_mult)
    source = (0, 0, 0)
    sink = (2, 9, 9)
    via_cost = 0.5
    d, _ = sweep_sssp_3d(w_h, source, via_cost=via_cost, w_v=w_v)
    path = backtrace_3d(
        d.cpu(), w_h.cpu(), source, sink, via_cost=via_cost, w_v=w_v.cpu()
    )
    assert path is not None
    assert path[0] == source
    assert path[-1] == sink
    # Walk the path summing axis-aware costs; must match d[sink] to within atol.
    total = 0.0
    for (l1, i1, j1), (l2, i2, j2) in zip(path, path[1:]):
        if l1 == l2 and i1 == i2 and abs(j1 - j2) == 1:
            total += float(w_h[l2, i2, j2])
        elif l1 == l2 and j1 == j2 and abs(i1 - i2) == 1:
            total += float(w_v[l2, i2, j2])
        elif abs(l1 - l2) == 1 and i1 == i2 and j1 == j2:
            total += via_cost
        else:
            raise AssertionError(f"non-adjacent step {(l1, i1, j1)} -> {(l2, i2, j2)}")
    assert abs(total - float(d[sink])) <= 5e-2, (
        f"path cost {total} disagrees with d[sink]={float(d[sink])}"
    )


def test_axis_costs_preserves_obstacles():
    """axis_costs must leave inf cells as inf in both output tensors regardless
    of the per-layer multiplier."""
    L, H, W = 3, 5, 5
    w = torch.ones(L, H, W)
    w[1, 2, 2] = math.inf
    w_h, w_v = axis_costs(w, [1.0, 10.0, 1.0], [10.0, 1.0, 10.0])
    assert math.isinf(float(w_h[1, 2, 2]))
    assert math.isinf(float(w_v[1, 2, 2]))
    assert float(w_h[0, 0, 0]) == 1.0
    assert float(w_v[0, 0, 0]) == 10.0
    assert float(w_h[1, 0, 0]) == 10.0
    assert float(w_v[1, 0, 0]) == 1.0


def test_extra_sources_seed_zero_distance():
    """Cells passed via extra_sources start at d=0 alongside the primary
    source; d at any extra source must be 0 after the sweep."""
    L, H, W = 2, 8, 8
    w = torch.ones(L, H, W)
    primary = (0, 0, 0)
    extras = [(0, 4, 4), (1, 7, 7)]
    d, _ = sweep_sssp_3d(w, primary, via_cost=1.0, extra_sources=extras)
    assert float(d[primary]) == 0.0
    for e in extras:
        assert float(d[e]) == 0.0, f"extra source {e} did not survive sweep"


def test_extra_sources_match_min_over_dijkstra_runs():
    """SSSP from multiple sources equals the element-wise min of SSSP runs
    from each source individually -- this is the canonical multi-source
    semantic."""
    torch.manual_seed(20)
    L, H, W = 3, 10, 10
    w = torch.rand(L, H, W) + 0.1
    w[1, 5, 3:8] = math.inf
    via_cost = 0.6
    sources = [(0, 0, 0), (2, 9, 9), (1, 3, 8)]
    d_multi, _ = sweep_sssp_3d(
        w, sources[0], via_cost=via_cost, extra_sources=sources[1:]
    )
    d_individual = [
        sweep_sssp_3d(w, s, via_cost=via_cost)[0] for s in sources
    ]
    d_stack = torch.stack(d_individual, dim=0)
    d_expected = d_stack.amin(dim=0)
    finite = torch.isfinite(d_expected).cpu()
    assert torch.allclose(
        d_multi.cpu()[finite], d_expected.cpu()[finite], atol=5e-2
    )


def test_sweep_sssp_3d_multi_matches_per_source():
    """K-batched 3D distance maps must equal the per-source single-source
    runs stacked. The leading K dim is the only difference; each slice
    is an independent SSSP."""
    torch.manual_seed(30)
    L, H, W = 3, 12, 12
    w = torch.rand(L, H, W) + 0.1
    w[1, 5, 2:10] = math.inf
    via_cost = 0.7
    sources = [(0, 0, 0), (2, 11, 11), (1, 3, 8), (0, 6, 4)]
    d_multi, _ = sweep_sssp_3d_multi(w, sources, via_cost=via_cost)
    assert d_multi.shape == (len(sources), L, H, W)
    for k, s in enumerate(sources):
        d_single, _ = sweep_sssp_3d(w, s, via_cost=via_cost)
        finite = torch.isfinite(d_single)
        assert torch.allclose(
            d_multi[k].cpu()[finite.cpu()], d_single.cpu()[finite.cpu()],
            atol=5e-2,
        ), f"source {k}={s}: K-batched diverges from single-source"
        assert torch.all(torch.isinf(d_multi[k].cpu()[~finite.cpu()]))


def test_sweep_sssp_3d_multi_anisotropic_matches_per_source():
    """K-batched 3D sweep with w_v anisotropy still tracks per-source runs."""
    torch.manual_seed(31)
    L, H, W = 3, 10, 10
    w = torch.rand(L, H, W) + 0.1
    w_h, w_v = axis_costs(w, [1.0, 8.0, 1.0], [8.0, 1.0, 8.0])
    via_cost = 0.5
    sources = [(0, 0, 0), (2, 9, 9), (1, 4, 4)]
    d_multi, _ = sweep_sssp_3d_multi(w_h, sources, via_cost=via_cost, w_v=w_v)
    for k, s in enumerate(sources):
        d_single, _ = sweep_sssp_3d(w_h, s, via_cost=via_cost, w_v=w_v)
        finite = torch.isfinite(d_single)
        assert torch.allclose(
            d_multi[k].cpu()[finite.cpu()], d_single.cpu()[finite.cpu()],
            atol=5e-2,
        ), f"source {k}={s}: anisotropic K-batched diverges from single-source"


def test_backtrace_extra_sources_terminates_at_any_seed():
    """When extra_sources is non-empty, backtrace walks back to whichever
    source cell is on the shortest predecessor chain, not just the named
    primary source."""
    L, H, W = 1, 6, 6
    w = torch.ones(L, H, W)
    # Sink is closer to the extra source than to the primary.
    primary = (0, 0, 0)
    extras = [(0, 5, 5)]
    sink = (0, 5, 4)
    d, _ = sweep_sssp_3d(w, primary, via_cost=1.0, extra_sources=extras)
    path = backtrace_3d(
        d.cpu(), w.cpu(), primary, sink, via_cost=1.0,
        extra_sources=extras,
    )
    assert path is not None
    assert path[-1] == sink
    # The first cell in the reversed (path-ordered) list is whichever
    # terminal we reached. With the extra source closer, we expect it
    # there, not the distant primary.
    assert path[0] == extras[0], (
        f"backtrace should attach at the nearer extra source {extras[0]}, "
        f"got {path[0]}"
    )


# --- per-pair via_cost tests -------------------------------------------------


def test_per_pair_via_cost_scalar_equivalence():
    """A length-(L-1) array of identical values must produce the same
    distances as the scalar of that value. Backwards-compat anchor."""
    torch.manual_seed(40)
    L, H, W = 5, 12, 12
    w = torch.rand(L, H, W) + 0.1
    w[2, 5, 3:9] = math.inf
    source = (0, 0, 0)
    scalar_v = 0.7
    d_scalar, _ = sweep_sssp_3d(w, source, via_cost=scalar_v)
    d_array, _ = sweep_sssp_3d(w, source, via_cost=[scalar_v] * (L - 1))
    _assert_distances_match(d_array, d_scalar, atol=1e-3)


def test_per_pair_via_cost_asymmetric_steers_routing():
    """When via_cost[0] is huge (M1-M2 expensive) but via_cost[1] is cheap
    (M2-M3 cheap), a path forced through both vias should reflect the
    different costs."""
    L, H, W = 3, 6, 6
    w = torch.ones(L, H, W)
    source = (0, 0, 0)
    sink = (2, 0, 0)
    # via_costs[0] = 100 (M1<->M2), via_costs[1] = 1 (M2<->M3)
    d, _ = sweep_sssp_3d(w, source, via_cost=[100.0, 1.0])
    # Distance must equal 100 + 1 = 101 (two vias, distinct costs).
    assert abs(float(d[sink]) - 101.0) < 1e-3, f"got d[sink]={float(d[sink])}"
    # Reverse: cheap first via, then expensive.
    d2, _ = sweep_sssp_3d(w, source, via_cost=[1.0, 100.0])
    assert abs(float(d2[sink]) - 101.0) < 1e-3, f"got d2[sink]={float(d2[sink])}"
    # Single layer-1 hop must reflect only the first via.
    assert abs(float(d[(1, 0, 0)]) - 100.0) < 1e-3
    assert abs(float(d2[(1, 0, 0)]) - 1.0) < 1e-3


def test_per_pair_via_cost_matches_dijkstra():
    """Random cost grid + asymmetric per-pair via_cost: sweep must match
    Dijkstra reference."""
    torch.manual_seed(41)
    L, H, W = 4, 10, 10
    w = torch.rand(L, H, W) + 0.1
    w[1, 4, 2:8] = math.inf
    w[2, 7, 1:5] = math.inf
    source = (0, 0, 0)
    via_costs = [3.0, 1.0, 5.0]
    d_sweep, _ = sweep_sssp_3d(w, source, via_cost=via_costs)
    d_ref = dijkstra_grid_3d(w, source, via_cost=via_costs)
    _assert_distances_match(d_sweep, d_ref, atol=5e-2)


def test_per_pair_via_cost_backtrace_uses_correct_per_step_cost():
    """Backtrace under asymmetric per-pair via_cost reconstructs a path
    whose summed cost matches d[sink]. The via_target check must index
    via_cost[cur_l-1] when stepping down, via_cost[cur_l] when stepping up."""
    torch.manual_seed(42)
    L, H, W = 4, 8, 8
    w = torch.rand(L, H, W) + 0.1
    w_h, w_v = axis_costs(w, [1.0, 5.0, 1.0, 5.0], [5.0, 1.0, 5.0, 1.0])
    source = (0, 0, 0)
    sink = (3, 7, 7)
    via_costs = [2.0, 0.5, 4.0]
    d, _ = sweep_sssp_3d(w_h, source, via_cost=via_costs, w_v=w_v)
    path = backtrace_3d(
        d.cpu(), w_h.cpu(), source, sink, via_cost=via_costs, w_v=w_v.cpu()
    )
    assert path is not None
    assert path[0] == source
    assert path[-1] == sink
    total = 0.0
    for (l1, i1, j1), (l2, i2, j2) in zip(path, path[1:]):
        if l1 == l2 and i1 == i2 and abs(j1 - j2) == 1:
            total += float(w_h[l2, i2, j2])
        elif l1 == l2 and j1 == j2 and abs(i1 - i2) == 1:
            total += float(w_v[l2, i2, j2])
        elif abs(l1 - l2) == 1 and i1 == i2 and j1 == j2:
            total += via_costs[min(l1, l2)]
        else:
            raise AssertionError(f"non-adjacent step {(l1, i1, j1)} -> {(l2, i2, j2)}")
    assert abs(total - float(d[sink])) <= 5e-2, (
        f"path cost {total} disagrees with d[sink]={float(d[sink])}"
    )


def test_sweep_sssp_3d_multi_per_pair_matches_per_source():
    """K-batched 3D sweep with per-pair via_cost must equal per-source
    runs with the same per-pair via_cost."""
    torch.manual_seed(43)
    L, H, W = 4, 10, 10
    w = torch.rand(L, H, W) + 0.1
    via_costs = [2.0, 0.5, 4.0]
    sources = [(0, 0, 0), (3, 9, 9), (1, 4, 4)]
    d_multi, _ = sweep_sssp_3d_multi(w, sources, via_cost=via_costs)
    for k, s in enumerate(sources):
        d_single, _ = sweep_sssp_3d(w, s, via_cost=via_costs)
        finite = torch.isfinite(d_single)
        assert torch.allclose(
            d_multi[k].cpu()[finite.cpu()], d_single.cpu()[finite.cpu()],
            atol=5e-2,
        ), f"source {k}={s}: K-batched per-pair diverges from single-source"


def test_per_pair_via_cost_wrong_length_raises():
    """Per-pair array of wrong shape must raise."""
    L, H, W = 3, 4, 4
    w = torch.ones(L, H, W)
    try:
        sweep_sssp_3d(w, (0, 0, 0), via_cost=[1.0, 2.0, 3.0])  # need L-1=2
    except ValueError as e:
        assert "via_cost must be scalar or shape (2,)" in str(e)
    else:
        raise AssertionError("expected ValueError")
