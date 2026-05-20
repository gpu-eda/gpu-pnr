"""Sweep-based SSSP on a 2D grid (4-connected), Bellman-Ford via Gauss-Seidel.

Each "iteration" runs four directional axis sweeps (H-forward, H-backward,
V-forward, V-backward). Each sweep is implemented as a segmented cumsum +
segmented cummin per axis, which dispatches as a parallel scan on GPU
rather than N sequential kernel launches.

Forward-sweep derivation (within a segment, i.e., a maximal run of
non-obstacle cells along the axis):
  d_new[j] = min over k<=j of (d[k] + sum w[k+1..j])
           = seg_cw[j] + min over k<=j in same segment of (d[k] - seg_cw[k])
where seg_cw[j] = cumsum of w from the current segment's start to j.

Obstacles are handled with a segmented scan, not a finite proxy:
  - cumsum(w_clean) where w_clean treats obstacles as 0; magnitudes stay
    proportional to real path weight (no INF_PROXY * N inflation).
  - seg_cw[j] = cw[j] - cw_at_most_recent_obstacle[j] (the latter via
    cummax of cw masked at obstacle positions).
  - cummin's input is offset by seg_id * SEG_BARRIER, where seg_id is the
    cumulative obstacle count along the axis. Earlier segments have a
    smaller offset subtracted, so their values are larger; cummin
    therefore can never pick across a segment boundary. The offset is
    subtracted back exactly to recover segment-restricted minima.

Float32 precision budget: max(|seg_cw|) is bounded by per-segment path
weight (small); max(|seg_id * SEG_BARRIER|) is the new dominant term.
With SEG_BARRIER=2e4 and max ~200 obstacles per row (5% density at 4096),
worst-case magnitude is ~4e6; float32 ULP ~0.25, leaving comfortable
headroom for unit-weight distances on grids well past 4096^2.

Convergence: O(diameter) iterations; typically a handful for sparse
obstacles.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

FLOAT32_PRECISION_BUDGET = 1e7  # ULP at 1e7 is ~0.6; safe headroom for autotune


@dataclass(frozen=True)
class _ScanState:
    """Loop-invariant per-(axis, direction) state for the masked scan.

    seg_cw[j] = cumsum of finite-only weight from the current segment's start
    through j. seg_id_barrier[j] = seg_id[j] * seg_barrier, the offset that
    keeps cummin from picking across segment boundaries. obstacle_mask is the
    same orientation as seg_cw / seg_id_barrier (flipped along axis for the
    backward direction). seg_barrier is carried so _sweep_forward can compute
    the polluted-mask threshold (= seg_barrier / 2) without a module global.
    """

    seg_cw: torch.Tensor
    seg_id_barrier: torch.Tensor
    obstacle_mask: torch.Tensor
    seg_barrier: float


def _obstacle_mask(w: torch.Tensor) -> torch.Tensor:
    return torch.isinf(w)


def _normalize_via_cost(
    via_cost: float | Sequence[float] | torch.Tensor,
    L: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Normalize via_cost to a length-(L-1) tensor on (device, dtype).

    Per-pair via_cost[k] is the edge weight between layer k and layer k+1
    (covers both directions). A scalar input broadcasts to a uniform array.
    For L=1 the result is a length-0 tensor (no vias).
    """
    n = L - 1
    if isinstance(via_cost, (int, float)):
        return torch.full((n,), float(via_cost), device=device, dtype=dtype)
    vc = torch.as_tensor(via_cost, device=device, dtype=dtype)
    if vc.shape != (n,):
        raise ValueError(
            f"via_cost must be scalar or shape ({n},), got {tuple(vc.shape)}"
        )
    return vc


def _autotune_seg_barrier(
    w: torch.Tensor,
    obstacle_mask: torch.Tensor,
    via_cost: float | Sequence[float] | torch.Tensor = 0.0,
    w_v: torch.Tensor | None = None,
    obstacle_mask_v: torch.Tensor | None = None,
) -> float:
    """Pick SEG_BARRIER from grid shape and obstacle distribution.

    Constraints (see docs/adr/0005-mask-based-segmented-scan.md and
    docs/spikes/phase32-hazard3-real-fixture.md):
      lower: SEG_BARRIER > 2 * max_legit_distance
        (so polluted-mask threshold = SEG_BARRIER/2 cleanly separates legit
         distances from cross-segment pollution shifted by SEG_BARRIER).
      upper: SEG_BARRIER * max_seg_id < FLOAT32_PRECISION_BUDGET
        (so float32 ULP at the seg_id*SEG_BARRIER product stays well below 1
         and doesn't corrupt distances during the cummin reconstruction).

    Synthetic 4096^2 grids with 5% obstacles want SEG_BARRIER ~2e4; real
    per-net guides with ~93% obstacle density want ~5e3. A single module
    constant can't cover both -- this function picks the geometric mean of
    the valid range for the actual grid being routed.

    When `w_v` / `obstacle_mask_v` are provided (anisotropic case), the
    max-cost estimate uses max(max_w_finite_h, max_w_finite_v) and the
    seg-id estimate scans both masks. A path of length H+W edges has cost
    at most (H+W)*max(max_w_h, max_w_v) + L*via_cost regardless of which
    edges are H vs V, so the joint upper bound is the right legit-distance
    estimate; using just one axis would underestimate.

    Cost: 1 cumsum-along-each-spatial-axis + 1 max + 1 sync per axis, plus 1
    masked max(w_finite) + 1 sync. ~3 syncs total at ~0.5ms each on MPS.
    """
    max_w_h = float(torch.where(obstacle_mask, 0.0, w).max().item()) if w.numel() else 0.0
    if w_v is not None:
        mask_v = obstacle_mask_v if obstacle_mask_v is not None else obstacle_mask
        max_w_v = float(torch.where(mask_v, 0.0, w_v).max().item()) if w_v.numel() else 0.0
    else:
        max_w_v = 0.0
    max_w_finite = max(max_w_h, max_w_v, 1.0)
    spatial_dims = w.shape[-2:]
    layer_dim = w.shape[0] if w.ndim == 3 else 1
    if isinstance(via_cost, torch.Tensor):
        via_max = float(via_cost.max().item()) if via_cost.numel() else 0.0
    elif isinstance(via_cost, (int, float)):
        via_max = float(via_cost)
    else:
        seq = list(via_cost)
        via_max = max(seq) if seq else 0.0
    max_legit_hint = sum(spatial_dims) * max_w_finite + layer_dim * via_max

    # max_seg_id is the largest cumulative obstacle count along any axis; since
    # cumsum is non-decreasing, max(cumsum(mask, axis)) == max(sum(mask, axis)).
    # Using sum instead of cumsum avoids an O(N) GPU pass and the temp alloc.
    masks = [obstacle_mask]
    if obstacle_mask_v is not None and obstacle_mask_v is not obstacle_mask:
        masks.append(obstacle_mask_v)
    max_seg_id = 0
    for mask in masks:
        for axis in range(mask.ndim - 2, mask.ndim):
            max_seg_id = max(
                max_seg_id, int(mask.sum(dim=axis).max().item())
            )
    if max_seg_id == 0:
        return 2.0 * max_legit_hint + 1.0
    upper = FLOAT32_PRECISION_BUDGET / max_seg_id
    lower = 2.0 * max_legit_hint
    if lower >= upper:
        # Workload exceeds the float32 precision budget; the polluted-mask is
        # going to false-positive on legit distances. Pick a hair below the
        # upper bound (1% headroom) so we don't bake a value at exactly the
        # ULP boundary and to keep the failure mode "some legit cells go inf"
        # rather than "wrong distances on cells just below the threshold."
        return upper * 0.99
    return (lower * upper) ** 0.5


def _precompute_scan(
    w: torch.Tensor,
    obstacle_mask: torch.Tensor,
    axis: int,
    seg_barrier: float,
) -> _ScanState:
    """Compute the parts of the segmented scan that depend only on (w, mask).

    Hoisting these out of the convergence loop matters: they're recomputed
    O(diameter) times otherwise, but they don't change as `d` evolves. With
    the hoist, the per-iter inner sweep collapses to one cummin and a few
    arithmetic ops on (d - seg_cw - seg_id_barrier).
    """
    w_clean = torch.where(obstacle_mask, 0.0, w)
    cw = torch.cumsum(w_clean, dim=axis)
    seg_id = torch.cumsum(obstacle_mask.to(w.dtype), dim=axis)
    cw_at_obs = torch.where(obstacle_mask, cw, 0.0)
    cw_recent_obs, _ = torch.cummax(cw_at_obs, dim=axis)
    return _ScanState(
        seg_cw=cw - cw_recent_obs,
        seg_id_barrier=seg_id * seg_barrier,
        obstacle_mask=obstacle_mask,
        seg_barrier=seg_barrier,
    )


def _precompute_axis(
    w: torch.Tensor,
    obstacle_mask: torch.Tensor,
    axis: int,
    seg_barrier: float,
) -> tuple[_ScanState, _ScanState]:
    """Forward + backward state for one axis. Backward state is precomputed on
    the flipped (w, mask) so the per-iter backward sweep just flips `d`."""
    fwd = _precompute_scan(w, obstacle_mask, axis, seg_barrier)
    w_f = torch.flip(w, dims=[axis])
    obstacle_mask_f = torch.flip(obstacle_mask, dims=[axis])
    bwd = _precompute_scan(w_f, obstacle_mask_f, axis, seg_barrier)
    return fwd, bwd


def _converge_or_max(
    d: torch.Tensor,
    body: Callable[[torch.Tensor], torch.Tensor],
    max_iters: int,
    check_every: int,
) -> tuple[torch.Tensor, int]:
    """Iterate `body(d)` until fixed point or `max_iters`, checking every K.

    Reuses `d_check`'s storage via in-place `copy_` instead of cloning each
    check, since `d` itself is reassigned to a fresh tensor every iteration
    (the sweep helpers return new tensors) -- only `d_check` needs persistence.
    """
    d_check = d.clone()
    for it in range(max_iters):
        d = body(d)
        if (it + 1) % check_every == 0:
            if torch.equal(d, d_check):
                return d, it + 1
            d_check.copy_(d)
    return d, max_iters


def _sweep_forward(
    d: torch.Tensor, state: _ScanState, axis: int
) -> torch.Tensor:
    """Forward axis sweep using the precomputed segmented-scan state.

    When every cell in the current segment is unreachable (d=inf), v is inf
    there, so cummin propagates the prior segment's running min forward; the
    reconstruction shifts that value by (S-S')*seg_barrier, producing a
    finite-but-large polluted distance instead of inf. The polluted-mask
    step (d > seg_barrier/2) returns those to inf -- legit distances are
    bounded by the segment's finite path weight, well under seg_barrier/2
    once seg_barrier has been picked by the autotune.
    """
    inf_scalar = float("inf")
    v = d - state.seg_cw - state.seg_id_barrier
    v = torch.where(state.obstacle_mask, inf_scalar, v)
    cm, _ = torch.cummin(v, dim=axis)
    d_new = state.seg_cw + cm + state.seg_id_barrier
    polluted = d_new > state.seg_barrier / 2
    return torch.where(state.obstacle_mask | polluted, inf_scalar, d_new)


def _sweep_backward(
    d: torch.Tensor, state: _ScanState, axis: int
) -> torch.Tensor:
    """Backward axis sweep. `state` must be the *flipped*-direction state
    produced by `_precompute_axis`; the polluted-mask is applied inside
    `_sweep_forward`."""
    d_f = torch.flip(d, dims=[axis])
    return torch.flip(_sweep_forward(d_f, state, axis), dims=[axis])


def sweep_sssp(
    w: torch.Tensor,
    source: tuple[int, int],
    max_iters: int = 200,
    check_every: int = 8,
    seg_barrier: float | None = None,
) -> tuple[torch.Tensor, int]:
    """Compute shortest-path distances on a 2D grid via alternating axis sweeps.

    Convergence is checked every `check_every` iterations rather than every
    iteration -- the per-iter `torch.equal` forces a CPU<->GPU sync that
    serialises the GPU pipeline. Checking every K iters lets K iterations
    run async between syncs.

    Args:
        w: (H, W) tensor, cost to enter each cell. Use float('inf') for obstacles.
        source: (row, col) of the source cell.
        max_iters: cap on outer-loop iterations.
        check_every: how often (in iterations) to test for convergence.
        seg_barrier: optional override for the segmented-scan barrier constant.
            Default None auto-tunes from grid shape and obstacle density.

    Returns:
        (d, iters) where d is the (H, W) distance tensor and iters is the
        number of outer iterations executed.
    """
    d = torch.full_like(w, float("inf"))
    sr, sc = source
    d[sr, sc] = 0.0
    obstacle_mask = _obstacle_mask(w)
    if seg_barrier is None:
        seg_barrier = _autotune_seg_barrier(w, obstacle_mask)
    fwd_h, bwd_h = _precompute_axis(w, obstacle_mask, axis=1, seg_barrier=seg_barrier)
    fwd_v, bwd_v = _precompute_axis(w, obstacle_mask, axis=0, seg_barrier=seg_barrier)

    def step(d: torch.Tensor) -> torch.Tensor:
        d = _sweep_forward(d, fwd_h, axis=1)
        d = _sweep_backward(d, bwd_h, axis=1)
        d = _sweep_forward(d, fwd_v, axis=0)
        return _sweep_backward(d, bwd_v, axis=0)

    return _converge_or_max(d, step, max_iters, check_every)


def sweep_sssp_multi(
    w: torch.Tensor,
    sources: list[tuple[int, int]],
    max_iters: int = 200,
    check_every: int = 8,
    seg_barrier: float | None = None,
) -> tuple[torch.Tensor, int]:
    """Compute K shortest-path distance maps from K sources concurrently.

    Generalises `sweep_sssp` to a batch dim (K). Each source gets its own
    distance map; sources in the same call do NOT see each other's wires
    (no obstacle update between them). The intended use is throughput:
    one multi-sweep replaces K sequential sweeps for batched routing.

    Args:
        w: (H, W) tensor of cell-entry costs.
        sources: list of K (row, col) source pins.
        max_iters, check_every: as in `sweep_sssp`.

    Returns:
        (d, iters) where d is (K, H, W).
    """
    K = len(sources)
    H, W = w.shape
    d = torch.full((K, H, W), float("inf"), device=w.device, dtype=w.dtype)
    for k, (sr, sc) in enumerate(sources):
        d[k, sr, sc] = 0.0
    obstacle_mask = _obstacle_mask(w)
    if seg_barrier is None:
        seg_barrier = _autotune_seg_barrier(w, obstacle_mask)
    w_b = w.unsqueeze(0)
    obstacle_mask_b = obstacle_mask.unsqueeze(0)
    fwd_h, bwd_h = _precompute_axis(w_b, obstacle_mask_b, axis=2, seg_barrier=seg_barrier)
    fwd_v, bwd_v = _precompute_axis(w_b, obstacle_mask_b, axis=1, seg_barrier=seg_barrier)

    def step(d: torch.Tensor) -> torch.Tensor:
        d = _sweep_forward(d, fwd_h, axis=2)
        d = _sweep_backward(d, bwd_h, axis=2)
        d = _sweep_forward(d, fwd_v, axis=1)
        return _sweep_backward(d, bwd_v, axis=1)

    return _converge_or_max(d, step, max_iters, check_every)


def sweep_sssp_3d(
    w: torch.Tensor,
    source: tuple[int, int, int],
    via_cost: float | Sequence[float] | torch.Tensor = 1.0,
    max_iters: int = 200,
    check_every: int = 8,
    seg_barrier: float | None = None,
    w_v: torch.Tensor | None = None,
    extra_sources: Sequence[tuple[int, int, int]] = (),
) -> tuple[torch.Tensor, int]:
    """Compute shortest-path distances on a multi-layer grid via sweep iteration.

    Each layer is 4-connected for in-layer wires; adjacent layers connect at
    the same (r, c) via an edge of weight `via_cost` (a via). Within a layer,
    obstacles are float('inf') in `w` / `w_v`. Vias are unobstructed and have
    constant cost regardless of (r, c) -- a deliberate simplification of real
    ASIC via cells (which can be DRC-blocked).

    Edge model: arrival at (l, r, c) along the column axis (axis=2 sweep,
    "horizontal" moves changing column) pays `w[l, r, c]`; arrival along the
    row axis (axis=1 sweep, "vertical" moves changing row) pays
    `w_v[l, r, c]` if `w_v` is given else `w[l, r, c]`. Arrival via a via
    pays only `via_cost` (the destination cell's wire cost is not charged).

    Per iteration:
        1. Four intra-layer sweeps (axis=2 fwd/bwd, axis=1 fwd/bwd).
           Vectorised over L: every layer is scanned in parallel.
        2. Mask INF_PROXY pollution back to inf.
        3. Sequential per-layer min relaxation along axis=0 (up then down).
           Each step is `d[l] = min(d[l], d[l-1] + via_cost)` followed by an
           obstacle re-mask so via paths neither land on nor chain through
           blocked cells. A naive cumsum-cummin scan along axis=0 would let
           vias "pass through" intermediate obstacles by adding via_cost*|dl|
           regardless of whether those cells exist; the sequential form costs
           2(L-1) min/where ops per iter and is correct under obstacles.
           The via-mask is `mask_h & mask_v` -- a cell is via-blocked only if
           it's blocked in both directional cost tensors, since a via that
           lands on a cell with one finite axis can continue from there.

    Args:
        w: (L, H, W) tensor, cost to enter each cell along axis=2 ("H") moves.
            Also used for axis=1 ("V") moves when `w_v` is not given. inf for
            obstacles.
        source: (layer, row, col).
        via_cost: edge weight for via transitions. Either a scalar (uniform
            cost on every via pair) or a length-(L-1) sequence/tensor giving
            per-pair costs, where `via_cost[k]` is the edge weight between
            layer `k` and layer `k+1` (covers both directions). Used to model
            DRC/resistance differences between Via1, Via2, ... on real PDKs.
        max_iters, check_every: as in `sweep_sssp`.
        seg_barrier: optional override; otherwise autotuned from both axes.
        w_v: optional (L, H, W) cost tensor for axis=1 ("V") moves. If None,
            `w` is used for both axes (isotropic). Use `axis_costs` to build
            (w, w_v) from a base tensor and per-layer preferred-direction
            multipliers.
        extra_sources: additional cells to seed at distance 0, beyond the
            primary `source`. Used by the multi-pin router to seed the
            already-committed tree as a source set: a sweep from a tree of
            many cells computes "distance to the nearest tree cell" at every
            other point, which is the correct quantity for picking the next
            attachment edge in an incremental maze route.

    Returns:
        (d, iters) where d is the (L, H, W) distance tensor.
    """
    L = w.shape[0]
    via_costs = _normalize_via_cost(via_cost, L, w.device, w.dtype)
    # Materialise as Python floats once; the hot-path via-relax in `step`
    # is a tensor + scalar broadcast (fast) when `vc_f[k]` is a float, but
    # a tensor + 0-d-tensor broadcast (slow on MPS, an extra kernel launch
    # per layer per direction per iteration) if we index `via_costs[k]`
    # directly. See docs/spikes/tier-b-envelope-throughput.md.
    vc_f: list[float] = via_costs.tolist()
    d = torch.full_like(w, float("inf"))
    sl, sr, sc = source
    d[sl, sr, sc] = 0.0
    for el, er, ec in extra_sources:
        d[el, er, ec] = 0.0
    mask_h = _obstacle_mask(w)
    if w_v is not None:
        mask_v = _obstacle_mask(w_v)
        mask_via = mask_h & mask_v
    else:
        mask_v = mask_h
        mask_via = mask_h
    inf_scalar = float("inf")
    if seg_barrier is None:
        seg_barrier = _autotune_seg_barrier(
            w,
            mask_h,
            via_cost=via_costs,
            w_v=w_v,
            obstacle_mask_v=mask_v if w_v is not None else None,
        )
    fwd_h, bwd_h = _precompute_axis(w, mask_h, axis=2, seg_barrier=seg_barrier)
    w_for_v = w_v if w_v is not None else w
    fwd_v, bwd_v = _precompute_axis(w_for_v, mask_v, axis=1, seg_barrier=seg_barrier)

    def step(d: torch.Tensor) -> torch.Tensor:
        d = _sweep_forward(d, fwd_h, axis=2)
        d = _sweep_backward(d, bwd_h, axis=2)
        d = _sweep_forward(d, fwd_v, axis=1)
        d = _sweep_backward(d, bwd_v, axis=1)
        for lyr in range(1, L):
            d[lyr] = torch.minimum(d[lyr], d[lyr - 1] + vc_f[lyr - 1])
            d[lyr] = torch.where(mask_via[lyr], inf_scalar, d[lyr])
        for lyr in range(L - 2, -1, -1):
            d[lyr] = torch.minimum(d[lyr], d[lyr + 1] + vc_f[lyr])
            d[lyr] = torch.where(mask_via[lyr], inf_scalar, d[lyr])
        return d

    return _converge_or_max(d, step, max_iters, check_every)


def axis_costs(
    w: torch.Tensor,
    h_mult: Sequence[float],
    v_mult: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-axis cost tensors from a base cost and per-layer multipliers.

    For preferred-direction routing on a PDK like gf180mcuD (M1=H, M2=V,
    M3=H, ...), pass `h_mult[l] = 1.0` and `v_mult[l] = off_mult` on
    horizontal-preferred layers, and the reverse on vertical-preferred
    layers. Obstacles (inf cells) stay inf in both tensors. The resulting
    (w_h, w_v) plug directly into `sweep_sssp_3d`'s (w=w_h, w_v=w_v) inputs.

    Args:
        w: (L, H, W) base cost tensor.
        h_mult: length-L sequence; per-layer axis=2 ("H") cost multiplier.
        v_mult: length-L sequence; per-layer axis=1 ("V") cost multiplier.

    Returns:
        (w_h, w_v) tensors of shape (L, H, W), same device and dtype as w.
    """
    L = w.shape[0]
    if len(h_mult) != L or len(v_mult) != L:
        raise ValueError(
            f"h_mult and v_mult must have length L={L}, "
            f"got {len(h_mult)} and {len(v_mult)}"
        )
    h_t = torch.tensor(h_mult, device=w.device, dtype=w.dtype).view(L, 1, 1)
    v_t = torch.tensor(v_mult, device=w.device, dtype=w.dtype).view(L, 1, 1)
    obstacle = _obstacle_mask(w)
    w_h = torch.where(obstacle, w, w * h_t)
    w_v = torch.where(obstacle, w, w * v_t)
    return w_h, w_v


def sweep_sssp_3d_multi(
    w: torch.Tensor,
    sources: list[tuple[int, int, int]],
    via_cost: float | Sequence[float] | torch.Tensor = 1.0,
    max_iters: int = 200,
    check_every: int = 8,
    seg_barrier: float | None = None,
    w_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Compute K shortest-path distance maps on a multi-layer grid concurrently.

    Generalises `sweep_sssp_3d` to a leading batch dim K, the way
    `sweep_sssp_multi` generalises `sweep_sssp`. Sources are independent;
    each one gets its own (L, H, W) distance map. The point is GPU
    throughput: one fused kernel call replaces K sequential calls.
    Intended use is per-tile batched routing once Phase 3.3's tile
    decomposition lands -- K independent 2-pin routes within the same
    tile can be solved in one sweep.

    Args:
        w: (L, H, W) cost tensor for axis=2 ("H") moves.
        sources: list of K (layer, row, col) source coords.
        via_cost: scalar or length-(L-1) per-pair via costs; see
            `sweep_sssp_3d` for the indexing convention.
        max_iters, check_every, seg_barrier, w_v: as in `sweep_sssp_3d`.

    Returns:
        (d, iters) where d is (K, L, H, W).
    """
    K = len(sources)
    L, H, W = w.shape
    via_costs = _normalize_via_cost(via_cost, L, w.device, w.dtype)
    # Materialise as Python floats once; see `sweep_sssp_3d` comment +
    # docs/spikes/tier-b-envelope-throughput.md for the perf rationale.
    vc_f: list[float] = via_costs.tolist()
    d = torch.full((K, L, H, W), float("inf"), device=w.device, dtype=w.dtype)
    for k, (sl, sr, sc) in enumerate(sources):
        d[k, sl, sr, sc] = 0.0
    mask_h = _obstacle_mask(w)
    if w_v is not None:
        mask_v = _obstacle_mask(w_v)
        mask_via = mask_h & mask_v
    else:
        mask_v = mask_h
        mask_via = mask_h
    inf_scalar = float("inf")
    if seg_barrier is None:
        seg_barrier = _autotune_seg_barrier(
            w,
            mask_h,
            via_cost=via_costs,
            w_v=w_v,
            obstacle_mask_v=mask_v if w_v is not None else None,
        )
    w_b = w.unsqueeze(0)
    mask_h_b = mask_h.unsqueeze(0)
    mask_v_b = mask_v.unsqueeze(0)
    mask_via_b = mask_via.unsqueeze(0)
    fwd_h, bwd_h = _precompute_axis(w_b, mask_h_b, axis=3, seg_barrier=seg_barrier)
    w_for_v = (w_v if w_v is not None else w).unsqueeze(0)
    fwd_v, bwd_v = _precompute_axis(w_for_v, mask_v_b, axis=2, seg_barrier=seg_barrier)

    def step(d: torch.Tensor) -> torch.Tensor:
        d = _sweep_forward(d, fwd_h, axis=3)
        d = _sweep_backward(d, bwd_h, axis=3)
        d = _sweep_forward(d, fwd_v, axis=2)
        d = _sweep_backward(d, bwd_v, axis=2)
        for lyr in range(1, L):
            d[:, lyr] = torch.minimum(d[:, lyr], d[:, lyr - 1] + vc_f[lyr - 1])
            d[:, lyr] = torch.where(mask_via_b[:, lyr], inf_scalar, d[:, lyr])
        for lyr in range(L - 2, -1, -1):
            d[:, lyr] = torch.minimum(d[:, lyr], d[:, lyr + 1] + vc_f[lyr])
            d[:, lyr] = torch.where(mask_via_b[:, lyr], inf_scalar, d[:, lyr])
        return d

    return _converge_or_max(d, step, max_iters, check_every)


def backtrace(
    d: torch.Tensor,
    w: torch.Tensor,
    source: tuple[int, int],
    sink: tuple[int, int],
    atol: float = 1e-5,
) -> list[tuple[int, int]] | None:
    """Reconstruct a shortest path from source to sink given the distance map.

    Walks backward from sink: at each step, pick a 4-neighbor n with
    d[n] + w[current] ~= d[current].
    """
    sr, sc = source
    si, sj = sink
    H, W = d.shape

    if not torch.isfinite(d[si, sj]):
        return None

    path: list[tuple[int, int]] = [(si, sj)]
    cur_i, cur_j = si, sj

    while (cur_i, cur_j) != (sr, sc):
        target = (d[cur_i, cur_j] - w[cur_i, cur_j]).item()
        moved = False
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = cur_i + di, cur_j + dj
            if 0 <= ni < H and 0 <= nj < W and torch.isfinite(d[ni, nj]):
                if abs(d[ni, nj].item() - target) <= atol:
                    path.append((ni, nj))
                    cur_i, cur_j = ni, nj
                    moved = True
                    break
        if not moved:
            return None

    path.reverse()
    return path


def backtrace_3d(
    d: torch.Tensor,
    w: torch.Tensor,
    source: tuple[int, int, int],
    sink: tuple[int, int, int],
    via_cost: float | Sequence[float] | torch.Tensor = 1.0,
    atol: float = 1e-5,
    w_v: torch.Tensor | None = None,
    extra_sources: Sequence[tuple[int, int, int]] = (),
) -> list[tuple[int, int, int]] | None:
    """Reconstruct a shortest 3D path from source to sink.

    At each step, prefer in-layer 4-neighbors. The predecessor-distance check
    is axis-aware: a column-changing neighbor must satisfy `d[neighbor] ==
    d[cur] - w[cur]` (axis=2 entry cost), and a row-changing neighbor must
    satisfy `d[neighbor] == d[cur] - w_v[cur]` (axis=1 entry cost). If `w_v`
    is None it falls back to `w` for both directions (isotropic).

    Falls back to cross-layer via neighbors at the same (r, c) on the layer
    above or below. Per-pair via_cost: a step from cur_l to neighbor cur_l-1
    uses `via_cost[cur_l - 1]`; a step to cur_l+1 uses `via_cost[cur_l]`.

    With `extra_sources` non-empty, the walk terminates upon reaching `source`
    OR any cell in `extra_sources` -- the first complete predecessor chain
    back to any seed cell wins. Used by the multi-pin router to attach a new
    pin to the nearest cell of the existing committed tree, rather than
    walking all the way back to a designated root.
    """
    if w_v is None:
        w_v = w
    sl, sr, sc = source
    tl, ti, tj = sink
    L, H, W = d.shape
    if isinstance(via_cost, torch.Tensor):
        via_costs_list = [float(v) for v in via_cost.tolist()]
    elif isinstance(via_cost, (int, float)):
        via_costs_list = [float(via_cost)] * (L - 1)
    else:
        via_costs_list = [float(v) for v in via_cost]
    if len(via_costs_list) != L - 1:
        raise ValueError(
            f"via_cost must be scalar or length ({L - 1},), got {len(via_costs_list)}"
        )
    terminal_cells: set[tuple[int, int, int]] = {(sl, sr, sc), *extra_sources}

    if not torch.isfinite(d[tl, ti, tj]):
        return None

    path: list[tuple[int, int, int]] = [(tl, ti, tj)]
    cur_l, cur_i, cur_j = tl, ti, tj

    while (cur_l, cur_i, cur_j) not in terminal_cells:
        h_target = (d[cur_l, cur_i, cur_j] - w[cur_l, cur_i, cur_j]).item()
        v_target = (d[cur_l, cur_i, cur_j] - w_v[cur_l, cur_i, cur_j]).item()
        cur_d = d[cur_l, cur_i, cur_j].item()
        moved = False
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = cur_i + di, cur_j + dj
            if 0 <= ni < H and 0 <= nj < W and torch.isfinite(d[cur_l, ni, nj]):
                target = v_target if di != 0 else h_target
                if abs(d[cur_l, ni, nj].item() - target) <= atol:
                    path.append((cur_l, ni, nj))
                    cur_i, cur_j = ni, nj
                    moved = True
                    break
        if moved:
            continue
        for dl in (-1, 1):
            nl = cur_l + dl
            if 0 <= nl < L and torch.isfinite(d[nl, cur_i, cur_j]):
                # via index between layers min(cur_l, nl) and that+1
                vc = via_costs_list[cur_l - 1] if dl == -1 else via_costs_list[cur_l]
                if abs(d[nl, cur_i, cur_j].item() - (cur_d - vc)) <= atol:
                    path.append((nl, cur_i, cur_j))
                    cur_l = nl
                    moved = True
                    break
        if not moved:
            return None

    path.reverse()
    return path
