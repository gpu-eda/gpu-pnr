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

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import torch

from gpu_pnr.guides import (
    GuideRect,
    GuideRegion,
    clamp_region_bounds,
    guide_region,
)
from gpu_pnr.router import MultiPin3DResult
from gpu_pnr.sweep import backtrace_3d, sweep_sssp_3d_batched

# Per-net sub-grid preparation hook: `(w_sub, local_pins) -> None`, applied
# in-place to a *clone* of each net's sub-grid before routing. Lets a caller
# inject PDK structural rules (e.g. `apply_pin_access_rules`) without coupling
# the router to any PDK. `local_pins` are in sub-grid-local coords.
PrepSubgrid = Callable[["torch.Tensor", list[tuple[int, int, int]]], None]


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


def _cell_index_tensors(
    cells: Sequence[tuple[int, int, int]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a list of `(l, r, c)` cells into per-axis long tensors for one
    batched `w[ls, rs, cs] = ...` index-assignment. Per-cell scalar writes are
    a kernel launch + sync each on MPS — death by O(cells) launches — so both
    the commit and the rip-up restore route through this single batched form.
    """
    return tuple(  # type: ignore[return-value]
        torch.tensor(axis, device=device, dtype=torch.long)
        for axis in zip(*cells)
    )


@dataclass(eq=False)
class _NetWork:
    """Mutable per-net routing state advanced one attachment per round.

    Internal to `GuideRouter.route`'s round-batched loop (Slice 3). Mirrors the
    sequential tree-growth locals in `route_multipin_nets_3d`, transposed so the
    outer axis is the attachment round and the inner axis is the net: `tree`,
    `unrouted`, and `paths` are in sub-grid-local coords; a net is still growing
    while `unrouted` is non-empty and `failed` is False.

    Slice 4 adds `committed_cells` (chip-global), the footprint this net last
    committed — the cell set rip-up restores from `w_chip` when the net loses a
    conflict — `global_paths`, the committed paths in chip-global coords (built
    once at commit, reused for the result), and `ripups`, the count of times it
    has been ripped up (capped).
    """

    plan: NetPlan
    region: GuideRegion
    local_pins: list[tuple[int, int, int]]
    tree: set[tuple[int, int, int]]
    unrouted: set[tuple[int, int, int]]
    paths: list[list[tuple[int, int, int]]]
    failed: bool
    committed_cells: set[tuple[int, int, int]]
    global_paths: list[list[tuple[int, int, int]]]
    ripups: int


class GuideRouter:
    """Chip-scale guide-constrained router (ADR 0012 as amended).

    Slices 1–2: classification + single-stream routing. Batched routing,
    conflict/ripup, and the coarsened tail land in Slices 3–6 per
    `docs/plans/ws33-tile-router-implementation.md`.

    `chip_shape` is taken from `w_chip` when a tensor is given, else from the
    explicit `chip_shape` argument (so the classifier is testable without a
    cost tensor). `prep_subgrid`, if given, is applied in-place to a clone of
    each net's sub-grid before routing — the PDK-injection hook (see
    `PrepSubgrid`).
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
        prep_subgrid: PrepSubgrid | None = None,
        bucket_size: int | None = None,
    ) -> None:
        if pitch_dbu <= 0:
            raise ValueError(f"pitch_dbu must be positive; got {pitch_dbu}")
        if axis_cap <= 0:
            raise ValueError(f"axis_cap must be positive; got {axis_cap}")
        if bucket_size is not None and bucket_size <= 0:
            raise ValueError(f"bucket_size must be positive; got {bucket_size}")
        self.w_chip = w_chip
        self.w_v_chip = w_v_chip
        self.chip_origin = chip_origin
        self.layer_order = layer_order
        self.pitch_dbu = pitch_dbu
        self.margin = margin
        self.axis_cap = axis_cap
        self.prep_subgrid = prep_subgrid
        self.bucket_size = bucket_size
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

    def _attach_batch(
        self,
        batch: list[_NetWork],
        w_cur: torch.Tensor,
        w_v_cur: torch.Tensor | None,
        committed: torch.Tensor | None,
        via_cost: float,
    ) -> None:
        """One batched sweep + backtrace over `batch`, growing each net's tree
        by one attachment. Slices each net's sub-grid from the shared `w_cur`
        snapshot, applies `prep_subgrid` + the committed re-block per net, pads
        to the batch's common `(L,H,W)`, runs one `sweep_sssp_3d_batched`, and
        backtraces each slice against its own sub-grid. Mutates each `_NetWork`
        in place (grows `tree`/`paths`, shrinks `unrouted`, or sets `failed`).

        Callers bucket `batch` by sub-grid size: padding every net to the
        batch's largest sub-grid wastes ~31× cells when sizes are mixed, which
        made one-giant-batch routing 70× slower than size-bucketed at the sweep
        level (`docs/spikes/size-bucketed-batching.md`).
        """
        inf = float("inf")
        device = w_cur.device
        # Slice + prep + committed-re-block PER NET, then pad+stack. prep_subgrid
        # is per-sub-grid (it cannot run on the stacked tensor), and the committed
        # re-block must carry into the batched path: prep rewrites landing-pad
        # cells to finite, resurrecting prior nets' committed wires unless we
        # re-block after prep (watch-outs 1 & 2).
        subgrids_h: list[torch.Tensor] = []
        subgrids_v: list[torch.Tensor] = []
        for nw in batch:
            reg = nw.region
            rl = (
                slice(reg.l0, reg.l1),
                slice(reg.r0, reg.r1),
                slice(reg.c0, reg.c1),
            )
            # Clone: prep mutates in place and the slice is a view into the
            # shared w_cur — prepping the view would corrupt the snapshot.
            w_sub = w_cur[rl].clone()
            w_v_sub = w_v_cur[rl].clone() if w_v_cur is not None else None
            if self.prep_subgrid is not None:
                assert committed is not None
                sub_committed = committed[rl]
                self.prep_subgrid(w_sub, nw.local_pins)
                w_sub[sub_committed] = inf  # prep must not resurrect wires
                if w_v_sub is not None:
                    self.prep_subgrid(w_v_sub, nw.local_pins)
                    w_v_sub[sub_committed] = inf
            subgrids_h.append(w_sub)
            if w_v_sub is not None:
                subgrids_v.append(w_v_sub)

        lmax = max(g.shape[0] for g in subgrids_h)
        hmax = max(g.shape[1] for g in subgrids_h)
        wmax = max(g.shape[2] for g in subgrids_h)
        K = len(batch)
        w_batch = torch.full(
            (K, lmax, hmax, wmax), inf, device=device, dtype=w_cur.dtype
        )
        for k, g in enumerate(subgrids_h):
            gl, gh, gw = g.shape
            w_batch[k, :gl, :gh, :gw] = g
        w_v_batch: torch.Tensor | None = None
        if subgrids_v:
            w_v_batch = torch.full(
                (K, lmax, hmax, wmax), inf, device=device, dtype=w_cur.dtype
            )
            for k, g in enumerate(subgrids_v):
                gl, gh, gw = g.shape
                w_v_batch[k, :gl, :gh, :gw] = g

        # Each net's primary source is its seed pin (tree[0] == local_pins[0]);
        # its extra_sources are the rest of its current tree. A multi-source
        # sweep gives "distance to nearest tree cell", the attachment quantity.
        sources = [nw.local_pins[0] for nw in batch]
        extra_sources = [
            [c for c in nw.tree if c != nw.local_pins[0]] for nw in batch
        ]
        d_batch, _ = sweep_sssp_3d_batched(
            w_batch, sources, via_cost=via_cost, w_v=w_v_batch,
            extra_sources=extra_sources,
        )
        d_cpu = d_batch.cpu()

        # Per-net backtrace against its OWN sub-grid (CPU-side).
        # NOTE (post-bucketing bottleneck): with size-bucketing removing the
        # padding waste, this serial CPU backtrace — per-net `.cpu()` + the
        # per-pin `.item()` argmin — is the *next* dominant cost. Within a bucket
        # the slices are uniform-shaped, so the best-pin search vectorises to one
        # gather + `torch.min` per bucket; pushing backtrace onto the GPU is the
        # ADR-0012 lever beyond that. See ADR 0013 + docs/spikes/size-bucketed-batching.md.
        for k, nw in enumerate(batch):
            gl, gh, gw = subgrids_h[k].shape
            d_k = d_cpu[k, :gl, :gh, :gw]
            w_sub_cpu = subgrids_h[k].cpu()
            w_v_sub_cpu = subgrids_v[k].cpu() if subgrids_v else None
            seed = nw.local_pins[0]
            extras = extra_sources[k]  # same tree-minus-seed used for the sweep
            best_pin: tuple[int, int, int] | None = None
            best_dist = inf
            for p in nw.unrouted:
                dp = float(d_k[p].item())
                if dp < best_dist:
                    best_dist = dp
                    best_pin = p
            if best_pin is None or best_dist == inf:
                nw.failed = True
                continue
            path = backtrace_3d(
                d_k, w_sub_cpu, seed, best_pin, via_cost=via_cost,
                w_v=w_v_sub_cpu, extra_sources=extras,
            )
            if path is None:
                nw.failed = True
                continue
            nw.paths.append(path)
            nw.tree.update(path)
            nw.unrouted.discard(best_pin)

    # Max rip-up rounds before a still-conflicting net is left failed (Slice 4,
    # plan §"Slice 4"). Walk-back: raise to 5 if >1% of nets fail to converge.
    MAX_RIPUPS = 3

    def _seed_work(self, plan: NetPlan) -> _NetWork | None:
        """Build a fresh unrouted `_NetWork` for `plan`, or None if off-region.

        Used both for the initial population and to reset a net on rip-up (tree
        back to the seed pin, paths cleared, committed footprint emptied).
        """
        reg = plan.region
        if not all(reg.contains(p) for p in plan.pins):
            return None  # off-region: caller leaves the unrouted sentinel
        local = [reg.rebase(p) for p in plan.pins]
        return _NetWork(
            plan=plan,
            region=reg,
            local_pins=local,
            tree={local[0]},
            unrouted=set(local[1:]),
            paths=[[local[0]]],
            failed=False,
            committed_cells=set(),
            global_paths=[],
            ripups=0,
        )

    def _route_population(
        self,
        work: Iterable[_NetWork],
        w_cur: torch.Tensor,
        w_v_cur: torch.Tensor | None,
        committed: torch.Tensor | None,
        via_cost: float,
    ) -> None:
        """Drain `work` to completion: round-batched attachment + per-round
        commit. Each fully-routed net's footprint is committed to `w_cur` (inf
        obstacle for later rounds) and recorded in `nw.committed_cells` (Slice 4
        rip-up restores from that set). Mutates `work` items + the grids in
        place; off-/failed nets stay unrouted.
        """
        inf = float("inf")
        device = w_cur.device
        # Round-batched attachment loop. A net is "still growing" while it has
        # unrouted pins and hasn't failed. The same w_cur snapshot bounds every
        # net in a round; commits land in the per-round commit step below.
        while True:
            active = [nw for nw in work if nw.unrouted and not nw.failed]
            if not active:
                break

            # Size-bucket within the round so small nets aren't padded up to the
            # round's largest sub-grid. Mixed sizes in one batch waste ~31× cells
            # and made one-giant-batch routing 70× slower than bucketed at the
            # sweep level (docs/spikes/size-bucketed-batching.md). bucket_size=None
            # keeps the single-batch behaviour. Every bucket shares this round's
            # w_cur snapshot (no commit between buckets) — same-snapshot semantics
            # preserved; commit lands after all buckets, below.
            if self.bucket_size is None:
                batches: list[list[_NetWork]] = [active]
            else:
                ordered = sorted(active, key=lambda nw: nw.region.cell_count)
                bs = self.bucket_size
                batches = [
                    ordered[i:i + bs] for i in range(0, len(ordered), bs)
                ]
            for batch in batches:
                self._attach_batch(batch, w_cur, w_v_cur, committed, via_cost)

            # --- Commit fully-routed nets between rounds, HPWL-ascending. ---
            # A net is done when it has no unrouted pins left. Committing its
            # cells to inf makes them obstacles for the *next* round's snapshot
            # (so later-round nets detour); within-round nets shared this round's
            # snapshot and may collide — the rip-up loop in `route` resolves that.
            # `active` is already HPWL-ascending (in_cap order is preserved).
            for nw in active:
                if nw.failed or nw.unrouted:
                    continue
                reg = nw.region
                global_paths = [
                    [(lyr + reg.l0, r + reg.r0, c + reg.c0) for (lyr, r, c) in path]
                    for path in nw.paths
                ]
                cells = [c for path in global_paths for c in path]
                ls, rs, cs = _cell_index_tensors(cells, device)
                w_cur[ls, rs, cs] = inf
                if w_v_cur is not None:
                    w_v_cur[ls, rs, cs] = inf
                if committed is not None:
                    committed[ls, rs, cs] = True
                nw.committed_cells = set(cells)
                nw.global_paths = global_paths  # reused at result assembly

    def _ripup_net(
        self,
        nw: _NetWork,
        keep: set[tuple[int, int, int]],
        w_cur: torch.Tensor,
        w_v_cur: torch.Tensor | None,
        committed: torch.Tensor | None,
    ) -> None:
        """Un-commit `nw`: restore its loser-exclusive cells from the original
        chip grid and clear their committed bits, so a reroute sees true chip
        cost (PDK pin-access included), not a stale inf or a clear-to-finite
        shortcut that would resurrect the conflict (handoff watch-out).

        `keep` is the set of cells still claimed by a surviving net (the
        conflict winner): those stay committed to the winner and are NOT
        restored. Only this net's exclusive cells revert.
        """
        assert self.w_chip is not None
        device = w_cur.device
        restore = [c for c in nw.committed_cells if c not in keep]
        if restore:
            # Full restore from the original chip grid (NOT a clear-to-finite):
            # keeps PDK pin-access values prep_subgrid relies on.
            ls, rs, cs = _cell_index_tensors(restore, device)
            w_cur[ls, rs, cs] = self.w_chip[ls, rs, cs]
            if w_v_cur is not None and self.w_v_chip is not None:
                w_v_cur[ls, rs, cs] = self.w_v_chip[ls, rs, cs]
            if committed is not None:
                committed[ls, rs, cs] = False
        nw.committed_cells = set()
        nw.global_paths = []

    def route(
        self,
        nets: list[Net3D],
        guides: Sequence[Sequence[GuideRect]],
        *,
        via_cost: float = 1.0,
    ) -> list[MultiPin3DResult]:
        """Route in-cap nets via round-batched guide-constrained sweeps with
        cross-net conflict rip-up / reroute (Slices 3–4).

        In-cap nets route together on the shared chip cost grid, batched by
        *attachment round* (`docs/spikes/multi-pin-batching-strategy.md`,
        option b). Each round, every net still missing a pin has its sub-grid
        sliced from the *current* shared `w_cur`, prepped + committed-re-blocked
        per net, padded to the batch's common `(L, H, W)` shape (pad = `inf`),
        and seeded with its current tree as `extra_sources`; a batched
        `sweep_sssp_3d_batched` distances them; each slice is backtraced
        against its own sub-grid to attach its nearest unrouted pin, growing
        that net's tree. Rounds repeat until no net is still growing.

        When `bucket_size` is set, each round's nets are size-sorted and
        chunked into `bucket_size`-net buckets, one batched sweep per bucket —
        so a small net isn't padded up to the round's largest sub-grid. This is
        a pure throughput optimisation (routes are byte-identical to the single
        giant batch, `bucket_size=None`) worth ~22–33× on Hazard3
        (`docs/spikes/size-bucketed-batching.md`, ADR 0013 Amendment 1).

        Nets in one round route against the **same `w_cur` snapshot**, so two
        can claim the same cell. After the population drains, **Slice 4**
        detects cells claimed by ≥2 committed nets; the lowest-HPWL net keeps
        each contested cell (ADR 0007) and the losers are ripped up — their
        loser-exclusive cells restored from the original chip grid — and
        rerouted against the updated `w_cur` in a further pass. Bounded to
        `MAX_RIPUPS` (3) passes; a net still conflicting after the cap is left
        failed (unrouted sentinel). This is the [ADR 0008] deferred-net unlock
        on guide sub-grids. Tail nets (over-cap / no-guide) and nets whose
        region doesn't contain all their pins are returned unrouted; the
        coarsened-pass fallback for the tail lands in Slice 5.

        Results are returned in input order. Paths and pins are in chip-global
        `(layer, row, col)` coordinates. API mirrors `route_multipin_nets_3d`
        plus the per-net `guides` (parallel to `nets`).
        """
        if self.w_chip is None:
            raise ValueError("route requires a w_chip cost tensor")

        # tail nets (over-cap / no-guide) keep the unrouted sentinel below.
        in_cap, _ = self.classify(nets, guides)
        w_cur = self.w_chip.clone()
        w_v_cur = self.w_v_chip.clone() if self.w_v_chip is not None else None
        # Tracks cells committed by earlier nets. Needed only with prep_subgrid:
        # a PDK prep (e.g. pin-access) rewrites landing-pad cells to finite,
        # which would *resurrect* a prior net's committed wire and let two nets
        # share a cell. We re-block committed cells after prep to prevent that.
        # Rip-up (`_ripup_net`) clears the bits it restores, or a rerouted net
        # could be wrongly re-blocked.
        committed = (
            torch.zeros_like(w_cur, dtype=torch.bool)
            if self.prep_subgrid is not None else None
        )

        # Build the per-net work-items for the still-growing population, keyed
        # by input index. Each carries its region + a CPU-side mutable
        # tree/unrouted/paths state that round-batching advances one attachment
        # per round (transpose of the sequential tree-growth loop in
        # route_multipin_nets_3d). Dict insertion order is in_cap order
        # (HPWL-ascending), preserved through the conflict winner-selection
        # below; reassigning a key on rip-up keeps its position.
        net_to_work: dict[int, _NetWork] = {}
        for plan in in_cap:  # HPWL-ascending
            nw = self._seed_work(plan)
            if nw is None:
                continue  # off-region: leave the unrouted sentinel
            net_to_work[plan.index] = nw

        # Rip-up / reroute loop. Route the population, detect cross-net
        # conflicts in the committed footprints, keep the lowest-HPWL claimant
        # of each contested cell, rip up + requeue the losers, and reroute.
        # Bounded to MAX_RIPUPS passes (else the loser is left failed).
        self._route_population(net_to_work.values(), w_cur, w_v_cur, committed, via_cost)
        for _ in range(self.MAX_RIPUPS):
            # A budget-exhausted loser can't reroute again; drop it from the
            # requeue set so it's left to fail rather than re-detected forever.
            losers = {
                nw for nw in self._collect_losers(net_to_work.values())
                if nw.ripups < self.MAX_RIPUPS
            }
            if not losers:
                break
            # Winners keep their cells; restore only loser-exclusive cells.
            keep: set[tuple[int, int, int]] = set()
            for nw in net_to_work.values():
                if nw not in losers and not nw.failed:
                    keep |= nw.committed_cells
            for plan_index, nw in list(net_to_work.items()):
                if nw not in losers:
                    continue
                self._ripup_net(nw, keep, w_cur, w_v_cur, committed)
                # Reset to a fresh unrouted state (tree back to seed, paths
                # cleared) and requeue in place. off-region is impossible here
                # (it routed once) but guard anyway.
                fresh = self._seed_work(nw.plan)
                if fresh is None:
                    del net_to_work[plan_index]
                    continue
                fresh.ripups = nw.ripups + 1
                net_to_work[plan_index] = fresh  # reassign keeps dict position
            self._route_population(net_to_work.values(), w_cur, w_v_cur, committed, via_cost)

        # Any net that still conflicts after the rip-up budget is left failed
        # (unrouted sentinel). Detect a final time including budget-exhausted
        # losers so they're caught and dropped here.
        final_losers = self._collect_losers(net_to_work.values())

        # Assemble results in input order: a routed net is one that committed a
        # footprint and is not a residual conflict loser; everything else keeps
        # the unrouted sentinel (tail, off-region, route-fail, over-cap loser).
        # `global_paths` was built once at commit (`_route_population`).
        results = [MultiPin3DResult(list(pins), None) for pins in nets]
        for plan_index, nw in net_to_work.items():
            if nw.failed or nw.unrouted or not nw.committed_cells:
                continue
            if nw in final_losers:
                continue  # still conflicting after the cap → leave failed
            results[plan_index] = MultiPin3DResult(
                list(nw.plan.pins), nw.global_paths
            )
        return results

    def _collect_losers(self, work: Iterable[_NetWork]) -> set[_NetWork]:
        """Return every conflict-loser `_NetWork` in `work`.

        A cell claimed by ≥2 committed nets is a conflict; the lowest-HPWL
        claimant (ADR 0007) keeps it, the rest are losers. A net is a loser if
        it loses ANY of its committed cells. Callers filter budget-exhausted
        losers themselves (the requeue pass drops them so they aren't
        re-detected forever; the final-assembly pass keeps them so they fall to
        the unrouted sentinel).
        """
        # cell -> list of (hpwl, NetWork) committed claimants.
        claims: dict[tuple[int, int, int], list[tuple[int, _NetWork]]] = {}
        for nw in work:
            if nw.failed or nw.unrouted or not nw.committed_cells:
                continue
            h = net_hpwl(nw.plan.pins)
            for cell in nw.committed_cells:
                claims.setdefault(cell, []).append((h, nw))
        losers: set[_NetWork] = set()
        for cell, claimants in claims.items():
            if len(claimants) < 2:
                continue
            # Lowest HPWL wins; ties broken by input index (stable, ADR 0007).
            winner = min(claimants, key=lambda hc: (hc[0], hc[1].plan.index))[1]
            for _, nw in claimants:
                if nw is not winner:
                    losers.add(nw)
        return losers
