"""Sequential multi-net routing on 2D and 3D (multi-layer) grids.

Reserves all pin cells (sources + sinks of every net) as obstacles before
routing starts; temporarily un-reserves a net's own pins while it routes.
This stops earlier nets from running their wires through later nets'
pins, which would otherwise force the later nets to fail.

A net's path cells (including its endpoints, which become wires once
routed) are committed to a separate `routed_cells` set. Once a cell is
in that set we never un-reserve it -- two nets touching the same wire
would be an electrical short.

The 3D variant treats (layer, row, col) as the cell coordinate; pins on
different layers at the same (row, col) are distinct cells.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gpu_pnr.sweep import backtrace, backtrace_3d, sweep_sssp, sweep_sssp_3d


@dataclass
class NetResult:
    source: tuple[int, int]
    sink: tuple[int, int]
    path: list[tuple[int, int]] | None

    @property
    def routed(self) -> bool:
        return self.path is not None

    @property
    def length(self) -> int:
        return len(self.path) - 1 if self.path else 0


@dataclass
class Net3DResult:
    source: tuple[int, int, int]
    sink: tuple[int, int, int]
    path: list[tuple[int, int, int]] | None

    @property
    def routed(self) -> bool:
        return self.path is not None

    @property
    def length(self) -> int:
        return len(self.path) - 1 if self.path else 0


def _is_finite(w: torch.Tensor, ij: tuple[int, ...]) -> bool:
    return bool(torch.isfinite(w[ij]).item())


def route_nets(
    w: torch.Tensor,
    nets: list[tuple[tuple[int, int], tuple[int, int]]],
    reserve_pins: bool = True,
) -> list[NetResult]:
    """Route nets sequentially on a working copy of `w`.

    Args:
        w: (H, W) tensor of cell-entry costs. float('inf') for obstacles.
        nets: ordered list of (source, sink) pin pairs.
        reserve_pins: if True (default), all pin cells are reserved as
            obstacles before routing begins; each net's own pins are
            temporarily un-reserved while it routes. Set False to get
            the naive baseline that ignores pin protection.

    Returns:
        List of NetResult in the same order as `nets`. A net with `path=None`
        either had an originally-blocked endpoint, an endpoint already
        committed to a prior net's wire, or no feasible route.
    """
    inf_val = torch.tensor(float("inf"), device=w.device, dtype=w.dtype)
    w_cur = w.clone()
    routed_cells: set[tuple[int, int]] = set()

    if reserve_pins:
        pin_cells: set[tuple[int, int]] = set()
        for s, t in nets:
            pin_cells.add(s)
            pin_cells.add(t)
        for ij in pin_cells:
            if _is_finite(w_cur, ij):
                w_cur[ij] = inf_val

    results: list[NetResult] = []
    for source, sink in nets:
        if source in routed_cells or sink in routed_cells:
            results.append(NetResult(source, sink, None))
            continue
        if not _is_finite(w, source) or not _is_finite(w, sink):
            results.append(NetResult(source, sink, None))
            continue

        if reserve_pins:
            w_cur[source] = w[source]
            w_cur[sink] = w[sink]

        d, _ = sweep_sssp(w_cur, source)
        path = backtrace(d.cpu(), w_cur.cpu(), source, sink)

        if path is not None:
            for ij in path:
                w_cur[ij] = inf_val
                routed_cells.add(ij)
        elif reserve_pins:
            w_cur[source] = inf_val
            w_cur[sink] = inf_val

        results.append(NetResult(source, sink, path))

    return results


def route_nets_3d(
    w: torch.Tensor,
    nets: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    via_cost: float = 1.0,
    reserve_pins: bool = True,
    w_v: torch.Tensor | None = None,
) -> list[Net3DResult]:
    """Route nets sequentially on a multi-layer grid with via transitions.

    Args:
        w: (L, H, W) tensor of cell-entry costs for axis=2 ("H") moves.
            Also used for axis=1 ("V") moves when `w_v` is not given.
            inf for obstacles.
        nets: ordered list of ((layer, row, col), (layer, row, col)) pairs.
        via_cost: edge weight for one via transition between adjacent layers.
        reserve_pins: if True, all pin cells (across all layers) are reserved
            as obstacles before routing; each net's own pins are temporarily
            un-reserved while it routes.
        w_v: optional (L, H, W) cost tensor for axis=1 ("V") moves. Use
            `gpu_pnr.sweep.axis_costs` to build (w, w_v) from per-layer
            preferred-direction multipliers. When set, pin reservation and
            committed-route obstacles are applied to both tensors so pin
            and wire blocking is consistent across axes.

    Returns:
        list of Net3DResult in input order.
    """
    inf_val = torch.tensor(float("inf"), device=w.device, dtype=w.dtype)
    w_cur = w.clone()
    w_v_cur = w_v.clone() if w_v is not None else None
    routed_cells: set[tuple[int, int, int]] = set()

    def _set_inf(ijk: tuple[int, int, int]) -> None:
        w_cur[ijk] = inf_val
        if w_v_cur is not None:
            w_v_cur[ijk] = inf_val

    def _restore(ijk: tuple[int, int, int]) -> None:
        w_cur[ijk] = w[ijk]
        if w_v_cur is not None:
            assert w_v is not None
            w_v_cur[ijk] = w_v[ijk]

    if reserve_pins:
        pin_cells: set[tuple[int, int, int]] = set()
        for s, t in nets:
            pin_cells.add(s)
            pin_cells.add(t)
        for ijk in pin_cells:
            if _is_finite(w_cur, ijk):
                _set_inf(ijk)

    results: list[Net3DResult] = []
    for source, sink in nets:
        if source in routed_cells or sink in routed_cells:
            results.append(Net3DResult(source, sink, None))
            continue
        if not _is_finite(w, source) or not _is_finite(w, sink):
            results.append(Net3DResult(source, sink, None))
            continue

        if reserve_pins:
            _restore(source)
            _restore(sink)

        d, _ = sweep_sssp_3d(w_cur, source, via_cost=via_cost, w_v=w_v_cur)
        w_v_for_backtrace = w_v_cur.cpu() if w_v_cur is not None else None
        path = backtrace_3d(
            d.cpu(),
            w_cur.cpu(),
            source,
            sink,
            via_cost=via_cost,
            w_v=w_v_for_backtrace,
        )

        if path is not None:
            for ijk in path:
                _set_inf(ijk)
                routed_cells.add(ijk)
        elif reserve_pins:
            _set_inf(source)
            _set_inf(sink)

        results.append(Net3DResult(source, sink, path))

    return results
