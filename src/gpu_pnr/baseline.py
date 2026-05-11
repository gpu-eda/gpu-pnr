"""Reference Dijkstra on 2D and 3D grids for ground-truth comparison against sweep SSSP."""

from __future__ import annotations

import heapq
import math

import torch


def dijkstra_grid(
    w: torch.Tensor,
    source: tuple[int, int],
) -> torch.Tensor:
    """Standard Dijkstra on a 4-connected grid.

    Args:
        w: (H, W) tensor (any device); cost to enter each cell. inf for obstacles.
        source: (row, col).

    Returns:
        (H, W) float32 CPU tensor of shortest distances. inf where unreachable.
    """
    w_np = w.detach().cpu().numpy()
    H, W = w_np.shape
    sr, sc = source

    d = [[math.inf] * W for _ in range(H)]
    d[sr][sc] = 0.0

    pq: list[tuple[float, int, int]] = [(0.0, sr, sc)]
    while pq:
        cur_d, i, j = heapq.heappop(pq)
        if cur_d > d[i][j]:
            continue
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                cost = float(w_np[ni, nj])
                if math.isinf(cost):
                    continue
                new_d = cur_d + cost
                if new_d < d[ni][nj]:
                    d[ni][nj] = new_d
                    heapq.heappush(pq, (new_d, ni, nj))

    return torch.tensor(d, dtype=torch.float32)


def dijkstra_grid_3d(
    w: torch.Tensor,
    source: tuple[int, int, int],
    via_cost: float = 1.0,
    w_v: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standard Dijkstra on a multi-layer 4-connected grid with via edges.

    Each layer is 4-connected for in-layer wires. The cost to enter a cell
    depends on which axis the move was along: column-changing ("H") moves
    pay `w[neighbor]`; row-changing ("V") moves pay `w_v[neighbor]` (or
    `w[neighbor]` if `w_v` is None). Adjacent layers connect at the same
    (r, c) via a via edge of weight `via_cost`; a via can land iff at least
    one of `w` / `w_v` is finite at the destination cell.

    Args:
        w: (L, H, W) tensor; inf for obstacles. Cost for axis=2 ("H") moves.
        source: (layer, row, col).
        via_cost: edge weight for one via transition.
        w_v: optional (L, H, W); cost for axis=1 ("V") moves. If None, `w` is
            used for both axes.

    Returns:
        (L, H, W) float32 CPU tensor of shortest distances.
    """
    w_h_np = w.detach().cpu().numpy()
    w_v_np = w_v.detach().cpu().numpy() if w_v is not None else w_h_np
    L, H, W = w_h_np.shape
    sl, sr, sc = source

    d = [[[math.inf] * W for _ in range(H)] for _ in range(L)]
    d[sl][sr][sc] = 0.0

    pq: list[tuple[float, int, int, int]] = [(0.0, sl, sr, sc)]
    while pq:
        cur_d, lyr, i, j = heapq.heappop(pq)
        if cur_d > d[lyr][i][j]:
            continue
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                cost = float(w_v_np[lyr, ni, nj] if di != 0 else w_h_np[lyr, ni, nj])
                if math.isinf(cost):
                    continue
                new_d = cur_d + cost
                if new_d < d[lyr][ni][nj]:
                    d[lyr][ni][nj] = new_d
                    heapq.heappush(pq, (new_d, lyr, ni, nj))
        for dl in (-1, 1):
            nl = lyr + dl
            if 0 <= nl < L:
                # Via can land on a cell if at least one axis is finite there;
                # the wire continues from there along whichever axis is usable.
                if math.isinf(float(w_h_np[nl, i, j])) and math.isinf(
                    float(w_v_np[nl, i, j])
                ):
                    continue
                new_d = cur_d + via_cost
                if new_d < d[nl][i][j]:
                    d[nl][i][j] = new_d
                    heapq.heappush(pq, (new_d, nl, i, j))

    return torch.tensor(d, dtype=torch.float32)
