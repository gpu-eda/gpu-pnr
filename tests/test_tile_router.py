"""Tests for the tile-router geometry + partition module.

Covers Slice 1 of the WS3.3 tile router (data classes, partitioning, and
net-to-tile assignment). Routing tests land in later slices. See
`docs/plans/ws33-tile-router-implementation.md` §Slice 1 and
`docs/adr/0012-tile-decomposition.md` §3, §6 for the design.
"""

from __future__ import annotations

import pytest

from gpu_pnr.tile_router import (
    Tile,
    TileRouter,
    assign_net_to_tile,
    classify_nets,
    net_bbox,
    partition_chip,
)


def test_partition_covers_chip():
    """Every owned cell is owned by exactly one tile; union = whole chip."""
    chip_h, chip_w = 512, 384
    tile_size = 128
    halo = 8
    tiles = partition_chip(chip_h, chip_w, tile_size=tile_size, halo=halo)

    # Walk every chip cell, count ownerships.
    owners: dict[tuple[int, int], int] = {}
    for t in tiles:
        for r in range(t.r0, t.r1):
            for c in range(t.c0, t.c1):
                owners[(r, c)] = owners.get((r, c), 0) + 1

    # Cover: every chip cell owned at least once.
    assert len(owners) == chip_h * chip_w
    # No overlap: every chip cell owned exactly once.
    assert all(count == 1 for count in owners.values())
    # No off-chip cells.
    for (r, c) in owners:
        assert 0 <= r < chip_h
        assert 0 <= c < chip_w


def test_partition_partial_edge_tiles():
    """Right/bottom edges produce partial tiles when chip dims don't divide evenly."""
    # 300 = 2*128 + 44, so 3 tiles along each axis with a 44-wide remainder.
    tiles = partition_chip(300, 300, tile_size=128, halo=8)
    rows = sorted({t.r0 for t in tiles})
    cols = sorted({t.c0 for t in tiles})
    assert rows == [0, 128, 256]
    assert cols == [0, 128, 256]
    # The last-row, last-col tile is 44x44 owned.
    last = next(t for t in tiles if t.r0 == 256 and t.c0 == 256)
    assert last.r1 - last.r0 == 44
    assert last.c1 - last.c0 == 44


def test_assign_net_bbox_fits_owned():
    """Net entirely inside one tile's owned region → assigned to that tile."""
    chip_h, chip_w = 512, 512
    tile_size = 128
    halo = 8
    tiles = partition_chip(chip_h, chip_w, tile_size=tile_size, halo=halo)

    # Pins all inside the tile with r0=128, c0=128.
    pins = [(0, 150, 150), (1, 180, 200), (2, 200, 170)]
    bbox = net_bbox(pins)
    t = assign_net_to_tile(bbox, tiles, halo=halo)

    assert t is not None
    assert t.r0 == 128 and t.c0 == 128
    assert t.r1 == 256 and t.c1 == 256


def test_assign_net_bbox_in_halo_only():
    """Bbox center in A's owned region, bbox extends into B's region but within A's halo → assigned to A.

    Tile A = owned [128, 256) × [128, 256), halo=8 → envelope [120, 264) × [120, 264).
    Tile B = owned [256, 384) × [128, 256), halo=8 → envelope [248, 392) × [120, 264).
    A bbox of rows [200, 260], cols [150, 170] has center (230, 160) which lies
    in A's owned region. The bbox row max 260 is inside A's halo envelope
    (rmax 260 < 264) and inside B's halo envelope (rmin 200 < 248 → NOT in B).
    So actually only A fits here. Construct a case where bbox+halo fits BOTH:
    rows [255, 257] → center 256 (boundary), in B's owned region. Use rows
    [250, 256] → center 253, in A's owned (r1=256 exclusive), and rmax 256
    is inside both A's envelope (256 < 264) and B's envelope (256 >= 248).
    """
    chip_h, chip_w = 512, 512
    tile_size = 128
    halo = 8
    tiles = partition_chip(chip_h, chip_w, tile_size=tile_size, halo=halo)

    # Pins straddling the A/B border but center in A's owned region.
    # A owned: rows [128, 256). B owned: rows [256, 384).
    # bbox rows [250, 256], center 253 → in A's owned region.
    # bbox rmax=256 inside A's envelope (rmax_envelope = 256+8 = 264) ✓
    # bbox rmin=250 inside B's envelope (rmin_envelope = 256-8 = 248) ✓
    # → both A and B's envelopes contain the bbox → tiebreak picks A.
    pins = [(0, 250, 150), (1, 256, 180)]
    bbox = net_bbox(pins)
    assert bbox == (250, 150, 256, 180)

    t = assign_net_to_tile(bbox, tiles, halo=halo)
    assert t is not None
    # Tiebreak: center (253, 165) is in A's owned region [128,256)×[128,256).
    assert t.r0 == 128 and t.c0 == 128


def test_assign_net_multi_tile_spanning():
    """Bbox+halo crosses multiple owned regions such that no single tile's envelope contains it → None."""
    chip_h, chip_w = 512, 512
    tile_size = 128
    halo = 8
    tiles = partition_chip(chip_h, chip_w, tile_size=tile_size, halo=halo)

    # Pins spanning rows 100..400 — far wider than any tile's halo envelope
    # (128 + 2*8 = 144). No single tile can contain this bbox.
    pins = [(0, 100, 100), (1, 400, 100)]
    bbox = net_bbox(pins)
    t = assign_net_to_tile(bbox, tiles, halo=halo)
    assert t is None


def test_classify_partitions_all_nets():
    """Every input net index appears in exactly one of {per-tile lists, spanning list}."""
    chip_h, chip_w = 512, 512
    tile_size = 128
    halo = 8
    tiles = partition_chip(chip_h, chip_w, tile_size=tile_size, halo=halo)

    nets: list[list[tuple[int, int, int]]] = [
        # Index 0: fits in tile (0,0).
        [(0, 10, 10), (1, 50, 50)],
        # Index 1: fits in tile (128, 128).
        [(0, 150, 150), (2, 200, 200)],
        # Index 2: spans many tiles (rows 50..450).
        [(0, 50, 50), (1, 450, 450)],
        # Index 3: fits in tile (384, 384) — bottom-right.
        [(0, 400, 400), (1, 480, 480)],
        # Index 4: spans two tiles vertically (rows 100..300).
        [(0, 100, 100), (1, 300, 100)],
    ]

    per_tile, spanning = classify_nets(nets, tiles, halo=halo)

    # Every net index appears exactly once across all the buckets.
    seen: list[int] = []
    for indices in per_tile.values():
        seen.extend(indices)
    seen.extend(spanning)
    assert sorted(seen) == list(range(len(nets)))
    # And no duplicates.
    assert len(seen) == len(nets)

    # Sanity: nets 0,1,3 are per-tile; net 2 is spanning (way too wide).
    all_per_tile = [i for indices in per_tile.values() for i in indices]
    assert 0 in all_per_tile
    assert 1 in all_per_tile
    assert 3 in all_per_tile
    assert 2 in spanning


def test_tile_router_route_is_stub():
    """Slice 1: TileRouter.route is a placeholder; routing lands in Slice 3+."""
    router = TileRouter(w_chip=None, w_v_chip=None, tile_size=128, halo=8)
    with pytest.raises(NotImplementedError):
        router.route([[(0, 0, 0), (1, 1, 1)]])


def test_tile_is_frozen_and_hashable():
    """Tile must be hashable (used as dict key in classify_nets)."""
    t1 = Tile(r0=0, c0=0, r1=128, c1=128, halo=8)
    t2 = Tile(r0=0, c0=0, r1=128, c1=128, halo=8)
    assert t1 == t2
    assert hash(t1) == hash(t2)
    # Frozen → can't mutate.
    with pytest.raises((AttributeError, TypeError)):
        t1.r0 = 1  # type: ignore[misc]
