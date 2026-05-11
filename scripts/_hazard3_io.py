"""Shared parsers + grid construction for the Hazard3 GF180 LibreLane fixture.

Used by spike_route_one_net.py and spike_route_many_nets.py. The fixture is the
pre-computed LibreLane run at ~/Code/Apitronix/hazard-test (see the
`hazard3_fixture` memory entry); see docs/spikes/phase32-hazard3-real-fixture.md for design notes.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

GUIDE = Path(
    "/Users/roberttaylor/Code/Apitronix/hazard-test/hazard3/librelane/runs/"
    "RUN_2026-05-08_22-32-24/39-openroad-globalrouting/after_grt.guide"
)
FINAL_DEF = Path(
    "/Users/roberttaylor/Code/Apitronix/hazard-test/hazard3/librelane/runs/"
    "RUN_2026-05-08_22-32-24/final/def/synth_top_level_3.def"
)
LAYER_ORDER = ["Metal1", "Metal2", "Metal3", "Metal4", "Metal5"]
PITCH_DBU = 200  # gf180mcuD: 0.20um wire pitch, 1 DBU = 1 nm


def parse_guides(path: Path) -> dict[str, list[tuple[int, int, int, int, str]]]:
    """Read all nets from a LibreLane after_grt.guide file.

    Format: net name on its own line, '(' on the next, then one
    'xlo ylo xhi yhi LayerName' line per guide rectangle, terminated by ')'.
    """
    nets: dict[str, list[tuple[int, int, int, int, str]]] = {}
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        name = lines[i].strip()
        if i + 1 < len(lines) and lines[i + 1].strip() == "(":
            rects: list[tuple[int, int, int, int, str]] = []
            j = i + 2
            while j < len(lines) and lines[j].strip() != ")":
                parts = lines[j].split()
                if len(parts) == 5:
                    rects.append(
                        (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), parts[4])
                    )
                j += 1
            nets[name] = rects
            i = j + 1
        else:
            i += 1
    return nets


def build_grid(
    rects: list[tuple[int, int, int, int, str]],
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Convert per-layer guide rectangles to an (L, H, W) cost tensor.

    Cells inside the union of guide rectangles for a layer get cost 1.0;
    cells outside get inf (obstacle). The grid origin is the bbox lower-left
    corner; quantization is PITCH_DBU per cell.

    Returns (w, (origin_x, origin_y)) where w has shape (5, H, W).
    """
    xlo = min(r[0] for r in rects)
    ylo = min(r[1] for r in rects)
    xhi = max(r[2] for r in rects)
    yhi = max(r[3] for r in rects)
    H = (yhi - ylo) // PITCH_DBU
    W = (xhi - xlo) // PITCH_DBU
    L = len(LAYER_ORDER)
    w = torch.full((L, H, W), float("inf"))
    for x0, y0, x1, y1, layer in rects:
        if layer not in LAYER_ORDER:
            continue
        lyr = LAYER_ORDER.index(layer)
        gx0 = (x0 - xlo) // PITCH_DBU
        gy0 = (y0 - ylo) // PITCH_DBU
        gx1 = (x1 - xlo) // PITCH_DBU
        gy1 = (y1 - ylo) // PITCH_DBU
        w[lyr, gy0:gy1, gx0:gx1] = 1.0
    return w, (xlo, ylo)


def rect_center_to_grid(
    rect: tuple[int, int, int, int, str], origin: tuple[int, int]
) -> tuple[int, int, int]:
    """Center of a guide rectangle, mapped to (layer, row, col)."""
    x0, y0, x1, y1, layer = rect
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    return (
        LAYER_ORDER.index(layer),
        (cy - origin[1]) // PITCH_DBU,
        (cx - origin[0]) // PITCH_DBU,
    )


_COORD_RE = re.compile(r"\(\s*(-?\d+|\*)\s+(-?\d+|\*)\s*\)")
_NET_HEADER_RE = re.compile(r"^\s*-\s+(\S+)\s")
_DIEAREA_RE = re.compile(
    r"DIEAREA\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)"
)


def parse_def_diearea(def_path: Path) -> tuple[int, int, int, int]:
    """Parse the DIEAREA line; returns (xlo, ylo, xhi, yhi) in DEF DBU."""
    m = _DIEAREA_RE.search(def_path.read_text())
    if m is None:
        raise ValueError(f"DIEAREA not found in {def_path}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


# Wire segment: (layer_name, x0, y0, x1, y1) -- endpoints in DEF DBU.
DefSegment = tuple[str, int, int, int, int]
# Via point: (layer_name_anchor, x, y, via_name).
DefVia = tuple[str, int, int, str]


def parse_def_net_geometry(
    def_path: Path,
) -> dict[str, tuple[list[DefSegment], list[DefVia]]]:
    """Parse the NETS section for per-segment geometry.

    Returns {net_name: (wire_segments, vias)}. Same DEF semantics as
    `parse_def_nets`: `*` resolves against `last_x`/`last_y`, RECT
    annotations are pin shapes (ignored). The via name (e.g. Via1_HV)
    implies the two layers it connects, but we don't decode that here --
    we just store the anchor coordinate and the via name; the renderer
    can paint a marker on the anchor layer if it wants to.
    """
    text = def_path.read_text()
    nets_start = text.find("\nNETS ")
    nets_end = text.find("\nEND NETS")
    if nets_start < 0 or nets_end < 0:
        raise ValueError(f"NETS section not found in {def_path}")
    section = text[nets_start:nets_end]

    nets: dict[str, tuple[list[DefSegment], list[DefVia]]] = {}
    cur_name: str | None = None
    cur_segments: list[DefSegment] = []
    cur_vias: list[DefVia] = []
    cur_layer: str | None = None
    last_x = 0
    last_y = 0

    for line in section.splitlines():
        m = _NET_HEADER_RE.match(line)
        if m is not None:
            if cur_name is not None:
                nets[cur_name] = (cur_segments, cur_vias)
            cur_name = m.group(1)
            cur_segments = []
            cur_vias = []
            cur_layer = None
            last_x = 0
            last_y = 0
            continue
        if cur_name is None:
            continue
        # Lines that introduce a routing layer carry a "Metal<n>" token
        # (after "+ ROUTED" or "NEW"). Lift it into cur_layer; subsequent
        # "*"-coord lines on the same net stay on that layer until a new
        # Metal token appears.
        for tok in line.split():
            if tok.startswith("Metal"):
                cur_layer = tok
                break
        coords = _COORD_RE.findall(line)
        if not coords:
            continue
        x0_tok, y0_tok = coords[0]
        x0 = last_x if x0_tok == "*" else int(x0_tok)
        y0 = last_y if y0_tok == "*" else int(y0_tok)
        last_x, last_y = x0, y0
        if "RECT" in line:
            continue
        if len(coords) >= 2:
            x1_tok, y1_tok = coords[1]
            x1 = x0 if x1_tok == "*" else int(x1_tok)
            y1 = y0 if y1_tok == "*" else int(y1_tok)
            if cur_layer is not None:
                cur_segments.append((cur_layer, x0, y0, x1, y1))
            last_x, last_y = x1, y1
        else:
            tokens = line.rstrip(" ;").split()
            if tokens and tokens[-1].startswith("Via") and cur_layer is not None:
                cur_vias.append((cur_layer, x0, y0, tokens[-1]))

    if cur_name is not None:
        nets[cur_name] = (cur_segments, cur_vias)
    return nets


def parse_def_nets(def_path: Path) -> dict[str, tuple[int, int]]:
    """Parse the NETS section of a DEF, returning per-net (wirelength_dbu, via_count).

    Format reminder:
        - net_name ( inst pin ) ( inst pin ) + USE SIGNAL
          + ROUTED Metal3 ( x y ) ( x' y' )    <- wire segment, two coord pairs
          NEW Metal1 ( x y ) Via1_HV           <- via, single coord + Via* token
          NEW Metal2 ( x y ) RECT ( a b c d )  <- pin shape, ignored for wirelength
          ...
          ;

    `*` in a coord position means "same value as the previous explicit coord
    along that axis"; resolved against `last_x` / `last_y` carried within
    each net's routing.

    Wirelength is summed in DEF DBU (1 nm for gf180mcuD); divide by PITCH_DBU
    to get grid cells. Via count is the number of segments whose final token
    matches Via*.
    """
    text = def_path.read_text()
    nets_start = text.find("\nNETS ")
    nets_end = text.find("\nEND NETS")
    if nets_start < 0 or nets_end < 0:
        raise ValueError(f"NETS section not found in {def_path}")
    section = text[nets_start:nets_end]

    nets: dict[str, tuple[int, int]] = {}
    cur_name: str | None = None
    cur_wire = 0
    cur_vias = 0
    last_x = 0
    last_y = 0

    for line in section.splitlines():
        m = _NET_HEADER_RE.match(line)
        if m is not None:
            if cur_name is not None:
                nets[cur_name] = (cur_wire, cur_vias)
            cur_name = m.group(1)
            cur_wire = 0
            cur_vias = 0
            last_x = 0
            last_y = 0
            continue
        if cur_name is None:
            continue
        coords = _COORD_RE.findall(line)
        if not coords:
            continue
        x0_tok, y0_tok = coords[0]
        x0 = last_x if x0_tok == "*" else int(x0_tok)
        y0 = last_y if y0_tok == "*" else int(y0_tok)
        last_x, last_y = x0, y0
        # RECT lines carry a second coord pair, but it's a relative bbox
        # (pin-shape annotation), not a wire endpoint -- skip after the anchor.
        if "RECT" in line:
            continue
        if len(coords) >= 2:
            x1_tok, y1_tok = coords[1]
            x1 = x0 if x1_tok == "*" else int(x1_tok)
            y1 = y0 if y1_tok == "*" else int(y1_tok)
            cur_wire += abs(x1 - x0) + abs(y1 - y0)
            last_x, last_y = x1, y1
        else:
            # Last segment of a net ends with ";", strip before via-name check.
            tokens = line.rstrip(" ;").split()
            if tokens and tokens[-1].startswith("Via"):
                cur_vias += 1

    if cur_name is not None:
        nets[cur_name] = (cur_wire, cur_vias)
    return nets

