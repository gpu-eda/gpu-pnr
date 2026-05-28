"""Tests for the post-DR DEF NETS-section parser used by the spike scripts.

The parser is hand-rolled (the lefdef PyPI package ships only Linux/Windows
binaries; see docs/spikes/phase32-hazard3-real-fixture.md). It picks up:
  - wire segments (two coord pairs on one line; sum Manhattan length),
  - vias (single coord + Via* trailing token),
  - `*` placeholders that reuse the previous explicit coord,
  - RECT pin-shape annotations (skipped after the anchor coord),
  - multi-line connection lists (lines with `( instname pin )` parens that look
    like coords but aren't -- the regex only matches `( int int )`).

These cases are exercised here against a synthetic DEF snippet so the canonical
hand-traced numbers from docs/spikes/phase32-hazard3-real-fixture.md are locked in.
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/ isn't on sys.path under pytest by default.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from _hazard3_io import (  # noqa: E402
    PITCH_DBU,
    build_chip_grid,
    parse_def_nets,
    rect_center_to_grid,
)


def _write_def(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "synth.def"
    p.write_text(body)
    return p


def test_simple_wire_and_vias(tmp_path: Path):
    """Trivial 2-pin net with a horizontal wire and two access vias."""
    body = """\
DESIGN test ;
NETS 1 ;
    - _0001_ ( inst1 D ) ( inst2 Z ) + USE SIGNAL
      + ROUTED Metal3 ( 100 200 ) ( 500 200 )
      NEW Metal1 ( 100 200 ) Via1_HV
      NEW Metal1 ( 500 200 ) Via1_HV ;
END NETS
"""
    nets = parse_def_nets(_write_def(tmp_path, body))
    assert nets == {"_0001_": (400, 2)}


def test_star_placeholder_reuses_previous_coord(tmp_path: Path):
    body = """\
DESIGN test ;
NETS 1 ;
    - _0002_ ( a A ) ( b B ) + USE SIGNAL
      + ROUTED Metal2 ( 100 200 ) ( * 350 )
      NEW Metal2 ( 100 350 ) ( 480 * ) ;
END NETS
"""
    nets = parse_def_nets(_write_def(tmp_path, body))
    # First seg vertical 200 -> 350 (length 150), then horizontal 100 -> 480 (380).
    assert nets == {"_0002_": (150 + 380, 0)}


def test_rect_annotation_skipped(tmp_path: Path):
    body = """\
DESIGN test ;
NETS 1 ;
    - _0003_ ( a A ) ( b B ) + USE SIGNAL
      + ROUTED Metal2 ( 100 100 ) ( 200 100 )
      NEW Metal2 ( 200 100 ) RECT ( -10 -10 10 10 ) ;
END NETS
"""
    nets = parse_def_nets(_write_def(tmp_path, body))
    # Only the first wire segment counts (length 100); RECT line ignored.
    assert nets == {"_0003_": (100, 0)}


def test_multi_line_connection_list_not_misread_as_coords(tmp_path: Path):
    """Multi-pin nets break their connection list onto continuation lines like
    `( _27644_ A1 ) ( _26906_ A1 )`, which contain parens but aren't coords.
    The regex requires int-or-* tokens so these are skipped."""
    body = """\
DESIGN test ;
NETS 1 ;
    - _0004_ ( inst1 ZN ) ( inst2 A1 )
      ( inst3 A1 ) ( inst4 A1 ) ( inst5 A2 ) + USE SIGNAL
      + ROUTED Metal2 ( 0 0 ) ( 100 0 ) ;
END NETS
"""
    nets = parse_def_nets(_write_def(tmp_path, body))
    assert nets == {"_0004_": (100, 0)}


def test_trailing_semicolon_on_via_line(tmp_path: Path):
    """Last segment of a net ends with `;`, which would shadow the trailing
    `Via*` token if not stripped."""
    body = """\
DESIGN test ;
NETS 1 ;
    - _0005_ ( a A ) ( b B ) + USE SIGNAL
      + ROUTED Metal3 ( 0 0 ) ( 200 0 )
      NEW Metal1 ( 200 0 ) Via1_HV ;
END NETS
"""
    nets = parse_def_nets(_write_def(tmp_path, body))
    assert nets == {"_0005_": (200, 1)}


def test_multiple_nets(tmp_path: Path):
    body = """\
DESIGN test ;
NETS 2 ;
    - _0006_ ( a A ) ( b B ) + USE SIGNAL
      + ROUTED Metal2 ( 0 0 ) ( 100 0 ) ;
    - _0007_ ( c C ) ( d D ) + USE SIGNAL
      + ROUTED Metal3 ( 1000 1000 ) ( 1000 2000 )
      NEW Metal2 ( 1000 1000 ) Via2_VH ;
END NETS
"""
    nets = parse_def_nets(_write_def(tmp_path, body))
    assert nets == {"_0006_": (100, 0), "_0007_": (1000, 1)}


def test_build_chip_grid_default_pitch_is_200():
    """Omitting pitch_dbu reproduces the legacy 200 DBU sampling."""
    nets = {"n0": [(0, 0, 1120, 1120, "Metal1")]}
    default = build_chip_grid(nets, 0, 0, 11200, 11200)
    explicit = build_chip_grid(nets, 0, 0, 11200, 11200, pitch_dbu=PITCH_DBU)
    assert PITCH_DBU == 200
    assert default.shape == explicit.shape
    # H = (11200 - 0) // 200 + 1 = 57
    assert default.shape[1] == 57 and default.shape[2] == 57


def test_build_chip_grid_track_pitch_shrinks_axes():
    """At the 1120 DBU track pitch the grid is ~5.6x coarser per axis."""
    nets = {"n0": [(0, 0, 1120, 1120, "Metal1")]}
    track = build_chip_grid(nets, 0, 0, 11200, 11200, pitch_dbu=1120)
    # H = (11200 - 0) // 1120 + 1 = 11
    assert track.shape[1] == 11 and track.shape[2] == 11
    # The single guide rect occupies one track cell, not a 5x5 block.
    assert float(track[0, 0, 0]) == 1.0
    assert float(track[0, 1, 1]) == float("inf")


def test_rect_center_to_grid_respects_pitch():
    """Pin mapping coarsens with pitch; default stays at 200 DBU."""
    rect = (1000, 2000, 1400, 2400, "Metal2")
    # center = (1200, 2200); layer index 1.
    assert rect_center_to_grid(rect, (0, 0)) == (1, 11, 6)
    assert rect_center_to_grid(rect, (0, 0), pitch_dbu=1120) == (1, 1, 1)


def test_missing_nets_section_raises(tmp_path: Path):
    p = tmp_path / "broken.def"
    p.write_text("DESIGN foo ;\nEND DESIGN\n")
    try:
        parse_def_nets(p)
    except ValueError:
        return
    raise AssertionError("expected ValueError for missing NETS section")
