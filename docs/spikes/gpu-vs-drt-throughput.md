# Spike — GPU sweep vs TritonRoute DRT throughput comparison

**Status:** Open (2026-05-28).

## Question

Is GPU-accelerated sweep routing competitive with TritonRoute's detailed
router on the same design, and if not, what architectural changes would
close the gap?

## Context

The E5 GPU sweep router does full-grid Dijkstra (SSSP) on a 3D cost grid.
TritonRoute (DRT) routes within global-routing guide regions using a
DRC-aware maze router on a fine grid. Both target the same problem
(detailed routing), but their search spaces per net are very different.

We have a direct comparison point: the **dualcore chip_top** design
(containing two Hazard3 cores), routed by both TritonRoute (via
LibreLane/OpenROAD) and our bench sweep on the same M4 Pro hardware.

## Measurements

### TritonRoute DRT — dualcore chip_top (M4 Pro, 14 threads)

Source: `/tmp/claude/dualcore-librelane-signoff-5.log` (2026-05-27).

- **Design:** 36,541 nets, 94,335 components, 5 metal layers (gf180mcuD)
- **Die:** 7864 × 10244 µm
- **GCELLGRID:** 468 × 609 cells at 16,800-unit pitch
- **Guides:** 337,404 (avg ~9.2 guides/net)

| Phase | CPU time | Elapsed (14 thr) |
|---|---|---|
| Pin access | 2m 07s | 10s |
| Track assignment | 16s | 5s |
| Iter 0 (initial route) | 6m 31s | 35s |
| Iter 1 (rip-up) | 5m 54s | 33s |
| Iter 2 (rip-up) | 5m 50s | 34s |
| Iter 3 (rip-up) | 2m 48s | 19s |
| Iter 4 (cleanup) | 5s | 1s |
| **Total DRT** | **21m 10s** | **2m 04s** |

Result: 0 DRC violations, 3.87 mm total wire, 266k vias.

**Per-net throughput (initial pass):** 36.5k nets in 35s elapsed →
**~1.0 ms/net** (14 threads) or **~10.7 ms/net** single-threaded CPU.

### GPU sweep — Hazard3 subset (~20k routable nets, M4 Pro)

Source: local bench runs (2026-05-27), CI golden bench (M2 Mac Mini).

**Sequential ms/source on M4 Pro:**

| Envelope | MPS | CPU | GPU speedup |
|---:|---:|---:|---:|
| 256² | 18 ms | 134 ms | 7.4× |
| 448² | 29 ms | 324 ms | 11.2× |
| 512² | 38 ms | 480 ms | 12.6× |

**Sequential ms/source on M2 Mac Mini (CI golden):**

| Envelope | MPS | CPU | GPU speedup |
|---:|---:|---:|---:|
| 256² | 31 ms | 78 ms | 2.5× |
| 448² | 91 ms | 205 ms | 2.3× |
| 512² | 125 ms | 432 ms | 3.5× |

### Head-to-head: per-net cost

| Router | ms/net | Device | Notes |
|---|---|---|---|
| DRT iter 0 | ~1.0 | 14× CPU threads | Guide-constrained search |
| DRT iter 0 | ~10.7 | 1× CPU (estimated) | Guide-constrained search |
| GPU sweep | 18 | MPS (M4 Pro) | Full 256² × 5-layer grid |
| GPU sweep | 134 | CPU (M4 Pro) | Full 256² × 5-layer grid |

**GPU MPS is ~1.7× slower than DRT single-threaded** on a per-net basis,
and **~18× slower than DRT 14-threaded**. But the search spaces are
radically different.

## DRT architecture (from source and published papers)

Sources: Kahng, Wang & Xu, "TritonRoute: The Open Source Detailed
Router," IEEE TCAD 2020; Kahng & Wang, ICCAD 2018; OpenROAD DRT
source (`src/drt/`).

### Algorithm

DRT uses **A\* search** on a non-uniform 3D grid graph. Multi-pin nets
are decomposed via Steiner-tree approximation into **two-pin segments**
routed sequentially. The grid graph has nodes at every track intersection
with bit-packed edge costs and blocked-edge flags (`FlexGridGraph`),
exploiting grid regularity for O(1) neighbour lookup. A key contribution
is correct-by-construction minimum-area constraint satisfaction during
the search itself.

### Guide-following

Global routing produces **route guides** — the union of GCells a net
passes through. DRT materialises the grid graph only inside guide
regions plus a small expansion margin. Nets can escape guides: DRT adds
adjacent GCells when pin access or inter-guide connectivity requires it.
GCell size is typically ~15×15 M1-track pitches. The fine grid pitch
comes from LEF TRACKS definitions, with off-track grid lines added for
non-standard pin geometry.

### Rip-up and reroute

Two phases: **initial detailed routing** (sequential, greedy), then
**search-and-repair** (iterative rip-up-and-reroute). Cost function has
two components:

- **Object cost:** applied around existing routed shapes; steers A\*
  away from potential DRC violations.
- **Marker cost:** accumulated at DRC violation locations as a
  **history penalty** across iterations (negotiation-based, à la
  PathFinder).

Outer loop: 7 iterations with per-net ripup limits (1, 4, 4, 4, 4, 4,
4). Each iteration scans DRC markers and reroutes offending nets within
a **standard box** (default 7×7 GCells).

### Threading

**Region-parallel bulk synchronous parallel (BSP).** The chip is
partitioned into non-overlapping rectangular regions (standard boxes).
Within a BSP superstep, independent regions are routed in parallel — no
synchronisation needed. Between supersteps, a barrier fires and DRC
markers are globally updated. `FlexDRWorker` encapsulates one region's
routing state.

### Memory

Only one region's grid graph and local net data are fully expanded per
thread at a time. R-trees (Boost.Geometry) for spatial queries are built
once from initial routing and updated incrementally. For very large
designs, OpenROAD supports distributed execution across multiple
processes.

## Analysis

### Why DRT is faster per net

DRT doesn't do full-grid Dijkstra. For each net it:

1. Materialises the grid only inside guide regions (~9 guides/net avg,
   each ~15×15 track pitches → ~2,000 grid cells per guide)
2. Runs A\* (heuristic-guided, not full SSSP) on that sub-grid
3. Uses pattern routing (pre-computed L/Z shapes) for simple 2-pin
   segments, falling back to maze A\* only for complex cases

A typical 2-pin net with 2–3 guides searches maybe **1,000–5,000
grid cells**. Our sweep explores **327,680 cells** (256 × 256 × 5)
regardless of net complexity. That's **65–328× more work per net.**

The GPU's 7–13× speedup over CPU partially compensates, but can't
overcome a 65–328× search space disadvantage.

### Why this isn't fatal

1. **DRT's guide quality depends on global routing.** When guides are
   wrong or too tight, DRT needs expensive rip-up iterations (we
   observed 5 iterations / 2 min on dualcore). Our full-grid approach
   always finds the optimal path within the cost grid.

2. **GPU parallelism scales differently.** DRT's BSP threading is
   near-linear but bounded by core count (14 threads → ~10× on the
   dualcore run). GPU parallelism scales with memory bandwidth and
   compute units — an M4 Pro GPU has ~16× the memory bandwidth of a
   single CPU core, and CUDA hardware much more.

3. **The search space gap is fixable.** We can constrain the sweep to
   guide regions or adaptive bounding boxes without changing the
   algorithm. A net with a 50×30×2 guide region would sweep 3,000 cells
   instead of 327,680 — putting us in DRT's ballpark per-net, with
   GPU acceleration on top.

4. **DRT's region-parallel model maps to GPU tiles.** DRT's standard-box
   partition (7×7 GCells per worker) is conceptually identical to our
   tile decomposition. The difference is that DRT expands one box per
   CPU thread while we could sweep many boxes simultaneously on GPU.

### The path to competitiveness

The architectural change needed is **guide-constrained sweep**:

1. Accept global-routing guides (from OpenROAD GRT or our own)
2. For each net, compute the bounding box of its guides + margin
3. Allocate a per-net sub-grid (or slice the chip grid) to that bbox
4. Run the GPU sweep on the sub-grid, not the full tile

Expected impact: a net with a 50×30 guide bbox on 2 layers sweeps
3,000 cells instead of 327,680. At our current MPS throughput that's
~0.16 ms/net vs 18 ms/net — **~112× faster**, putting us well ahead
of DRT's single-threaded 10.7 ms/net.

The real GPU win comes from **batching many small sub-grids**: unlike
K-batching on a single large grid (which is dead per Tier B), batching
many small independent grids should parallelise well — each net's
sub-grid is independent, and small grids fit in GPU cache. This is
architecturally similar to DRT's BSP region-parallelism, but executed
on GPU tensor hardware instead of CPU threads.

### What this means for WS3.3

The current tile decomposition plan (ADR 0012) assumes fixed 256² tiles.
Guide-constrained sweep would replace the fixed-tile model with
**per-net adaptive regions** — closer to ADR 0012's "Path C" walk-back
option (adaptive envelope).

The plan revision should:
- Drop fixed-tile K-batching (already dead per Tier B)
- Adopt guide-following as the primary search space constraint
- Keep the cost-grid infrastructure (obstacle encoding, via costs)
- Add a GRT guide ingestion path (read OpenROAD guide DEF output)
- Explore batched small-grid sweep as the GPU parallelism model

## What this spike does NOT cover

- **Guide ingestion implementation.** Reading OpenROAD's guide format
  and mapping to our grid coordinates is a separate workstream.
- **DRC-aware routing.** DRT checks DRC during routing; we don't. The
  quality comparison (wire/via ratios) is tracked in `results.md`.
- **CUDA scaling.** All measurements are Apple Silicon MPS. CUDA's
  memory bandwidth advantage could significantly change the economics.
- **Multi-pin net topology.** DRT decomposes multi-pin nets via Steiner
  trees; we use incremental tree growth. The 0.55× via ratio gap is
  attributed to this difference (see `results.md` Phase 3.2).

## References

- Kahng, Wang & Xu, "TritonRoute: The Open Source Detailed Router,"
  IEEE TCAD 2020. <https://vlsicad.ucsd.edu/Publications/Journals/j133.pdf>
- Kahng & Wang, "TritonRoute: An Initial Detailed Router for Advanced
  VLSI Technologies," ICCAD 2018.
- OpenROAD DRT docs: <https://openroad.readthedocs.io/en/latest/main/src/drt/README.html>

## Links

- [Tier B spike](tier-b-envelope-throughput.md) — K-batching dead,
  sequential is the design parameter
- [ADR 0012](../adr/0012-tile-decomposition.md) — tile decomposition
  design (to be amended)
- [`results.md`](../results.md) §Phase 3.2 — wire/via quality comparison
  vs TritonRoute
- [Phase 3.2 Hazard3 spike](phase32-hazard3-real-fixture.md) — detailed
  quality analysis including guide-escape finding
