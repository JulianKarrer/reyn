# Neighbour Search Benchmarks

Auto-generated summary of `nanobench` results for the spatial acceleration datastructure that implements the Uniform grid described by <a href="https://ramakarl.com/pdfs/2014_Hoetzlein_FastFixedRadius_Neighbors.pdf" target="_blank">[Hoetlzlein 2014]</a> that uses a single prefix sum and atomic increments and decrements to construct a sorted particle buffer in $\mathcal{O}(N)$, but has memory requirements that scale in the volume of the entire scene, not just the portion filled with fluid.

This benchmark is used to make changes to the datastructure and memory layout measurable and be able to test changes such as the switch to a strict AoS layout, or to measure small optimization opportunities such as culling of neighbours due to conservative but quick heuristics (AABB checks etc.), use of intrinsics, branchless programming for less warp divergence, use of local and shared memory etc. 

The test scene is a $100^3$ cube of particles on a regular grid with a half-jitter (uniformly randomly offset by half the grid spacing) and grid construction with reordering is given 4 extra arrays to resort in the order of the space-filling curve to emulate resorting of masses and each component of velocity during actual simulations and make to comparison fair. The test-kernel computes $\sum_{j\in\mathcal{N}_i} W(x_{ij}) x_{i,x} x_{j,x}$ for each particle $i$.

Each `nanobench` epoch is set to at least 50 iterations to keep fluctuations in the measurements low. Speedups are relative to the initial version with or without reordering respectively!

## Variations
`Reorder ✅` indicates that all relevant buffers (positions, velocities, masses) are reordered along the space-filling curve each time the grid is constructed to improve memory coherency - otherwise (❌) the buffers are randomly shuffled and there is little coalescing to be expected.

- `Initial_Version`: baseline implementation where the cell size is the search radius
- `Current`: the most recent version - used by the implementation at the time of generation of this site
- `AABB_Check`: load each component of $x_j$ seperately, checking if $(x_{i,x} - x_{j,x})^2 \leq r^2$ for search radius $r$ before loading $x_{j,y}, x_{j,z}$ etc. from global memory
- `Check_8of27`: increase cell size of uniform grid to twice the search radius, so only $2^d = 8$ cells instead of $3^d = 27$ cells in $d=3$ dimensions must be checked, but each cell contains more candidate positions. Checks the integer and fractional component of $\frac{x}{r}$ to find the octant of the cell that the query point is in, only checking cells adjacent to that octant.
- `125Cells`: opposite of the previous strategy, cell size is half the search radius, with fewer positions per cell to query but there are $5^d = 125$ cells to check. <a href="https://dual.sphysics.org/2ndusersworkshop/Dominguez_DualSPHysics_Workshop_2015_Keynote_Optimisation_and_SPH_Tricks.pdf" target="_blank">[Reportedly]</a> can improve query runtime performance at the cost of increased memory usage. If no more than one particle per cell is to be expected, then subsequent threads will traverse the sorted list from subsequent initial indices to subsequent final indices, yielding perfect colalescing in the best case.
- `Branchless`: use a conditional of the form `acc += x_ij_l2 <= r_c_2 ? map(...) : 0` instead of an `if`-condition for pruning out-of range neighbour candidates in hopes of decreasing warp divergence. The compiler might be expected do something equivalent automatically, so this is not expected to have much impact.

## Query Overview

| Version | Reorder Buffers | Speedup | Time | MdAPE |
|---|:---:|---:|---:|---|
| `125Cells` | ✅ | <span style="color:#80EF80">+40.13%</span> | 597.4μs | 0.65% |
| `125Branchless` | ✅ | <span style="color:#80EF80">+38.66%</span> | 612.1μs | 0.44% |
| `125Branchless` | ❌ | <span style="color:#80EF80">+34.56%</span> | 14387.3μs | 0.10% |
| `Current` | ✅ | <span style="color:#80EF80">+34.43%</span> | 654.3μs | 0.89% |
| `125Cells` | ❌ | <span style="color:#80EF80">+34.28%</span> | 14448.6μs | 0.07% |
| `Current` | ❌ | <span style="color:#80EF80">+28.79%</span> | 15657.0μs | 0.35% |
| `AABB_Check` | ❌ | <span style="color:#FFEE8C">+2.76%</span> | 21378.4μs | 3.75% |
| `Initial_Version` | ❌ | <span style="color:#FFEE8C">+0.00%</span> | 21985.7μs | 0.52% |
| `Initial_Version` | ✅ | <span style="color:#FFEE8C">+0.00%</span> | 997.8μs | 0.57% |
| `AABB_Check` | ✅ | <span style="color:#FF746C">-38.45%</span> | 1381.5μs | 3.43% |
| `Check_8of27` | ❌ | <span style="color:#FF746C">-183.16%</span> | 62254.4μs | 0.01% |
| `Check_8of27` | ✅ | <span style="color:#FF746C">-490.77%</span> | 5894.7μs | 0.32% |

## Construction Overview

| Version | Reorder Buffers | Speedup | Time | MdAPE |
|---|:---:|---:|---:|---|
| `Check_8of27` | ❌ | <span style="color:#FFEE8C">+0.11%</span> | 176.0μs | 0.53% |
| `Initial_Version` | ❌ | <span style="color:#FFEE8C">+0.00%</span> | 176.2μs | 2.64% |
| `Initial_Version` | ✅ | <span style="color:#FFEE8C">+0.00%</span> | 197.2μs | 3.28% |
| `125Branchless` | ✅ | <span style="color:#FFEE8C">-0.30%</span> | 197.8μs | 0.24% |
| `125Cells` | ✅ | <span style="color:#FFEE8C">-2.84%</span> | 202.8μs | 1.88% |
| `Current` | ✅ | <span style="color:#FF746C">-9.84%</span> | 216.6μs | 6.90% |
| `AABB_Check` | ✅ | <span style="color:#FF746C">-10.09%</span> | 217.1μs | 2.79% |
| `AABB_Check` | ❌ | <span style="color:#FF746C">-17.48%</span> | 207.0μs | 2.37% |
| `Current` | ❌ | <span style="color:#FF746C">-33.88%</span> | 235.9μs | 5.27% |
| `125Branchless` | ❌ | <span style="color:#FF746C">-36.49%</span> | 240.5μs | 3.46% |
| `Check_8of27` | ✅ | <span style="color:#FF746C">-38.03%</span> | 272.2μs | 5.57% |
| `125Cells` | ❌ | <span style="color:#FF746C">-52.67%</span> | 269.0μs | 6.53% |

## Details

| Benchmark | median time/iter | iters | MdAPE | source |
|---|---:|---:|---:|---|
| Construction (No Reordering) | 240.5μs| 597 | 3.46% | `125Branchless` |
| Query (No Reordering) | 14387.3μs| 597 | 0.10% | `125Branchless` |
| Construction (Reordering) | 197.8μs| 597 | 0.24% | `125Branchless` |
| Query (Reordering) | 612.1μs| 597 | 0.44% | `125Branchless` |
| Construction (No Reordering) | 269.0μs| 597 | 6.53% | `125Cells` |
| Query (No Reordering) | 14448.6μs| 597 | 0.07% | `125Cells` |
| Construction (Reordering) | 202.8μs| 597 | 1.88% | `125Cells` |
| Query (Reordering) | 597.4μs| 597 | 0.65% | `125Cells` |
| Construction (No Reordering) | 207.0μs| 597 | 2.37% | `AABB_Check` |
| Query (No Reordering) | 21378.4μs| 597 | 3.75% | `AABB_Check` |
| Construction (Reordering) | 217.1μs| 597 | 2.79% | `AABB_Check` |
| Query (Reordering) | 1381.5μs| 597 | 3.43% | `AABB_Check` |
| Construction (No Reordering) | 176.0μs| 597 | 0.53% | `Check_8of27` |
| Query (No Reordering) | 62254.4μs| 597 | 0.01% | `Check_8of27` |
| Construction (Reordering) | 272.2μs| 597 | 5.57% | `Check_8of27` |
| Query (Reordering) | 5894.7μs| 597 | 0.32% | `Check_8of27` |
| Construction (No Reordering) | 235.9μs| 597 | 5.27% | `Current` |
| Query (No Reordering) | 15657.0μs| 597 | 0.35% | `Current` |
| Construction (Reordering) | 216.6μs| 597 | 6.90% | `Current` |
| Query (Reordering) | 654.3μs| 597 | 0.89% | `Current` |
| Construction (No Reordering) | 176.2μs| 597 | 2.64% | `Initial_Version` |
| Query (No Reordering) | 21985.7μs| 597 | 0.52% | `Initial_Version` |
| Construction (Reordering) | 197.2μs| 597 | 3.28% | `Initial_Version` |
| Query (Reordering) | 997.8μs| 597 | 0.57% | `Initial_Version` |
