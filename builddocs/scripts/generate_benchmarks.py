import argparse
import json
from pathlib import Path


def fmt_μs(x):
    return f"{float(x*1_000_000):.1f}μs"


def fmt_ms(x):
    return f"{float(x*1_000):.1f}ms"


def fmt_percent(x):
    return f"{float(x*100):.2f}%"


def parse_file(p: Path):
    j = json.loads(p.read_text(encoding="utf-8"))
    # nanobench uses top-level "results": [ ... ]
    entries = j.get("results")
    out = []
    for r in entries:
        name = r.get("name")
        median = r.get("median(elapsed)")
        medianAbsolutePercentError = r.get(
            "medianAbsolutePercentError(elapsed)", "")
        measurements = r.get("measurements", [])
        # sum iterations if present
        total_iters = 0
        if isinstance(measurements, list):
            for m in measurements:
                try:
                    total_iters += int(m.get("iterations", 0))
                except Exception:
                    pass
        out.append({
            "name": str(name),
            "median": fmt_µs(median),
            "total_iters": str(total_iters),
            "medianAbsolutePercentError": fmt_percent(medianAbsolutePercentError),
            "source": p.name.split(".json")[0],
        })
    return out


def color_percent(speedup):
    significant = 0.05
    if speedup > significant:
        color = "#80EF80"
    elif speedup < -significant:
        color = "#FF746C"
    else:
        color = "#FFEE8C"
    return f'<span style="color:{color}">{"+" if speedup >= 0 else ""}{fmt_percent(speedup)}</span>'


def generate_overview_table(rows, benchmark_type="Construction"):
    relevant_names = {
        "Construction": ["Construction (No Reordering)", "Construction (Reordering)"],
        "Query": ["Query (No Reordering)", "Query (Reordering)"]
    }[benchmark_type]
    table_rows = [r for r in rows if r['name'] in relevant_names]
    baseline_times = {
        r['name']: float(r['median'][:-2])
        for r in table_rows
        if r['source'] == "Initial_Version"
    }

    md = []
    md.append(f"## {benchmark_type} Overview\n")
    md.append("| Version | Reorder Buffers | Speedup | Time | MdAPE |")
    md.append("|---|:---:|---:|---:|---|")

    # sort from biggest improvement to greatest slowdown
    table_rows = sorted(
        table_rows,
        key=lambda r: (baseline_times.get(r['name'], float(r['median'][:-2])) - float(r['median'][:-2])) /
        baseline_times.get(r['name'], float(r['median'][:-2])),
        reverse=True
    )
    for r in table_rows:
        time_val = float(r['median'][:-2])  # remove μs/ms suffix
        baseline_time = baseline_times.get(r['name'], time_val)
        speedup_val = (baseline_time - time_val) / \
            baseline_time if baseline_time else 0

        reordered = "❌" if "No" in r['name'] else "✅"
        speedup_colored = color_percent(speedup_val)

        md.append(
            f"""| `{r['source']}` | {reordered} | {speedup_colored} | {r['median']} | {r['medianAbsolutePercentError']} |""")

    md.append("")
    return md


if __name__ == "__main__":
    json_dir = Path("builddocs/_staticc/benchmarks")
    out_path = Path("builddocs/benchmarks/index.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for jf in sorted(json_dir.glob("*.json")):
        rows.extend(parse_file(jf))

    md = []
    md.append("# Neighbour Search Benchmarks\n")
    md.append(
        """Auto-generated summary of `nanobench` results for the spatial acceleration datastructure that implements the Uniform grid described by <a href="https://ramakarl.com/pdfs/2014_Hoetzlein_FastFixedRadius_Neighbors.pdf" target="_blank">[Hoetlzlein 2014]</a> that uses a single prefix sum and atomic increments and decrements to construct a sorted particle buffer in $\\mathcal{O}(N)$, but has memory requirements that scale in the volume of the entire scene, not just the portion filled with fluid.

This benchmark is used to make changes to the datastructure and memory layout measurable and be able to test changes such as the switch to a strict AoS layout, or to measure small optimization opportunities such as culling of neighbours due to conservative but quick heuristics (AABB checks etc.), use of intrinsics, branchless programming for less warp divergence, use of local and shared memory etc. 

The test scene is a $100^3$ cube of particles on a regular grid with a half-jitter (uniformly randomly offset by half the grid spacing) and grid construction with reordering is given 4 extra arrays to resort in the order of the space-filling curve to emulate resorting of masses and each component of velocity during actual simulations and make to comparison fair. The test-kernel computes $\\sum_{j\\in\\mathcal{N}_i} W(x_{ij}) x_{i,x} x_{j,x}$ for each particle $i$.

Each `nanobench` epoch is set to at least 50 iterations to keep fluctuations in the measurements low. Speedups are relative to the initial version with or without reordering respectively!

## Variations
`Reorder ✅` indicates that all relevant buffers (positions, velocities, masses) are reordered along the space-filling curve each time the grid is constructed to improve memory coherency - otherwise (❌) the buffers are randomly shuffled and there is little coalescing to be expected.

- `Initial_Version`: baseline implementation where the cell size is the search radius
- `Current`: the most recent version - used by the implementation at the time of generation of this site
- `AABB_Check`: load each component of $x_j$ seperately, checking if $(x_{i,x} - x_{j,x})^2 \\leq r^2$ for search radius $r$ before loading $x_{j,y}, x_{j,z}$ etc. from global memory
- `Check_8of27`: increase cell size of uniform grid to twice the search radius, so only $2^d = 8$ cells instead of $3^d = 27$ cells in $d=3$ dimensions must be checked, but each cell contains more candidate positions. Checks the integer and fractional component of $\\frac{x}{r}$ to find the octant of the cell that the query point is in, only checking cells adjacent to that octant.
- `125Cells`: opposite of the previous strategy, cell size is half the search radius, with fewer positions per cell to query but there are $5^d = 125$ cells to check. <a href="https://dual.sphysics.org/2ndusersworkshop/Dominguez_DualSPHysics_Workshop_2015_Keynote_Optimisation_and_SPH_Tricks.pdf" target="_blank">[Reportedly]</a> can improve query runtime performance at the cost of increased memory usage. If no more than one particle per cell is to be expected, then subsequent threads will traverse the sorted list from subsequent initial indices to subsequent final indices, yielding perfect colalescing in the best case.
- `Branchless`: use a conditional of the form `acc += x_ij_l2 <= r_c_2 ? map(...) : 0` instead of an `if`-condition for pruning out-of range neighbour candidates in hopes of decreasing warp divergence. The compiler might be expected do something equivalent automatically, so this is not expected to have much impact.
""")

    md.extend(generate_overview_table(rows, "Query"))
    md.extend(generate_overview_table(rows, "Construction"))

    md.append("## Details\n")
    md.append(
        "| Benchmark | median time/iter | iters | MdAPE | source |")
    md.append("|---|---:|---:|---:|---|")
    for r in rows:
        md.append(
            f"""| {r['name']} | {r['median']}| {r['total_iters']} | {r['medianAbsolutePercentError']} | `{r['source']}` |"""
        )
    md.append("")

    out_path.write_text("\n".join(md), encoding="utf-8")
