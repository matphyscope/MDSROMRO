#!/usr/bin/env python3
"""
sicn_rings.py — Ring statistics for amorphous SiCN networks (King's criterion).

WHY:
  Rings (closed cycles in the bond network) are the most direct fingerprint
  of medium-range topology in amorphous covalent networks:
    - 6-membered rings → graphenic/aromatic clusters in free C, or Si3N4-like
    - 5- and 7-membered → "frustration" indicator
    - 3-, 4-rings      → strained over-coordination defects
  See: Wright (1994), Le Roux & Jund (2010), Yuan & Cormack (2002).

ALGORITHM:
  King's shortest-path criterion: for each bond (i,j), find the shortest
  cycle through (i,j) by removing the edge and computing shortest path
  i→j. Ring size = path length + 1.
  A given ring of size n is found n times (once per edge), so we divide
  by n at the end to get the actual ring count.

INPUT:
  LAMMPS data file or dump trajectory.

OUTPUT:
  <out>/rings.txt              — table: size, count, count/atom, composition
  <out>/rings.png              — 3-panel plot
  <out>/ring_composition.txt   — Si/C/N breakdown per ring size

USAGE:
  python3 sicn_rings.py SiCN_300K_final.data --out analysis/MRO/
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sicn_common import (
    TYPE_SI, TYPE_C, TYPE_N, TYPE_NAME, TYPE_COLOR,
    load_structure, build_bond_list, density_g_per_cc,
)

try:
    import networkx as nx
except ImportError:
    print("ERROR: networkx required. Install with: pip install networkx")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Ring detection — King's criterion
# ─────────────────────────────────────────────────────────────────────────────
def find_kings_rings(G, max_size=12, verbose=True):
    """For each edge (i,j) in G, find the shortest ring through it.

    Returns:
        rings  : list of dicts, one per (UNIQUE) ring found.
                 keys: 'size', 'atoms' (sorted tuple), 'count_via_edges' (n hits)
    """
    edges = list(G.edges())
    if verbose:
        print(f"  [rings] searching King's rings over {len(edges)} edges "
              f"(max size {max_size}) ...")

    # canonical ring → count
    seen = {}   # frozenset(atoms) -> dict
    n_done = 0
    n_print = max(1, len(edges) // 10)

    for u, v in edges:
        # Temporarily remove edge
        G.remove_edge(u, v)
        try:
            path = nx.shortest_path(G, u, v)
            size = len(path)  # node count == ring size when including closure
            if 3 <= size <= max_size:
                atoms = frozenset(path)
                if atoms in seen:
                    seen[atoms]["count_via_edges"] += 1
                else:
                    seen[atoms] = {
                        "size": size,
                        "atoms": tuple(sorted(path)),
                        "path": path,
                        "count_via_edges": 1,
                    }
        except nx.NetworkXNoPath:
            pass
        # Restore edge
        G.add_edge(u, v)

        n_done += 1
        if verbose and n_done % n_print == 0:
            print(f"    {100*n_done/len(edges):.0f}% — {len(seen)} unique rings so far")

    rings = list(seen.values())
    if verbose:
        print(f"  [rings] found {len(rings)} unique rings (sizes 3–{max_size})")
    return rings


def classify_ring(ring, types):
    """Count Si, C, N atoms in ring; return dict + label."""
    cnt = {TYPE_SI: 0, TYPE_C: 0, TYPE_N: 0}
    for a in ring["atoms"]:
        cnt[types[a]] += 1
    nSi, nC, nN = cnt[TYPE_SI], cnt[TYPE_C], cnt[TYPE_N]
    # Heuristic classification
    if nC == ring["size"]:
        label = "pure-C"          # free carbon ring (graphenic)
    elif nC == 0:
        label = "Si-N"            # silicon nitride ring
    elif nSi == 0:
        label = "C-N"             # rare (Tersoff C-N is repulsive)
    elif nN == 0:
        label = "Si-C"            # silicon carbide ring
    else:
        label = "mixed-SiCN"      # mixed
    return {"nSi": nSi, "nC": nC, "nN": nN, "label": label}


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────
def ring_statistics(rings, types, n_atoms, max_size=12):
    """Aggregate ring counts by size and by composition."""
    by_size = Counter()
    by_size_class = defaultdict(Counter)
    for r in rings:
        cls = classify_ring(r, types)
        by_size[r["size"]] += 1
        by_size_class[r["size"]][cls["label"]] += 1

    rows = []
    for n in range(3, max_size + 1):
        total = by_size[n]
        per_atom = total / n_atoms
        composition = dict(by_size_class[n])
        rows.append({
            "size": n,
            "count": total,
            "count_per_atom": per_atom,
            "composition": composition,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_rings(rows, struct, outfile, max_size=12):
    """3-panel plot: total count, count/atom, composition stack."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sizes = [r["size"] for r in rows]
    counts = [r["count"] for r in rows]
    per_atom = [r["count_per_atom"] for r in rows]

    # Panel 1: total count
    ax = axes[0]
    bars = ax.bar(sizes, counts, color="#4a7ab8", edgecolor="black", linewidth=0.5)
    for b, c in zip(bars, counts):
        if c > 0:
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    str(c), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Ring size", fontsize=11)
    ax.set_ylabel("Number of rings", fontsize=11)
    ax.set_title("Ring count by size (King's)", fontsize=12, fontweight="bold")
    ax.set_xticks(sizes)
    ax.grid(alpha=0.3, axis="y")

    # Panel 2: count per atom
    ax = axes[1]
    ax.bar(sizes, per_atom, color="#d65f5f", edgecolor="black", linewidth=0.5)
    for s, p in zip(sizes, per_atom):
        if p > 0:
            ax.text(s, p, f"{p:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Ring size", fontsize=11)
    ax.set_ylabel("Rings per atom", fontsize=11)
    ax.set_title("Normalized ring count", fontsize=12, fontweight="bold")
    ax.set_xticks(sizes)
    ax.grid(alpha=0.3, axis="y")

    # Panel 3: composition stack
    ax = axes[2]
    classes = ["pure-C", "Si-C", "Si-N", "C-N", "mixed-SiCN"]
    color_map = {
        "pure-C":      "#2ca02c",  # green - free carbon ring
        "Si-C":        "#1f77b4",  # blue
        "Si-N":        "#d62728",  # red
        "C-N":         "#9467bd",  # purple
        "mixed-SiCN":  "#ff7f0e",  # orange
    }
    bottoms = np.zeros(len(sizes))
    for cls in classes:
        vals = np.array([r["composition"].get(cls, 0) for r in rows])
        if vals.sum() == 0:
            continue
        ax.bar(sizes, vals, bottom=bottoms, color=color_map[cls],
               label=cls, edgecolor="black", linewidth=0.4)
        bottoms += vals
    ax.set_xlabel("Ring size", fontsize=11)
    ax.set_ylabel("Number of rings", fontsize=11)
    ax.set_title("Composition breakdown per ring size", fontsize=12, fontweight="bold")
    ax.set_xticks(sizes)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Ring statistics — a-SiCN  (N = {struct['n_atoms']}, "
                 f"ρ = {density_g_per_cc(struct):.3f} g/cc)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(outfile, dpi=120, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────────────────────────────────────
def save_outputs(rows, struct, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # rings.txt
    L = []
    L.append(f"Ring statistics (King's criterion) — a-SiCN N={struct['n_atoms']}")
    L.append("=" * 72)
    L.append(f"{'size':<6}{'count':>10}{'per atom':>15}    composition")
    L.append("-" * 72)
    total = sum(r["count"] for r in rows)
    for r in rows:
        comp = ", ".join(f"{k}:{v}" for k, v in r["composition"].items() if v > 0)
        L.append(f"{r['size']:<6}{r['count']:>10}{r['count_per_atom']:>15.4f}    {comp or '-'}")
    L.append("-" * 72)
    L.append(f"{'TOTAL':<6}{total:>10}")
    L.append("")
    L.append("Notes:")
    L.append("  King's criterion: shortest cycle through each bond.")
    L.append("  pure-C   = ring composed entirely of carbon (graphenic/free-C)")
    L.append("  Si-N     = ring composed of Si+N only (no carbon)")
    L.append("  Si-C     = ring composed of Si+C only (no nitrogen)")
    L.append("  mixed    = ring containing all three elements")
    (out_dir / "rings.txt").write_text("\n".join(L))

    # ring_composition.txt — table by class
    L2 = []
    L2.append(f"Ring composition breakdown — a-SiCN N={struct['n_atoms']}")
    L2.append("=" * 80)
    classes = ["pure-C", "Si-N", "Si-C", "C-N", "mixed-SiCN"]
    head = f"{'size':<6}" + "".join(f"{c:>14}" for c in classes) + f"{'total':>10}"
    L2.append(head)
    L2.append("-" * 80)
    for r in rows:
        line = f"{r['size']:<6}"
        for c in classes:
            line += f"{r['composition'].get(c, 0):>14}"
        line += f"{r['count']:>10}"
        L2.append(line)
    (out_dir / "ring_composition.txt").write_text("\n".join(L2))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run(input_file, out_dir="analysis/MRO", max_size=12, verbose=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[rings] Loading {input_file} ...")
    struct = load_structure(input_file)
    if verbose:
        print(f"[rings] {struct['n_atoms']} atoms, ρ = {density_g_per_cc(struct):.3f} g/cc")

    bonds, _, _ = build_bond_list(struct)
    if verbose:
        print(f"[rings] {len(bonds)} bonds (avg CN = {2*len(bonds)/struct['n_atoms']:.2f})")

    G = nx.Graph()
    G.add_nodes_from(range(struct["n_atoms"]))
    G.add_edges_from(bonds)

    rings = find_kings_rings(G, max_size=max_size, verbose=verbose)
    rows = ring_statistics(rings, struct["types"], struct["n_atoms"], max_size)

    plot_rings(rows, struct, out_dir / "rings.png", max_size)
    save_outputs(rows, struct, out_dir)
    if verbose:
        print(f"[rings] → Saved {out_dir/'rings.png'}, rings.txt, ring_composition.txt")

    # Print summary
    if verbose:
        print("")
        print(f"{'size':<6}{'count':>10}{'/atom':>10}    dominant class")
        print("-" * 60)
        for r in rows:
            if r["count"] == 0:
                continue
            dom = max(r["composition"].items(), key=lambda x: x[1]) if r["composition"] else ("-", 0)
            print(f"{r['size']:<6}{r['count']:>10}{r['count_per_atom']:>10.4f}    "
                  f"{dom[0]} ({dom[1]}/{r['count']})")
    return rows, struct


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="LAMMPS data file or dump trajectory")
    ap.add_argument("--out", default="analysis/MRO", help="output directory")
    ap.add_argument("--max_size", type=int, default=12, help="max ring size to detect")
    a = ap.parse_args()
    run(a.input, a.out, max_size=a.max_size)
