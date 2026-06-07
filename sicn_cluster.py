#!/usr/bin/env python3
"""
sicn_cluster.py — Cluster connectivity, free-C L_a, Si-N network percolation.

WHY:
  In PDC-SiCN, free carbon segregates into graphenic sp² clusters whose
  lateral extent L_a is the direct structural quantity Raman measures (via
  D/G band ratio, ID/IG ∝ 1/L_a).  Saha 2005 reports L_a ≈ 1–2 nm at
  1000–1400 °C, growing with annealing.

  This module:
    1. Builds the bond graph
    2. Identifies free-C clusters (connected sub-graph of C–C bonds, no Si/N)
    3. For each cluster: atoms, R_g, L_a estimate
    4. Identifies Si-N "network" (Si+N atoms with Si–N bonds + Si–Si/Si–C)
    5. Reports percolation: is there a spanning Si–N network?
    6. Reports cluster-size histogram (free-C and overall)

INPUT:  LAMMPS data file or dump trajectory (last frame).
OUTPUT:
  <out>/clusters.txt      — summary text
  <out>/clusters.png      — cluster-size histograms + L_a distribution
  <out>/freeC_clusters.csv — per-cluster: size, R_g, L_a, dominant_element

USAGE:
  python3 sicn_cluster.py SiCN_300K_final.data --out analysis/MRO/
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sicn_common import (
    TYPE_SI, TYPE_C, TYPE_N, TYPE_NAME, TYPE_COLOR,
    load_structure, build_bond_list, density_g_per_cc, composition,
    pbc_displacement,
)

try:
    import networkx as nx
except ImportError:
    print("ERROR: networkx required. Install with: pip install networkx")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Cluster geometry — radius of gyration + L_a with PBC
# ─────────────────────────────────────────────────────────────────────────────
def cluster_geometry(atoms, coords_w, types, L):
    """Compute size, R_g, L_a, composition of a cluster (PBC-correct).

    PBC handling: pick first atom as reference, accumulate unwrapped coords
    by minimum-image traversal of the cluster's spanning tree.
    """
    atoms = list(atoms)
    if len(atoms) < 1:
        return None
    # BFS unwrap
    pos = {atoms[0]: coords_w[atoms[0]].copy()}
    queue = [atoms[0]]
    visited = {atoms[0]}
    # Build local adjacency from list (already a connected component, no graph needed)
    # We just need to walk it; pass-in neighbor list via global graph context
    return pos  # placeholder, real walking done in caller


def cluster_geometry_full(component_atoms, G, coords_w, types, L):
    """PBC-correct geometry of a cluster (component_atoms = set of atom indices).
    Walks the cluster's spanning tree to unwrap positions, then computes
    R_g, L_a, composition.
    """
    atoms = list(component_atoms)
    if len(atoms) < 1:
        return None

    pos = {atoms[0]: coords_w[atoms[0]].copy()}
    queue = [atoms[0]]
    visited = {atoms[0]}
    while queue:
        u = queue.pop(0)
        for v in G.neighbors(u):
            if v not in component_atoms or v in visited:
                continue
            dr = coords_w[v] - coords_w[u]
            dr = pbc_displacement(dr, L)
            pos[v] = pos[u] + dr
            visited.add(v)
            queue.append(v)

    P = np.array([pos[a] for a in atoms])
    com = P.mean(axis=0)
    Rg2 = float(np.mean(np.sum((P - com) ** 2, axis=1)))
    Rg = np.sqrt(max(Rg2, 0.0))

    # L_a estimate: principal-axis spread (2 × √λ_1)
    # using gyration tensor
    diffs = P - com
    G_tensor = diffs.T @ diffs / len(atoms)
    eigs = np.linalg.eigvalsh(G_tensor)
    eigs = np.sort(eigs)[::-1]  # descending
    L_a = 2 * np.sqrt(max(eigs[0], 0))    # extent along longest axis
    # Aspect ratio: λ_1 / λ_3 (large for planar/sheet-like, ≈1 for compact)
    aspect = eigs[0] / max(eigs[2], 1e-9)

    # Composition
    cnt = Counter(types[a] for a in atoms)

    return {
        "size":    len(atoms),
        "R_g":     Rg,
        "L_a":     L_a,
        "aspect":  aspect,
        "n_Si":    cnt[TYPE_SI],
        "n_C":     cnt[TYPE_C],
        "n_N":     cnt[TYPE_N],
        "atoms":   atoms,
        "eigs":    eigs.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build graphs: full, C-only, Si-N-only
# ─────────────────────────────────────────────────────────────────────────────
def build_subgraph(struct, bonds, types_allowed=None, restrict_pair_types=None):
    """Build a subgraph containing only atoms of `types_allowed` and bonds
    whose pair of types is in restrict_pair_types (set of frozensets).
    """
    G = nx.Graph()
    types = struct["types"]
    if types_allowed is None:
        nodes = list(range(struct["n_atoms"]))
    else:
        nodes = [i for i, t in enumerate(types) if t in types_allowed]
    G.add_nodes_from(nodes)
    nodeset = set(nodes)
    for i, j in bonds:
        if i not in nodeset or j not in nodeset:
            continue
        if restrict_pair_types is not None:
            key = frozenset((types[i], types[j]))
            if key not in restrict_pair_types:
                continue
        G.add_edge(i, j)
    return G


# ─────────────────────────────────────────────────────────────────────────────
# Percolation check
# ─────────────────────────────────────────────────────────────────────────────
def check_percolation(component_atoms, G, coords_w, L, thresh_frac=0.85):
    """Heuristic percolation check: does the unwrapped cluster span
    more than `thresh_frac × L` in any axis?
    """
    atoms = list(component_atoms)
    if len(atoms) < 2:
        return False, 0.0
    # Unwrap (same as cluster_geometry_full)
    pos = {atoms[0]: coords_w[atoms[0]].copy()}
    queue = [atoms[0]]
    visited = {atoms[0]}
    while queue:
        u = queue.pop(0)
        for v in G.neighbors(u):
            if v not in component_atoms or v in visited:
                continue
            dr = pbc_displacement(coords_w[v] - coords_w[u], L)
            pos[v] = pos[u] + dr
            visited.add(v)
            queue.append(v)
    P = np.array([pos[a] for a in atoms])
    extent = P.max(axis=0) - P.min(axis=0)
    max_frac = float(np.max(extent / L))
    return max_frac >= thresh_frac, max_frac


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_clusters(struct, freeC_clusters, all_clusters, sin_components, outfile):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.32)

    # Panel A: free-C cluster size histogram
    ax = fig.add_subplot(gs[0, 0])
    sizes = [c["size"] for c in freeC_clusters]
    if sizes:
        max_s = max(sizes)
        bins = np.arange(0.5, max(max_s + 1.5, 11.5))
        ax.hist(sizes, bins=bins, color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Cluster size (# atoms)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"Free-C cluster size distribution ({len(freeC_clusters)} clusters)",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    # Panel B: L_a histogram for free-C clusters (size >= 4 only)
    ax = fig.add_subplot(gs[0, 1])
    La_vals = [c["L_a"] for c in freeC_clusters if c["size"] >= 4]
    if La_vals:
        bins = np.linspace(0, max(max(La_vals), 5), 25)
        ax.hist(La_vals, bins=bins, color="#ff7f0e", edgecolor="black", linewidth=0.5)
        ax.axvline(np.mean(La_vals), color="red", ls="--", lw=1.5,
                   label=f"mean = {np.mean(La_vals):.2f} Å")
        ax.legend(fontsize=9)
    ax.set_xlabel(r"$L_a$ (Å) — principal axis extent", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"Free-C $L_a$ distribution ({len(La_vals)} clusters, size≥4)",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    # Panel C: R_g vs size scatter (mass-fractal check)
    ax = fig.add_subplot(gs[0, 2])
    s_arr = np.array([c["size"] for c in freeC_clusters if c["size"] >= 3])
    R_arr = np.array([c["R_g"] for c in freeC_clusters if c["size"] >= 3])
    if s_arr.size > 0:
        ax.scatter(s_arr, R_arr, s=30, alpha=0.6, color="#1f77b4", edgecolor="black")
        # log-log fit to estimate D_f: R_g ∝ size^(1/D_f) → size ∝ R_g^D_f
        m = (s_arr >= 3) & (R_arr > 0)
        if m.sum() >= 3:
            slope, intercept = np.polyfit(np.log(R_arr[m]), np.log(s_arr[m]), 1)
            Df = slope
            R_fit = np.linspace(R_arr[m].min(), R_arr[m].max(), 50)
            ax.plot(np.exp(intercept) * R_fit ** Df, R_fit,
                    "r--", lw=1.2, label=f"$N \\propto R_g^{{D_f}}$, $D_f$={Df:.2f}")
            ax.legend(fontsize=9)
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Cluster size (# atoms)", fontsize=10)
    ax.set_ylabel("$R_g$ (Å)", fontsize=10)
    ax.set_title("Mass-fractal: $R_g$ vs size  (Saha: $D_f$≈2.5)",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, which="both")

    # Panel D: all-network connected component sizes
    ax = fig.add_subplot(gs[1, 0])
    all_sizes = [c["size"] for c in all_clusters]
    if all_sizes:
        ax.bar(range(len(all_sizes)), sorted(all_sizes, reverse=True),
               color="navy", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Component rank (largest first)", fontsize=10)
    ax.set_ylabel("Component size (# atoms)", fontsize=10)
    ax.set_title(f"All bonds — connected component sizes "
                 f"({len(all_clusters)} components)",
                 fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, axis="y")

    # Panel E: Si-N network component sizes
    ax = fig.add_subplot(gs[1, 1])
    sin_sizes = [c["size"] for c in sin_components]
    if sin_sizes:
        ax.bar(range(min(20, len(sin_sizes))),
               sorted(sin_sizes, reverse=True)[:20],
               color="#d62728", edgecolor="black", linewidth=0.5)
    n_total_sin = sum(1 for t in struct["types"] if t in (TYPE_SI, TYPE_N))
    largest_frac = max(sin_sizes) / n_total_sin if sin_sizes else 0
    ax.set_xlabel("Component rank", fontsize=10)
    ax.set_ylabel("Component size (# atoms)", fontsize=10)
    ax.set_title(f"Si-N network components — largest covers "
                 f"{largest_frac*100:.1f}% of Si+N",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    # Panel F: cluster composition (free-C purity)
    ax = fig.add_subplot(gs[1, 2])
    sizes_pure = [c["size"] for c in freeC_clusters]
    composition_data = {"Pure-C ≥3": 0, "Pure-C =2": 0, "Pure-C =1": 0}
    for c in freeC_clusters:
        if c["size"] >= 3:
            composition_data["Pure-C ≥3"] += 1
        elif c["size"] == 2:
            composition_data["Pure-C =2"] += 1
        else:
            composition_data["Pure-C =1"] += 1
    labels = list(composition_data.keys())
    counts = list(composition_data.values())
    ax.bar(labels, counts, color=["#2ca02c", "#90c890", "#c8e8c8"],
           edgecolor="black", linewidth=0.5)
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Free-C cluster size class", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    x_Si, x_C, x_N = composition(struct)
    fig.suptitle(f"Cluster connectivity & free-C analysis — a-SiCN  "
                 f"(N={struct['n_atoms']}, ρ={density_g_per_cc(struct):.3f} g/cc, "
                 f"Si{x_Si*100:.1f}/C{x_C*100:.1f}/N{x_N*100:.1f})",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.savefig(outfile, dpi=120, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────────────────────────────────────
def save_outputs(freeC_clusters, all_clusters, sin_components, sin_percolates,
                 sin_max_frac, struct, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # clusters.txt — summary
    n_C = int(np.sum(struct["types"] == TYPE_C))
    n_SiN = int(np.sum(np.isin(struct["types"], [TYPE_SI, TYPE_N])))

    free_sizes = [c["size"] for c in freeC_clusters]
    large_clusters = [c for c in freeC_clusters if c["size"] >= 4]
    isolated = sum(1 for c in freeC_clusters if c["size"] == 1)
    dimers = sum(1 for c in freeC_clusters if c["size"] == 2)
    n_C_in_clusters = sum(c["size"] for c in freeC_clusters if c["size"] >= 4)

    L = []
    L.append(f"Cluster connectivity analysis — a-SiCN N={struct['n_atoms']}")
    L.append("=" * 72)
    L.append("")
    L.append("FREE-CARBON ANALYSIS")
    L.append("-" * 72)
    L.append(f"  Total C atoms                 : {n_C}")
    L.append(f"  Free-C clusters total         : {len(freeC_clusters)}")
    L.append(f"  Isolated C (carbidic, size=1) : {isolated}  ({isolated/n_C*100:.1f}%)")
    L.append(f"  C-C dimers (size=2)           : {dimers}")
    L.append(f"  Clusters size ≥ 4 (graphenic) : {len(large_clusters)}")
    L.append(f"  C atoms in clusters ≥ 4       : {n_C_in_clusters}  "
             f"({n_C_in_clusters/n_C*100:.1f}%)")
    if large_clusters:
        L.append(f"  Mean L_a (size ≥ 4)           : "
                 f"{np.mean([c['L_a'] for c in large_clusters]):.3f} Å")
        L.append(f"  Max  L_a                      : "
                 f"{max(c['L_a'] for c in large_clusters):.3f} Å")
        L.append(f"  Largest cluster               : "
                 f"{max(free_sizes)} atoms")
    L.append("")

    L.append("SI-N NETWORK PERCOLATION")
    L.append("-" * 72)
    L.append(f"  Total Si+N atoms              : {n_SiN}")
    L.append(f"  Si-N components               : {len(sin_components)}")
    if sin_components:
        largest = max(sin_components, key=lambda c: c["size"])
        L.append(f"  Largest Si-N component        : {largest['size']} atoms  "
                 f"({largest['size']/n_SiN*100:.1f}% of Si+N)")
        L.append(f"  Si-N percolates (spans box)?  : "
                 f"{'YES ✓' if sin_percolates else 'NO ✗'}  "
                 f"(max axial fraction = {sin_max_frac:.2f})")
    L.append("")

    L.append("ALL-BONDS NETWORK")
    L.append("-" * 72)
    L.append(f"  Total components              : {len(all_clusters)}")
    if all_clusters:
        all_sorted = sorted(all_clusters, key=lambda c: -c["size"])
        L.append(f"  Largest (% of all atoms)      : "
                 f"{all_sorted[0]['size']}  "
                 f"({all_sorted[0]['size']/struct['n_atoms']*100:.1f}%)")
    (out_dir / "clusters.txt").write_text("\n".join(L))

    # freeC_clusters.csv — per-cluster details
    lines = ["size,n_C,R_g(Å),L_a(Å),aspect"]
    for c in sorted(freeC_clusters, key=lambda x: -x["size"]):
        lines.append(f"{c['size']},{c['n_C']},{c['R_g']:.3f},"
                     f"{c['L_a']:.3f},{c['aspect']:.3f}")
    (out_dir / "freeC_clusters.csv").write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run(input_file, out_dir="analysis/MRO"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cluster] Loading {input_file} ...")
    struct = load_structure(input_file)
    print(f"[cluster] {struct['n_atoms']} atoms, ρ = {density_g_per_cc(struct):.3f} g/cc")

    bonds, coords_w, L = build_bond_list(struct)
    print(f"[cluster] {len(bonds)} bonds")

    # Build subgraphs
    # 1. Free-C: only C atoms, only C-C bonds
    cc_pair_set = {frozenset((TYPE_C, TYPE_C))}
    G_C = build_subgraph(struct, bonds, types_allowed={TYPE_C},
                         restrict_pair_types=cc_pair_set)
    # 2. Si-N network: Si+N atoms, all bonds among them
    G_SiN = build_subgraph(struct, bonds, types_allowed={TYPE_SI, TYPE_N})
    # 3. All bonds, all atoms
    G_all = nx.Graph()
    G_all.add_nodes_from(range(struct["n_atoms"]))
    G_all.add_edges_from(bonds)

    # Connected components
    print(f"[cluster] Free-C components ...")
    cc_components = list(nx.connected_components(G_C))
    freeC_clusters = [cluster_geometry_full(c, G_C, coords_w, struct["types"], L)
                      for c in cc_components]
    freeC_clusters = [c for c in freeC_clusters if c is not None]

    print(f"[cluster] Si-N components ...")
    sin_components_raw = list(nx.connected_components(G_SiN))
    sin_components = [cluster_geometry_full(c, G_SiN, coords_w, struct["types"], L)
                      for c in sin_components_raw]
    sin_components = [c for c in sin_components if c is not None]

    # Check percolation of largest Si-N component
    sin_percolates = False
    sin_max_frac = 0.0
    if sin_components:
        biggest_raw = max(sin_components_raw, key=len)
        sin_percolates, sin_max_frac = check_percolation(
            biggest_raw, G_SiN, coords_w, L)

    print(f"[cluster] All-bonds components ...")
    all_comps_raw = list(nx.connected_components(G_all))
    all_clusters = [cluster_geometry_full(c, G_all, coords_w, struct["types"], L)
                    for c in all_comps_raw]
    all_clusters = [c for c in all_clusters if c is not None]

    # Plot & save
    plot_clusters(struct, freeC_clusters, all_clusters, sin_components,
                  out_dir / "clusters.png")
    save_outputs(freeC_clusters, all_clusters, sin_components,
                 sin_percolates, sin_max_frac, struct, out_dir)
    print(f"[cluster] → Saved clusters.png, clusters.txt, freeC_clusters.csv")

    # Quick CLI report
    n_C = int(np.sum(struct["types"] == TYPE_C))
    n_iso = sum(1 for c in freeC_clusters if c["size"] == 1)
    large = [c for c in freeC_clusters if c["size"] >= 4]
    print("")
    print(f"  Free-C: {n_iso}/{n_C} isolated, "
          f"{len(large)} clusters ≥ 4 atoms")
    if large:
        print(f"  Mean L_a (size≥4) = {np.mean([c['L_a'] for c in large]):.2f} Å, "
              f"max L_a = {max(c['L_a'] for c in large):.2f} Å")
    print(f"  Si-N percolation: {'YES' if sin_percolates else 'NO'} "
          f"(axial fraction {sin_max_frac:.2f})")
    return freeC_clusters, all_clusters, sin_components, struct


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="LAMMPS data file or dump trajectory")
    ap.add_argument("--out", default="analysis/MRO")
    a = ap.parse_args()
    run(a.input, a.out)
