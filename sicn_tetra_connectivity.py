#!/usr/bin/env python3
"""
sicn_tetra_connectivity.py — Si tetrahedra connectivity analysis (IRO)

THEORY (Intermediate-Range Order, IRO):
  Si is sp³-coordinated to 4 first neighbors forming a tetrahedron.
  Two Si tetrahedra can connect via shared atoms in 3 distinct ways:

    Corner sharing: 1 atom shared (vertex)    → most common, "flexible joint"
    Edge sharing  : 2 atoms shared (edge)     → rigid, denser
    Face sharing  : 3 atoms shared (face)     → very rigid, very dense

  Crystalline references:
    α-Si₃N₄    : 100% corner-sharing
    β-SiC       : 100% corner-sharing (diamond-type)
    Stishovite (SiO₂): edge-sharing (high-pressure)
    Face-sharing: rare, only under extreme compression

  Geometric signature (Si-Si distance between two tetrahedra centers):
    Corner shared: d_SiSi ≈ 2 × d_SiX × sin(θ/2)
                  For Si-N (1.75 Å) at θ=120° → d_SiSi ≈ 3.03 Å
                  For Si-C (1.87 Å) at θ=120° → d_SiSi ≈ 3.24 Å
    Edge shared  : ~2.5–2.7 Å (closer)
    Face shared  : ~2.0–2.3 Å (very close)

ALGORITHM:
  For each pair of Si atoms (i, j) within d_Si-Si < 4 Å:
    1. Find shared first neighbors (atoms bonded to BOTH i and j)
    2. Classify by count:
       n_shared = 1 → corner
       n_shared = 2 → edge
       n_shared = 3 → face
       n_shared = 0 → not connected via shared atoms (separate clusters)

  Also classify by what atom is shared (Si, C, or N).

OUTPUT:
  tetra_conn_summary.png      — bar chart + Si-Si distance distribution
  tetra_conn_summary.txt      — counts + percentages
  tetra_conn_pairs.csv        — every Si-Si pair classification

USAGE:
  python3 sicn_tetra_connectivity.py SiCN_300K_final.data --out analysis/11_tetra_conn/
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sicn_common import (
    TYPE_SI, TYPE_C, TYPE_N, TYPE_NAME,
    load_structure, build_bond_list, pbc_distance,
    density_g_per_cc, composition,
)


def compute_si_tetra_connectivity(struct, d_si_si_max=4.0):
    """For every Si-Si pair within d_si_si_max, determine connectivity type.

    Args:
        d_si_si_max : max Si-Si distance to consider as 'potentially connected'.
                      (4 Å covers corner, edge, face sharing comfortably).

    Returns:
        results : list of dicts with per-pair info.
        stats   : aggregated counts.
    """
    types = struct["types"]
    coords = struct["coords"]
    L = struct["box"][:, 1] - struct["box"][:, 0]

    bonds, _, _ = build_bond_list(struct)
    neighbors = {i: set() for i in range(struct["n_atoms"])}
    for i, j in bonds:
        neighbors[i].add(j)
        neighbors[j].add(i)

    # All Si atom indices
    si_idx = np.where(types == TYPE_SI)[0]
    n_si = len(si_idx)
    print(f"[tetra_conn]   {n_si} Si atoms")

    # For efficiency, only check Si-Si pairs within d_si_si_max
    # Use simple O(n²) since n_si ~3500 is fine
    print(f"[tetra_conn]   Computing Si-Si pair connectivity ...")
    results = []

    # Compute all pairwise Si-Si distances (vectorized chunks)
    si_coords = coords[si_idx]
    # Loop in chunks to limit memory; n=3500 fits direct calc
    for k, i in enumerate(si_idx):
        d_arr = pbc_distance(si_coords, coords[i:i+1], L)  # (n_si,)
        # Find Si neighbors within d_si_si_max
        within = np.where((d_arr > 1e-6) & (d_arr < d_si_si_max))[0]
        for w_idx in within:
            j = si_idx[w_idx]
            if j <= i:  # avoid double-counting (i, j) and (j, i)
                continue
            d_ij = d_arr[w_idx]
            # Find shared 1st neighbors (atoms bonded to BOTH i and j)
            shared = neighbors[i] & neighbors[j]
            n_shared = len(shared)
            if n_shared == 0:
                continue  # Si-Si pair but not connected via shared bridging atom
            # Classify shared-atom types
            n_shared_Si = sum(1 for s in shared if types[s] == TYPE_SI)
            n_shared_C  = sum(1 for s in shared if types[s] == TYPE_C)
            n_shared_N  = sum(1 for s in shared if types[s] == TYPE_N)

            # Connectivity type
            if   n_shared == 1: conn = "corner"
            elif n_shared == 2: conn = "edge"
            elif n_shared == 3: conn = "face"
            else:               conn = f"higher_{n_shared}"

            results.append({
                "i": i, "j": j, "d_SiSi": d_ij,
                "n_shared": n_shared, "conn": conn,
                "n_shared_Si": n_shared_Si,
                "n_shared_C":  n_shared_C,
                "n_shared_N":  n_shared_N,
            })

    # Aggregate stats
    stats = {
        "n_si": n_si,
        "n_pairs_total": len(results),
        "by_conn": {},
        "by_shared_type": {},
    }
    for cls in ["corner", "edge", "face"]:
        items = [r for r in results if r["conn"] == cls]
        stats["by_conn"][cls] = len(items)
    # Higher (n_shared ≥ 4) — should be 0 normally
    higher = [r for r in results if r["n_shared"] >= 4]
    stats["by_conn"]["higher (≥4)"] = len(higher)

    # By shared-atom species (for corner sharing only, dominant case)
    corner = [r for r in results if r["conn"] == "corner"]
    stats["corner_by_bridge"] = {
        "via_N": sum(1 for r in corner if r["n_shared_N"] == 1),
        "via_C": sum(1 for r in corner if r["n_shared_C"] == 1),
        "via_Si": sum(1 for r in corner if r["n_shared_Si"] == 1),
    }
    return results, stats


def plot_connectivity(results, stats, struct, outfile):
    """4-panel: connectivity counts + Si-Si distance distribution + bridge-atom species."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel 1: connectivity type counts
    ax = axes[0, 0]
    conn_types = ["corner", "edge", "face", "higher (≥4)"]
    counts = [stats["by_conn"].get(c, 0) for c in conn_types]
    n_total = sum(counts)
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#8c564b"]
    bars = ax.bar(conn_types, counts, color=colors, edgecolor="black", linewidth=0.5)
    for b, c in zip(bars, counts):
        if c > 0:
            pct = c / n_total * 100 if n_total > 0 else 0
            ax.text(b.get_x() + b.get_width()/2, c + n_total * 0.01,
                    f"{c}\n({pct:.1f}%)",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of Si-Si pairs", fontsize=11)
    ax.set_title(f"Si tetrahedra connectivity types  (total = {n_total})",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    # Panel 2: Si-Si distance distribution by connectivity type
    ax = axes[0, 1]
    bins = np.linspace(1.5, 4.0, 50)
    for cls, color in zip(["corner", "edge", "face"],
                           ["#1f77b4", "#ff7f0e", "#d62728"]):
        d_vals = [r["d_SiSi"] for r in results if r["conn"] == cls]
        if len(d_vals) > 0:
            ax.hist(d_vals, bins=bins, alpha=0.55, color=color,
                    label=f"{cls} (n={len(d_vals)})",
                    edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Si–Si distance (Å)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Si–Si distance by connectivity type",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 3: Corner-sharing bridge-atom species
    ax = axes[1, 0]
    bridge = stats["corner_by_bridge"]
    species = ["via_N", "via_C", "via_Si"]
    cnts = [bridge[s] for s in species]
    n_corner = sum(cnts)
    colors2 = ["#d62728", "#2ca02c", "#1f77b4"]
    bars = ax.bar(species, cnts, color=colors2, edgecolor="black", linewidth=0.5)
    for b, c in zip(bars, cnts):
        if c > 0:
            pct = c / n_corner * 100 if n_corner > 0 else 0
            ax.text(b.get_x() + b.get_width()/2, c + n_corner * 0.01,
                    f"{c}\n({pct:.1f}%)",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of Si-Si pairs", fontsize=11)
    ax.set_title(f"Corner-sharing: bridging atom species  (total = {n_corner})",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    # Panel 4: Summary text
    ax = axes[1, 1]
    ax.axis("off")
    lines = ["Si tetrahedra connectivity (IRO)", "─" * 38]
    lines.append(f"  Si atoms          : {stats['n_si']}")
    lines.append(f"  Connected Si-Si   : {stats['n_pairs_total']}")
    lines.append("")
    lines.append("Connectivity types:")
    for cls in conn_types:
        c = stats["by_conn"].get(cls, 0)
        pct = c / n_total * 100 if n_total > 0 else 0
        lines.append(f"  {cls:<12}: {c:>5} ({pct:>5.1f}%)")
    lines.append("")
    lines.append("Corner-sharing (dominant):")
    for s in species:
        c = bridge[s]
        pct = c / n_corner * 100 if n_corner > 0 else 0
        lines.append(f"  {s:<12}: {c:>5} ({pct:>5.1f}%)")
    lines.append("")
    lines.append("Reference (crystalline):")
    lines.append("  α-Si₃N₄, β-SiC: 100% corner")
    lines.append("  Stishovite SiO₂: edge")
    lines.append("  Face: only extreme compression")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=10, verticalalignment="top", family="monospace")

    x = composition(struct)
    fig.suptitle(f"Si Tetrahedra Connectivity (IRO analysis) — a-SiCN  "
                 f"(N = {struct['n_atoms']}, ρ = {density_g_per_cc(struct):.3f} g/cc, "
                 f"Si{x[0]*100:.1f}/C{x[1]*100:.1f}/N{x[2]*100:.1f})",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(outfile, dpi=120, bbox_inches="tight")
    plt.close()


def save_summary(results, stats, struct, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    L = []
    L.append("Si tetrahedra connectivity (Intermediate-Range Order analysis)")
    L.append("=" * 70)
    L.append(f"  System: {struct['n_atoms']} atoms ({stats['n_si']} Si)")
    L.append(f"  Total Si-Si pairs (connected via shared atoms): {stats['n_pairs_total']}")
    L.append("")
    L.append("[Connectivity type distribution]")
    L.append("-" * 70)
    n_total = stats["n_pairs_total"]
    for cls in ["corner", "edge", "face", "higher (≥4)"]:
        cnt = stats["by_conn"].get(cls, 0)
        pct = cnt / n_total * 100 if n_total > 0 else 0
        d_vals = [r["d_SiSi"] for r in results if r["conn"] == cls]
        if d_vals:
            mean_d = np.mean(d_vals)
            L.append(f"  {cls:<14}: {cnt:>6}  ({pct:>5.1f}%)  "
                      f"avg Si-Si = {mean_d:.3f} Å")
        else:
            L.append(f"  {cls:<14}: {cnt:>6}  ({pct:>5.1f}%)")
    L.append("")
    L.append("[Corner-sharing: bridging atom species]")
    L.append("-" * 70)
    bridge = stats["corner_by_bridge"]
    n_corner = sum(bridge.values())
    for s in ["via_N", "via_C", "via_Si"]:
        cnt = bridge[s]
        pct = cnt / n_corner * 100 if n_corner > 0 else 0
        L.append(f"  {s:<14}: {cnt:>6}  ({pct:>5.1f}%)")
    L.append("")
    L.append("─ Physical interpretation ─")
    L.append("  Corner sharing  → flexible Si network (typical for a-SiN, SiC)")
    L.append("  Edge sharing    → rigid, denser packing (rare in amorphous covalent)")
    L.append("  Face sharing    → very rare, highly compressed (extreme defect)")
    L.append("")
    L.append("Reference crystals:")
    L.append("  α-Si₃N₄  : 100% corner via N")
    L.append("  β-SiC     : 100% corner via C")
    L.append("  Stishovite SiO₂ (high-pressure) : edge-sharing")
    (out_dir / "tetra_conn_summary.txt").write_text("\n".join(L))

    # CSV
    csv = ["i,j,d_SiSi,n_shared,conn,n_shared_Si,n_shared_C,n_shared_N"]
    for r in results:
        csv.append(f"{r['i']},{r['j']},{r['d_SiSi']:.4f},{r['n_shared']},"
                    f"{r['conn']},{r['n_shared_Si']},{r['n_shared_C']},"
                    f"{r['n_shared_N']}")
    (out_dir / "tetra_conn_pairs.csv").write_text("\n".join(csv))


def run(input_file, out_dir="analysis/11_tetra_conn", d_si_si_max=4.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[tetra_conn] Loading {input_file} ...")
    struct = load_structure(input_file)
    print(f"[tetra_conn] {struct['n_atoms']} atoms")
    results, stats = compute_si_tetra_connectivity(struct, d_si_si_max=d_si_si_max)

    # Quick summary
    n_total = stats["n_pairs_total"]
    if n_total > 0:
        pct_corner = stats["by_conn"].get("corner", 0) / n_total * 100
        pct_edge = stats["by_conn"].get("edge", 0) / n_total * 100
        pct_face = stats["by_conn"].get("face", 0) / n_total * 100
        print(f"[tetra_conn]   Total connected Si-Si pairs: {n_total}")
        print(f"[tetra_conn]   Corner: {stats['by_conn'].get('corner', 0)} "
              f"({pct_corner:.1f}%)")
        print(f"[tetra_conn]   Edge  : {stats['by_conn'].get('edge', 0)} "
              f"({pct_edge:.1f}%)")
        print(f"[tetra_conn]   Face  : {stats['by_conn'].get('face', 0)} "
              f"({pct_face:.1f}%)")
        bridge = stats["corner_by_bridge"]
        n_c = sum(bridge.values())
        if n_c > 0:
            print(f"[tetra_conn]   Corner via N: {bridge['via_N']} "
                  f"({bridge['via_N']/n_c*100:.1f}%)")
            print(f"[tetra_conn]   Corner via C: {bridge['via_C']} "
                  f"({bridge['via_C']/n_c*100:.1f}%)")
            print(f"[tetra_conn]   Corner via Si: {bridge['via_Si']} "
                  f"({bridge['via_Si']/n_c*100:.1f}%)")

    plot_connectivity(results, stats, struct, out_dir / "tetra_conn_summary.png")
    save_summary(results, stats, struct, out_dir)
    print(f"[tetra_conn] → Saved tetra_conn_summary.png, .txt, tetra_conn_pairs.csv")
    return results, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="LAMMPS data file")
    ap.add_argument("--out", default="analysis/11_tetra_conn",
                    help="output directory")
    ap.add_argument("--d_max", type=float, default=4.0,
                    help="max Si-Si distance to check (Å)")
    a = ap.parse_args()
    run(a.input, a.out, d_si_si_max=a.d_max)
