#!/usr/bin/env python3
"""
sicn_dihedral.py — Dihedral (torsion) angle distribution

THEORY:
  For a 4-body chain i-j-k-l, the dihedral angle φ is the angle between
  the two planes (i,j,k) and (j,k,l). φ ∈ [0°, 180°] (using absolute value).

  Conformational interpretations:
    φ ≈ 0°    : cis / eclipsed
    φ ≈ 60°   : gauche
    φ ≈ 180°  : trans / anti

  In amorphous covalent networks, dihedral distribution reveals IRO
  (intermediate-range order) — flat distribution = no IRO, sharp peaks = order.

KEY QUADRUPLETS for a-SiCN:
  - Si-N-Si-N : N tetrahedra puckering / ring shape
  - Si-C-Si-C : carbidic SiC backbone
  - C-C-C-C  : sp² graphene-like dihedrals (180° for flat 6-ring)
  - Si-N-Si-C : mixed-network IRO

USAGE:
  python3 sicn_dihedral.py SiCN_300K_final.data --out analysis/07_dihedral/
"""
from __future__ import annotations
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sicn_common import (
    TYPE_SI, TYPE_C, TYPE_N, TYPE_NAME,
    load_structure, build_bond_list, pbc_displacement,
    density_g_per_cc, composition,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dihedral angle calculation (PBC-aware)
# ─────────────────────────────────────────────────────────────────────────────
def dihedral_angle(p1, p2, p3, p4, box_L):
    """Compute dihedral angle (deg) for atoms p1-p2-p3-p4 (PBC-corrected).

    φ = angle between the (p1,p2,p3) plane and (p2,p3,p4) plane.

    Returns:
        φ in degrees, [0, 180].
    """
    # Bond vectors, minimum-image
    b1 = pbc_displacement(p2 - p1, box_L)
    b2 = pbc_displacement(p3 - p2, box_L)
    b3 = pbc_displacement(p4 - p3, box_L)
    # Normals to the two planes
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-9 or n2_norm < 1e-9:
        return float("nan")
    cos_phi = np.dot(n1, n2) / (n1_norm * n2_norm)
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi = np.degrees(np.arccos(abs(cos_phi)))  # → [0°, 90°] always positive
    # Actually we want [0°, 180°] — use unsigned cos
    phi = np.degrees(np.arccos(cos_phi))
    return phi


# ─────────────────────────────────────────────────────────────────────────────
# Enumerate dihedrals i-j-k-l where j, k are bonded
# ─────────────────────────────────────────────────────────────────────────────
def enumerate_dihedrals(struct):
    """Find all unique i-j-k-l quadruplets where:
      - i bonded to j, j bonded to k, k bonded to l
      - i ≠ k, j ≠ l (no 3-atom rings forming loop)

    Returns:
        list of (i, j, k, l) tuples (unique by canonical ordering).
    """
    types = struct["types"]
    n_atoms = struct["n_atoms"]
    bonds, _, _ = build_bond_list(struct)

    # Per-atom neighbor list
    neighbors = {i: [] for i in range(n_atoms)}
    for i, j in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)

    # Build dihedrals: for each bond (j, k), pair i ∈ N(j)\{k} with l ∈ N(k)\{j}
    dihedrals = []
    for j, k in bonds:
        for i in neighbors[j]:
            if i == k: continue
            for l in neighbors[k]:
                if l == j or l == i: continue
                # Canonical ordering: (j, k) is the central bond
                # Ensure each dihedral is unique by enforcing i < l canonical when j > k
                if j > k:
                    quad = (l, k, j, i)
                else:
                    quad = (i, j, k, l)
                dihedrals.append(quad)
    # Deduplicate (set requires hashable)
    return list(set(dihedrals))


def classify_dihedral(quad, types):
    """Return label like 'Si-N-Si-N' for a dihedral, with canonical ordering
    (smaller end label first to avoid duplicate symmetric labels)."""
    t = [TYPE_NAME[types[a]] for a in quad]
    label = "-".join(t)
    # Symmetric counterpart: reverse
    rev = "-".join(reversed(t))
    return min(label, rev)


# ─────────────────────────────────────────────────────────────────────────────
# Main computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_dihedrals(struct, sample_max=None):
    """Compute all dihedrals + classify by type.

    Args:
        sample_max : if given, randomly subsample to this many dihedrals
                     for speed on very large systems.

    Returns:
        results : dict {label: list of angles (deg)}
        total_count : int
    """
    types = struct["types"]
    coords = struct["coords"]
    box_L = struct["box"][:, 1] - struct["box"][:, 0]

    print(f"[dihedral] Enumerating quadruplets ...")
    quads = enumerate_dihedrals(struct)
    print(f"[dihedral]   Found {len(quads)} dihedrals")

    if sample_max is not None and len(quads) > sample_max:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(quads), size=sample_max, replace=False)
        quads = [quads[i] for i in idx]
        print(f"[dihedral]   Subsampled to {len(quads)} for speed")

    results = {}
    for quad in quads:
        i, j, k, l = quad
        phi = dihedral_angle(coords[i], coords[j], coords[k], coords[l], box_L)
        if np.isnan(phi):
            continue
        label = classify_dihedral(quad, types)
        results.setdefault(label, []).append(phi)
    return results, len(quads)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_dihedrals(results, struct, outfile, top_n=8):
    """Histograms of top-N most populated dihedral types."""
    # Sort by population
    sorted_items = sorted(results.items(), key=lambda kv: -len(kv[1]))
    top = sorted_items[:top_n]
    n_plots = len(top)

    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    bins = np.linspace(0, 180, 37)  # 5° bins

    for idx, (label, angles) in enumerate(top):
        ax = axes[idx // n_cols, idx % n_cols]
        ax.hist(angles, bins=bins, color="steelblue", edgecolor="black",
                linewidth=0.4, alpha=0.7)
        ax.axvline(60,  color="green",  ls=":", lw=0.7, alpha=0.5)
        ax.axvline(180, color="orange", ls=":", lw=0.7, alpha=0.5)
        ax.set_title(f"{label}  (n={len(angles)})", fontsize=10, fontweight="bold")
        ax.set_xlabel("φ (deg)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)

    # Hide extra subplots
    for idx in range(n_plots, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    x = composition(struct)
    fig.suptitle(f"Dihedral (torsion) angle distribution — a-SiCN  "
                 f"(N = {struct['n_atoms']}, Top {n_plots} populated types)",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(outfile, dpi=120, bbox_inches="tight")
    plt.close()


def save_dihedral_summary(results, total, struct, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    L = []
    L.append("Dihedral (torsion) angle distribution")
    L.append("=" * 70)
    L.append(f"  System: {struct['n_atoms']} atoms")
    L.append(f"  Total unique dihedrals enumerated: {total}")
    L.append("")
    L.append(f"{'Type':<18}{'Count':>10}{'Mean (deg)':>14}{'Std':>10}{'Peak (deg)':>14}")
    L.append("-" * 70)
    bins = np.linspace(0, 180, 37)
    sorted_items = sorted(results.items(), key=lambda kv: -len(kv[1]))
    for label, angles in sorted_items:
        if len(angles) < 5: continue
        arr = np.array(angles)
        mean = arr.mean()
        std = arr.std()
        hist, edges = np.histogram(arr, bins=bins)
        peak = (edges[:-1] + edges[1:])[np.argmax(hist)] / 2
        peak = edges[np.argmax(hist)] + (edges[1] - edges[0]) / 2
        L.append(f"{label:<18}{len(angles):>10}{mean:>14.2f}{std:>10.2f}"
                  f"{peak:>14.2f}")
    (out_dir / "dihedral_summary.txt").write_text("\n".join(L))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run(input_file, out_dir="analysis/07_dihedral", sample_max=200000):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dihedral] Loading {input_file} ...")
    struct = load_structure(input_file)
    print(f"[dihedral] {struct['n_atoms']} atoms")
    results, total = compute_dihedrals(struct, sample_max=sample_max)
    print(f"[dihedral] Found {len(results)} distinct dihedral types")

    plot_dihedrals(results, struct, out_dir / "dihedral_dist.png")
    save_dihedral_summary(results, total, struct, out_dir)
    print(f"[dihedral] → Saved dihedral_dist.png, dihedral_summary.txt")

    # Console summary of top types
    sorted_items = sorted(results.items(), key=lambda kv: -len(kv[1]))
    print(f"[dihedral] Top 5 types:")
    for label, angles in sorted_items[:5]:
        arr = np.array(angles)
        print(f"           {label:<14}: n={len(angles):>6}, mean={arr.mean():.1f}°")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="LAMMPS data file")
    ap.add_argument("--out", default="analysis/07_dihedral",
                    help="output directory")
    ap.add_argument("--sample_max", type=int, default=200000,
                    help="max dihedrals to compute (for speed)")
    a = ap.parse_args()
    run(a.input, a.out, sample_max=a.sample_max)
