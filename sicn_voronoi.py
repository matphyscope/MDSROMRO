#!/usr/bin/env python3
"""
sicn_voronoi.py — Voronoi tessellation + free volume analysis

THEORY:
  Voronoi cell V_i of atom i: region of space closer to atom i than any other.
  V_i is a convex polyhedron with face_count f_i (number of nearest neighbors).

  Quantities:
    • V_i (cell volume)         → free volume per atom
    • f_i (face count)          → Voronoi coordination (geometric, not bonded)
    • voronoi index (n3,n4,n5,n6,…)  → polyhedron shape fingerprint
    • V_i histogram by type      → density inhomogeneity (Si-rich vs C-rich domains)

  Difference from bonded CN:
    Voronoi CN counts ALL nearest neighbors (not just bonded). Generally:
      f_i > CN_bonded   (typically 12-14 vs 4 for tetrahedral)

USAGE:
  python3 sicn_voronoi.py SiCN_300K_final.data --out analysis/08_voronoi/
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import freud
except ImportError:
    raise ImportError("freud is required: pip install freud-analysis  "
                       "(or conda install -c conda-forge freud)")

from sicn_common import (
    TYPE_SI, TYPE_C, TYPE_N, TYPE_NAME,
    load_structure, density_g_per_cc, composition,
)


# ─────────────────────────────────────────────────────────────────────────────
# Voronoi computation using freud
# ─────────────────────────────────────────────────────────────────────────────
def compute_voronoi(struct):
    """Compute Voronoi tessellation with freud.

    Returns:
        volumes      : (N,) per-atom cell volume (Å³)
        face_counts  : (N,) per-atom face count (Voronoi coordination)
        nlist        : freud NeighborList (faces ↔ neighbor edges)
    """
    coords = struct["coords"]
    box = struct["box"]
    L = box[:, 1] - box[:, 0]

    # freud expects coordinates in [-L/2, L/2]
    centered = coords - (box[:, 1] + box[:, 0]) / 2

    fbox = freud.box.Box.from_box(L)
    voro = freud.locality.Voronoi()
    voro.compute(system=(fbox, centered))

    volumes = np.array(voro.volumes)  # (N,)
    # Face count via neighbor list
    nlist = voro.nlist
    face_counts = nlist.neighbor_counts  # (N,)
    return volumes, face_counts, nlist


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_voronoi(volumes, face_counts, struct, outfile):
    """4-panel: volume distribution per type + face count histogram + summary."""
    types = struct["types"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel 1: Volume distribution per atom type
    ax = axes[0, 0]
    for t, name, color in [(TYPE_SI, "Si", "#1f77b4"),
                            (TYPE_C,  "C",  "#2ca02c"),
                            (TYPE_N,  "N",  "#d62728")]:
        mask = (types == t)
        if mask.sum() > 0:
            v = volumes[mask]
            ax.hist(v, bins=50, alpha=0.55, label=f"{name}  (μ={v.mean():.2f} Å³)",
                    color=color, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Voronoi cell volume V$_i$ (Å³)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Per-atom Voronoi cell volume by type",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 2: Face count distribution per atom type
    ax = axes[0, 1]
    max_fc = int(face_counts.max())
    bins = np.arange(-0.5, max_fc + 1.5, 1)
    for t, name, color in [(TYPE_SI, "Si", "#1f77b4"),
                            (TYPE_C,  "C",  "#2ca02c"),
                            (TYPE_N,  "N",  "#d62728")]:
        mask = (types == t)
        if mask.sum() > 0:
            ax.hist(face_counts[mask], bins=bins, alpha=0.55,
                    label=f"{name}  (μ={face_counts[mask].mean():.1f})",
                    color=color, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Voronoi face count (geometric coord. #)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Voronoi face count by type",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(range(0, max_fc + 2, 2))
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 3: V vs face count scatter (density inhomogeneity)
    ax = axes[1, 0]
    colors_map = {TYPE_SI: "#1f77b4", TYPE_C: "#2ca02c", TYPE_N: "#d62728"}
    names_map = {TYPE_SI: "Si", TYPE_C: "C", TYPE_N: "N"}
    for t in [TYPE_SI, TYPE_C, TYPE_N]:
        mask = (types == t)
        if mask.sum() > 0:
            ax.scatter(face_counts[mask], volumes[mask], s=2, alpha=0.3,
                       color=colors_map[t], label=names_map[t])
    ax.set_xlabel("Face count", fontsize=11)
    ax.set_ylabel("Volume (Å³)", fontsize=11)
    ax.set_title("V vs face count (per atom)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 4: Summary statistics
    ax = axes[1, 1]
    ax.axis("off")
    lines = ["Voronoi summary:", "─" * 38]
    total_V = volumes.sum()
    avg_V = total_V / struct["n_atoms"]
    lines.append(f"  Total V    : {total_V:.1f} Å³")
    lines.append(f"  Avg V/atom : {avg_V:.3f} Å³")
    lines.append(f"  Avg faces  : {face_counts.mean():.2f}")
    lines.append("")
    for t in [TYPE_SI, TYPE_C, TYPE_N]:
        mask = (types == t)
        if mask.sum() > 0:
            v = volumes[mask]
            f = face_counts[mask]
            name = names_map[t]
            lines.append(f"  {name} (n={mask.sum()})")
            lines.append(f"    V : {v.mean():.3f} ± {v.std():.3f} Å³")
            lines.append(f"    f : {f.mean():.2f} ± {f.std():.2f}")
    lines.append("")
    # Voronoi vs bond CN comparison hint
    lines.append("Note: Voronoi face count")
    lines.append("  ≠ bonded CN")
    lines.append("  Includes ALL nearest")
    lines.append("  neighbors (geometric).")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=10, verticalalignment="top", family="monospace")

    x = composition(struct)
    fig.suptitle(f"Voronoi tessellation — a-SiCN  "
                 f"(N = {struct['n_atoms']}, ρ = {density_g_per_cc(struct):.3f} g/cc, "
                 f"Si{x[0]*100:.1f}/C{x[1]*100:.1f}/N{x[2]*100:.1f})",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(outfile, dpi=120, bbox_inches="tight")
    plt.close()


def save_voronoi_summary(volumes, face_counts, struct, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    types = struct["types"]
    L = []
    L.append("Voronoi tessellation summary")
    L.append("=" * 70)
    L.append(f"  System: {struct['n_atoms']} atoms, "
             f"ρ = {density_g_per_cc(struct):.3f} g/cc")
    L.append(f"  Box volume: {volumes.sum():.2f} Å³")
    L.append(f"  Avg V/atom: {volumes.mean():.3f} Å³")
    L.append(f"  Avg face count: {face_counts.mean():.2f}")
    L.append("")
    L.append(f"{'Type':<8}{'Count':>8}{'V mean (Å³)':>14}{'V std':>10}"
              f"{'f mean':>10}{'f std':>10}")
    L.append("-" * 70)
    for t in [TYPE_SI, TYPE_C, TYPE_N]:
        mask = (types == t)
        if mask.sum() > 0:
            v = volumes[mask]
            f = face_counts[mask]
            L.append(f"{TYPE_NAME[t]:<8}{mask.sum():>8}"
                      f"{v.mean():>14.3f}{v.std():>10.3f}"
                      f"{f.mean():>10.2f}{f.std():>10.2f}")
    L.append("")
    L.append("Note: Voronoi face count is GEOMETRIC coordination (includes all")
    L.append("      nearest neighbors via Voronoi faces, not just bonded ones).")
    L.append("      Compare with bond-based CN from cn_lammps.txt.")
    (out_dir / "voronoi_summary.txt").write_text("\n".join(L))

    # Per-atom CSV
    csv_lines = ["idx,type,volume_A3,face_count"]
    for i in range(struct["n_atoms"]):
        csv_lines.append(f"{i},{TYPE_NAME[types[i]]},{volumes[i]:.4f},"
                          f"{face_counts[i]}")
    (out_dir / "voronoi_per_atom.csv").write_text("\n".join(csv_lines))


def run(input_file, out_dir="analysis/08_voronoi"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[voronoi] Loading {input_file} ...")
    struct = load_structure(input_file)
    print(f"[voronoi] {struct['n_atoms']} atoms, "
          f"ρ = {density_g_per_cc(struct):.3f} g/cc")
    print(f"[voronoi] Computing Voronoi tessellation (freud) ...")
    volumes, face_counts, _ = compute_voronoi(struct)
    print(f"[voronoi]   Avg V/atom    = {volumes.mean():.3f} Å³")
    print(f"[voronoi]   Avg face count = {face_counts.mean():.2f}")

    plot_voronoi(volumes, face_counts, struct, out_dir / "voronoi_summary.png")
    save_voronoi_summary(volumes, face_counts, struct, out_dir)
    print(f"[voronoi] → Saved voronoi_summary.png, voronoi_summary.txt, "
          f"voronoi_per_atom.csv")
    return volumes, face_counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="LAMMPS data file")
    ap.add_argument("--out", default="analysis/08_voronoi",
                    help="output directory")
    a = ap.parse_args()
    run(a.input, a.out)
