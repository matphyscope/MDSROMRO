#!/usr/bin/env python3
"""
sicn_boo.py — Steinhardt Bond-Orientational Order (BOO) parameters Q4, Q6, W4, W6

THEORY:
  For atom i with N_b neighbors at angles (θ_j, φ_j):

    q_lm(i) = (1/N_b) Σ_j Y_lm(θ_j, φ_j)        ← raw moments

    Q_l(i) = sqrt[ (4π/(2l+1)) Σ_m |q_lm(i)|² ]   ← rotationally invariant
    W_l(i) = Wigner-3j weighted product of q_lm    ← signed invariant

  Ideal local geometry signatures:
    Q4 ≈ 0.50, Q6 ≈ 0.63   for FCC
    Q4 ≈ 0.04, Q6 ≈ 0.51   for BCC
    Q4 ≈ 0.51, Q6 ≈ 0.66   for tetrahedral (diamond/Si)
    Q4 ≈ 0,   Q6 ≈ 0      for fully random (gas-like)
    Q6 ≈ 0.4              for icosahedral (liquid metal)

PHYSICAL MEANING (a-SiCN):
  Q_l per atom → local symmetry classifier
  Si atoms in sp³ network → Q6 should peak ~0.5-0.6
  C atoms in graphenic → Q4 distinct shape
  Broad Q6 distribution → amorphous / glass-like
  Sharp peaks → crystalline order (NOT expected in amorphous)

USAGE:
  python3 sicn_boo.py SiCN_300K_final.data --out analysis/09_boo/
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
# BOO computation via freud
# ─────────────────────────────────────────────────────────────────────────────
def compute_boo(struct, num_neighbors=4):
    """Compute Q4, Q6, W4, W6 for every atom using freud.Steinhardt.

    Args:
        num_neighbors : how many nearest neighbors per atom to use.
                        4 = tetrahedral; 6 = octahedral; 12 = FCC.
                        For SiCN, 4 is natural for sp³ Si tetrahedra.
    Returns:
        Q4, Q6, W4, W6 : (N,) arrays of per-atom values.
    """
    coords = struct["coords"]
    box = struct["box"]
    L = box[:, 1] - box[:, 0]
    centered = coords - (box[:, 1] + box[:, 0]) / 2

    fbox = freud.box.Box.from_box(L)

    # Build neighbor list (k-nearest neighbors)
    aabb = freud.locality.AABBQuery(fbox, centered)
    query = aabb.query(centered, dict(num_neighbors=num_neighbors,
                                       exclude_ii=True))
    nlist = query.toNeighborList()

    # Steinhardt Q_l and W_l
    q4 = freud.order.Steinhardt(l=4, wl=False).compute(
        (fbox, centered), neighbors=nlist).particle_order
    q6 = freud.order.Steinhardt(l=6, wl=False).compute(
        (fbox, centered), neighbors=nlist).particle_order
    w4 = freud.order.Steinhardt(l=4, wl=True).compute(
        (fbox, centered), neighbors=nlist).particle_order
    w6 = freud.order.Steinhardt(l=6, wl=True).compute(
        (fbox, centered), neighbors=nlist).particle_order

    return np.array(q4), np.array(q6), np.array(w4), np.array(w6)


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_boo(Q4, Q6, W4, W6, struct, outfile, num_neighbors):
    """4-panel histograms + Q4-Q6 scatter overlaid with crystal reference."""
    types = struct["types"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    colors_map = {TYPE_SI: "#1f77b4", TYPE_C: "#2ca02c", TYPE_N: "#d62728"}
    names_map = {TYPE_SI: "Si", TYPE_C: "C", TYPE_N: "N"}

    # Panel 1: Q4 histogram
    ax = axes[0, 0]
    for t in [TYPE_SI, TYPE_C, TYPE_N]:
        mask = (types == t)
        if mask.sum() > 0:
            ax.hist(Q4[mask], bins=50, alpha=0.55, density=True,
                    label=f"{names_map[t]} (μ={Q4[mask].mean():.3f})",
                    color=colors_map[t], edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Q$_4$", fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.set_title(f"Q$_4$ distribution  (k = {num_neighbors})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: Q6 histogram
    ax = axes[0, 1]
    for t in [TYPE_SI, TYPE_C, TYPE_N]:
        mask = (types == t)
        if mask.sum() > 0:
            ax.hist(Q6[mask], bins=50, alpha=0.55, density=True,
                    label=f"{names_map[t]} (μ={Q6[mask].mean():.3f})",
                    color=colors_map[t], edgecolor="black", linewidth=0.3)
    # Reference markers
    ax.axvline(0.510, color="orange",   ls="--", lw=1.0, alpha=0.6,
                label="FCC (0.51)")
    ax.axvline(0.628, color="purple",   ls="--", lw=1.0, alpha=0.6,
                label="Tetrahedral (0.63)")
    ax.axvline(0.485, color="brown",    ls="--", lw=1.0, alpha=0.6,
                label="Icosahedral (0.49)")
    ax.set_xlabel("Q$_6$", fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.set_title(f"Q$_6$ distribution  (k = {num_neighbors})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: W4 vs W6 scatter
    ax = axes[0, 2]
    for t in [TYPE_SI, TYPE_C, TYPE_N]:
        mask = (types == t)
        if mask.sum() > 0:
            ax.scatter(W4[mask], W6[mask], s=2, alpha=0.3,
                       color=colors_map[t], label=names_map[t])
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("W$_4$", fontsize=11)
    ax.set_ylabel("W$_6$", fontsize=11)
    ax.set_title(f"W$_4$ vs W$_6$  (k = {num_neighbors})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 4: Q4-Q6 scatter (most useful for structure ID)
    ax = axes[1, 0]
    for t in [TYPE_SI, TYPE_C, TYPE_N]:
        mask = (types == t)
        if mask.sum() > 0:
            ax.scatter(Q4[mask], Q6[mask], s=2, alpha=0.3,
                       color=colors_map[t], label=names_map[t])
    # Ideal crystal markers
    ax.scatter([0.510], [0.628], color="orange", marker="*", s=180,
                edgecolor="black", linewidth=1, label="Tetrahedral", zorder=10)
    ax.scatter([0.190], [0.575], color="purple", marker="*", s=180,
                edgecolor="black", linewidth=1, label="FCC", zorder=10)
    ax.scatter([0.036], [0.510], color="brown", marker="*", s=180,
                edgecolor="black", linewidth=1, label="BCC", zorder=10)
    ax.set_xlabel("Q$_4$", fontsize=11)
    ax.set_ylabel("Q$_6$", fontsize=11)
    ax.set_title("Q$_4$–Q$_6$ map (local structure ID)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # Panel 5: W4 histogram
    ax = axes[1, 1]
    for t in [TYPE_SI, TYPE_C, TYPE_N]:
        mask = (types == t)
        if mask.sum() > 0:
            ax.hist(W4[mask], bins=50, alpha=0.55, density=True,
                    label=f"{names_map[t]} (μ={W4[mask].mean():.3f})",
                    color=colors_map[t], edgecolor="black", linewidth=0.3)
    ax.set_xlabel("W$_4$", fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.set_title("W$_4$ distribution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 6: Summary
    ax = axes[1, 2]
    ax.axis("off")
    lines = ["BOO summary:", "─" * 38]
    lines.append(f"  Neighbors (k) : {num_neighbors}")
    lines.append("")
    for t in [TYPE_SI, TYPE_C, TYPE_N]:
        mask = (types == t)
        if mask.sum() > 0:
            name = names_map[t]
            lines.append(f"  {name} (n={mask.sum()})")
            lines.append(f"    Q4: {Q4[mask].mean():.4f} ± {Q4[mask].std():.4f}")
            lines.append(f"    Q6: {Q6[mask].mean():.4f} ± {Q6[mask].std():.4f}")
            lines.append(f"    W4: {W4[mask].mean():.4f} ± {W4[mask].std():.4f}")
            lines.append(f"    W6: {W6[mask].mean():.4f} ± {W6[mask].std():.4f}")
    lines.append("")
    lines.append("Reference values (perfect):")
    lines.append("  Tetra (sp³ Si) Q4=0.51, Q6=0.63")
    lines.append("  FCC            Q4=0.19, Q6=0.58")
    lines.append("  BCC            Q4=0.04, Q6=0.51")
    lines.append("  Random         Q_l → 0")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, verticalalignment="top", family="monospace")

    x = composition(struct)
    fig.suptitle(f"Steinhardt Bond-Orientational Order — a-SiCN  "
                 f"(N = {struct['n_atoms']}, "
                 f"Si{x[0]*100:.1f}/C{x[1]*100:.1f}/N{x[2]*100:.1f})",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(outfile, dpi=120, bbox_inches="tight")
    plt.close()


def save_boo_summary(Q4, Q6, W4, W6, struct, out_dir, num_neighbors):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    types = struct["types"]
    L = []
    L.append("Steinhardt Bond-Orientational Order (BOO)")
    L.append("=" * 70)
    L.append(f"  System: {struct['n_atoms']} atoms")
    L.append(f"  Neighbors per atom (k): {num_neighbors}")
    L.append("")
    L.append(f"{'Type':<8}{'Count':>8}{'Q4':>16}{'Q6':>16}{'W4':>16}{'W6':>16}")
    L.append("-" * 90)
    for t in [TYPE_SI, TYPE_C, TYPE_N]:
        mask = (types == t)
        if mask.sum() > 0:
            row = f"{TYPE_NAME[t]:<8}{mask.sum():>8}"
            for arr in (Q4, Q6, W4, W6):
                v = arr[mask]
                row += f"  {v.mean():>7.4f}±{v.std():>5.4f}"
            L.append(row)
    L.append("")
    L.append("Reference values (perfect crystals):")
    L.append("  Tetrahedral (sp³ Si/diamond) : Q4 = 0.510, Q6 = 0.628")
    L.append("  FCC                          : Q4 = 0.191, Q6 = 0.575")
    L.append("  BCC                          : Q4 = 0.036, Q6 = 0.510")
    L.append("  Icosahedral (liquid metal)   : Q4 = 0.187, Q6 = 0.485")
    L.append("  Random / gas                 : Q4, Q6 → 0")
    (out_dir / "boo_summary.txt").write_text("\n".join(L))


def run(input_file, out_dir="analysis/09_boo", num_neighbors=4):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[boo] Loading {input_file} ...")
    struct = load_structure(input_file)
    print(f"[boo] {struct['n_atoms']} atoms")
    print(f"[boo] Computing Q4, Q6, W4, W6 (k = {num_neighbors}) ...")
    Q4, Q6, W4, W6 = compute_boo(struct, num_neighbors=num_neighbors)
    print(f"[boo]   <Q4> = {Q4.mean():.4f}  (tetra ref: 0.510)")
    print(f"[boo]   <Q6> = {Q6.mean():.4f}  (tetra ref: 0.628)")

    plot_boo(Q4, Q6, W4, W6, struct, out_dir / "boo_summary.png",
             num_neighbors=num_neighbors)
    save_boo_summary(Q4, Q6, W4, W6, struct, out_dir,
                      num_neighbors=num_neighbors)
    print(f"[boo] → Saved boo_summary.png, boo_summary.txt")
    return Q4, Q6, W4, W6


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="LAMMPS data file")
    ap.add_argument("--out", default="analysis/09_boo", help="output directory")
    ap.add_argument("--k", type=int, default=4,
                    help="number of nearest neighbors per atom (default 4)")
    a = ap.parse_args()
    run(a.input, a.out, num_neighbors=a.k)
