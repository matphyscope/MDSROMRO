#!/usr/bin/env python3
"""
sicn_csro.py — Warren-Cowley Chemical Short-Range Order parameter α_AB

THEORY:
  Warren-Cowley α_AB measures chemical clustering vs anti-clustering:

    α_AB = 1 - P(B|A)/x_B

  where:
    P(B|A) = probability that an A atom has a B neighbor in 1st shell
    x_B    = global mole fraction of B

  Interpretation:
    α_AB = 0  → random mixing (Bernoulli)
    α_AB > 0  → A and B avoid each other (anti-clustering / chemical ordering)
    α_AB < 0  → A and B cluster together (preference for A-B bonds)

  Bounds:
    α_AB ∈ [-x_B/(1-x_B),  +1]  if x_A ≥ x_B
    α_AB ∈ [-(1-x_A)/x_A,  +1]  otherwise

PHYSICAL MEANING (a-SiCN):
  α_Si-N < 0: Si preferentially bonds N (expected: Si-N is strong bond)
  α_C-C  < 0: C-C clustering (free-C / graphenic domains)
  α_Si-C > 0: Si avoids C? or < 0: Si-C carbidic preference?

INPUT:
  LAMMPS .data file (single frame). Uses v7 bond cutoffs from sicn_common.

OUTPUT:
  csro_matrix.png        — α_AB heatmap
  csro_summary.txt       — text table

USAGE:
  python3 sicn_csro.py SiCN_300K_final.data --out analysis/06_csro/
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
    load_structure, build_bond_list, density_g_per_cc, composition,
)

TYPES = [TYPE_SI, TYPE_C, TYPE_N]
NAMES = ["Si", "C", "N"]


def compute_csro(struct):
    """Compute Warren-Cowley α_AB matrix from bond list.

    Returns:
        alpha : (3, 3) ndarray of α_AB (A=row, B=col).
        P     : (3, 3) probability matrix P(B|A) — neighbors of A that are B.
        x     : (3,) mole fractions.
    """
    types = struct["types"]
    n_atoms = struct["n_atoms"]

    # Get bond list (from v7 cutoffs in sicn_common)
    bonds, _, _ = build_bond_list(struct)

    # Per-atom neighbor type counts: counts[i, t] = # of type-t neighbors of atom i
    counts = np.zeros((n_atoms, 3), dtype=int)
    for i, j in bonds:
        ti = types[i] - 1  # TYPE_SI=1 → index 0, C→1, N→2
        tj = types[j] - 1
        counts[i, tj] += 1
        counts[j, ti] += 1

    # Group by central atom type
    x = np.array(composition(struct))  # mole fractions [Si, C, N]
    P = np.zeros((3, 3))
    for a_idx, a_type in enumerate(TYPES):
        mask = (types == a_type)
        total_neighbors = counts[mask].sum(axis=1)  # per-atom total CN
        # Skip atoms with no neighbors
        valid = total_neighbors > 0
        if valid.sum() == 0:
            continue
        # Average P(B|A) = sum_i (n_iB / CN_i) for valid atoms / N_valid
        for b_idx in range(3):
            ratios = counts[mask, b_idx][valid] / total_neighbors[valid]
            P[a_idx, b_idx] = ratios.mean()

    # Warren-Cowley α_AB = 1 - P(B|A)/x_B
    alpha = np.zeros((3, 3))
    for a_idx in range(3):
        for b_idx in range(3):
            if x[b_idx] > 0:
                alpha[a_idx, b_idx] = 1.0 - P[a_idx, b_idx] / x[b_idx]
            else:
                alpha[a_idx, b_idx] = np.nan
    return alpha, P, x


def plot_csro(alpha, P, x, struct, outfile):
    """Two-panel: α matrix + neighbor-probability matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel 1: α matrix
    ax = axes[0]
    vmax = max(0.5, np.nanmax(np.abs(alpha)))
    im = ax.imshow(alpha, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(NAMES, fontsize=12); ax.set_yticklabels(NAMES, fontsize=12)
    ax.set_xlabel("B (neighbor)", fontsize=12)
    ax.set_ylabel("A (central)", fontsize=12)
    ax.set_title("Warren-Cowley  α$_{AB}$\n"
                 "(blue = clustering A–B,  red = avoidance)",
                 fontsize=12, fontweight="bold")
    for i in range(3):
        for j in range(3):
            val = alpha[i, j]
            if np.isnan(val): continue
            color = "white" if abs(val) > vmax * 0.5 else "black"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="α$_{AB}$")

    # Panel 2: P(B|A) matrix
    ax = axes[1]
    im2 = ax.imshow(P, cmap="viridis", vmin=0, vmax=max(P.max(), 0.6))
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(NAMES, fontsize=12); ax.set_yticklabels(NAMES, fontsize=12)
    ax.set_xlabel("B (neighbor)", fontsize=12)
    ax.set_ylabel("A (central)", fontsize=12)
    ax.set_title("P(B|A) — fraction of A's neighbors that are B\n"
                 "(diagonal — like-bonds dominant)",
                 fontsize=12, fontweight="bold")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{P[i, j]:.3f}", ha="center", va="center",
                    color="white", fontsize=12, fontweight="bold")
    plt.colorbar(im2, ax=ax, label="P(B|A)")

    fig.suptitle(f"Chemical short-range order (Warren-Cowley) — a-SiCN  "
                 f"(N={struct['n_atoms']}, "
                 f"Si{x[0]*100:.1f}/C{x[1]*100:.1f}/N{x[2]*100:.1f})",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(outfile, dpi=120, bbox_inches="tight")
    plt.close()


def save_csro_summary(alpha, P, x, struct, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    L = []
    L.append("Warren-Cowley Chemical Short-Range Order (CSRO)")
    L.append("=" * 70)
    L.append(f"  System: {struct['n_atoms']} atoms")
    L.append(f"  Composition: Si {x[0]*100:.2f}%, C {x[1]*100:.2f}%, "
             f"N {x[2]*100:.2f}%")
    L.append("")
    L.append("  Interpretation:")
    L.append("    α_AB =  0    → random (Bernoulli) mixing")
    L.append("    α_AB > 0    → A avoids B (chemical ordering / repulsion)")
    L.append("    α_AB < 0    → A prefers B (chemical clustering)")
    L.append("")
    L.append("[α_AB matrix]")
    L.append(f"{'':<8}" + "".join(f"{n:>12}" for n in NAMES))
    for i, name in enumerate(NAMES):
        row = f"{name:<8}"
        for j in range(3):
            row += f"{alpha[i,j]:>+12.4f}"
        L.append(row)
    L.append("")
    L.append("[P(B|A) matrix — fraction of A's neighbors that are B]")
    L.append(f"{'':<8}" + "".join(f"{n:>12}" for n in NAMES))
    for i, name in enumerate(NAMES):
        row = f"{name:<8}"
        for j in range(3):
            row += f"{P[i,j]:>12.4f}"
        L.append(row)
    L.append("")

    # Key insights
    L.append("─ Key results ─")
    pairs = [(0, 1, "Si-C"), (0, 2, "Si-N"), (1, 2, "C-N"),
              (0, 0, "Si-Si"), (1, 1, "C-C"), (2, 2, "N-N")]
    for i, j, label in pairs:
        a = alpha[i, j]
        sign = "+" if a > 0 else "" if a == 0 else ""
        interp = "(clustering)" if a < -0.05 else \
                 "(avoidance)" if a > 0.05 else "(near-random)"
        L.append(f"  α({label}) = {a:+.4f}  {interp}")

    (out_dir / "csro_summary.txt").write_text("\n".join(L))


def run(input_file, out_dir="analysis/06_csro"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[csro] Loading {input_file} ...")
    struct = load_structure(input_file)
    print(f"[csro] {struct['n_atoms']} atoms, ρ = {density_g_per_cc(struct):.3f} g/cc")
    print(f"[csro] Computing Warren-Cowley α_AB ...")
    alpha, P, x = compute_csro(struct)
    plot_csro(alpha, P, x, struct, out_dir / "csro_matrix.png")
    save_csro_summary(alpha, P, x, struct, out_dir)
    # Quick console summary
    print(f"[csro]   α(Si-N) = {alpha[0,2]:+.4f}  "
          f"(< 0 → Si-N clustering, > 0 → avoidance)")
    print(f"[csro]   α(Si-C) = {alpha[0,1]:+.4f}")
    print(f"[csro]   α(C-C)  = {alpha[1,1]:+.4f}  "
          f"(< 0 → free-C clustering)")
    print(f"[csro] → Saved {out_dir/'csro_matrix.png'}, csro_summary.txt")
    return alpha, P, x


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="LAMMPS data file")
    ap.add_argument("--out", default="analysis/06_csro", help="output directory")
    a = ap.parse_args()
    run(a.input, a.out)
