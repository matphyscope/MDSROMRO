#!/usr/bin/env python3
"""
sicn_sp23.py — sp² vs sp³ hybridization classification of C and Si atoms

WHY:
  Si-C-Si peak at 119.5° (not 109.5°) suggests planar C bonding.
  This module classifies every C and Si by CN + planarity to confirm
  the sp²/sp³ ratio quantitatively.

CRITERIA:
  C atoms:
    CN=4 → sp³ (tetrahedral)
    CN=3 → check planarity (improper dihedral ω):
       ω < 15°  → sp² (planar, graphenic)
       ω ≥ 15°  → sp³-defective (bent / pyramidal)
    CN=2 → chain end (rare)
  Si atoms:
    CN=4 → sp³ (standard)
    CN=3 → defective (3-coord)
    CN=5 → over-coordinated (Tersoff potential characteristic)

PLANARITY (ω = improper angle):
    ω = 0°    → planar
    ω ≈ 35.3° → perfect tetrahedral

USAGE:
  python3 sicn_sp23.py SiCN_300K_final.data --out analysis/10_sp23/
"""
from __future__ import annotations
import argparse
import math
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


def improper_angle(center, neighbors, box_L):
    """Improper-dihedral / out-of-plane angle ω (degrees).

    0° = planar, ~35° = tetrahedral.
    """
    if len(neighbors) != 3:
        return float("nan")
    a, b, c = neighbors
    va = pbc_displacement(a - center, box_L)
    vb = pbc_displacement(b - center, box_L)
    vc = pbc_displacement(c - center, box_L)
    e1 = vb - va
    e2 = vc - va
    n = np.cross(e1, e2)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        return float("nan")
    n /= n_norm
    d = abs(np.dot(n, va))
    bond_len = (np.linalg.norm(va) + np.linalg.norm(vb)
                + np.linalg.norm(vc)) / 3.0
    if bond_len < 1e-9:
        return float("nan")
    sin_omega = min(1.0, d / bond_len)
    return math.degrees(math.asin(sin_omega))


def classify_atoms(struct, planar_thresh_deg=15.0):
    """Classify every C and Si atom."""
    types = struct["types"]
    coords = struct["coords"]
    box_L = struct["box"][:, 1] - struct["box"][:, 0]

    bonds, _, _ = build_bond_list(struct)
    neighbors = {i: [] for i in range(struct["n_atoms"])}
    for i, j in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)

    results = []
    for idx in range(struct["n_atoms"]):
        t = types[idx]
        if t not in (TYPE_C, TYPE_SI):
            continue
        nei = neighbors[idx]
        cn = len(nei)
        n_si = sum(1 for j in nei if types[j] == TYPE_SI)
        n_c  = sum(1 for j in nei if types[j] == TYPE_C)
        n_n  = sum(1 for j in nei if types[j] == TYPE_N)

        omega = float("nan")
        if t == TYPE_C:
            if cn == 4:
                cls = "C-sp3"
            elif cn == 3:
                nei_coords = [coords[j] for j in nei]
                omega = improper_angle(coords[idx], nei_coords, box_L)
                if not math.isnan(omega) and omega < planar_thresh_deg:
                    cls = "C-sp2"
                else:
                    cls = "C-sp3-defective"
            elif cn == 2:
                cls = "C-chain"
            elif cn <= 1:
                cls = "C-undercoord"
            else:
                cls = "C-overcoord"
        else:  # Si
            if cn == 4:
                cls = "Si-sp3"
            elif cn == 3:
                nei_coords = [coords[j] for j in nei]
                omega = improper_angle(coords[idx], nei_coords, box_L)
                cls = "Si-CN3-defective"
            elif cn >= 5:                      # ← FIXED: was `cn == 5`, missed CN≥6
                cls = "Si-overcoord"
            else:                              # CN ≤ 2 only
                cls = "Si-undercoord"

        results.append({
            "idx": idx, "type": int(t), "CN": cn, "omega": omega,
            "class": cls, "n_si": n_si, "n_c": n_c, "n_n": n_n,
        })
    return results


def plot_summary(results, struct, outfile):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    # Panel 1: C classes
    ax = axes[0, 0]
    c_classes_order = ["C-sp2", "C-sp3", "C-sp3-defective",
                        "C-chain", "C-undercoord", "C-overcoord"]
    c_counts = {cls: 0 for cls in c_classes_order}
    for r in results:
        if r["type"] == TYPE_C:
            c_counts[r["class"]] = c_counts.get(r["class"], 0) + 1
    n_C = sum(c_counts.values())
    colors_C = {"C-sp2": "#2ca02c", "C-sp3": "#1f77b4",
                "C-sp3-defective": "#9467bd",
                "C-chain": "#ff7f0e", "C-undercoord": "#d62728",
                "C-overcoord": "#8c564b"}
    bars = ax.bar(c_classes_order, [c_counts[c] for c in c_classes_order],
                  color=[colors_C[c] for c in c_classes_order],
                  edgecolor="black", linewidth=0.5)
    for b, c in zip(bars, c_classes_order):
        v = c_counts[c]
        if v > 0:
            ax.text(b.get_x() + b.get_width()/2, v + n_C*0.01,
                    f"{v}\n({v/n_C*100:.1f}%)",
                    ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Carbon classification (total = {n_C})",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.grid(alpha=0.3, axis="y")

    # Panel 2: Si classes
    ax = axes[0, 1]
    si_classes_order = ["Si-sp3", "Si-CN3-defective",
                         "Si-undercoord", "Si-overcoord"]
    si_counts = {cls: 0 for cls in si_classes_order}
    for r in results:
        if r["type"] == TYPE_SI:
            si_counts[r["class"]] = si_counts.get(r["class"], 0) + 1
    n_Si = sum(si_counts.values())
    colors_Si = {"Si-sp3": "#1f77b4", "Si-CN3-defective": "#9467bd",
                 "Si-undercoord": "#d62728", "Si-overcoord": "#ff7f0e"}
    bars = ax.bar(si_classes_order, [si_counts[c] for c in si_classes_order],
                  color=[colors_Si[c] for c in si_classes_order],
                  edgecolor="black", linewidth=0.5)
    for b, c in zip(bars, si_classes_order):
        v = si_counts[c]
        if v > 0:
            ax.text(b.get_x() + b.get_width()/2, v + n_Si*0.01,
                    f"{v}\n({v/n_Si*100:.1f}%)",
                    ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Silicon classification (total = {n_Si})",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.grid(alpha=0.3, axis="y")

    # Panel 3: ω distribution for CN=3 C
    ax = axes[1, 0]
    omegas = [r["omega"] for r in results
              if r["type"] == TYPE_C and r["CN"] == 3
              and not math.isnan(r["omega"])]
    if len(omegas) > 0:
        ax.hist(omegas, bins=40, range=(0, 60),
                color="#2ca02c", edgecolor="black", linewidth=0.5, alpha=0.7)
        ax.axvline(15.0, color="red", ls="--", lw=1.5,
                   label="planarity threshold (15°)")
        ax.axvline(35.26, color="orange", ls="--", lw=1.0, alpha=0.7,
                   label="ideal tetrahedral (35.3°)")
        ax.set_title(f"Improper angle ω for CN=3 carbon (n = {len(omegas)})",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("ω (deg)  — 0=planar, 35°=tetrahedral")
        ax.set_ylabel("Count")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no CN=3 C atoms", ha="center", va="center",
                transform=ax.transAxes)

    # Panel 4: Neighbor composition of C-class
    ax = axes[1, 1]
    classes_to_show = ["C-sp2", "C-sp3", "C-sp3-defective"]
    width = 0.25
    x_pos = np.arange(len(classes_to_show))
    si_means, c_means, n_means = [], [], []
    for cls in classes_to_show:
        items = [r for r in results if r["class"] == cls]
        if items:
            si_means.append(np.mean([r["n_si"] for r in items]))
            c_means.append(np.mean([r["n_c"] for r in items]))
            n_means.append(np.mean([r["n_n"] for r in items]))
        else:
            si_means.append(0); c_means.append(0); n_means.append(0)
    ax.bar(x_pos - width, si_means, width, label="Si neighbors", color="#1f77b4")
    ax.bar(x_pos,         c_means, width, label="C neighbors",  color="#2ca02c")
    ax.bar(x_pos + width, n_means, width, label="N neighbors",  color="#d62728")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes_to_show, rotation=15, fontsize=10)
    ax.set_ylabel("Avg # of neighbors per atom")
    ax.set_title("Neighbor composition of C-class",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    x = composition(struct)
    fig.suptitle(f"sp²/sp³ classification — a-SiCN  "
                 f"(N = {struct['n_atoms']}, ρ = {density_g_per_cc(struct):.3f} g/cc, "
                 f"Si{x[0]*100:.1f}/C{x[1]*100:.1f}/N{x[2]*100:.1f})",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(outfile, dpi=120, bbox_inches="tight")
    plt.close()


def save_summary(results, struct, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_C = sum(1 for r in results if r["type"] == TYPE_C)
    n_Si = sum(1 for r in results if r["type"] == TYPE_SI)
    c_counts, si_counts = {}, {}
    for r in results:
        if r["type"] == TYPE_C:
            c_counts[r["class"]] = c_counts.get(r["class"], 0) + 1
        else:
            si_counts[r["class"]] = si_counts.get(r["class"], 0) + 1
    L = []
    L.append("sp²/sp³ hybridization classification")
    L.append("=" * 70)
    L.append(f"  System: {struct['n_atoms']} atoms")
    L.append("")
    L.append(f"[CARBON CLASSIFICATION] (total {n_C} C atoms)")
    L.append("-" * 70)
    L.append(f"{'Class':<22}{'Count':>10}{'Fraction':>12}    Description")
    L.append("-" * 70)
    descriptions = {
        "C-sp2":            "planar 3-coord (graphene-like)",
        "C-sp3":            "tetrahedral 4-coord (SiC-like)",
        "C-sp3-defective":  "3-coord but bent (ω≥15°)",
        "C-chain":          "2-coord (chain end)",
        "C-undercoord":     "0 or 1 coord (very rare)",
        "C-overcoord":      "≥5 coord (defect)",
    }
    for cls in ["C-sp2", "C-sp3", "C-sp3-defective", "C-chain",
                 "C-undercoord", "C-overcoord"]:
        cnt = c_counts.get(cls, 0)
        pct = cnt / n_C * 100 if n_C > 0 else 0
        L.append(f"{cls:<22}{cnt:>10}{pct:>11.2f}%    {descriptions[cls]}")
    L.append("")
    L.append(f"[SILICON CLASSIFICATION] (total {n_Si} Si atoms)")
    L.append("-" * 70)
    L.append(f"{'Class':<22}{'Count':>10}{'Fraction':>12}    Description")
    L.append("-" * 70)
    si_desc = {
        "Si-sp3":            "tetrahedral 4-coord (standard)",
        "Si-CN3-defective":  "3-coord (defect / dangling)",
        "Si-undercoord":     "0–2 coord (rare defect)",
        "Si-overcoord":      "≥5 coord (Tersoff characteristic)",
    }
    for cls in ["Si-sp3", "Si-CN3-defective", "Si-undercoord", "Si-overcoord"]:
        cnt = si_counts.get(cls, 0)
        pct = cnt / n_Si * 100 if n_Si > 0 else 0
        L.append(f"{cls:<22}{cnt:>10}{pct:>11.2f}%    {si_desc[cls]}")

    n_sp2 = c_counts.get("C-sp2", 0)
    n_sp3 = c_counts.get("C-sp3", 0) + c_counts.get("C-sp3-defective", 0)
    n_total_C = n_sp2 + n_sp3
    L.append("")
    L.append("─ KEY RESULT ─")
    if n_total_C > 0:
        L.append(f"  sp² fraction of C : {n_sp2/n_total_C*100:.2f}%")
        L.append(f"  sp³ fraction of C : {n_sp3/n_total_C*100:.2f}%")
    omegas = [r["omega"] for r in results
              if r["type"] == TYPE_C and r["CN"] == 3
              and not math.isnan(r["omega"])]
    if omegas:
        L.append(f"  Avg ω (CN=3 C)    : {np.mean(omegas):.2f}°")
        L.append(f"  ω std            : {np.std(omegas):.2f}°")
    (out_dir / "sp23_summary.txt").write_text("\n".join(L))

    # CSV
    csv_lines = ["idx,type,CN,omega_deg,class,n_Si,n_C,n_N"]
    for r in results:
        omega_str = f"{r['omega']:.3f}" if not math.isnan(r['omega']) else "nan"
        csv_lines.append(f"{r['idx']},{TYPE_NAME[r['type']]},{r['CN']},"
                          f"{omega_str},{r['class']},{r['n_si']},"
                          f"{r['n_c']},{r['n_n']}")
    (out_dir / "sp23_per_atom.csv").write_text("\n".join(csv_lines))


def run(input_file, out_dir="analysis/10_sp23", planar_thresh_deg=15.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sp23] Loading {input_file} ...")
    struct = load_structure(input_file)
    print(f"[sp23] {struct['n_atoms']} atoms, "
          f"ρ = {density_g_per_cc(struct):.3f} g/cc")
    print(f"[sp23] Classifying (planarity threshold = {planar_thresh_deg}°) ...")
    results = classify_atoms(struct, planar_thresh_deg)
    n_C_total = sum(1 for r in results if r["type"] == TYPE_C)
    n_sp2 = sum(1 for r in results if r["class"] == "C-sp2")
    n_sp3 = sum(1 for r in results if r["class"] == "C-sp3")
    n_sp3_def = sum(1 for r in results if r["class"] == "C-sp3-defective")
    print(f"[sp23]   C-sp²            : {n_sp2:>5} ({n_sp2/n_C_total*100:.2f}%)")
    print(f"[sp23]   C-sp³            : {n_sp3:>5} ({n_sp3/n_C_total*100:.2f}%)")
    print(f"[sp23]   C-sp³-defective : {n_sp3_def:>5} ({n_sp3_def/n_C_total*100:.2f}%)")
    plot_summary(results, struct, out_dir / "sp23_summary.png")
    save_summary(results, struct, out_dir)
    print(f"[sp23] → Saved sp23_summary.png, .txt, sp23_per_atom.csv")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="LAMMPS data file")
    ap.add_argument("--out", default="analysis/10_sp23",
                    help="output directory")
    ap.add_argument("--planar_thresh", type=float, default=15.0,
                    help="ω threshold (deg) for sp²/sp³ classification")
    a = ap.parse_args()
    run(a.input, a.out, a.planar_thresh)
