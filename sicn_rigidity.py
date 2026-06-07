#!/usr/bin/env python3
"""
sicn_rigidity.py — Maxwell-Phillips-Thorpe constraint counting

THEORY (Phillips 1979, Thorpe 1983):
  In an amorphous covalent network, each atom contributes:
    - Bond-stretching constraints: r/2  (r = coordination number)
    - Bond-bending constraints   : 2r - 3 (for r ≥ 2)

  Total constraints per atom:  n_c = r/2 + (2r - 3) = 5r/2 - 3
  Degrees of freedom (3D):     n_d = 3
  Rigidity index:              f = n_d - n_c = 3 - (5r/2 - 3) = 6 - 5r/2

    f > 0 : under-constrained ('floppy'), can deform freely
    f = 0 : isostatic, critical (Maxwell criterion, r = 2.4)
    f < 0 : over-constrained ('rigid' / stressed)

  Examples:
    Polymer (r=2): f = +1   → very floppy
    Chalcogenide glass (r=2.4): f = 0  → optimum 'rigidity transition'
    Si-network (r=4): f = -4  → highly stressed (typical for SiC, Si3N4)
    Diamond (r=4): f = -4   → rigid

PHYSICAL MEANING (a-SiCN):
  Si network is heavily over-constrained → high modulus, brittle
  Free-C clusters (sp², r=3): f = -1.5 → mildly stressed
  Network as a whole: f weighted by composition

EXTENSIONS:
  - Per-atom f → spatial distribution of stress
  - Excess constraints (-f) correlates with hardness, elastic modulus

USAGE:
  python3 sicn_rigidity.py SiCN_300K_final.data --out analysis/12_rigidity/
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
    load_structure, build_bond_list,
    density_g_per_cc, composition,
)


def compute_rigidity(struct):
    """Compute Maxwell constraint counts per atom.

    Returns:
        cn_per_atom  : (N,) CN of each atom
        n_c_per_atom : (N,) constraints per atom
        f_per_atom   : (N,) rigidity index (3 - n_c)
        avg_r        : <r> over all atoms
        avg_f        : <f> over all atoms
    """
    types = struct["types"]
    n_atoms = struct["n_atoms"]
    bonds, _, _ = build_bond_list(struct)

    cn_per_atom = np.zeros(n_atoms, dtype=int)
    for i, j in bonds:
        cn_per_atom[i] += 1
        cn_per_atom[j] += 1

    # Phillips-Thorpe: n_c = r/2 (stretching) + (2r-3) (bending) for r>=2
    # For r=0 or 1: no bending, only stretching count
    n_c_per_atom = np.zeros(n_atoms, dtype=float)
    for i in range(n_atoms):
        r = cn_per_atom[i]
        if r == 0:
            n_c_per_atom[i] = 0
        elif r == 1:
            n_c_per_atom[i] = 0.5  # only stretching, no bending
        else:
            n_c_per_atom[i] = r / 2.0 + (2 * r - 3)
    f_per_atom = 3.0 - n_c_per_atom

    avg_r = cn_per_atom.mean()
    avg_f = f_per_atom.mean()
    return cn_per_atom, n_c_per_atom, f_per_atom, avg_r, avg_f


def plot_rigidity(cn_per_atom, n_c_per_atom, f_per_atom, struct, outfile):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    types = struct["types"]

    colors_map = {TYPE_SI: "#1f77b4", TYPE_C: "#2ca02c", TYPE_N: "#d62728"}
    names_map = {TYPE_SI: "Si", TYPE_C: "C", TYPE_N: "N"}

    # Panel 1: CN distribution per type
    ax = axes[0, 0]
    cn_max = int(cn_per_atom.max()) + 1
    bins = np.arange(-0.5, cn_max + 0.5, 1)
    for t in [TYPE_SI, TYPE_C, TYPE_N]:
        mask = (types == t)
        if mask.sum() > 0:
            ax.hist(cn_per_atom[mask], bins=bins, alpha=0.55,
                    label=f"{names_map[t]} (⟨r⟩={cn_per_atom[mask].mean():.3f})",
                    color=colors_map[t], edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Coordination number r", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("CN distribution per type", fontsize=12, fontweight="bold")
    ax.set_xticks(range(0, cn_max + 1))
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 2: f distribution
    ax = axes[0, 1]
    f_min = int(np.floor(f_per_atom.min()))
    f_max = int(np.ceil(f_per_atom.max()))
    bins = np.arange(f_min - 0.5, f_max + 1.5, 0.5)
    for t in [TYPE_SI, TYPE_C, TYPE_N]:
        mask = (types == t)
        if mask.sum() > 0:
            ax.hist(f_per_atom[mask], bins=bins, alpha=0.55,
                    label=f"{names_map[t]} (⟨f⟩={f_per_atom[mask].mean():+.3f})",
                    color=colors_map[t], edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="red", ls="--", lw=1.2, label="Isostatic (f=0)")
    ax.set_xlabel("Rigidity index f = 3 − n$_c$  (per atom)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Rigidity index distribution\n(f>0 floppy, f<0 over-constrained)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3: per-element <f> bar chart
    ax = axes[1, 0]
    names = []
    avg_fs = []
    avg_rs = []
    colors_list = []
    for t in [TYPE_SI, TYPE_C, TYPE_N]:
        mask = (types == t)
        if mask.sum() > 0:
            names.append(names_map[t])
            avg_fs.append(f_per_atom[mask].mean())
            avg_rs.append(cn_per_atom[mask].mean())
            colors_list.append(colors_map[t])
    # Whole-system avg
    names.append("Total")
    avg_fs.append(f_per_atom.mean())
    avg_rs.append(cn_per_atom.mean())
    colors_list.append("#888888")

    bars = ax.bar(names, avg_fs, color=colors_list, edgecolor="black", linewidth=0.5)
    for b, f_v, r_v in zip(bars, avg_fs, avg_rs):
        ax.text(b.get_x() + b.get_width()/2,
                f_v + (0.1 if f_v >= 0 else -0.3),
                f"⟨f⟩={f_v:+.2f}\n⟨r⟩={r_v:.2f}",
                ha="center", va="bottom" if f_v >= 0 else "top", fontsize=10)
    ax.axhline(0, color="red", ls="--", lw=1.0)
    ax.set_ylabel("Avg rigidity index ⟨f⟩", fontsize=11)
    ax.set_title("Element-wise & total rigidity",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    # Panel 4: Summary text
    ax = axes[1, 1]
    ax.axis("off")
    lines = ["Maxwell-Phillips-Thorpe summary:", "─" * 38]
    lines.append(f"  Atoms total       : {struct['n_atoms']}")
    lines.append(f"  Mean ⟨r⟩          : {cn_per_atom.mean():.4f}")
    lines.append(f"  Mean ⟨n_c⟩        : {n_c_per_atom.mean():.4f}")
    lines.append(f"  Mean ⟨f⟩          : {f_per_atom.mean():+.4f}")
    lines.append("")
    lines.append("Per element:")
    for t in [TYPE_SI, TYPE_C, TYPE_N]:
        mask = (types == t)
        if mask.sum() > 0:
            r_v = cn_per_atom[mask].mean()
            f_v = f_per_atom[mask].mean()
            lines.append(f"  {names_map[t]:<5}: ⟨r⟩={r_v:.3f},  ⟨f⟩={f_v:+.3f}")
    lines.append("")
    n_floppy = (f_per_atom > 0).sum()
    n_iso = (f_per_atom == 0).sum()
    n_rigid = (f_per_atom < 0).sum()
    N = struct["n_atoms"]
    lines.append("Fraction by class:")
    lines.append(f"  floppy (f>0)      : {n_floppy} ({n_floppy/N*100:.2f}%)")
    lines.append(f"  isostatic (f=0)   : {n_iso} ({n_iso/N*100:.2f}%)")
    lines.append(f"  over-constrained  : {n_rigid} ({n_rigid/N*100:.2f}%)")
    lines.append("")
    lines.append("Reference values:")
    lines.append("  Maxwell isostatic : ⟨r⟩ = 2.4, ⟨f⟩ = 0")
    lines.append("  Polymer (r=2)     : ⟨f⟩ = +1.0 (floppy)")
    lines.append("  Si network (r=4)  : ⟨f⟩ = -4.0 (rigid/stressed)")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=10, verticalalignment="top", family="monospace")

    x = composition(struct)
    fig.suptitle(f"Network Rigidity (Maxwell counting) — a-SiCN  "
                 f"(N={struct['n_atoms']}, ρ={density_g_per_cc(struct):.3f} g/cc, "
                 f"Si{x[0]*100:.1f}/C{x[1]*100:.1f}/N{x[2]*100:.1f})",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(outfile, dpi=120, bbox_inches="tight")
    plt.close()


def save_summary(cn_per_atom, n_c_per_atom, f_per_atom, struct, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    types = struct["types"]
    N = struct["n_atoms"]

    L = []
    L.append("Maxwell-Phillips-Thorpe Network Rigidity")
    L.append("=" * 70)
    L.append(f"  System: {N} atoms")
    L.append(f"  Mean coordination ⟨r⟩  : {cn_per_atom.mean():.4f}")
    L.append(f"  Mean constraints ⟨n_c⟩: {n_c_per_atom.mean():.4f}")
    L.append(f"  Mean rigidity ⟨f⟩     : {f_per_atom.mean():+.4f}")
    L.append("")
    L.append("[Per element]")
    L.append("-" * 70)
    L.append(f"{'Element':<10}{'Count':>8}{'⟨r⟩':>12}{'⟨n_c⟩':>12}{'⟨f⟩':>12}")
    for t in [TYPE_SI, TYPE_C, TYPE_N]:
        mask = (types == t)
        if mask.sum() > 0:
            L.append(f"{TYPE_NAME[t]:<10}{mask.sum():>8}"
                      f"{cn_per_atom[mask].mean():>12.4f}"
                      f"{n_c_per_atom[mask].mean():>12.4f}"
                      f"{f_per_atom[mask].mean():>+12.4f}")
    L.append(f"{'Total':<10}{N:>8}"
              f"{cn_per_atom.mean():>12.4f}"
              f"{n_c_per_atom.mean():>12.4f}"
              f"{f_per_atom.mean():>+12.4f}")
    L.append("")
    L.append("[Rigidity class fractions]")
    L.append("-" * 70)
    n_floppy = (f_per_atom > 0).sum()
    n_iso = (f_per_atom == 0).sum()
    n_rigid = (f_per_atom < 0).sum()
    L.append(f"  floppy (f>0)         : {n_floppy:>6} ({n_floppy/N*100:>5.2f}%)")
    L.append(f"  isostatic (f=0)      : {n_iso:>6} ({n_iso/N*100:>5.2f}%)")
    L.append(f"  over-constrained (f<0): {n_rigid:>6} ({n_rigid/N*100:>5.2f}%)")
    L.append("")
    L.append("─ Interpretation ─")
    L.append("  Maxwell criterion (r=2.4, f=0) is the rigidity transition")
    L.append("  Si network (r=4): heavily over-constrained → high modulus, brittle")
    L.append("  Free-C clusters (sp², r=3): mildly stressed")
    L.append("  ⟨f⟩ ≪ 0 → glass is rigid & internally stressed")
    L.append("")
    L.append("Reference systems:")
    L.append("  Polymer       (r=2)   : ⟨f⟩ = +1.0  (floppy)")
    L.append("  Chalcogenide  (r=2.4) : ⟨f⟩ =  0    (isostatic — ideal glass)")
    L.append("  Si network    (r=4)   : ⟨f⟩ = -4.0  (highly stressed)")
    (out_dir / "rigidity_summary.txt").write_text("\n".join(L))


def run(input_file, out_dir="analysis/12_rigidity"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rigidity] Loading {input_file} ...")
    struct = load_structure(input_file)
    print(f"[rigidity] {struct['n_atoms']} atoms")
    print(f"[rigidity] Computing Maxwell constraint counting ...")
    cn, n_c, f, avg_r, avg_f = compute_rigidity(struct)
    print(f"[rigidity]   ⟨r⟩ = {avg_r:.4f}")
    print(f"[rigidity]   ⟨n_c⟩ = {n_c.mean():.4f}")
    print(f"[rigidity]   ⟨f⟩ = {avg_f:+.4f}  "
          f"({'over-constrained' if avg_f < 0 else 'floppy' if avg_f > 0 else 'isostatic'})")
    plot_rigidity(cn, n_c, f, struct, out_dir / "rigidity.png")
    save_summary(cn, n_c, f, struct, out_dir)
    print(f"[rigidity] → Saved rigidity.png, rigidity_summary.txt")
    return cn, n_c, f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="LAMMPS data file")
    ap.add_argument("--out", default="analysis/12_rigidity",
                    help="output directory")
    a = ap.parse_args()
    run(a.input, a.out)
