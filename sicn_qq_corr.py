#!/usr/bin/env python3
"""
sicn_qq_corr.py — Q-site spatial correlation g_QQ(r) and mass-fractal D_f

THEORY:
  Each Si atom in a-SiCN has a "Q-site" label based on its bonded neighbors:
    SiC4   : 4 C neighbors (carbidic Si)
    SiNC3  : 3 C + 1 N
    SiN2C2 : 2 N + 2 C  (mixed)
    SiN3C  : 1 C + 3 N
    SiN4   : 4 N neighbors (nitridic Si, α-Si₃N₄ environment)

  For each Q-Q pair (e.g., SiN4-SiN4), compute the spatial correlation g_QQ(r):
    g_QQ(r) = (rho local around Q' atom near Q) / (rho_random)

  Interpretation:
    g_QQ(r) > 1 at small r : same Q-sites cluster (nano-domain)
    g_QQ(r) ≈ 1            : random spatial distribution
    g_QQ(r) < 1            : Q-Q avoid each other (anti-clustering)

MASS-FRACTAL DIMENSION D_f (Saha 2006):
  In a nano-domain glass, the running coordination N(r) of like-Q sites
  follows a fractal scaling:
      N(r) ∝ r^D_f
  where D_f < 3 means fractal (clustered), D_f = 3 = uniform.

  Saha (2006) reported D_f ≈ 2.5 for polymer-derived a-SiCN from SAXS.
  This module extracts D_f from MD structure as a direct check.

USAGE:
  python3 sicn_qq_corr.py SiCN_300K_final.data --out analysis/13_qq_corr/
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


# Q-site labels (SiC4 → SiNC3 → SiN2C2 → SiN3C → SiN4)
QSITE_LABELS = ["SiC4", "SiNC3", "SiN2C2", "SiN3C", "SiN4"]


def classify_qsites(struct):
    """For each Si atom, classify by (n_C, n_N) neighbors.

    Q-site label = SiN_(n_N)C_(n_C) if n_N + n_C == 4 (canonical SP3 Si).
    For other CN, return 'other'.
    """
    types = struct["types"]
    bonds, _, _ = build_bond_list(struct)
    n_atoms = struct["n_atoms"]

    nei_type_counts = np.zeros((n_atoms, 3), dtype=int)  # [Si, C, N]
    for i, j in bonds:
        nei_type_counts[i, types[j] - 1] += 1
        nei_type_counts[j, types[i] - 1] += 1

    labels = {}  # label → list of Si atom indices
    for lbl in QSITE_LABELS + ["other"]:
        labels[lbl] = []

    si_idx = np.where(types == TYPE_SI)[0]
    for i in si_idx:
        n_Si, n_C, n_N = nei_type_counts[i]
        # Use canonical Si(N,C) classification — ignoring Si-Si neighbors
        if n_C == 4 and n_N == 0: lbl = "SiC4"
        elif n_C == 3 and n_N == 1: lbl = "SiNC3"
        elif n_C == 2 and n_N == 2: lbl = "SiN2C2"
        elif n_C == 1 and n_N == 3: lbl = "SiN3C"
        elif n_C == 0 and n_N == 4: lbl = "SiN4"
        else: lbl = "other"
        labels[lbl].append(int(i))
    return labels


def compute_gqq(struct, labels, r_max=15.0, n_bins=80):
    """Compute g_QQ(r) for each Q-Q pair using PBC.

    Returns dict { (Q_a, Q_b): (r_bins, g_r) }.
    """
    coords = struct["coords"]
    box = struct["box"]
    L = box[:, 1] - box[:, 0]
    coords_w = (coords - box[:, 0][None, :]) % L[None, :]
    V_box = float(np.prod(L))
    # Use only Q-sites with at least a few atoms
    valid_labels = [lbl for lbl in QSITE_LABELS if len(labels[lbl]) >= 5]
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    shell_vol = 4 * np.pi / 3 * (r_edges[1:]**3 - r_edges[:-1]**3)

    print(f"[qq_corr] Computing g_QQ(r) for {len(valid_labels)} valid Q-sites ...")
    results = {}
    for i_a, Qa in enumerate(valid_labels):
        idx_a = np.array(labels[Qa], dtype=int)
        N_a = len(idx_a)
        for Qb in valid_labels[i_a:]:
            idx_b = np.array(labels[Qb], dtype=int)
            N_b = len(idx_b)
            counts = np.zeros(n_bins, dtype=float)
            for ai in idx_a:
                d = pbc_distance(coords_w[ai], coords_w[idx_b], L)
                d = d[d > 1e-6]  # remove self-pair
                # Avoid double-counting when Qa == Qb (each pair counted twice in d)
                if Qa == Qb:
                    h, _ = np.histogram(d, bins=r_edges)
                    counts += h / 2.0
                else:
                    h, _ = np.histogram(d, bins=r_edges)
                    counts += h
            # Normalize to g(r):  g = counts / (shell_vol * rho_b * N_pairs_norm)
            if Qa == Qb:
                # Each Qa pair: rho_b = (N_a - 1) / V; reference pair count = N_a * (N_a-1) / 2
                rho_b = (N_a - 1) / V_box if N_a > 1 else 1
                norm_pairs = N_a / 2 if N_a > 0 else 1  # we already divided counts by 2 above
            else:
                rho_b = N_b / V_box
                norm_pairs = N_a
            with np.errstate(divide="ignore", invalid="ignore"):
                g_r = counts / (shell_vol * rho_b * norm_pairs)
            results[(Qa, Qb)] = (r_centers, g_r)
    return results, valid_labels


def compute_fractal_dim(struct, labels, target_label="SiN4",
                         r_min=2.0, r_max=10.0, n_bins=40):
    """Extract mass-fractal dimension D_f from N(r) ∝ r^D_f scaling of
    the running coordination of `target_label` Si atoms.

    For each target Si, count how many `target_label` neighbors at distance ≤ r,
    average over all target Si.
    """
    coords = struct["coords"]
    box = struct["box"]
    L = box[:, 1] - box[:, 0]
    coords_w = (coords - box[:, 0][None, :]) % L[None, :]

    idx_t = np.array(labels[target_label], dtype=int)
    if len(idx_t) < 10:
        return None, None, None, None

    r_grid = np.linspace(r_min, r_max, n_bins)
    N_r = np.zeros_like(r_grid)
    for ai in idx_t:
        d = pbc_distance(coords_w[ai], coords_w[idx_t], L)
        d = d[d > 1e-6]
        for k, r in enumerate(r_grid):
            N_r[k] += np.sum(d <= r)
    N_r /= len(idx_t)

    # Log-log fit slope = D_f
    valid = N_r > 0
    if valid.sum() < 5:
        return r_grid, N_r, None, None
    log_r = np.log(r_grid[valid])
    log_N = np.log(N_r[valid])
    # Use middle region (avoid r near 0 and r near box/2 to mitigate FSE)
    n = valid.sum()
    start = max(2, n // 6)
    end = max(start + 3, 2 * n // 3)
    slope, intercept = np.polyfit(log_r[start:end], log_N[start:end], 1)
    return r_grid, N_r, slope, (log_r[start:end], log_N[start:end])


def plot_qq(results, valid_labels, struct, outfile,
             fractal_data=None, fractal_label="SiN4"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel 1: Q-site population bar
    ax = axes[0, 0]
    labels = classify_qsites(struct)
    pops = [len(labels[lbl]) for lbl in QSITE_LABELS]
    bars = ax.bar(QSITE_LABELS, pops, color="steelblue",
                   edgecolor="black", linewidth=0.5)
    n_total_si = sum(pops)
    for b, p in zip(bars, pops):
        if p > 0:
            ax.text(b.get_x() + b.get_width()/2, p + n_total_si * 0.01,
                    f"{p}\n({p/n_total_si*100:.1f}%)",
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of Si atoms", fontsize=11)
    ax.set_title(f"Q-site population (canonical Si(N,C)₄ tetrahedra)\n"
                 f"Total canonical = {n_total_si}",
                 fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(alpha=0.3, axis="y")

    # Panel 2: Diagonal g_QQ(r) for each Q
    ax = axes[0, 1]
    colors_q = plt.cm.viridis(np.linspace(0.2, 0.95, len(valid_labels)))
    for i, Q in enumerate(valid_labels):
        r, g = results[(Q, Q)]
        ax.plot(r, g, color=colors_q[i], lw=1.5, label=Q)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("r (Å)", fontsize=11)
    ax.set_ylabel("g$_{QQ}$(r)", fontsize=11)
    ax.set_title("Like-Q spatial correlation g$_{QQ}$(r)\n"
                 "(g>1 = same Q-site clustering)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max([results[(Q, Q)][1].max() for Q in valid_labels]) * 1.1)
    ax.grid(alpha=0.3)

    # Panel 3: Mass-fractal D_f log-log
    ax = axes[1, 0]
    if fractal_data is not None and fractal_data[0] is not None:
        r_grid, N_r, slope, fit_data = fractal_data
        valid = N_r > 0
        ax.loglog(r_grid[valid], N_r[valid], "o-", color="darkblue",
                  markersize=4, label=f"N(r) for {fractal_label}-{fractal_label}")
        if slope is not None and fit_data is not None:
            log_r_fit, log_N_fit = fit_data
            r_fit = np.exp(log_r_fit)
            N_fit_line = np.exp(log_N_fit[0] + slope * (log_r_fit - log_r_fit[0]))
            ax.loglog(r_fit, N_fit_line, "r-", lw=2.5,
                      label=f"power-law fit, D_f = {slope:.3f}")
        # Reference lines
        r_ref = np.array([2.5, 10.0])
        for Df_ref, ls, lbl in [(3.0, ":", "uniform (D_f=3)"),
                                  (2.5, "--", "Saha D_f=2.5 (SAXS)")]:
            N_ref = (r_ref / r_ref[0]) ** Df_ref * N_r[valid][0] if valid.sum() > 0 else r_ref ** Df_ref
            ax.loglog(r_ref, N_ref, ls, color="gray", lw=1.0, label=lbl)
        ax.set_xlabel("r (Å)", fontsize=11)
        ax.set_ylabel("⟨N(r)⟩  cumulative like-Q neighbors", fontsize=11)
        ax.set_title(f"Mass-fractal scaling for {fractal_label}\n"
                     f"N(r) ∝ r^D_f", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which="both")
    else:
        ax.text(0.5, 0.5, f"Insufficient {fractal_label} atoms for D_f",
                ha="center", va="center", transform=ax.transAxes)

    # Panel 4: Summary text
    ax = axes[1, 1]
    ax.axis("off")
    L_text = ["Q-site spatial correlation summary:", "─" * 38]
    L_text.append(f"  Total Si        : {sum(len(labels[l]) for l in QSITE_LABELS) + len(labels['other'])}")
    L_text.append(f"  Canonical Q-Si  : {n_total_si}")
    L_text.append(f"  'other' Si      : {len(labels['other'])}  (non-canonical)")
    L_text.append("")
    L_text.append("Q-site populations:")
    for lbl in QSITE_LABELS:
        n = len(labels[lbl])
        pct = n / n_total_si * 100 if n_total_si > 0 else 0
        L_text.append(f"  {lbl:<8}: {n:>5} ({pct:>5.1f}%)")
    L_text.append("")
    # g_QQ(r) peaks
    L_text.append("Like-Q clustering @ r=3-5 Å:")
    for Q in valid_labels:
        r, g = results[(Q, Q)]
        # Mean g(r) in 3-5 Å region
        mask = (r > 3) & (r < 5)
        if mask.sum() > 0:
            g_avg = g[mask].mean()
            note = "CLUSTERED" if g_avg > 1.2 else \
                   "near-random" if 0.85 < g_avg < 1.15 else "anti-clustered"
            L_text.append(f"  {Q:<8}: ⟨g⟩={g_avg:.3f}  ({note})")
    L_text.append("")
    if fractal_data and fractal_data[2] is not None:
        L_text.append(f"Mass-fractal D_f ({fractal_label}):")
        L_text.append(f"  Measured : {fractal_data[2]:.3f}")
        L_text.append(f"  Saha 2006: ≈ 2.5 (SAXS)")
        L_text.append(f"  Uniform  : 3.0")
    ax.text(0.05, 0.95, "\n".join(L_text), transform=ax.transAxes,
            fontsize=10, verticalalignment="top", family="monospace")

    x = composition(struct)
    fig.suptitle(f"Q-site spatial correlation — a-SiCN  "
                 f"(N={struct['n_atoms']}, "
                 f"Si{x[0]*100:.1f}/C{x[1]*100:.1f}/N{x[2]*100:.1f})",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(outfile, dpi=120, bbox_inches="tight")
    plt.close()


def save_summary(results, valid_labels, labels, fractal_data,
                  fractal_label, struct, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_total_si = sum(len(labels[l]) for l in QSITE_LABELS)
    L_lines = []
    L_lines.append("Q-site spatial correlation g_QQ(r) and mass-fractal D_f")
    L_lines.append("=" * 70)
    L_lines.append(f"  System: {struct['n_atoms']} atoms")
    L_lines.append(f"  Canonical Q-site Si atoms: {n_total_si}")
    L_lines.append(f"  Non-canonical 'other' Si : {len(labels['other'])}")
    L_lines.append("")
    L_lines.append("[Q-site population]")
    L_lines.append(f"{'Q-site':<12}{'Count':>10}{'Fraction':>12}")
    for lbl in QSITE_LABELS:
        n = len(labels[lbl])
        pct = n / n_total_si * 100 if n_total_si > 0 else 0
        L_lines.append(f"  {lbl:<10}{n:>10}{pct:>11.2f}%")
    L_lines.append("")
    L_lines.append("[Like-Q spatial correlation g_QQ(r) at short range (3-5 Å)]")
    for Q in valid_labels:
        r, g = results[(Q, Q)]
        mask = (r > 3) & (r < 5)
        if mask.sum() > 0:
            g_avg = g[mask].mean()
            note = "CLUSTERED" if g_avg > 1.2 else \
                   "near-random" if 0.85 < g_avg < 1.15 else "anti-clustered"
            L_lines.append(f"  g({Q}-{Q})  ⟨g⟩(3-5 Å) = {g_avg:.4f}  ({note})")
    L_lines.append("")
    if fractal_data and fractal_data[2] is not None:
        L_lines.append(f"[Mass-fractal D_f from {fractal_label}-{fractal_label} scaling]")
        L_lines.append(f"  Measured D_f : {fractal_data[2]:.3f}")
        L_lines.append(f"  Reference    : Saha 2006 (SAXS) ≈ 2.5 (nano-domain)")
        L_lines.append(f"  Reference    : Uniform = 3.0")
    L_lines.append("")
    L_lines.append("─ Physical interpretation ─")
    L_lines.append("  g_QQ(r) > 1 → same Q-sites cluster spatially (nano-domain)")
    L_lines.append("  D_f < 3      → fractal mass distribution")
    L_lines.append("  D_f ≈ 2.5    → consistent with Saha (2006) SAXS for PDC-SiCN")
    (out_dir / "qq_corr_summary.txt").write_text("\n".join(L_lines))


def run(input_file, out_dir="analysis/13_qq_corr",
         r_max=15.0, n_bins=80, fractal_label="SiN4"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[qq_corr] Loading {input_file} ...")
    struct = load_structure(input_file)
    print(f"[qq_corr] {struct['n_atoms']} atoms")
    print(f"[qq_corr] Classifying Q-sites ...")
    labels = classify_qsites(struct)
    for lbl in QSITE_LABELS:
        print(f"[qq_corr]   {lbl}: {len(labels[lbl])} atoms")

    results, valid_labels = compute_gqq(struct, labels, r_max=r_max, n_bins=n_bins)
    print(f"[qq_corr] Computed g_QQ for {len(valid_labels)} valid Q-sites")

    # Pick fractal_label automatically if it's not populated enough
    if len(labels[fractal_label]) < 50:
        # Pick the most populated label
        fractal_label = max(QSITE_LABELS,
                             key=lambda l: len(labels[l]))
        print(f"[qq_corr] Insufficient SiN4 atoms; using {fractal_label} for D_f instead")
    fractal_data = compute_fractal_dim(struct, labels, target_label=fractal_label)
    if fractal_data and fractal_data[2] is not None:
        print(f"[qq_corr]   Mass-fractal D_f ({fractal_label}-{fractal_label}): "
              f"{fractal_data[2]:.3f}  (Saha 2006: ~2.5, uniform: 3.0)")

    plot_qq(results, valid_labels, struct, out_dir / "qq_corr.png",
             fractal_data=fractal_data, fractal_label=fractal_label)
    save_summary(results, valid_labels, labels, fractal_data,
                  fractal_label, struct, out_dir)
    print(f"[qq_corr] → Saved qq_corr.png, qq_corr_summary.txt")
    return results, valid_labels, fractal_data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="LAMMPS data file")
    ap.add_argument("--out", default="analysis/13_qq_corr",
                    help="output directory")
    ap.add_argument("--r_max", type=float, default=15.0)
    ap.add_argument("--n_bins", type=int, default=80)
    ap.add_argument("--fractal", default="SiN4",
                    help="Q-site to use for D_f extraction (auto-fallback if empty)")
    a = ap.parse_args()
    run(a.input, a.out, r_max=a.r_max, n_bins=a.n_bins, fractal_label=a.fractal)
