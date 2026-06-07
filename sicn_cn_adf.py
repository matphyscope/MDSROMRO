#!/usr/bin/env python3
"""
sicn_cn_adf.py — LAMMPS 시간평균 CN_*.txt + angles_*.txt 표시

WHY:
  LAMMPS의 fix ave/time 로 800,000 step 평균한 CN과 ADF가 가장 정확한
  '시스템 평균' 값이다. .data 파일 단일 frame에서 계산한 값과 비교하여
  thermal 효과 추정 가능.

INPUT:
  --cn       : CN_300K.txt    (LAMMPS fix ave/time 출력)
  --adf      : angles_300K.txt (LAMMPS fix ave/time 출력)
  --data     : SiCN_*.data    (구조 정보 — atoms, composition)

CN_300K.txt 형식 (LAMMPS thermo, 10개 column):
  TimeStep  CN_Si  CN_Si_SRO  CN_C  CN_N  pSi_Si  pSi_C  pSi_N  pC_C  pC_Si  pN_Si

angles_300K.txt 형식 (LAMMPS fix ave/time mode vector):
  block 별로:  TimeStep Nrows
    row, theta, g_NSiN, cn, g_CSiC, cn, g_CSiN, cn, g_SiCSi, cn, g_SiNSi, cn, g_CCC, cn

OUTPUT:
  <out>/cn_lammps.txt      — text summary
  <out>/cn_lammps.png      — bar chart of CN values
  <out>/adf_lammps.png     — 6-panel ADF plot
  <out>/adf_peaks.txt      — peak positions of each ADF

USAGE:
  python3 sicn_cn_adf.py --cn CN_300K.txt --adf angles_300K.txt \\
                         --data SiCN_300K_final.data --out analysis/SRO/
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sicn_common import (
    TYPE_SI, TYPE_C, TYPE_N,
    load_structure, density_g_per_cc, composition,
)

# ─────────────────────────────────────────────────────────────────────────────
# CN_300K.txt parser
# ─────────────────────────────────────────────────────────────────────────────
def read_cn_file(path):
    """Read LAMMPS fix ave/time output for CN.
    Format: TS CN_Si CN_Si_SRO CN_C CN_N pSi_Si pSi_C pSi_N pC_C pC_Si pN_Si
    Returns dict of {key: mean_value}.
    """
    rows = []
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            pass
    if not rows:
        raise ValueError(f"No data rows in {path}")
    arr = np.array(rows)
    # Time-average all the columns
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    keys = ["TS", "CN_Si", "CN_Si_SRO", "CN_C", "CN_N",
            "pSi_Si", "pSi_C", "pSi_N", "pC_C", "pC_Si", "pN_Si"]
    result = {}
    for i, k in enumerate(keys):
        if i < len(means):
            result[k] = float(means[i])
            result[k + "_std"] = float(stds[i])
    result["_n_rows"] = len(rows)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# angles_300K.txt parser (LAMMPS fix ave/time mode vector)
# ─────────────────────────────────────────────────────────────────────────────
def read_adf_file(path):
    """Read LAMMPS fix ave/time mode vector output for ADF.
    Multiple time blocks; each block:
        TimeStep Nrows
        row theta count1 cn1 count2 cn2 ...
    Returns: theta (Nbins,), adfs dict {label: array}
    """
    blocks, cur = [], []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                try:
                    int(parts[0]); int(parts[1])
                    if cur: blocks.append(np.array(cur, dtype=float))
                    cur = []
                    continue
                except ValueError:
                    pass
            try:
                cur.append([float(x) for x in parts])
            except ValueError:
                pass
    if cur:
        blocks.append(np.array(cur, dtype=float))
    if not blocks:
        raise ValueError(f"No ADF blocks in {path}")

    # Time-average all blocks
    arr = np.mean(np.stack(blocks), axis=0)
    theta = arr[:, 1]
    # LAMMPS 'compute adf' triplet columns per angle type (verified by data inspection):
    #   [1] = normalized angle distribution function  ← what we want
    #   [2] = cumulative N(θ)
    #   [3] = θ (redundant, same as theta column)
    # 6 ADFs in order (as defined in v8 LAMMPS):
    #   N-Si-N, C-Si-C, C-Si-N, Si-C-Si, Si-N-Si, C-C-C
    labels = ["N-Si-N", "C-Si-C", "C-Si-N", "Si-C-Si", "Si-N-Si", "C-C-C"]
    adfs = {}
    for k, lab in enumerate(labels):
        # row = arr[:,0], θ = arr[:,1], then triplets start at col 2:
        #   col 2+3k+0 = ADF [1]  ← THIS
        #   col 2+3k+1 = N(θ)  [2]
        #   col 2+3k+2 = θ     [3]
        col_adf = 2 + 3 * k
        if col_adf < arr.shape[1]:
            adfs[lab] = arr[:, col_adf]
        else:
            adfs[lab] = np.zeros_like(theta)
    return theta, adfs, len(blocks)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_cn(cn_data, struct, outfile):
    """Bar chart of per-atom and partial CNs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: total CN per atom type
    ax = axes[0]
    labels = ["Si total", "Si SRO\n(C+N only)", "C total", "N total"]
    keys = ["CN_Si", "CN_Si_SRO", "CN_C", "CN_N"]
    vals = [cn_data[k] for k in keys]
    stds = [cn_data[k + "_std"] for k in keys]
    bars = ax.bar(labels, vals, yerr=stds, capsize=4,
                  color=["#1f77b4", "#5fa8d8", "#2ca02c", "#d62728"],
                  edgecolor="black", linewidth=0.5)
    for b, v, s in zip(bars, vals, stds):
        ax.text(b.get_x() + b.get_width()/2, v + s + 0.05,
                f"{v:.3f}\n±{s:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(4, color="gray", ls="--", lw=0.8, alpha=0.5, label="ideal 4 (Si)")
    ax.axhline(3, color="brown", ls="--", lw=0.8, alpha=0.5, label="ideal 3 (N)")
    ax.set_ylabel("Coordination number", fontsize=11)
    ax.set_title(f"Time-averaged CN per atom type ({cn_data['_n_rows']} samples)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, max(vals) * 1.25)

    # Panel 2: partial CNs
    ax = axes[1]
    labels2 = ["Si-Si", "Si-C", "Si-N", "C-C", "C-Si", "N-Si"]
    keys2 = ["pSi_Si", "pSi_C", "pSi_N", "pC_C", "pC_Si", "pN_Si"]
    vals2 = [cn_data[k] for k in keys2]
    stds2 = [cn_data[k + "_std"] for k in keys2]
    colors = ["#1f77b4", "#ff7f0e", "#d62728",
              "#2ca02c", "#9467bd", "#8c564b"]
    bars = ax.bar(labels2, vals2, yerr=stds2, capsize=4, color=colors,
                  edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals2):
        ax.text(b.get_x() + b.get_width()/2, v + 0.05,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Partial coordination number", fontsize=11)
    ax.set_title("Partial CN (per central atom)", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, max(vals2) * 1.20 if max(vals2) > 0 else 1)

    x = composition(struct)
    fig.suptitle(f"LAMMPS Time-Averaged CN — a-SiCN  "
                 f"(N={struct['n_atoms']}, ρ={density_g_per_cc(struct):.3f} g/cc, "
                 f"Si{x[0]*100:.1f}/C{x[1]*100:.1f}/N{x[2]*100:.1f})",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(outfile, dpi=120, bbox_inches="tight")
    plt.close()


def plot_adf(theta, adfs, struct, outfile, n_blocks=1):
    """6-panel plot of ADFs."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.flatten()

    # Tetrahedral ideal: 109.47°
    labels_order = ["N-Si-N", "C-Si-C", "C-Si-N", "Si-C-Si", "Si-N-Si", "C-C-C"]
    expected = {"N-Si-N": 109.47, "C-Si-C": 109.47, "C-Si-N": 109.47,
                "Si-C-Si": 109.47, "Si-N-Si": 120.0, "C-C-C": 120.0}

    for k, lab in enumerate(labels_order):
        ax = axes[k]
        p = adfs[lab]
        if p.max() < 1e-9:
            ax.text(0.5, 0.5, "(no signal)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            ax.set_title(lab, fontsize=11, fontweight="bold")
            ax.set_xlim(0, 180)
            continue
        ax.plot(theta, p, color="navy", lw=1.5)
        # Peak detection (max value, theta > 60° to avoid noise)
        mask = theta > 60
        if mask.any() and p[mask].max() > 0:
            ipeak = np.argmax(p[mask])
            theta_peak = theta[mask][ipeak]
            ax.axvline(theta_peak, color="red", ls="--", lw=1.0, alpha=0.7)
            ax.text(theta_peak + 5, p.max() * 0.9,
                    f"peak: {theta_peak:.1f}°", color="red",
                    fontsize=9, fontweight="bold")
        # Mark expected angle
        ax.axvline(expected[lab], color="green", ls=":", lw=1.0, alpha=0.5)
        ax.text(expected[lab] + 5, p.max() * 0.05,
                f"ideal: {expected[lab]:.1f}°", color="green",
                fontsize=8, alpha=0.7)
        ax.set_xlabel("θ (degree)", fontsize=10)
        ax.set_ylabel("ADF", fontsize=10)
        ax.set_title(lab, fontsize=11, fontweight="bold")
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 60, 90, 120, 150, 180])
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"LAMMPS Time-Averaged ADF — a-SiCN  "
                 f"(N={struct['n_atoms']}, time-avg over {n_blocks} blocks)",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(outfile, dpi=120, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Save text outputs
# ─────────────────────────────────────────────────────────────────────────────
def save_cn_summary(cn_data, struct, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    L = []
    L.append("LAMMPS time-averaged coordination numbers")
    L.append("=" * 70)
    L.append(f"  System: {struct['n_atoms']} atoms, "
             f"ρ = {density_g_per_cc(struct):.3f} g/cc")
    L.append(f"  Samples (time blocks): {cn_data['_n_rows']}")
    L.append("")
    L.append(f"{'Quantity':<22}{'Mean':>12}{'Std':>12}    의미")
    L.append("-" * 70)
    items = [
        ("CN_Si",      "Si total (Si+C+N)"),
        ("CN_Si_SRO",  "Si SRO (C+N only)"),
        ("CN_C",       "C total (C+Si)"),
        ("CN_N",       "N total (N-Si only)"),
        ("pSi_Si",     "Si의 Si neighbors"),
        ("pSi_C",      "Si의 C neighbors"),
        ("pSi_N",      "Si의 N neighbors"),
        ("pC_C",       "C의 C neighbors"),
        ("pC_Si",      "C의 Si neighbors"),
        ("pN_Si",      "N의 Si neighbors"),
    ]
    for k, desc in items:
        if k in cn_data:
            L.append(f"{k:<22}{cn_data[k]:>12.4f}{cn_data[k+'_std']:>12.4f}    {desc}")
    (out_dir / "cn_lammps.txt").write_text("\n".join(L))


def save_adf_peaks(theta, adfs, out_dir, n_blocks=1):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    L = []
    L.append("LAMMPS time-averaged ADF peak analysis")
    L.append("=" * 70)
    L.append(f"  Time samples: {n_blocks}")
    L.append(f"  θ bins: {len(theta)} (range {theta[0]:.1f}–{theta[-1]:.1f}°)")
    L.append("")
    L.append(f"{'Triplet':<12}{'Peak θ (°)':>14}{'Peak height':>14}{'FWHM (°)':>14}")
    L.append("-" * 70)
    for lab in ["N-Si-N", "C-Si-C", "C-Si-N", "Si-C-Si", "Si-N-Si", "C-C-C"]:
        p = adfs[lab]
        if p.max() < 1e-9:
            L.append(f"{lab:<12}{'—':>14}{'—':>14}{'—':>14}  (no signal)")
            continue
        # peak only above 60 deg
        mask = theta > 60
        if not mask.any():
            continue
        ipeak = np.argmax(p[mask])
        theta_peak = theta[mask][ipeak]
        peak_h = p[mask][ipeak]
        # FWHM around the peak
        half = peak_h / 2
        # Find the bins where p crosses half on either side of peak
        idx_peak_full = np.argmax(p)
        left = np.where(p[:idx_peak_full] < half)[0]
        right = np.where(p[idx_peak_full:] < half)[0]
        fwhm_str = "—"
        if len(left) > 0 and len(right) > 0:
            fwhm = theta[idx_peak_full + right[0]] - theta[left[-1]]
            fwhm_str = f"{fwhm:.2f}"
        L.append(f"{lab:<12}{theta_peak:>14.2f}{peak_h:>14.5f}{fwhm_str:>14}")
    (out_dir / "adf_peaks.txt").write_text("\n".join(L))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run(data_file, cn_file=None, adf_file=None, out_dir="analysis/SRO"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cn_adf] Loading structure: {data_file}")
    struct = load_structure(data_file)
    print(f"[cn_adf] {struct['n_atoms']} atoms, ρ = {density_g_per_cc(struct):.3f} g/cc")

    if cn_file is not None:
        print(f"[cn_adf] Reading CN file: {cn_file}")
        try:
            cn_data = read_cn_file(cn_file)
            plot_cn(cn_data, struct, out_dir / "cn_lammps.png")
            save_cn_summary(cn_data, struct, out_dir)
            print(f"[cn_adf] → Saved cn_lammps.png, cn_lammps.txt")
            print(f"           N samples: {cn_data['_n_rows']}")
            print(f"           CN_Si = {cn_data['CN_Si']:.3f} ± {cn_data['CN_Si_std']:.3f}")
            print(f"           CN_C  = {cn_data['CN_C']:.3f} ± {cn_data['CN_C_std']:.3f}")
            print(f"           CN_N  = {cn_data['CN_N']:.3f} ± {cn_data['CN_N_std']:.3f}")
        except Exception as e:
            print(f"[cn_adf] CN read 실패: {e}")

    if adf_file is not None:
        print(f"[cn_adf] Reading ADF file: {adf_file}")
        try:
            theta, adfs, n_blocks = read_adf_file(adf_file)
            plot_adf(theta, adfs, struct, out_dir / "adf_lammps.png", n_blocks)
            save_adf_peaks(theta, adfs, out_dir, n_blocks)
            print(f"[cn_adf] → Saved adf_lammps.png, adf_peaks.txt")
            print(f"           N time blocks: {n_blocks}, θ bins: {len(theta)}")
        except Exception as e:
            print(f"[cn_adf] ADF read 실패: {e}")

    return struct


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="LAMMPS .data file (structure)")
    ap.add_argument("--cn", default=None, help="CN_300K.txt (LAMMPS fix ave/time)")
    ap.add_argument("--adf", default=None, help="angles_300K.txt")
    ap.add_argument("--out", default="analysis/SRO", help="output directory")
    a = ap.parse_args()
    if a.cn is None and a.adf is None:
        print("at least one of --cn or --adf required")
        sys.exit(1)
    run(a.data, cn_file=a.cn, adf_file=a.adf, out_dir=a.out)
