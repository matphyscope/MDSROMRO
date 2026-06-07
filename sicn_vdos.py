#!/usr/bin/env python3
"""
sicn_vdos.py — Vibrational Density of States via harmonic Hessian

THEORY:
  At the local minimum, the second derivatives of the potential energy
  form the Hessian matrix H of size 3N × 3N. The mass-weighted form:
      D_iα,jβ = H_iα,jβ / sqrt(m_i · m_j)
  is the "dynamical matrix" at Γ (k=0).

  Its eigenvalues are ω² (squared angular frequencies); negative eigenvalues
  indicate saddle-point modes (imaginary frequency).

  The vibrational density of states (VDOS):
      g(ω) = Σ_i δ(ω - ω_i)
  Sum rules: ∫ g(ω) dω = 3N

WORKFLOW (this is a 2-step process):
  Step 1: Generate Hessian with LAMMPS
      $ lmp -in in_compute_hessian.lmp
      → produces dynmat.dat (text or binary)

  Step 2: Diagonalize and plot
      $ python3 sicn_vdos.py SiCN_300K_final.data --hessian dynmat.dat

BOSON PEAK:
  Amorphous solids show a characteristic excess g(ω)/ω² peak at low ω
  (typically 0.5-5 THz). This is the "boson peak" — a fingerprint of
  amorphous structure.

USAGE:
  python3 sicn_vdos.py SiCN_300K_final.data --hessian dynmat.dat --out analysis/14_vdos/

NOTE: LAMMPS dynamical_matrix output format (text mode, binary no):
  Sequential rows: 3N lines, each with 3N space-separated floats.
  Units: eV/(Å² × amu)  (LAMMPS metal units → ω² in eV/(amu·Å²))
  To convert to THz: ω [rad/s] = sqrt(ω² × eV/(amu·Å²) × conversion)
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
    TYPE_SI, TYPE_C, TYPE_N, TYPE_NAME, ATOMIC_MASS,
    load_structure, density_g_per_cc, composition,
)


# Conversion: eV/(amu·Å²) → (rad/s)²
#   1 eV/(amu·Å²) = (1.602e-19 J) / (1.66e-27 kg × 1e-20 m²)
#                 = 9.65e21  s⁻²
# So ω [rad/s] = sqrt(9.65e21 × ω²_in_LAMMPS)
# And ν [THz] = ω / (2π × 1e12)
_LAMMPS_FREQ_TO_RADS = np.sqrt(9.6485e21)  # rad/s
_RADS_TO_THZ = 1.0 / (2 * np.pi * 1e12)
LAMMPS_TO_THZ = _LAMMPS_FREQ_TO_RADS * _RADS_TO_THZ  # multiply sqrt(ω²) by this


def load_hessian(path, n_atoms):
    """Load LAMMPS dynamical_matrix output (text mode).

    Expected format:
        3N lines, each with 3N space-separated floats.
        Mass-weighted Hessian in eV/(amu·Å²).
    """
    n_dim = 3 * n_atoms
    print(f"[vdos] Loading Hessian: expected {n_dim} × {n_dim} = "
          f"{n_dim*n_dim} entries...")
    try:
        data = np.loadtxt(path)
    except Exception as e:
        print(f"[vdos] ERROR loading {path}: {e}")
        sys.exit(1)
    if data.ndim == 1:
        # Flat: reshape
        if data.size != n_dim * n_dim:
            print(f"[vdos] ERROR: Hessian size mismatch: got {data.size}, "
                  f"expected {n_dim*n_dim}")
            sys.exit(1)
        D = data.reshape((n_dim, n_dim))
    else:
        D = data
    print(f"[vdos]   Loaded shape: {D.shape}")
    # Symmetrize (numerical)
    D = 0.5 * (D + D.T)
    return D


def compute_vdos(D, n_atoms, n_bins=300, eigvecs=False):
    """Diagonalize dynamical matrix, return eigenvalues and VDOS.

    Args:
        D : (3N, 3N) symmetric mass-weighted Hessian.
        n_bins : number of frequency bins for VDOS.
    Returns:
        freqs_thz   : (3N,) eigenfrequencies in THz
        vdos        : (n_bins,) histogram counts
        bin_centers : (n_bins,) bin centers in THz
        modes       : (3N, 3N) eigenvectors  if eigvecs=True, else None
    """
    print(f"[vdos] Diagonalizing {D.shape[0]} × {D.shape[1]} Hessian ...")
    if eigvecs:
        w, V = np.linalg.eigh(D)
    else:
        w = np.linalg.eigvalsh(D)
        V = None
    # w is ω² in eV/(amu·Å²). Convert sqrt(|w|) × LAMMPS_TO_THZ → THz
    # Negative w → imaginary frequencies (give negative freq for plotting)
    sign = np.sign(w)
    freqs_thz = sign * np.sqrt(np.abs(w)) * LAMMPS_TO_THZ

    n_imag = (freqs_thz < -0.05).sum()
    n_zero = ((freqs_thz >= -0.05) & (freqs_thz < 0.05)).sum()
    print(f"[vdos]   Total modes      : {len(freqs_thz)}")
    print(f"[vdos]   Imaginary (<−0.05 THz): {n_imag}")
    print(f"[vdos]   Near-zero (±0.05 THz) : {n_zero}  (≈ 3 acoustic + drift)")
    print(f"[vdos]   Max frequency    : {freqs_thz.max():.2f} THz")

    # Build VDOS (positive frequencies only; ignore imaginary modes)
    pos = freqs_thz[freqs_thz > 0.05]
    f_max = float(pos.max()) * 1.05
    bin_edges = np.linspace(0, f_max, n_bins + 1)
    hist, _ = np.histogram(pos, bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # Normalize: integral over positive freqs should be ~ 3N - 3 (subtract translations)
    df = bin_edges[1] - bin_edges[0]
    vdos = hist / df  # density per THz

    return freqs_thz, vdos, bin_centers, V


def plot_vdos(freqs_thz, vdos, bin_centers, struct, outfile):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel 1: VDOS
    ax = axes[0, 0]
    ax.fill_between(bin_centers, vdos, alpha=0.5, color="steelblue",
                     edgecolor="black", linewidth=0.5)
    ax.set_xlabel("ν (THz)", fontsize=11)
    ax.set_ylabel("g(ν) (modes / THz)", fontsize=11)
    ax.set_title("Vibrational density of states", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    # Mark frequencies in cm⁻¹ on secondary axis: 1 THz = 33.36 cm⁻¹
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()[0] * 33.36, ax.get_xlim()[1] * 33.36)
    ax2.set_xlabel("ν (cm⁻¹)", fontsize=10)

    # Panel 2: Boson peak — g(ω)/ω²
    ax = axes[0, 1]
    pos_mask = bin_centers > 0.5
    bp = np.zeros_like(vdos)
    bp[pos_mask] = vdos[pos_mask] / (bin_centers[pos_mask] ** 2)
    ax.plot(bin_centers[pos_mask], bp[pos_mask], "-", color="darkred", lw=1.5)
    ax.set_xlabel("ν (THz)", fontsize=11)
    ax.set_ylabel("g(ν) / ν²  (boson peak indicator)", fontsize=11)
    ax.set_title("Boson peak — amorphous fingerprint\n"
                 "(excess over Debye ν² law)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, min(15, bin_centers[pos_mask].max()))
    ax.grid(alpha=0.3)
    # Mark boson peak position
    if pos_mask.sum() > 5:
        bp_smooth = np.convolve(bp[pos_mask], np.ones(5)/5, mode="same")
        if bp_smooth.max() > 0:
            i_peak = np.argmax(bp_smooth)
            ax.axvline(bin_centers[pos_mask][i_peak], color="green", ls="--",
                        lw=1.2, label=f"Peak at ν={bin_centers[pos_mask][i_peak]:.2f} THz")
            ax.legend(fontsize=10)

    # Panel 3: Cumulative integrated DOS
    ax = axes[1, 0]
    cum = np.cumsum(vdos) * (bin_centers[1] - bin_centers[0])
    ax.plot(bin_centers, cum, "-", color="purple", lw=1.5)
    ax.axhline(3 * struct["n_atoms"] - 3, color="red", ls="--", lw=1.0,
                label=f"3N − 3 = {3*struct['n_atoms']-3}")
    ax.set_xlabel("ν (THz)", fontsize=11)
    ax.set_ylabel("Cumulative # modes", fontsize=11)
    ax.set_title("Cumulative integrated VDOS (should ≈ 3N−3)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 4: Summary
    ax = axes[1, 1]
    ax.axis("off")
    n_imag = (freqs_thz < -0.05).sum()
    n_zero = ((freqs_thz >= -0.05) & (freqs_thz < 0.05)).sum()
    n_real = (freqs_thz > 0.05).sum()
    lines = ["VDOS summary:", "─" * 38]
    lines.append(f"  N atoms     : {struct['n_atoms']}")
    lines.append(f"  3N modes    : {3*struct['n_atoms']}")
    lines.append(f"  Real modes  : {n_real}")
    lines.append(f"  Imaginary   : {n_imag}  (saddles / disorder)")
    lines.append(f"  Zero-modes  : {n_zero}  (~ 3 acoustic)")
    lines.append("")
    lines.append(f"  Max freq    : {freqs_thz.max():.2f} THz  "
                  f"({freqs_thz.max()*33.36:.0f} cm⁻¹)")
    if n_real > 0:
        pos_f = freqs_thz[freqs_thz > 0.05]
        lines.append(f"  Mean ν       : {pos_f.mean():.2f} THz")
        lines.append(f"  Median ν     : {np.median(pos_f):.2f} THz")
    lines.append("")
    pos_mask = bin_centers > 0.5
    if pos_mask.sum() > 5:
        bp = vdos[pos_mask] / (bin_centers[pos_mask] ** 2)
        bp_smooth = np.convolve(bp, np.ones(5)/5, mode="same")
        if bp_smooth.max() > 0:
            i_peak = np.argmax(bp_smooth)
            lines.append(f"Boson peak:")
            lines.append(f"  Position : ν = {bin_centers[pos_mask][i_peak]:.2f} THz")
            lines.append(f"           = {bin_centers[pos_mask][i_peak]*33.36:.0f} cm⁻¹")
    lines.append("")
    lines.append("Reference Raman (Mera 2010):")
    lines.append("  D-band of free C  : ~1350 cm⁻¹ = 40.5 THz")
    lines.append("  G-band of free C  : ~1580 cm⁻¹ = 47.4 THz")
    lines.append("  Si-N stretch       : ~850 cm⁻¹ = 25.5 THz")
    lines.append("  Si-C stretch       : ~700 cm⁻¹ = 21.0 THz")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=10, verticalalignment="top", family="monospace")

    x = composition(struct)
    fig.suptitle(f"Vibrational density of states — a-SiCN  "
                 f"(N={struct['n_atoms']}, ρ={density_g_per_cc(struct):.3f} g/cc, "
                 f"Si{x[0]*100:.1f}/C{x[1]*100:.1f}/N{x[2]*100:.1f})",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(outfile, dpi=120, bbox_inches="tight")
    plt.close()


def save_summary(freqs_thz, vdos, bin_centers, struct, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_real = (freqs_thz > 0.05).sum()
    n_imag = (freqs_thz < -0.05).sum()
    pos_f = freqs_thz[freqs_thz > 0.05]
    L = []
    L.append("Vibrational Density of States (VDOS)")
    L.append("=" * 70)
    L.append(f"  System: {struct['n_atoms']} atoms")
    L.append(f"  3N total modes : {3*struct['n_atoms']}")
    L.append(f"  Real modes     : {n_real}")
    L.append(f"  Imaginary      : {n_imag}  (negative eigenvalues)")
    L.append("")
    if n_real > 0:
        L.append(f"  Max freq     : {freqs_thz.max():.3f} THz   "
                  f"({freqs_thz.max()*33.36:.1f} cm⁻¹)")
        L.append(f"  Mean freq    : {pos_f.mean():.3f} THz   "
                  f"({pos_f.mean()*33.36:.1f} cm⁻¹)")
        L.append(f"  Median freq  : {np.median(pos_f):.3f} THz   "
                  f"({np.median(pos_f)*33.36:.1f} cm⁻¹)")
    L.append("")
    pos_mask = bin_centers > 0.5
    if pos_mask.sum() > 5:
        bp = vdos[pos_mask] / (bin_centers[pos_mask] ** 2)
        bp_smooth = np.convolve(bp, np.ones(5)/5, mode="same")
        if bp_smooth.max() > 0:
            i_peak = np.argmax(bp_smooth)
            L.append(f"Boson peak position:")
            L.append(f"  ν   = {bin_centers[pos_mask][i_peak]:.3f} THz")
            L.append(f"      = {bin_centers[pos_mask][i_peak]*33.36:.1f} cm⁻¹")
    L.append("")
    L.append("─ Interpretation ─")
    L.append("  Real VDOS peaks correspond to characteristic vibrations.")
    L.append("  Boson peak (g(ω)/ω² maximum at ν ≈ 1-5 THz) is the")
    L.append("  amorphous signature absent in crystalline counterparts.")
    L.append("")
    L.append("Reference Raman peaks for SiCN (Mera 2010, Saha 2005):")
    L.append("  Free-C D-band  ~ 1350 cm⁻¹ = 40.5 THz")
    L.append("  Free-C G-band  ~ 1580 cm⁻¹ = 47.4 THz")
    L.append("  Si-N stretch    ~ 850 cm⁻¹ = 25.5 THz")
    L.append("  Si-C stretch    ~ 700 cm⁻¹ = 21.0 THz")
    (out_dir / "vdos_summary.txt").write_text("\n".join(L))

    # Save VDOS data
    data = np.column_stack([bin_centers, vdos, bin_centers * 33.36])
    header = "nu_THz  g(nu)_per_THz  nu_cm-1"
    np.savetxt(out_dir / "vdos.txt", data, header=header, fmt="%.6e")


def run(input_file, hessian_file=None, out_dir="analysis/14_vdos"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[vdos] Loading {input_file} ...")
    struct = load_structure(input_file)
    print(f"[vdos] {struct['n_atoms']} atoms")

    if hessian_file is None:
        # Try to find Hessian file automatically
        candidates = [Path(input_file).parent / "dynmat.dat",
                       Path("dynmat.dat")]
        for c in candidates:
            if c.exists():
                hessian_file = str(c)
                print(f"[vdos] Auto-detected Hessian: {hessian_file}")
                break

    if hessian_file is None or not Path(hessian_file).exists():
        print(f"[vdos] ⚠️ No Hessian file found.")
        print(f"[vdos]    To compute VDOS, first run LAMMPS Hessian script:")
        print(f"[vdos]      $ lmp -in in_compute_hessian.lmp")
        print(f"[vdos]    Then re-run with: --hessian dynmat.dat")
        # Save a placeholder summary
        L = [
            "VDOS analysis SKIPPED — Hessian file not provided.",
            "",
            "To compute VDOS:",
            "  1. Copy in_compute_hessian.lmp + SiCN.tersoff + SiCN_300K_final.data",
            "  2. Run:  lmp -in in_compute_hessian.lmp",
            "  3. This produces dynmat.dat (~ several MB for 7775 atoms)",
            "  4. Re-run:  python3 sicn_vdos.py SiCN_300K_final.data --hessian dynmat.dat",
        ]
        (out_dir / "vdos_summary.txt").write_text("\n".join(L))
        return None

    print(f"[vdos] Loading Hessian from {hessian_file} ...")
    D = load_hessian(hessian_file, struct["n_atoms"])

    freqs_thz, vdos, bin_centers, _ = compute_vdos(D, struct["n_atoms"])

    plot_vdos(freqs_thz, vdos, bin_centers, struct, out_dir / "vdos.png")
    save_summary(freqs_thz, vdos, bin_centers, struct, out_dir)
    print(f"[vdos] → Saved vdos.png, vdos.txt, vdos_summary.txt")
    return freqs_thz, vdos, bin_centers


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="LAMMPS data file")
    ap.add_argument("--hessian", default=None,
                    help="Hessian file (dynmat.dat). Auto-detect if omitted.")
    ap.add_argument("--out", default="analysis/14_vdos",
                    help="output directory")
    a = ap.parse_args()
    run(a.input, a.hessian, a.out)
