#!/usr/bin/env python3
"""
sicn_gr_extended.py — Extended partial g(r) to 20 Å with shell analysis.

WHY:
  v7 LAMMPS RDF used cutoff 6 Å (small-box limit). For 2nd/3rd shell and
  long-range structure we need r up to ~L/2 ≈ 20 Å.  This module computes
  g_αβ(r) for all 6 pairs (Si-Si, Si-C, Si-N, C-C, C-N, N-N) from a
  snapshot or averaged over multiple frames.

  NOTE: SRO/IRO/MRO are structural definitions, NOT distance bins:
    - SRO: single structural unit (CN, bond length, bond angle)
    - IRO: connectivity between units (corner/edge/face sharing, dihedrals)
    - MRO: collective structure (rings, clusters)
  This module's r-range shading is descriptive only.

  Also extracts:
    - 1st, 2nd, 3rd, 4th coordination shell positions and heights
    - Integrated CN(r): N(r) = 4π ρ ∫_0^r r'² g(r') dr'
    - Per-pair higher-shell CN at 2nd-shell minimum

INPUT:
  LAMMPS data file (final) or dump trajectory (multi-frame averaging).

OUTPUT:
  <out>/gr_extended.txt    — table of r, g_partials, integrated CNs
  <out>/gr_extended.png    — 6-panel partial g(r) + total g(r)
  <out>/shells.txt         — text summary of shell positions and heights
  <out>/higher_cn.txt      — 2nd/3rd shell coordination numbers

USAGE:
  python3 sicn_gr_extended.py SiCN_300K_final.data --out analysis/MRO/
  python3 sicn_gr_extended.py dump_T0300.lammpstrj --out analysis/MRO/T0300 --frames 20
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sicn_common import (
    TYPE_SI, TYPE_C, TYPE_N, TYPE_NAME, PAIRS, PAIR_TYPES,
    parse_lammps_data, parse_lammps_dump, load_structure,
    pbc_distance, number_density, composition, density_g_per_cc,
    np_trapezoid,
)

# ─────────────────────────────────────────────────────────────────────────────
# RDF computation (KDTree-accelerated, type-aware)
# ─────────────────────────────────────────────────────────────────────────────
def compute_partial_rdf(struct, r_max=20.0, n_bins=800):
    """Compute partial g_αβ(r) for all 6 pairs.

    Returns:
        r       : (n_bins,) bin centers
        g_part  : dict {(t1,t2): g(r)}   t1 ≤ t2
        n_part  : dict {(t1,t2): N(r)}   running integral (per central atom)
    """
    from scipy.spatial import cKDTree

    coords = struct["coords"]
    types = struct["types"]
    box = struct["box"]
    L = box[:, 1] - box[:, 0]
    V = float(np.prod(L))
    N_total = len(types)

    # Wrap into [0, L)
    coords_w = (coords - box[:, 0][None, :]) % L[None, :]

    # Half-box safety
    r_max_safe = min(r_max, 0.49 * float(L.min()))
    if r_max_safe < r_max:
        print(f"  [gr_extended] WARNING: r_max {r_max} > L/2; capping at {r_max_safe:.2f} Å")
    r_max = r_max_safe

    dr = r_max / n_bins
    edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (edges[:-1] + edges[1:])

    tree = cKDTree(coords_w, boxsize=L)

    # Type masks → counts per type
    Nt = {t: int(np.sum(types == t)) for t in [TYPE_SI, TYPE_C, TYPE_N]}

    # Histogram per (ti, tj) pair, ti ≤ tj
    hists = {key: np.zeros(n_bins, dtype=np.int64) for key in PAIR_TYPES}

    # Loop over all pairs within r_max (efficient with KDTree)
    pairs = tree.query_pairs(r=r_max, output_type="ndarray")
    if pairs.size > 0:
        d = np.linalg.norm(
            (coords_w[pairs[:, 0]] - coords_w[pairs[:, 1]])
            - L * np.round((coords_w[pairs[:, 0]] - coords_w[pairs[:, 1]]) / L),
            axis=1,
        )
        bin_idx = np.clip((d / dr).astype(int), 0, n_bins - 1)
        ti = types[pairs[:, 0]]
        tj = types[pairs[:, 1]]
        # canonical (min,max) ordering
        t_lo = np.minimum(ti, tj)
        t_hi = np.maximum(ti, tj)
        for key in PAIR_TYPES:
            mask = (t_lo == key[0]) & (t_hi == key[1])
            np.add.at(hists[key], bin_idx[mask], 1)

    # Normalize to g(r) — Faber-Ziman partial:
    #   g_αβ(r) = V/(N_α N_β) × dN_αβ/(4π r² dr) × (1 + δ_αβ)
    # where dN_αβ is the count in shell.  For α=β we double-count internally,
    # so divide by 2; query_pairs already gives unique pairs so factor cancels.
    # Standard form: g_αβ(r) = V × hist_αβ / (N_α N_β × 4π r² dr × (1 if α≠β else ½))
    g_part = {}
    n_part = {}
    for key in PAIR_TYPES:
        t1, t2 = key
        Na, Nb = Nt[t1], Nt[t2]
        if Na == 0 or Nb == 0:
            g_part[key] = np.zeros(n_bins)
            n_part[key] = np.zeros(n_bins)
            continue
        shell_vol = 4 * np.pi * r_centers ** 2 * dr
        shell_vol[shell_vol < 1e-12] = 1e-12
        if t1 == t2:
            # query_pairs returns unique unordered pairs → count is already
            # the number of α–α pairs; ideal pair density = N(N-1)/(2V)
            ideal = Na * (Na - 1) / (2 * V) * shell_vol
        else:
            ideal = Na * Nb / V * shell_vol
        g_part[key] = hists[key] / np.maximum(ideal, 1e-12)
        # Running coordination number from atom-α perspective
        # N_α→β(r) = 4πρ_β ∫ r'² g_αβ(r') dr' = (2*hist/Na) for α=β, hist/Na for α≠β
        if t1 == t2:
            n_part[key] = np.cumsum(2 * hists[key] / Na)
        else:
            n_part[key] = np.cumsum(hists[key] / Na)

    return r_centers, g_part, n_part


def compute_partial_rdf_avg(structs, r_max=20.0, n_bins=800):
    """Average partial g(r) over multiple frames (structs is a list)."""
    r0 = None
    g_sum = {key: None for key in PAIR_TYPES}
    n_sum = {key: None for key in PAIR_TYPES}
    for s in structs:
        r, g, n = compute_partial_rdf(s, r_max=r_max, n_bins=n_bins)
        if r0 is None:
            r0 = r
            g_sum = {k: np.zeros_like(r) for k in PAIR_TYPES}
            n_sum = {k: np.zeros_like(r) for k in PAIR_TYPES}
        for k in PAIR_TYPES:
            g_sum[k] += g[k]
            n_sum[k] += n[k]
    n = len(structs)
    for k in PAIR_TYPES:
        g_sum[k] /= n
        n_sum[k] /= n
    return r0, g_sum, n_sum


# ─────────────────────────────────────────────────────────────────────────────
# Shell analysis
# ─────────────────────────────────────────────────────────────────────────────
def find_shells(r, g, max_shells=4, min_peak_height=1.1, smooth_window=5):
    """Find 1st, 2nd, ... shell peaks and following minima.
    Returns list of (r_peak, g_peak, r_min_after, g_min_after).
    """
    from scipy.signal import find_peaks
    # Light smoothing for robust peak picking
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        g_smooth = np.convolve(g, kernel, mode="same")
    else:
        g_smooth = g

    peaks, _ = find_peaks(g_smooth, height=min_peak_height, distance=10)
    mins,  _ = find_peaks(-g_smooth, distance=5)

    shells = []
    for p in peaks[:max_shells]:
        # find first minimum after this peak
        after = mins[mins > p]
        if len(after) == 0:
            rmin, gmin = r[-1], g[-1]
        else:
            mi = after[0]
            rmin, gmin = r[mi], g[mi]
        shells.append({
            "r_peak": float(r[p]),
            "g_peak": float(g[p]),
            "r_min":  float(rmin),
            "g_min":  float(gmin),
        })
    return shells


def integrate_cn_to_r(r, g, rho_partner, r_cutoff):
    """Integrate 4π ρ_β ∫ r'² g_αβ(r') dr' up to r_cutoff."""
    mask = r <= r_cutoff
    if not mask.any():
        return 0.0
    integ = np_trapezoid(r[mask] ** 2 * g[mask], r[mask])
    return 4 * np.pi * rho_partner * integ


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_gr_extended(r, g_part, n_part, struct, outfile, title_suffix=""):
    """6-panel partial g(r) + 1 panel total g(r) + integrated CN inset.
    NOTE: This is the original combined plot. See plot_gr_total() and
    plot_gr_partials_overlay() for additional dedicated plots.
    """
    from sicn_common import composition
    x = composition(struct)
    x_map = {TYPE_SI: x[0], TYPE_C: x[1], TYPE_N: x[2]}
    rho = number_density(struct)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.32)

    # 6 partials in 2×3 layout
    for k, (key, label) in enumerate(zip(PAIR_TYPES, PAIRS)):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        ax.plot(r, g_part[key], color="navy", lw=1.4)
        ax.axhline(1.0, color="gray", ls=":", lw=0.7)
        ax.set_title(f"$g_{{{label[0]}-{label[1]}}}(r)$", fontsize=11, fontweight="bold")
        ax.set_xlabel("r (Å)", fontsize=9)
        ax.set_xlim(0.5, r.max())
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
        # Annotate shells
        shells = find_shells(r, g_part[key])
        for i, sh in enumerate(shells[:3]):
            ax.axvline(sh["r_peak"], color=f"C{i}", ls="--", lw=0.7, alpha=0.6)
            ax.text(sh["r_peak"], ax.get_ylim()[1] * (0.92 - 0.08 * i),
                    f" {sh['r_peak']:.2f}", fontsize=7, color=f"C{i}")

    # Total g(r) (Faber-Ziman number-weighted)
    ax_t = fig.add_subplot(gs[2, :2])
    g_tot = compute_total_gr(r, g_part, struct)
    ax_t.plot(r, g_tot, color="black", lw=1.6, label="Total $g(r)$ (FZ-weighted)")
    ax_t.axhline(1.0, color="gray", ls=":", lw=0.7)
    ax_t.set_xlabel("r (Å)", fontsize=10)
    ax_t.set_ylabel("g(r)", fontsize=10)
    ax_t.set_title(f"Total $g(r)$ — 2nd/3rd shell + long-range region",
                   fontsize=11, fontweight="bold")
    ax_t.set_xlim(0.5, r.max())
    ax_t.grid(alpha=0.3)
    # Distance-based shading (purely descriptive; IRO/MRO are structural
    # definitions and need ring/dihedral analysis, not just r ranges)
    r_lim = r.max()
    ax_t.axvspan(5.0, min(10.0, r_lim), color="orange", alpha=0.10,
                 label="2nd shell (5–10 Å)")
    if r_lim > 10.0:
        ax_t.axvspan(10.0, r_lim, color="purple", alpha=0.10,
                     label=f"long-range (10–{r_lim:.0f} Å)")
    ax_t.legend(loc="upper right", fontsize=9)

    # Integrated CN(r) summary
    ax_n = fig.add_subplot(gs[2, 2])
    for key, label in zip(PAIR_TYPES, PAIRS):
        ax_n.plot(r, n_part[key], lw=1.2, label=f"{label[0]}–{label[1]}")
    ax_n.set_xlabel("r (Å)", fontsize=10)
    ax_n.set_ylabel(r"$N_{\alpha \rightarrow \beta}(r)$", fontsize=10)
    ax_n.set_title("Integrated coordination number", fontsize=11, fontweight="bold")
    ax_n.set_xlim(0.5, r.max())
    ax_n.grid(alpha=0.3)
    ax_n.legend(fontsize=8, ncol=2)

    fig.suptitle(f"Extended partial g(r) — a-SiCN  "
                 f"(ρ = {density_g_per_cc(struct):.3f} g/cc, "
                 f"N = {struct['n_atoms']}) {title_suffix}",
                 fontsize=13, fontweight="bold", y=0.995)
    plt.savefig(outfile, dpi=120, bbox_inches="tight")
    plt.close()


def compute_total_gr(r, g_part, struct):
    """Compute total g(r) — Faber-Ziman number-weighted average of partials.

        g_total(r) = Σ_αβ w_αβ × g_αβ(r) / Σ_αβ w_αβ
        w_αβ = x_α × x_β × (1 if α=β else 2)

    This is what neutron/X-ray total g(r) experiments measure (with
    proper coherent scattering length weighting; here we use x_α x_β
    for simplicity, which is the 'number-weighted' total).
    """
    from sicn_common import composition
    x = composition(struct)
    x_map = {TYPE_SI: x[0], TYPE_C: x[1], TYPE_N: x[2]}
    g_tot = np.zeros_like(r)
    w_sum = 0.0
    for key in PAIR_TYPES:
        t1, t2 = key
        w = x_map[t1] * x_map[t2] * (1 if t1 == t2 else 2)
        g_tot += w * g_part[key]
        w_sum += w
    if w_sum > 0:
        g_tot /= w_sum
    return g_tot


def plot_gr_total(r, g_part, struct, outfile, title_suffix="", split_r=None):
    """Dedicated plot — total g(r) only, large and clear.

    split_r : (optional) r 위치를 표시 — LAMMPS 시간평균 vs single-frame 경계
    """
    g_tot = compute_total_gr(r, g_part, struct)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(r, g_tot, color="black", lw=2.0, label="Total $g(r)$")
    ax.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.7)
    # Annotate first 4 peaks
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    g_smooth = gaussian_filter1d(g_tot, 1.0)
    peaks, _ = find_peaks(g_smooth, height=1.05, distance=10)
    for k, p in enumerate(peaks[:6]):
        ax.axvline(r[p], color=f"C{k}", ls="--", lw=0.8, alpha=0.6)
        ax.text(r[p], g_tot[p] + 0.1, f"{r[p]:.2f} Å",
                ha="center", fontsize=9, color=f"C{k}", fontweight="bold")
    # Distance-based shading (descriptive only — IRO/MRO are structural
    # definitions requiring ring/dihedral analysis, not just r ranges).
    # See sicn_dihedral.py (IRO) and sicn_rings.py (MRO) for structural analyses.
    r_lim = r.max()
    # 2nd shell region: 5–10 Å
    ax.axvspan(5.0, min(10.0, r_lim), color="orange", alpha=0.10,
               label="2nd shell region (5–10 Å)")
    # Long-range region: 10 Å 이상
    if r_lim > 10.0:
        ax.axvspan(10.0, r_lim, color="purple", alpha=0.10,
                   label=f"long-range region (10–{r_lim:.0f} Å)")
    # Mark split point between LAMMPS time-avg and single-frame data
    if split_r is not None and split_r < r_lim:
        ax.axvline(split_r, color="red", ls="-.", lw=1.2, alpha=0.6)
        ymax = ax.get_ylim()[1]
        ax.text(split_r + 0.2, ymax * 0.95,
                f"← LAMMPS time-avg | single-frame →",
                fontsize=8, color="red", style="italic")
    ax.set_xlabel("r (Å)", fontsize=12)
    ax.set_ylabel("Total g(r)  (Faber-Ziman number-weighted)", fontsize=12)
    ax.set_title(f"Total g(r) — a-SiCN  "
                 f"(ρ = {density_g_per_cc(struct):.3f} g/cc, N = {struct['n_atoms']}) "
                 f"{title_suffix}",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0.5, r.max())
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(outfile, dpi=120, bbox_inches="tight")
    plt.close()


def plot_gr_partials_overlay(r, g_part, struct, outfile, title_suffix=""):
    """Dedicated plot — all 6 partials on the same axes (overlay) for comparison."""
    fig, ax = plt.subplots(figsize=(12, 6.5))

    colors = {("Si","Si"): "#1f77b4",  ("Si","C"):  "#ff7f0e",
              ("Si","N"): "#d62728",   ("C","C"):   "#2ca02c",
              ("C","N"):  "#9467bd",   ("N","N"):   "#8c564b"}

    for key, label in zip(PAIR_TYPES, PAIRS):
        c = colors[label]
        ax.plot(r, g_part[key], color=c, lw=1.5,
                label=f"$g_{{{label[0]}-{label[1]}}}(r)$")
    ax.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.set_xlabel("r (Å)", fontsize=12)
    ax.set_ylabel("Partial g(r)", fontsize=12)
    ax.set_title(f"Partial g(r) overlay — a-SiCN  "
                 f"(ρ = {density_g_per_cc(struct):.3f} g/cc, N = {struct['n_atoms']}) "
                 f"{title_suffix}",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0.5, r.max())
    ax.legend(fontsize=11, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(outfile, dpi=120, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Save text summaries
# ─────────────────────────────────────────────────────────────────────────────
def save_summary(r, g_part, n_part, struct, out_dir):
    """Write gr_extended.txt, shells.txt, higher_cn.txt."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # gr_extended.txt — r, g_partials, n_partials
    header = "# r(Å) " + " ".join([f"g_{a}{b}" for a,b in PAIRS]) + \
             " " + " ".join([f"N_{a}{b}" for a,b in PAIRS])
    data = np.column_stack([r] + [g_part[k] for k in PAIR_TYPES]
                                + [n_part[k] for k in PAIR_TYPES])
    np.savetxt(out_dir / "gr_extended.txt", data, header=header[2:],
               fmt="%.6f")

    # shells.txt
    L = []
    L.append(f"Shell analysis — a-SiCN N={struct['n_atoms']}")
    L.append("=" * 78)
    L.append(f"{'pair':<6}{'shell':<6}{'r_peak (Å)':>12}{'g_peak':>10}"
             f"{'r_min (Å)':>12}{'g_min':>10}")
    L.append("-" * 78)
    for key, label in zip(PAIR_TYPES, PAIRS):
        shells = find_shells(r, g_part[key])
        for i, sh in enumerate(shells):
            L.append(f"{label[0]}-{label[1]:<4}{i+1:<6}{sh['r_peak']:>12.3f}"
                     f"{sh['g_peak']:>10.3f}{sh['r_min']:>12.3f}{sh['g_min']:>10.3f}")
        if not shells:
            L.append(f"{label[0]}-{label[1]:<4}    (no resolvable peaks)")
        L.append("")
    (out_dir / "shells.txt").write_text("\n".join(L))

    # higher_cn.txt — CN at 1st & 2nd minimum
    L2 = []
    L2.append("Higher-shell coordination numbers")
    L2.append("=" * 60)
    L2.append(f"{'pair':<8}{'1st-shell CN':>15}{'2nd-shell CN':>18}")
    L2.append("-" * 60)
    for key, label in zip(PAIR_TYPES, PAIRS):
        shells = find_shells(r, g_part[key])
        cn1 = cn2 = np.nan
        if shells:
            i1 = np.searchsorted(r, shells[0]["r_min"])
            cn1 = n_part[key][min(i1, len(r) - 1)]
        if len(shells) >= 2:
            i2 = np.searchsorted(r, shells[1]["r_min"])
            cn2 = n_part[key][min(i2, len(r) - 1)]
        L2.append(f"{label[0]}-{label[1]:<6}{cn1:>15.3f}{cn2:>18.3f}")
    (out_dir / "higher_cn.txt").write_text("\n".join(L2))

    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run(input_file, out_dir="analysis/MRO", r_max=20.0, n_bins=800, n_frames=1,
        rdf_file=None):
    """
    input_file : .data 파일 (구조)  OR  .lammpstrj (dump trajectory)
    rdf_file   : (선택) LAMMPS rdf 시간평균 .txt 파일.
                 이게 있으면 g(r)을 *자체 계산하지 않고* LAMMPS 평균을 직접 사용.
                 구조 정보 (composition, density)는 input_file에서 가져옴.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Branch A: LAMMPS 시간평균 rdf 파일이 주어진 경우 (가장 정확)
    if rdf_file is not None:
        print(f"[gr_extended] Reading LAMMPS time-averaged RDF: {rdf_file}")
        print(f"[gr_extended] Structure info from: {input_file}")
        struct = load_structure(input_file)
        print(f"[gr_extended] {struct['n_atoms']} atoms, ρ = {density_g_per_cc(struct):.3f} g/cc")

        # Read LAMMPS rdf — same parser as in sicn_bt.py
        from sicn_bt import read_lammps_rdf
        r_l, g_l = read_lammps_rdf(rdf_file)
        print(f"[gr_extended] LAMMPS rdf: {len(r_l)} bins, "
              f"r = {r_l[0]:.3f}–{r_l[-1]:.3f} Å (dr ≈ {r_l[1]-r_l[0]:.4f} Å)")

        # ── Hybrid mode: if user requests r_max > LAMMPS cutoff,
        #    extend with self-computed g(r) from .data (single frame)
        L_half = (struct["box"][0, 1] - struct["box"][0, 0]) / 2
        target_rmax = r_max if r_max > 0 else min(20.0, L_half * 0.95)

        if target_rmax > r_l[-1] + 0.5:  # need extension
            # Self-compute g(r) on the SAME bin grid up to target_rmax
            print(f"[gr_extended] ⭐ HYBRID mode:")
            print(f"               0 – {r_l[-1]:.2f} Å : LAMMPS time-averaged (700 frames)")
            print(f"               {r_l[-1]:.2f} – {target_rmax:.2f} Å : .data single frame (long-range region)")
            dr_l = r_l[1] - r_l[0]
            # Use the SAME bin width as LAMMPS rdf to ensure smooth join
            n_bins_full = int(np.round(target_rmax / dr_l))
            print(f"[gr_extended] Computing self-RDF up to {target_rmax:.2f} Å "
                  f"(n_bins={n_bins_full}, dr={dr_l:.4f})...")
            r_s, g_s, _ = compute_partial_rdf(struct, r_max=target_rmax,
                                               n_bins=n_bins_full)

            # Merge: 0 – r_l[-1] from LAMMPS, > r_l[-1] from self
            n_lammps = len(r_l)
            n_total = len(r_s)
            r = r_s.copy()
            g_part = {}
            for key in PAIR_TYPES:
                merged = np.zeros(n_total)
                # First n_lammps bins from LAMMPS
                merged[:n_lammps] = g_l[key][:n_lammps] if len(g_l[key]) >= n_lammps else g_l[key]
                # Beyond from self-computed
                merged[n_lammps:] = g_s[key][n_lammps:]
                g_part[key] = merged
            split_r = r_l[-1]  # for marking on plot
            title_suffix = (f"(0–{split_r:.1f} Å LAMMPS time-avg, "
                            f"{split_r:.1f}–{target_rmax:.1f} Å single frame)")
        else:
            # No extension needed — use LAMMPS rdf as-is
            r = r_l
            g_part = g_l
            split_r = None
            title_suffix = "(LAMMPS time-averaged)"
            print(f"[gr_extended] Using LAMMPS rdf as-is (r_max sufficient)")

        # Compute running CN from final g_part
        from sicn_common import number_density
        x = composition(struct)
        x_map = {TYPE_SI: x[0], TYPE_C: x[1], TYPE_N: x[2]}
        rho = number_density(struct)
        dr = r[1] - r[0]
        n_part = {}
        for key in PAIR_TYPES:
            t1, t2 = key
            shell_vol = 4 * np.pi * r**2 * dr
            rho_partner = rho * x_map[t2]
            # integrated CN per central α atom: 4π ρ_β ∫ r² g_αβ(r) dr
            n_part[key] = np.cumsum(shell_vol * rho_partner * g_part[key])

    # Branch B: 기존 동작 — .data 단일 frame OR .lammpstrj multi-frame
    else:
        print(f"[gr_extended] Loading {input_file} ...")
        is_dump = ".lammpstrj" in str(input_file)
        if is_dump and n_frames > 1:
            all_frames = parse_lammps_dump(input_file, frames="all")
            use = all_frames[-n_frames:] if n_frames < len(all_frames) else all_frames
            print(f"[gr_extended] Averaging over {len(use)} frames")
            r, g_part, n_part = compute_partial_rdf_avg(use, r_max=r_max, n_bins=n_bins)
            struct = use[-1]
            title_suffix = f"({len(use)} frames averaged)"
        else:
            struct = load_structure(input_file)
            print(f"[gr_extended] {struct['n_atoms']} atoms, ρ = {density_g_per_cc(struct):.3f} g/cc")
            r, g_part, n_part = compute_partial_rdf(struct, r_max=r_max, n_bins=n_bins)
            title_suffix = "(single frame from .data)"
        split_r = None  # no LAMMPS/self split in branch B

    print(f"[gr_extended] Computing partial g(r) to r_max = {r[-1]:.2f} Å ...")

    # 1) 6-partial + total + integrated-CN combined plot (기존)
    plot_gr_extended(r, g_part, n_part, struct, out_dir / "gr_extended.png",
                     title_suffix=title_suffix)
    # 2) ⭐ Dedicated Total g(r) plot (split_r marker if hybrid)
    plot_gr_total(r, g_part, struct, out_dir / "gr_total.png",
                  title_suffix=title_suffix, split_r=split_r)
    # 3) ⭐ Partials overlay (6개 partial 한 panel에 비교)
    plot_gr_partials_overlay(r, g_part, struct,
                              out_dir / "gr_partials_overlay.png",
                              title_suffix=title_suffix)

    save_summary(r, g_part, n_part, struct, out_dir)

    # Save total g(r) as separate text for easy access
    g_tot = compute_total_gr(r, g_part, struct)
    np.savetxt(out_dir / "gr_total.txt", np.column_stack([r, g_tot]),
               header="r(Å)  g_total(r)  [Faber-Ziman number-weighted]",
               fmt="%.6f")

    print(f"[gr_extended] → Saved {out_dir/'gr_extended.png'}  (combined 8-panel)")
    print(f"[gr_extended] → Saved {out_dir/'gr_total.png'}     ⭐ total g(r)")
    print(f"[gr_extended] → Saved {out_dir/'gr_partials_overlay.png'}  ⭐ 6 partials overlaid")
    print(f"[gr_extended] → Saved {out_dir/'gr_total.txt'}     ⭐ total g(r) raw data")
    print(f"[gr_extended] → Saved {out_dir/'gr_extended.txt'}, shells.txt, higher_cn.txt")
    return r, g_part, n_part, struct


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="LAMMPS data file or dump trajectory")
    ap.add_argument("--rdf", default=None,
                    help="(권장) LAMMPS 시간평균 rdf .txt 파일. "
                         "이게 있으면 자체 계산 안 하고 LAMMPS 평균 사용.")
    ap.add_argument("--out", default="analysis/MRO", help="output directory")
    ap.add_argument("--r_max", type=float, default=20.0)
    ap.add_argument("--n_bins", type=int, default=800)
    ap.add_argument("--frames", type=int, default=1,
                    help="number of last frames to average (dump only)")
    a = ap.parse_args()
    run(a.input, a.out, r_max=a.r_max, n_bins=a.n_bins, n_frames=a.frames,
        rdf_file=a.rdf)
