#!/usr/bin/env python3
"""
sicn_bt.py — Bhatia-Thornton structure factor decomposition for a-SiCN.

WHY:
  Total S(Q) mixes topological and chemical information.  The Bhatia-Thornton
  decomposition separates them:
      S_NN(Q)  : number-number correlation = total topology (network MRO)
      S_CC(Q)  : concentration-concentration = chemical clustering / de-mixing
      S_NC(Q)  : cross (Q→0 asymptote signals miscibility / phase separation)
  Reference: Bhatia & Thornton, PRB 2, 3004 (1970).

  For ternary a-SiCN, the most physically meaningful pseudo-binary is
      Network (Si+N)   vs   Free carbon (C)
  because the Tersoff potential builds the Si-N network and "expels" C into
  graphenic clusters.  S_CC(Q→0) > c_C(1-c_C) ⇒ C phase-separation.

INPUT:
  Either a LAMMPS data file (this module computes partial g(r) itself),
  or a pre-computed partial RDF text file in LAMMPS 'compute rdf' format.

OUTPUT:
  <out>/bt_partials.txt   — Q, partial S_αβ(Q) for 6 pairs
  <out>/bt_components.txt — Q, S_NN(Q), S_CC(Q), S_NC(Q) (pseudo-binary (Si+N)-C)
  <out>/bt.png            — 3-panel plot: partials, BT components, FSDP analysis

USAGE:
  python3 sicn_bt.py SiCN_300K_final.data --out analysis/MRO/
  python3 sicn_bt.py rdf_300K.txt --data SiCN_300K_final.data --out analysis/MRO/
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
    load_structure, parse_lammps_data, composition, number_density,
    density_g_per_cc, np_trapezoid,
)


# ─────────────────────────────────────────────────────────────────────────────
# Read partial g(r): from LAMMPS rdf file OR compute from data file
# ─────────────────────────────────────────────────────────────────────────────
def read_lammps_rdf(filepath):
    """Read LAMMPS 'compute rdf' time-averaged output.
    Block format: header line '# Row r g(Si-Si) cn(Si-Si) g(Si-C) cn(Si-C) ...'
    Multiple time blocks — average them.
    """
    blocks, cur = [], []
    with open(filepath) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                # block separator: "TimeStep Nrows"
                try:
                    int(parts[0]); int(parts[1])
                    if cur:
                        blocks.append(np.array(cur, dtype=float))
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
        raise ValueError(f"No data blocks in {filepath}")
    arr = np.mean(np.stack(blocks), axis=0)
    r = arr[:, 1]
    # columns: 2,4,6,8,10,12 are g(r) for the 6 partials in order
    g_part = {}
    for k, key in enumerate(PAIR_TYPES):
        col = 2 + 2 * k
        if col < arr.shape[1]:
            g_part[key] = arr[:, col]
        else:
            g_part[key] = np.zeros_like(r)
    return r, g_part


def compute_partial_gr_from_data(struct, r_max=10.0, n_bins=400):
    """Compute partial g(r) directly from a LAMMPS data file structure."""
    from sicn_gr_extended import compute_partial_rdf
    r, g, _ = compute_partial_rdf(struct, r_max=r_max, n_bins=n_bins)
    return r, g


# ─────────────────────────────────────────────────────────────────────────────
# Fourier transform: partial g(r) → partial S(Q)
# ─────────────────────────────────────────────────────────────────────────────
def sine_transform_partial(r, g_part, rho_total, c_alpha, c_beta,
                           q_grid=None, lorch_damp=True):
    """Compute partial S_αβ(Q) by sine transform of partial g(r).

    Faber-Ziman convention:
        S_αβ(Q) = δ_αβ + 4π ρ √(c_α c_β) ∫_0^∞ r [g_αβ(r) - 1] sin(Qr)/Q dr

    Lorch damping window suppresses termination ripples:
        M(r) = sin(π r/R)/(π r/R),  R = r_max

    Returns Q (1D), S(Q) (1D).
    """
    if q_grid is None:
        q_grid = np.linspace(0.1, 20.0, 400)
    r_max = r[-1]
    dr = r[1] - r[0]
    if lorch_damp:
        w = np.sinc(r / r_max)  # numpy sinc(x) = sin(πx)/(πx)
    else:
        w = np.ones_like(r)
    delta = 1.0 if abs(c_alpha - c_beta) < 1e-9 and c_alpha == c_beta else 0.0  # placeholder
    # We don't use delta here — caller adds δ_αβ explicitly.
    pre = 4 * np.pi * rho_total * np.sqrt(c_alpha * c_beta)
    integrand = r * (g_part - 1.0) * w  # (Nr,)
    # sin(Qr)/Q vector for all Q
    S = np.zeros_like(q_grid)
    for i, q in enumerate(q_grid):
        if q < 1e-9:
            S[i] = np_trapezoid(integrand * r, r) * pre
        else:
            S[i] = pre * np_trapezoid(integrand * np.sin(q * r) / q, r)
    return q_grid, S


def compute_partials_sq(r, g_part, struct, q_grid=None):
    """Compute partial S_αβ(Q) for all 6 pairs, Faber-Ziman convention."""
    if q_grid is None:
        q_grid = np.linspace(0.1, 20.0, 400)
    x_Si, x_C, x_N = composition(struct)
    x_map = {TYPE_SI: x_Si, TYPE_C: x_C, TYPE_N: x_N}
    rho = number_density(struct)
    S_part = {}
    for key in PAIR_TYPES:
        t1, t2 = key
        c1, c2 = x_map[t1], x_map[t2]
        q, S = sine_transform_partial(r, g_part[key], rho, c1, c2, q_grid)
        # Add Kronecker delta for diagonal
        if t1 == t2:
            S = S + 1.0   # δ_αα
        S_part[key] = S
    return q_grid, S_part


# ─────────────────────────────────────────────────────────────────────────────
# Bhatia-Thornton decomposition — pseudo-binary
# ─────────────────────────────────────────────────────────────────────────────
def bt_pseudobinary(q, S_part, struct, group_A=(TYPE_SI, TYPE_N), group_B=(TYPE_C,)):
    """Compute pseudo-binary Bhatia-Thornton S_NN, S_CC, S_NC.

    Aggregate types in group_A as species 'A', group_B as 'B'.
    Effective concentration and partials:
        c_A = sum of x_a for a in group_A
        g_AA = (1/c_A^2) Σ_{α,β∈A} x_α x_β g_αβ
        g_AB = (1/(c_A c_B)) Σ_{α∈A, β∈B} x_α x_β g_αβ
        g_BB = analogous

    Then standard binary BT:
        S_NN(Q) = c_A^2 S_AA + c_B^2 S_BB + 2 c_A c_B S_AB
        S_CC(Q) = c_A c_B [1 + c_A c_B (S_AA + S_BB - 2 S_AB)]
        S_NC(Q) = c_A c_B [c_A (S_AA - S_AB) - c_B (S_BB - S_AB)]
    """
    x_Si, x_C, x_N = composition(struct)
    x_map = {TYPE_SI: x_Si, TYPE_C: x_C, TYPE_N: x_N}

    c_A = sum(x_map[t] for t in group_A)
    c_B = sum(x_map[t] for t in group_B)
    if c_A < 1e-9 or c_B < 1e-9:
        raise ValueError("Pseudo-binary group has zero concentration")

    def S_eff(g1, g2):
        """Aggregate S for two type-groups by concentration-weighted average."""
        S_acc = np.zeros_like(q)
        w_acc = 0.0
        for a in g1:
            for b in g2:
                key = (min(a, b), max(a, b))
                if key not in S_part:
                    continue
                w = x_map[a] * x_map[b]
                S_acc += w * S_part[key]
                w_acc += w
        return S_acc / w_acc if w_acc > 0 else S_acc

    # Effective S for A-A, A-B, B-B
    # Note: S_part is already in Faber-Ziman form with c_α c_β baked in.
    # We need to UN-do that weighting before re-aggregating.
    # Easier path: define partial without √(c_α c_β) prefactor and rebuild.
    # → Use the relation: F_αβ(Q) = (S_αβ(Q) - δ_αβ) / √(c_α c_β) for FT,
    #   then aggregate. We'll just re-do the math at the level of g(r).

    # Aggregate g_AA, g_AB, g_BB directly from S_part (since the FT is linear)
    # Actually, since S_αβ already has √(c_α c_β) prefactor, we can build:
    #   c_α c_β S_αβ' where S_αβ' is partial WITHOUT the prefactor.
    # We re-derive partial without prefactor:
    rho = number_density(struct)
    # F_αβ = S_αβ_FZ - δ_αβ   has prefactor √(c_α c_β)
    # → "bare" partial:  F_αβ / √(c_α c_β)  (which is the FT of 4π ρ r[g-1])
    def S_FZ_to_bare(t1, t2):
        c1, c2 = x_map[t1], x_map[t2]
        F = S_part[(min(t1,t2), max(t1,t2))]
        if t1 == t2:
            F = F - 1.0
        return F / np.sqrt(c1 * c2)

    # S_AA, S_AB, S_BB (Faber-Ziman style, including δ_αβ for diagonal)
    def aggregate_S(gA, gB):
        bare = np.zeros_like(q)
        w = 0.0
        for a in gA:
            for b in gB:
                ca, cb = x_map[a], x_map[b]
                bare_ab = S_FZ_to_bare(a, b)
                bare += ca * cb * bare_ab
                w += ca * cb
        # Now: aggregate "bare" sum has c_α c_β weighting.
        # Convert back to FZ form for the aggregate species:
        # F_AB^aggregate * √(c_A c_B) = bare → S_AB^aggregate = bare * √(c_A c_B)/(c_A c_B) + δ
        c_a_agg = sum(x_map[a] for a in gA)
        c_b_agg = sum(x_map[b] for b in gB)
        S_agg = bare * np.sqrt(c_a_agg * c_b_agg) / (c_a_agg * c_b_agg)
        if gA == gB:
            S_agg = S_agg + 1.0
        return S_agg

    S_AA = aggregate_S(group_A, group_A)
    S_BB = aggregate_S(group_B, group_B)
    S_AB = aggregate_S(group_A, group_B)

    # Bhatia-Thornton — binary
    S_NN = c_A**2 * S_AA + c_B**2 * S_BB + 2 * c_A * c_B * S_AB
    S_CC = c_A * c_B * (1 + c_A * c_B * (S_AA + S_BB - 2 * S_AB))
    S_NC = c_A * c_B * (c_A * (S_AA - S_AB) - c_B * (S_BB - S_AB))

    # Ideal-mixing baseline for S_CC: c_A c_B (random mixing limit)
    S_CC_ideal = c_A * c_B

    return {
        "q": q,
        "S_NN": S_NN,
        "S_CC": S_CC,
        "S_NC": S_NC,
        "S_AA": S_AA, "S_BB": S_BB, "S_AB": S_AB,
        "c_A": c_A, "c_B": c_B,
        "S_CC_ideal": S_CC_ideal,
        "label_A": "+".join(TYPE_NAME[t] for t in group_A),
        "label_B": "+".join(TYPE_NAME[t] for t in group_B),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_bt(q, S_part, bt, struct, outfile):
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.32)

    # Top row: 6 partial S_αβ(Q)
    for k, (key, label) in enumerate(zip(PAIR_TYPES, PAIRS)):
        ax = fig.add_subplot(gs[0, k % 3]) if k < 3 else fig.add_subplot(gs[1, k % 3])
        ax.plot(q, S_part[key], color="navy", lw=1.3)
        ax.axhline(1.0 if key[0]==key[1] else 0.0, color="gray", ls=":", lw=0.7)
        ax.set_title(f"$S_{{{label[0]}-{label[1]}}}(Q)$ (FZ partial)",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Q (Å⁻¹)", fontsize=9)
        ax.set_xlim(0, q.max())
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)

    # Bottom-left: Bhatia-Thornton components
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(q, bt["S_NN"], color="#1f77b4", lw=1.6, label="$S_{NN}(Q)$  topology")
    ax.axhline(0, color="gray", ls=":", lw=0.7)
    ax.set_xlabel("Q (Å⁻¹)", fontsize=10)
    ax.set_ylabel("$S_{NN}(Q)$", fontsize=10)
    ax.set_title("Number-number (topology)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, q.max())
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(q, bt["S_CC"], color="#d62728", lw=1.6,
            label=f"$S_{{CC}}(Q)$  ({bt['label_A']})-({bt['label_B']})")
    ax.axhline(bt["S_CC_ideal"], color="gray", ls="--", lw=1.0,
               label=f"random-mix limit = {bt['S_CC_ideal']:.3f}")
    ax.set_xlabel("Q (Å⁻¹)", fontsize=10)
    ax.set_ylabel("$S_{CC}(Q)$", fontsize=10)
    ax.set_title("Concentration-concentration (chemistry)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0, q.max())
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 2])
    ax.plot(q, bt["S_NC"], color="#2ca02c", lw=1.6, label="$S_{NC}(Q)$")
    ax.axhline(0, color="gray", ls=":", lw=0.7)
    ax.set_xlabel("Q (Å⁻¹)", fontsize=10)
    ax.set_ylabel("$S_{NC}(Q)$", fontsize=10)
    ax.set_title("Cross (NC) — phase-separation signal",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0, q.max())
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # Mark FSDP region
    for ax in fig.axes:
        ax.axvspan(1.5, 3.5, color="orange", alpha=0.05)

    fig.suptitle(f"Bhatia-Thornton decomposition — a-SiCN  "
                 f"(N={struct['n_atoms']}, ρ={density_g_per_cc(struct):.3f} g/cc)\n"
                 f"Pseudo-binary: A = {bt['label_A']} (c={bt['c_A']:.3f}), "
                 f"B = {bt['label_B']} (c={bt['c_B']:.3f})",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.savefig(outfile, dpi=120, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────────────────────────────────────
def save_outputs(q, S_part, bt, struct, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # bt_partials.txt
    cols = [q]
    header = "Q(Å⁻¹) "
    for key, lab in zip(PAIR_TYPES, PAIRS):
        cols.append(S_part[key])
        header += f"S_{lab[0]}{lab[1]} "
    np.savetxt(out_dir / "bt_partials.txt", np.column_stack(cols),
               header=header.strip(), fmt="%.6f")

    # bt_components.txt
    data = np.column_stack([bt["q"], bt["S_NN"], bt["S_CC"], bt["S_NC"],
                            bt["S_AA"], bt["S_BB"], bt["S_AB"]])
    np.savetxt(out_dir / "bt_components.txt", data,
               header=f"Q  S_NN  S_CC  S_NC  S_AA  S_BB  S_AB"
                      f"  (A={bt['label_A']}, B={bt['label_B']}, "
                      f"c_A={bt['c_A']:.4f}, c_B={bt['c_B']:.4f}, "
                      f"S_CC_ideal={bt['S_CC_ideal']:.4f})",
               fmt="%.6f")

    # Summary text
    L = []
    L.append(f"Bhatia-Thornton analysis — a-SiCN N={struct['n_atoms']}")
    L.append("=" * 72)
    L.append(f"Pseudo-binary:  A = {bt['label_A']}  (c_A = {bt['c_A']:.4f})")
    L.append(f"                B = {bt['label_B']}  (c_B = {bt['c_B']:.4f})")
    L.append(f"S_CC random-mix limit = c_A·c_B = {bt['S_CC_ideal']:.4f}")
    L.append("")
    # Q=0 (low-Q) values
    q = bt["q"]
    iq_low = int(np.argmin(np.abs(q - 0.5)))
    iq_fsdp = int(np.argmin(np.abs(q - 2.5)))
    L.append(f"At Q ≈ 0.5 Å⁻¹  (low-Q, IRO indicator):")
    L.append(f"   S_NN = {bt['S_NN'][iq_low]:+.4f}")
    L.append(f"   S_CC = {bt['S_CC'][iq_low]:+.4f}    "
             f"({'CLUSTERING' if bt['S_CC'][iq_low] > bt['S_CC_ideal'] else 'ordering'})")
    L.append(f"   S_NC = {bt['S_NC'][iq_low]:+.4f}")
    L.append("")
    L.append(f"At Q ≈ 2.5 Å⁻¹ (FSDP region):")
    L.append(f"   S_NN = {bt['S_NN'][iq_fsdp]:+.4f}")
    L.append(f"   S_CC = {bt['S_CC'][iq_fsdp]:+.4f}")
    L.append(f"   S_NC = {bt['S_NC'][iq_fsdp]:+.4f}")
    L.append("")
    L.append("Interpretation:")
    L.append("  S_CC(Q→0) > c_A·c_B  ⇒  CHEMICAL CLUSTERING (free-C phase separation)")
    L.append("  S_CC(Q→0) < c_A·c_B  ⇒  chemical ordering (mixed bonds preferred)")
    L.append("  S_CC(Q→0) ≈ c_A·c_B  ⇒  random mixing")
    (out_dir / "bt_summary.txt").write_text("\n".join(L))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run(input_file, out_dir="analysis/MRO", data_file=None,
        r_max=10.0, n_bins=400, q_min=0.15, q_max=20.0, n_q=400):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load partial g(r): from rdf file if input ends in .txt, else compute
    if str(input_file).endswith(".txt"):
        print(f"[bt] Reading LAMMPS RDF file: {input_file}")
        r, g_part = read_lammps_rdf(input_file)
        if data_file is None:
            raise ValueError("If using RDF file, --data <data file> required for composition")
        struct = load_structure(data_file)
    else:
        print(f"[bt] Loading structure & computing g(r): {input_file}")
        struct = load_structure(input_file)
        r, g_part = compute_partial_gr_from_data(struct, r_max=r_max, n_bins=n_bins)

    print(f"[bt] {struct['n_atoms']} atoms, ρ={density_g_per_cc(struct):.3f} g/cc")
    print(f"[bt] g(r) range: {r[0]:.3f} – {r[-1]:.3f} Å ({len(r)} bins)")

    q_grid = np.linspace(q_min, q_max, n_q)
    print(f"[bt] Computing partial S(Q) ...")
    q, S_part = compute_partials_sq(r, g_part, struct, q_grid)

    print(f"[bt] Bhatia-Thornton decomposition (Si+N) vs C ...")
    bt = bt_pseudobinary(q, S_part, struct, group_A=(TYPE_SI, TYPE_N), group_B=(TYPE_C,))

    plot_bt(q, S_part, bt, struct, out_dir / "bt.png")
    save_outputs(q, S_part, bt, struct, out_dir)
    print(f"[bt] → Saved {out_dir/'bt.png'}, bt_partials.txt, bt_components.txt, bt_summary.txt")

    # Quick CLI report
    iq_low = int(np.argmin(np.abs(q - 0.5)))
    cluster = "CLUSTERING (free-C separation)" if bt["S_CC"][iq_low] > bt["S_CC_ideal"] else \
              "chemical ordering"
    print(f"\n  S_CC(Q≈0.5) = {bt['S_CC'][iq_low]:+.4f}  "
          f"vs ideal-mix {bt['S_CC_ideal']:.4f}  →  {cluster}")
    return q, S_part, bt, struct


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="LAMMPS data file OR partial RDF file (.txt)")
    ap.add_argument("--data", default=None, help="data file (required if input is .txt RDF)")
    ap.add_argument("--out", default="analysis/MRO")
    ap.add_argument("--r_max", type=float, default=10.0)
    ap.add_argument("--n_bins", type=int, default=400)
    ap.add_argument("--q_min", type=float, default=0.15)
    ap.add_argument("--q_max", type=float, default=20.0)
    ap.add_argument("--n_q", type=int, default=400)
    a = ap.parse_args()
    run(a.input, a.out, data_file=a.data,
        r_max=a.r_max, n_bins=a.n_bins,
        q_min=a.q_min, q_max=a.q_max, n_q=a.n_q)
