#!/usr/bin/env python3
"""
analyze.py — produce figures + txt for a single structure or a temperature set.

Every figure is written as PNG *and* its data as .txt; every numeric table is
written as .txt; the full console log is saved to report.txt. Output goes to
--out (default: results/<stem>/).

SINGLE structure (full SRO+MRO figure set):
  python3 scripts/analyze.py data/dump_T0300.lammpstrj --type-map 1:Si,2:C,3:N

TEMPERATURE SWEEP (auto-discovers dumps in a directory):
  python3 scripts/analyze.py data/ --sweep --type-map 1:Si,2:C,3:N
"""
from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import amorph
from amorph import core, presets, sweep
from amorph.report import Reporter
from amorph.sro import rdf, coordination
from amorph.mro import rings as mro_rings, clusters as mro_clusters
from amorph.mro import tetra_connectivity, bhatia_thornton

C_BY_SP = {"Si": "#e8a33d", "C": "#444444", "N": "#3a6ea5"}


def parse_type_map(s):
    return {int(k): v.strip() for k, v in (t.split(":") for t in s.split(","))} if s else None


def parse_prod(s):
    if not s:
        return None
    lo, hi = s.split(":")
    return (int(lo) if lo else 0, int(hi) if hi else None)


# ─────────────────────────────────────────────────────────────────────────────
def analyze_single(path, CM, rep, *, type_map, prod, stride, args):
    traj = core.select_frames(core.load(path, type_map=type_map, frames="all"),
                              frame_range=prod, stride=stride)
    sp = core.species_of(traj)
    all_pairs = core.unique_pairs(sp)              # g(r): EVERY pair (bonded or not)
    bonded = [p for p in all_pairs if CM.get(*p) > 0]  # info only

    comp = np.mean([[fr.count(e) / fr.n_atoms for e in sp] for fr in traj], axis=0)
    rho = np.mean([fr.mass_density() for fr in traj])
    rep.print(f"\n[{Path(path).name}] frames={len(traj)} atoms={traj[0].n_atoms} "
              f"species={sp}")
    rep.print("  composition mol%: " + ", ".join(f"{e} {c*100:.2f}" for e, c in zip(sp, comp)))
    rep.print(f"  mass density: {rho:.4f} g/cc")

    # 1) partial g(r) — ALL pairs, including non-bonded (C-N, N-N show avoidance)
    R = rdf.partial_rdf(traj, pairs=all_pairs, r_max=args.rmax, nbins=args.nbins,
                        n_blocks=args.nblocks)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    data = {"r": R["r"]}
    rep.print("\n[g(r)] first-peak (r, height, FWHM)   [* = non-bonded by potential]:")
    for (A, B) in all_pairs:
        g = R[(A, B)]["g"]
        nb = CM.get(A, B) <= 0
        ax.plot(R["r"], g, label=f"{A}-{B}" + ("*" if nb else ""),
                ls="--" if nb else "-")
        data[f"g_{A}{B}"] = g
        m = rdf.measure_peaks(R["r"], g, search=(0.8, None))
        rep.print(f"  {A}-{B}{'*' if nb else ' '}: r={m['peak_r']:.3f} Å  "
                  f"h={m['height']:.2f}  FWHM={m['fwhm']:.3f}")
    ax.set_xlabel("r (Å)"); ax.set_ylabel("g(r)")
    ax.set_title(f"Partial g(r) — {Path(path).stem}  (dashed* = non-bonded)")
    ax.legend(ncol=2); ax.grid(alpha=0.3)
    rep.save_fig(fig, "gr_partials", data=data); plt.close(fig)

    # 2) coordination numbers (stacked bar) ---------------------------------
    CN = coordination.coordination_numbers(traj, CM, n_blocks=args.nblocks)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    rep.print("\n[CN] coordination numbers:")
    cn_rows = [["center", *sp, "total"]]
    bottoms = np.zeros(len(sp))
    for j, B in enumerate(sp):
        vals = [CN["cn"][A][B][0] if CM.get(A, B) > 0 else 0.0 for A in sp]
        ax.bar(sp, vals, bottom=bottoms, label=f"–{B}", color=C_BY_SP.get(B))
        bottoms += np.array(vals)
    for A in sp:
        tot = CN["cn_total"][A]
        row = [A] + [f"{CN['cn'][A][B][0]:.3f}" if CM.get(A, B) > 0 else "-" for B in sp] + [f"{tot[0]:.3f}±{tot[1]:.3f}"]
        cn_rows.append(row)
        rep.print(f"  {A}: total {tot[0]:.3f}±{tot[1]:.3f}")
    ax.set_ylabel("coordination number"); ax.set_title("CN by neighbor species")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    rep.save_fig(fig, "coordination",
                 data={"center": np.arange(len(sp))} | {f"CN_{B}": [CN["cn"][A][B][0] if CM.get(A,B)>0 else 0 for A in sp] for B in sp})
    plt.close(fig)
    rep.save_table("coordination_table", "\n".join("  ".join(f"{c:>10}" for c in r) for r in cn_rows))

    # 3) ADF ----------------------------------------------------------------
    triplets = []
    for A in sp:
        bonded = [B for B in sp if CM.get(A, B) > 0]
        for i, B in enumerate(bonded):
            for C in bonded[i:]:
                triplets.append((B, A, C))
    if triplets:
        ADF = coordination.adf(traj, triplets, CM, nbins=180, n_blocks=args.nblocks)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        data = {"theta": ADF["theta"]}
        rep.print("\n[ADF] bond-angle peaks:")
        for t in triplets:
            p = ADF[t]["p"]
            if np.all(np.isnan(p)):
                continue
            ax.plot(ADF["theta"], p, label=f"{t[0]}-{t[1]}-{t[2]}")
            data[f"{t[0]}{t[1]}{t[2]}"] = p
            rep.print(f"  {t[0]}-{t[1]}-{t[2]}: {ADF['theta'][np.nanargmax(p)]:.1f}°")
        ax.set_xlabel("angle (deg)"); ax.set_ylabel("P(θ)"); ax.set_title("Bond-angle distributions")
        ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
        rep.save_fig(fig, "adf", data=data); plt.close(fig)

    # 4) rings --------------------------------------------------------------
    RG = mro_rings.ring_statistics(traj, CM, max_size=args.max_ring,
                                   max_frames=args.ring_frames,
                                   n_blocks=min(args.nblocks, args.ring_frames))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.bar(RG["sizes"], RG["count"]["mean"], yerr=RG["count"]["err"],
           color="#6a8caf", capsize=3)
    ax.set_xlabel("ring size"); ax.set_ylabel("rings per frame")
    ax.set_title(f"Ring statistics ({RG['n_frames']} frames)"); ax.grid(alpha=0.3, axis="y")
    rep.save_fig(fig, "rings", data={"size": RG["sizes"], "count": RG["count"]["mean"],
                                     "err": RG["count"]["err"]})
    plt.close(fig)
    rep.print("\n[rings] " + ", ".join(f"{int(s)}:{m:.0f}" for s, m in
              zip(RG["sizes"], RG["count"]["mean"]) if m > 0.5))

    # 5) tetra connectivity -------------------------------------------------
    if args.tetra_center in sp:
        TC = tetra_connectivity.tetra_connectivity(traj, CM, center=args.tetra_center,
                                                   d_max=4.0, n_blocks=args.nblocks)
        fr = TC["fractions"]
        fig, ax = plt.subplots(figsize=(4.5, 4.2))
        kinds = ["corner", "edge", "face"]
        ax.bar(kinds, [fr[k] * 100 for k in kinds], color=["#4c9f70", "#e0b13a", "#c0504d"])
        ax.set_ylabel("%"); ax.set_title(f"{args.tetra_center} tetrahedra sharing")
        ax.grid(alpha=0.3, axis="y")
        rep.save_fig(fig, "tetra_connectivity",
                     data={"kind": np.arange(3), "percent": [fr[k]*100 for k in kinds]})
        plt.close(fig)
        rep.print(f"[tetra] {args.tetra_center}: " +
                  ", ".join(f"{k} {fr[k]*100:.1f}%" for k in kinds))

    # 6) free-element clusters ---------------------------------------------
    if args.free_element in sp:
        FC = mro_clusters.free_clusters(traj, CM, element=args.free_element,
                                        n_blocks=args.nblocks)
        rep.print(f"[clusters] free-{args.free_element}: n={FC['n_clusters'][0]:.0f}, "
                  f"isolated={FC['n_isolated'][0]:.0f}, graphenic={FC['n_graphenic'][0]:.0f}, "
                  f"mean L_a={FC['mean_La'][0]:.2f} Å, max size={FC['max_size'][0]:.0f}")
    return R  # return g(r) for optional reuse


# ─────────────────────────────────────────────────────────────────────────────
def analyze_sweep(directory, CM, rep, *, type_map, prod, stride, args):
    dumps = sweep.discover_dumps(directory)
    rep.print(f"discovered {len(dumps)} temperature dumps:")
    for d in dumps:
        rep.print(f"  {d['label']:<14} {Path(d['path']).name}")
    if not dumps:
        rep.print("no dumps found"); return

    S = sweep.temperature_series(
        dumps, CM, type_map=type_map, prod_range=prod, stride=stride,
        tetra_center=args.tetra_center, free_element=args.free_element,
        network_group=["Si", "N"], bt_groups=(["Si", "N"], ["C"]),
        include_rings=args.include_rings, max_ring_size=args.max_ring,
        max_frames_rings=args.ring_frames, n_blocks=args.nblocks, verbose=False)

    rep.print("\n" + sweep.to_table(S))
    rep.save_table("sweep_table", sweep.to_table(S))

    # multipanel: every observable vs T
    cols = list(S["columns"].keys())
    heat = ~S["cool"]; cooled = S["cool"]
    ncol = 3; nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    sweep_data = {"T": S["T"], "cool": S["cool"].astype(int)}
    for ax, name in zip(axes, cols):
        m = S["columns"][name]["mean"]; e = S["columns"][name]["err"]
        ax.errorbar(S["T"][heat], m[heat], yerr=e[heat], fmt="o-", capsize=3, color="#1f77b4")
        if cooled.any():
            ax.scatter(S["T"][cooled], m[cooled], marker="*", s=150, color="#d62728", zorder=9)
        ax.set_xlabel("T (K)"); ax.set_title(name, fontsize=10); ax.grid(alpha=0.3)
        sweep_data[f"{name}_mean"] = m; sweep_data[f"{name}_err"] = e
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Properties vs temperature", y=1.0)
    fig.tight_layout()
    rep.save_fig(fig, "sweep_vs_T", data=sweep_data); plt.close(fig)

    # g(r) overlay across temperature, one figure per pair (ALL pairs)
    sp0 = None
    grT = {}
    for d in dumps:
        traj = core.select_frames(core.load(d["path"], type_map=type_map, frames="all"),
                                  frame_range=prod, stride=stride * 4)
        sp0 = sp0 or core.species_of(traj)
        all_pairs = core.unique_pairs(sp0)
        R = rdf.partial_rdf(traj, pairs=all_pairs, r_max=args.rmax, nbins=args.nbins, n_blocks=1)
        grT[d["label"]] = R
    for (A, B) in all_pairs:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        data = {"r": grT[dumps[0]["label"]]["r"]}
        for d in dumps:
            R = grT[d["label"]]
            ax.plot(R["r"], R[(A, B)]["g"], label=d["label"])
            data[f"g_{d['label']}"] = R[(A, B)]["g"]
        ax.set_xlabel("r (Å)"); ax.set_ylabel("g(r)")
        ax.set_title(f"{A}-{B} g(r) vs temperature"); ax.legend(); ax.grid(alpha=0.3)
        rep.save_fig(fig, f"grT_{A}{B}", data=data); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="dump/.data file (single) or directory (with --sweep)")
    ap.add_argument("--sweep", action="store_true", help="temperature-sweep mode")
    ap.add_argument("--out", default=None, help="output dir (default results/<stem>)")
    ap.add_argument("--type-map", default="1:Si,2:C,3:N")
    ap.add_argument("--cutoffs", default=None)
    ap.add_argument("--prod", default=None)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--rmax", type=float, default=8.0)
    ap.add_argument("--nbins", type=int, default=400)
    ap.add_argument("--nblocks", type=int, default=5)
    ap.add_argument("--max-ring", type=int, default=9)
    ap.add_argument("--ring-frames", type=int, default=3)
    ap.add_argument("--include-rings", action="store_true", help="rings in sweep (slow)")
    ap.add_argument("--tetra-center", default="Si")
    ap.add_argument("--free-element", default="C")
    args = ap.parse_args()

    cdict = (None if not args.cutoffs else
             {tuple(p.split("-")): float(rc) for p, rc in (t.split(":") for t in args.cutoffs.split(","))})
    CM = core.CutoffMatrix(cdict or presets.SICN_CUTOFFS, default=0.0)
    tmap = parse_type_map(args.type_map)
    prod = parse_prod(args.prod)

    stem = Path(args.path.rstrip("/")).stem or "sweep"
    out = args.out or f"results/{'sweep_' if args.sweep else ''}{stem}"
    rep = Reporter(out, title=f"amorph {amorph.__version__} analyze: {args.path}")

    if args.sweep:
        analyze_sweep(args.path, CM, rep, type_map=tmap, prod=prod, stride=args.stride, args=args)
    else:
        analyze_single(args.path, CM, rep, type_map=tmap, prod=prod, stride=args.stride, args=args)
    rep.flush()
    print(f"\nAll outputs in: {out}/")


if __name__ == "__main__":
    main()
