#!/usr/bin/env python3
"""
analyze_sro_sweep.py — temperature dependence of the FULL SRO analysis.

For a folder of per-temperature dumps, computes every structural SRO observable
at each T and plots it vs temperature so structures can be compared:
  g(r) peak position / height / FWHM (per pair, Gaussian fit, block errors),
  coordination numbers, ADF bond angles, Warren-Cowley alpha, tetrahedral q,
  Steinhardt Q4/Q6, Voronoi volume/faces, hybridisation fractions,
  plus the actual g(r) curves overlaid across temperature.

Every figure is saved as PNG + its data as .txt; the full table and log too.
Multi-core (--jobs, default all). nohup-friendly.

  python3 scripts/analyze_sro_sweep.py data/ --type-map 1:Si,2:C,3:N --out sro_T
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import amorph
from amorph import core, presets, sweep, sro_sweep
from amorph.report import Reporter


def parse_type_map(s):
    return {int(k): v.strip() for k, v in (t.split(":") for t in s.split(","))} if s else None


def _plot_group(rep, S, names, title, ylabel, fname):
    """Plot each column in `names` vs T on one axis (heating ● + cooled ★)."""
    names = [n for n in names if n in S["columns"]]
    if not names:
        return
    heat = ~S["cool"]; cooled = S["cool"]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    data = {"T": S["T"]}
    for n in names:
        m = S["columns"][n]["mean"]; e = S["columns"][n]["err"]
        line = ax.errorbar(S["T"][heat], m[heat], yerr=e[heat], fmt="o-", capsize=3,
                           label=n)
        if cooled.any():
            ax.scatter(S["T"][cooled], m[cooled], marker="*", s=140,
                       color=line[0].get_color(), edgecolor="k", zorder=9)
        data[f"{n}_mean"] = m; data[f"{n}_err"] = e
    ax.set_xlabel("T (K)"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    rep.save_fig(fig, fname, data=data); plt.close(fig)


def _cols_matching(S, prefix, suffix=""):
    return sorted(n for n in S["columns"]
                  if n.startswith(prefix) and n.endswith(suffix))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="directory of per-temperature dumps")
    ap.add_argument("--out", default=None)
    ap.add_argument("--type-map", default="1:Si,2:C,3:N")
    ap.add_argument("--cutoffs", default=None,
                    help="override bond cutoffs (Å), e.g. for ReaxFF: "
                         "C-C:1.75,C-Si:2.09,N-N:1.81,N-Si:2.35,Si-Si:2.78 "
                         "(unlisted pairs -> non-bonded). Default: SiCN/Tersoff preset.")
    ap.add_argument("--prod", default=None)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--rmax", type=float, default=8.0)
    ap.add_argument("--nbins", type=int, default=400)
    ap.add_argument("--bin-width", type=float, default=None,
                    help="g(r) bin width in Å (overrides --nbins). e.g. 0.005 or 0.001. "
                         "Finer = more detail but noisier per bin; peak r/FWHM come from "
                         "a Gaussian fit so they stay sub-bin precise regardless.")
    ap.add_argument("--nblocks", type=int, default=5)
    ap.add_argument("--jobs", "-j", type=int, default=0)
    ap.add_argument("--no-boo", action="store_true")
    ap.add_argument("--no-voronoi", action="store_true")
    ap.add_argument("--no-hyb", action="store_true")
    args = ap.parse_args()

    cdict = (None if not args.cutoffs else
             {tuple(p.split("-")): float(rc)
              for p, rc in (t.split(":") for t in args.cutoffs.split(","))})
    CM = core.CutoffMatrix(cdict or presets.SICN_CUTOFFS, default=0.0)
    tmap = parse_type_map(args.type_map)
    if args.bin_width:
        args.nbins = max(1, round(args.rmax / args.bin_width))
    prod = None
    if args.prod:
        lo, hi = args.prod.split(":")
        prod = (int(lo) if lo else 0, int(hi) if hi else None)

    stem = Path(args.path.rstrip("/")).stem or "sweep"
    out = args.out or f"results/sroT_{stem}"
    rep = Reporter(out, title=f"amorph {amorph.__version__} SRO T-sweep: {args.path}")

    dumps = sweep.discover_dumps(args.path)
    rep.print(f"discovered {len(dumps)} temperature dumps:")
    for d in dumps:
        rep.print(f"  {d['label']:<14} {Path(d['path']).name}")
    if not dumps:
        rep.print("no dumps found — check path / filename pattern"); rep.flush(); return

    rep.print(f"\ncutoffs: {CM}")
    rep.print(f"g(r): r_max={args.rmax} Å, nbins={args.nbins} "
              f"→ bin width = {args.rmax/args.nbins:.4f} Å "
              f"(peak r/FWHM are Gaussian-fit, sub-bin precise)")
    rep.print(f"computing full SRO at each temperature (jobs={args.jobs}) ...")
    S = sro_sweep.sro_temperature_series(
        dumps, CM, type_map=tmap, prod_range=prod, stride=args.stride,
        n_blocks=args.nblocks, jobs=args.jobs, r_max=args.rmax, nbins=args.nbins,
        do_boo=not args.no_boo, do_voronoi=not args.no_voronoi,
        do_hyb=not args.no_hyb, verbose=True)

    # full numeric table (defensive: tolerate any column shorter than the T axis)
    names = sorted(S["columns"])
    nT = len(S["T"])

    def _cell(n, i):
        m = S["columns"][n]["mean"]; e = S["columns"][n]["err"]
        if i < len(m) and np.isfinite(m[i]):
            return f"{m[i]:>8.4f}±{e[i]:<7.4f}"
        return f"{'nan':>8} {'':<7}"

    hdr = f"{'T(K)':>7} {'cool':>5} " + " ".join(f"{n:>16}" for n in names)
    lines = [hdr]
    for i, T in enumerate(S["T"]):
        row = f"{T:>7} {'y' if S['cool'][i] else '':>5} "
        row += " ".join(_cell(n, i) for n in names)
        lines.append(row)
    rep.save_table("sro_sweep_table", "\n".join(lines))
    rep.print(f"\ntracked {len(names)} structural metrics across {len(S['T'])} temperatures")

    # ---- grouped property-vs-T figures -----------------------------------
    rep.print("\nplotting property-vs-temperature ...")
    _plot_group(rep, S, _cols_matching(S, "gr_", "_r"),
                "g(r) first-peak position vs T", "peak r (Å)", "grpeak_position_vs_T")
    _plot_group(rep, S, _cols_matching(S, "gr_", "_fwhm"),
                "g(r) first-peak FWHM vs T (thermal broadening)", "FWHM (Å)", "grpeak_fwhm_vs_T")
    _plot_group(rep, S, _cols_matching(S, "gr_", "_h"),
                "g(r) first-peak height vs T", "height", "grpeak_height_vs_T")
    _plot_group(rep, S, _cols_matching(S, "rmin_"),
                "g(r) first-minimum (shell edge) vs T", "r_min (Å)", "shell_edge_vs_T")
    _plot_group(rep, S, [n for n in _cols_matching(S, "CN_") if n.count("_") == 1],
                "Coordination number (total) vs T", "CN", "CN_total_vs_T")
    _plot_group(rep, S, [n for n in _cols_matching(S, "CN_") if n.count("_") == 2],
                "Partial coordination A→B vs T", "CN", "CN_partial_vs_T")
    _plot_group(rep, S, _cols_matching(S, "adf_", "_deg"),
                "Bond-angle (ADF) Gaussian-fit peak vs T", "angle (deg)", "ADF_peak_vs_T")
    _plot_group(rep, S, _cols_matching(S, "adf_", "_fwhm"),
                "Bond-angle (ADF) width FWHM vs T", "FWHM (deg)", "ADF_fwhm_vs_T")
    _plot_group(rep, S, [n for n in _cols_matching(S, "alpha_") if n[6] != n[7]],
                "Warren–Cowley chemical SRO α vs T", "α", "CSRO_alpha_vs_T")
    _plot_group(rep, S, _cols_matching(S, "q_"),
                "Tetrahedral order q vs T", "q", "tetra_q_vs_T")
    _plot_group(rep, S, _cols_matching(S, "Q6_"),
                "Steinhardt Q6 vs T", "Q6", "BOO_Q6_vs_T")
    _plot_group(rep, S, _cols_matching(S, "Q4_"),
                "Steinhardt Q4 vs T", "Q4", "BOO_Q4_vs_T")
    _plot_group(rep, S, _cols_matching(S, "vorVol_"),
                "Voronoi cell volume vs T (thermal expansion)", "volume (Å³)", "voronoi_volume_vs_T")
    _plot_group(rep, S, _cols_matching(S, "vorFaces_"),
                "Voronoi face count vs T", "faces", "voronoi_faces_vs_T")
    _plot_group(rep, S, ["density_gcc"], "Mass density vs T", "g/cc", "density_vs_T")
    for cl in ("CN3-planar", "CN4", "CN3-pyramidal"):
        _plot_group(rep, S, _cols_matching(S, "hyb_", "_" + cl),
                    f"Hybridisation fraction [{cl}] vs T", "fraction", f"hyb_{cl}_vs_T")

    # ---- g(r) curve overlays across temperature, one figure per pair -----
    rep.print("plotting g(r) curve overlays across T ...")
    labels = S["labels"]
    r = S["curves"][labels[0]]["r"]
    pair_keys = [k for k in S["curves"][labels[0]] if k not in ("r", "adf")]
    for key in pair_keys:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        data = {"r": r}
        for lab in labels:
            ax.plot(r, S["curves"][lab][key], label=lab)
            data[f"g_{lab}"] = S["curves"][lab][key]
        ax.set_xlabel("r (Å)"); ax.set_ylabel("g(r)")
        ax.set_title(f"{key} g(r) vs temperature"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        rep.save_fig(fig, f"gr_curve_{key}", data=data); plt.close(fig)

    # ---- ADF histogram overlays across temperature, one figure per triplet
    adf0 = S["curves"][labels[0]].get("adf", {})
    if adf0 and "theta" in adf0:
        rep.print("plotting ADF histogram overlays across T ...")
        theta = adf0["theta"]
        tri_keys = [k for k in adf0 if k != "theta"]
        for tk in tri_keys:
            fig, ax = plt.subplots(figsize=(7, 4.4))
            data = {"theta_deg": theta}
            for lab in labels:
                p = S["curves"][lab].get("adf", {}).get(tk)
                if p is None:
                    continue
                ax.plot(theta, p, label=lab)
                data[f"P_{lab}"] = p
            ax.set_xlabel("angle (deg)"); ax.set_ylabel("P(θ)")
            ax.set_title(f"{tk[0]}-{tk[1]}-{tk[2]} bond-angle distribution vs T")
            ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(0, 180)
            rep.save_fig(fig, f"adf_curve_{tk}", data=data); plt.close(fig)

    rep.flush()
    print(f"\nAll SRO T-sweep outputs in: {out}/")


if __name__ == "__main__":
    main()
