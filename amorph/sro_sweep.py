"""
sro_sweep.py — temperature dependence of the FULL short-range-order analysis.

Where :mod:`amorph.sweep` tracks a few MRO scalars, this tracks every structural
SRO observable as a function of temperature so structures can be *compared* across
T:

  * g(r) total + each partial pair  → Gaussian-fit peak position r0, height, FWHM
    (block-averaged error bars), plus the first-minimum (shell edge);
  * coordination numbers (total and partial A→B);
  * ADF bond-angle peak positions (every B–A–C triplet);
  * Warren–Cowley chemical SRO α_AB;
  * tetrahedral order q, Steinhardt Q4/Q6/W4/W6, Voronoi volume/faces,
    hybridisation (CN/planarity) fractions — for EVERY species.

`sro_scalars` returns a flat {name: (mean, err)} dict for one trajectory;
`sro_temperature_series` runs it across a folder of per-T dumps (in parallel)
and also keeps the mean g(r) curves for overlay plots.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np

from .core.io import load
from .core.average import select_frames, block_average
from .core.frame import species_of, unique_pairs
from .core.neighbors import CutoffMatrix
from .core.parallel import pmap
from .core import cutoffs as cutmod
from .sro import rdf, coordination, csro, tetrahedra, boo, voronoi, hybridization


def _triplets(sp, cutoffs):
    out = []
    for A in sp:
        bonded = [B for B in sp if cutoffs.get(A, B) > 0]
        for i, B in enumerate(bonded):
            for C in bonded[i:]:
                out.append((B, A, C))
    return out


def sro_scalars(traj, cutoffs: CutoffMatrix, *, n_blocks=5, r_max=8.0, nbins=400,
                do_boo=True, do_voronoi=True, do_hyb=True, do_csro=True,
                return_curves=False):
    """Flat {name: (mean, err)} of all SRO structural metrics for one trajectory."""
    sp = species_of(traj)
    all_pairs = unique_pairs(sp)
    cols = {}

    # density
    cols["density_gcc"] = block_average([fr.mass_density() for fr in traj], n_blocks)
    cols["number_density"] = block_average([fr.number_density() for fr in traj], n_blocks)

    # g(r): mean curves (for overlay) + per-block peak fits (for error bars)
    Rmean = rdf.partial_rdf(traj, pairs=all_pairs, r_max=r_max, nbins=nbins, n_blocks=1)
    r = Rmean["r"]
    keys = list(all_pairs) + ["total"]
    nb = max(1, min(n_blocks, len(traj)))
    peak_acc = defaultdict(list)
    for blk in np.array_split(np.arange(len(traj)), nb):
        sub = [traj[i] for i in blk]
        Rb = rdf.partial_rdf(sub, pairs=all_pairs, r_max=r_max, nbins=nbins, n_blocks=1)
        for key in keys:
            nm = "total" if key == "total" else f"{key[0]}{key[1]}"
            m = rdf.measure_peaks(Rb["r"], Rb[key]["g"], search=(0.8, None))
            peak_acc[f"gr_{nm}_r"].append(m["peak_r"])
            peak_acc[f"gr_{nm}_h"].append(m["height"])
            peak_acc[f"gr_{nm}_fwhm"].append(m["fwhm"])
    for nm, vals in peak_acc.items():
        a = np.array(vals, float)
        cols[nm] = (float(np.nanmean(a)), float(np.nanstd(a)))

    # first-minimum (shell edge) per bonded pair, from mean g(r)
    for (A, B) in all_pairs:
        if cutoffs.get(A, B) > 0:
            rmin = cutmod.first_minimum(r, Rmean[(A, B)]["g"], search=(0.8, None))
            cols[f"rmin_{A}{B}"] = (float(rmin), 0.0)

    # coordination numbers (total + partial)
    CN = coordination.coordination_numbers(traj, cutoffs, n_blocks=n_blocks)
    for A in sp:
        cols[f"CN_{A}"] = CN["cn_total"][A]
        for B in sp:
            if cutoffs.get(A, B) > 0:
                cols[f"CN_{A}_{B}"] = CN["cn"][A][B]

    # ADF peak angles
    tri = _triplets(sp, cutoffs)
    if tri:
        ADF = coordination.adf(traj, tri, cutoffs, n_blocks=n_blocks)
        for t in tri:
            p = ADF[t]["p"]
            if not np.all(np.isnan(p)):
                cols[f"adf_{t[0]}{t[1]}{t[2]}_deg"] = (float(ADF["theta"][np.nanargmax(p)]), 0.0)

    # Warren–Cowley chemical SRO
    if do_csro:
        W = csro.warren_cowley(traj, cutoffs, n_blocks=n_blocks)
        spW = W["species"]
        for i, A in enumerate(spW):
            for j, B in enumerate(spW):
                cols[f"alpha_{A}{B}"] = (float(W["alpha"]["mean"][i, j]),
                                         float(W["alpha"]["err"][i, j]))

    # tetrahedral order — every species
    TQ = tetrahedra.tetrahedral_order(traj, n_blocks=n_blocks)
    for A in sp:
        if A in TQ["q"]:
            cols[f"q_{A}"] = TQ["q"][A]

    # Steinhardt BOO — every species
    if do_boo:
        BO = boo.steinhardt(traj, n_blocks=n_blocks)
        for key in ("Q4", "Q6", "W4", "W6"):
            for A in sp:
                if A in BO[key]:
                    cols[f"{key}_{A}"] = BO[key][A]

    # Voronoi — every species
    if do_voronoi:
        V = voronoi.voronoi(traj, n_blocks=n_blocks)
        for A in sp:
            cols[f"vorVol_{A}"] = V["volume"][A]
            cols[f"vorFaces_{A}"] = V["faces"][A]

    # hybridisation fractions — every species × class
    if do_hyb:
        H = hybridization.classify(traj, cutoffs, n_blocks=n_blocks)
        for A in sp:
            fm, fe = H["fractions"][A]
            for ci, cl in enumerate(H["classes"]):
                cols[f"hyb_{A}_{cl}"] = (float(fm[ci]), float(fe[ci]))

    if return_curves:
        curves = {"r": r, "total": Rmean["total"]["g"]}
        for (A, B) in all_pairs:
            curves[f"{A}{B}"] = Rmean[(A, B)]["g"]
        return cols, curves
    return cols


# ─────────────────────────────────────────────────────────────────────────────
def _sro_task(task):
    """Picklable worker: load one dump, compute SRO scalars + g(r) curves."""
    d, path, type_map, prod_range, stride, cutoffs, kw = task
    traj = load(path, type_map=type_map, frames="all")
    traj = select_frames(traj, frame_range=prod_range, stride=stride)
    cols, curves = sro_scalars(traj, cutoffs, return_curves=True, **kw)
    return d, cols, curves


def sro_temperature_series(dumps, cutoffs: CutoffMatrix, *, type_map=None,
                           prod_range=None, stride=1, include_cool=True,
                           jobs=1, verbose=True, **sro_kwargs):
    """Run :func:`sro_scalars` across temperature dumps (parallel over T).

    Returns
    -------
    dict
      "T", "labels", "cool"
      "columns" : {name: {"mean": (n,), "err": (n,)}}
      "curves"  : {label: {"r":…, "total":…, "AB":…}}   mean g(r) per temperature
    """
    entries = [d for d in dumps if include_cool or not d["cool"]]
    tasks = [(d, d["path"], type_map, prod_range, stride, cutoffs, sro_kwargs)
             for d in entries]
    n_tot = len(tasks)
    cb = (lambda k: print(f"  [{k}/{n_tot}] temperatures done", flush=True)) if verbose else None

    T, labels, cool = [], [], []
    curves = {}
    per_T = []                      # list of {name: (mean, err)} dicts, one per T
    for d, cols, cv in pmap(_sro_task, tasks, jobs=jobs, on_done=cb):
        T.append(d["T"]); labels.append(d["label"]); cool.append(d["cool"])
        curves[d["label"]] = cv
        per_T.append(cols)

    # union of all metric names (a metric may be absent at some temperatures,
    # e.g. a bond/triplet that does not occur at every T) — fill gaps with NaN
    names = []
    seen = set()
    for cols in per_T:
        for k in cols:
            if k not in seen:
                seen.add(k); names.append(k)
    columns = {}
    for k in names:
        ms = np.array([cols.get(k, (np.nan, np.nan))[0] for cols in per_T])
        es = np.array([cols.get(k, (np.nan, np.nan))[1] for cols in per_T])
        columns[k] = {"mean": ms, "err": es}
    return dict(T=np.array(T), labels=labels, cool=np.array(cool),
                columns=columns, curves=curves)
