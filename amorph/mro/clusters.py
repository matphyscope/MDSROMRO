"""
clusters.py — Connected-component (cluster) analysis & percolation.

Three views, element-agnostic:
  * free clusters of a chosen element X (X–X bonds only) — e.g. free carbon;
    per-cluster size, radius of gyration R_g, principal extent L_a (Raman L_a
    proxy), aspect ratio, and a mass-fractal dimension D_f from R_g vs size.
  * a network sub-graph over a group of elements (e.g. Si+N) — largest-component
    fraction and box-spanning percolation check.
  * the full bond network — component-size distribution.

Scalar summaries (largest-cluster fraction, mean L_a, percolation fraction) are
time-averaged with block errors; the final frame's per-cluster table is returned
for histograms.
"""
from __future__ import annotations
import numpy as np

from ..core.average import block_average
from ._common import build_graph, unwrap_component, gyration, require_networkx


def _components(G):
    nx = require_networkx()
    return [c for c in nx.connected_components(G)]


def _cluster_records(frame, G, perc_thresh=0.85):
    """Per-component geometry records for graph G on this frame.

    A component whose unwrapped extent spans the box (>= ``perc_thresh`` of L on
    any axis) PERCOLATES: under PBC it connects to its own periodic image, so its
    R_g / L_a are ill-defined. Such records are flagged ``percolates=True`` and
    their geometry set to NaN so callers can exclude them from size statistics.
    """
    recs = []
    for comp in _components(G):
        atoms = list(comp)
        if len(atoms) < 2:
            recs.append({"size": len(atoms), "R_g": 0.0, "L_a": 0.0,
                         "aspect": 1.0, "atoms": atoms, "percolates": False})
            continue
        coords = unwrap_component(atoms, G, frame)
        extent = coords.max(axis=0) - coords.min(axis=0)
        perc = bool(np.max(extent / frame.L) >= perc_thresh)
        if perc:
            Rg = L_a = aspect = np.nan
        else:
            Rg, L_a, aspect, _ = gyration(coords)
        recs.append({"size": len(atoms), "R_g": Rg, "L_a": L_a,
                     "aspect": aspect, "atoms": atoms, "percolates": perc})
    return recs


def _percolates(frame, G, comp, thresh=0.85):
    atoms = list(comp)
    if len(atoms) < 2:
        return False, 0.0
    coords = unwrap_component(atoms, G, frame)
    extent = coords.max(axis=0) - coords.min(axis=0)
    frac = float(np.max(extent / frame.L))
    return frac >= thresh, frac


def free_clusters(traj, cutoffs, element, n_blocks=5, min_graphenic=4):
    """Free clusters of one element (X–X bonds only).

    Returns time-averaged scalars + last-frame per-cluster records.
    """
    pf_nclus, pf_iso, pf_large, pf_meanLa, pf_maxsize = [], [], [], [], []
    pf_perc, pf_maxfinite = [], []
    last = None
    for fr in traj:
        G = build_graph(fr, cutoffs, nodes=np.where(fr.mask(element))[0],
                        pair_filter=lambda a, b: a == element and b == element)
        recs = _cluster_records(fr, G)
        sizes = [r["size"] for r in recs]
        large = [r for r in recs if r["size"] >= min_graphenic]
        # L_a only meaningful for finite (non-percolating) clusters
        finite_La = [r["L_a"] for r in large if not r["percolates"]]
        finite_sizes = [r["size"] for r in recs if not r["percolates"]]
        pf_nclus.append(len(recs))
        pf_iso.append(sum(1 for s in sizes if s == 1))
        pf_large.append(len(large))
        pf_meanLa.append(np.mean(finite_La) if finite_La else np.nan)
        pf_maxsize.append(max(sizes) if sizes else 0)
        pf_maxfinite.append(max(finite_sizes) if finite_sizes else 0)
        pf_perc.append(sum(1 for r in recs if r["percolates"]))
        last = recs
    def ba(x): return block_average(x, n_blocks)
    return dict(element=element,
                n_clusters=ba(pf_nclus), n_isolated=ba(pf_iso),
                n_graphenic=ba(pf_large), mean_La=ba(pf_meanLa),
                max_size=ba(pf_maxsize), max_finite_size=ba(pf_maxfinite),
                n_percolating=ba(pf_perc), last_frame=last,
                min_graphenic=min_graphenic)


def fractal_dimension(records, min_size=3):
    """Mass-fractal D_f from size ∝ R_g^{D_f} (log-log fit). Returns (Df, n)."""
    s = np.array([r["size"] for r in records if r["size"] >= min_size and r["R_g"] > 0])
    Rg = np.array([r["R_g"] for r in records if r["size"] >= min_size and r["R_g"] > 0])
    if len(s) < 3:
        return np.nan, len(s)
    slope, _ = np.polyfit(np.log(Rg), np.log(s), 1)
    return float(slope), len(s)


def network_percolation(traj, cutoffs, group, n_blocks=5):
    """Largest-component fraction and box-spanning percolation of a group network."""
    pf_frac, pf_perc, pf_axial, pf_ncomp = [], [], [], []
    gset = set(group)
    for fr in traj:
        nodes = np.where(np.isin(fr.elements, list(gset)))[0]
        n_group = len(nodes)
        G = build_graph(fr, cutoffs, nodes=nodes)
        comps = _components(G)
        if not comps:
            pf_frac.append(np.nan); pf_perc.append(0.0); pf_axial.append(0.0)
            pf_ncomp.append(0); continue
        biggest = max(comps, key=len)
        perc, axial = _percolates(fr, G, biggest)
        pf_frac.append(len(biggest) / n_group if n_group else np.nan)
        pf_perc.append(1.0 if perc else 0.0)
        pf_axial.append(axial)
        pf_ncomp.append(len(comps))
    def ba(x): return block_average(x, n_blocks)
    return dict(group=group, largest_fraction=ba(pf_frac),
                percolation_prob=ba(pf_perc), axial_fraction=ba(pf_axial),
                n_components=ba(pf_ncomp))
