"""
tetra_connectivity.py — Tetrahedra connectivity (corner / edge / face sharing).

For a chosen centre species (e.g. Si), any two centre atoms within ``d_max`` that
share bonded neighbours are connected:
  1 shared atom → corner, 2 → edge, 3 → face (≥4 → higher).
Corner sharing is further split by the bridging atom's species. This is a direct
intermediate-range-order fingerprint (α-Si₃N₄/β-SiC are 100% corner-sharing).

Counts (per frame) are time-averaged with block errors; the final frame's
per-pair Si–Si distances per class are returned for distance histograms.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree

from ..core.frame import species_of
from ..core.average import block_average
from ._common import neighbor_sets

_CLASSES = ["corner", "edge", "face", "higher"]


def _frame_connectivity(frame, cutoffs, center, d_max, bridge_species):
    nb = neighbor_sets(frame, cutoffs)
    elem = frame.elements
    cidx = np.where(elem == center)[0]
    if len(cidx) < 2:
        return ({c: 0 for c in _CLASSES},
                {b: 0 for b in bridge_species}, {c: [] for c in _CLASSES})
    L = frame.L
    p = frame.wrapped()
    tree = cKDTree(p[cidx], boxsize=L)
    pairs = tree.query_pairs(d_max, output_type="ndarray")
    by_conn = {c: 0 for c in _CLASSES}
    corner_bridge = {b: 0 for b in bridge_species}
    dist_by_class = {c: [] for c in _CLASSES}
    for a, b in pairs:
        i, j = int(cidx[a]), int(cidx[b])
        shared = nb[i] & nb[j]
        ns = len(shared)
        if ns == 0:
            continue
        cls = "corner" if ns == 1 else "edge" if ns == 2 else "face" if ns == 3 else "higher"
        by_conn[cls] += 1
        d = p[i] - p[j]; d -= L * np.round(d / L)
        dist_by_class[cls].append(float(np.linalg.norm(d)))
        if ns == 1:
            br = elem[next(iter(shared))]
            if br in corner_bridge:
                corner_bridge[br] += 1
    return by_conn, corner_bridge, dist_by_class


def tetra_connectivity(traj, cutoffs, center="Si", d_max=4.0, n_blocks=5):
    """Time-averaged tetrahedra connectivity for a centre species.

    Returns
    -------
    dict
      "center"        : species
      "by_conn"       : {class: (mean, err)}  pair counts per frame
      "corner_bridge" : {bridge_species: (mean, err)}
      "fractions"     : {class: mean fraction of connected pairs}
      "last_frame"    : {class: [Si-Si distances]}
    """
    bridge_species = species_of(traj)
    pf_conn = {c: [] for c in _CLASSES}
    pf_bridge = {b: [] for b in bridge_species}
    last_dist = None
    for fr in traj:
        by_conn, corner_bridge, dist_by_class = _frame_connectivity(
            fr, cutoffs, center, d_max, bridge_species)
        for c in _CLASSES:
            pf_conn[c].append(by_conn[c])
        for b in bridge_species:
            pf_bridge[b].append(corner_bridge[b])
        last_dist = dist_by_class
    by_conn = {c: block_average(pf_conn[c], n_blocks) for c in _CLASSES}
    corner_bridge = {b: block_average(pf_bridge[b], n_blocks) for b in bridge_species}
    total = sum(by_conn[c][0] for c in _CLASSES)
    fractions = {c: (by_conn[c][0] / total if total > 0 else np.nan) for c in _CLASSES}
    return dict(center=center, by_conn=by_conn, corner_bridge=corner_bridge,
                fractions=fractions, last_frame=last_dist)
