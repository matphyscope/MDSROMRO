"""
coordination.py — Coordination numbers and angular distribution functions.

* :func:`coordination_numbers` — cutoff-based CN: for each centre species A, the
  mean number of neighbours of each species B, the total CN, and the per-atom CN
  distribution. Time-averaged with block errors.
* :func:`adf` — angular (bond-angle) distribution for B–A–C triplets, centred on
  species A, using the same cutoff matrix. Time-averaged, normalised to a
  probability density over angle (degrees).

Both consume a :class:`~amorph.core.neighbors.CutoffMatrix`, so non-bonding
pairs (cutoff 0) are simply never counted — matching the SiCN C–N/N–N
convention when that matrix is used.
"""
from __future__ import annotations
from functools import partial
import numpy as np

from ..core.frame import species_of
from ..core.neighbors import NeighborCache, CutoffMatrix
from ..core.average import block_average
from ..core.parallel import pmap


def _directed_neighbors(frame, cutoffs: CutoffMatrix, cache=None):
    """For each atom, list of (neighbor_index, unit_vector_from_atom_to_nbr).

    Returns also the raw vectors (not unit) for angle computation.
    """
    rmax = cutoffs.rmax
    if cache is not None and cache.r_max >= rmax:
        i, j, rij, r = cache.pairs()
    else:
        i, j, rij, r = NeighborCache(frame, rmax).pairs()
    elem = frame.elements
    nbr = [[] for _ in range(frame.n_atoms)]
    for a, b, v, d in zip(i, j, rij, r):
        rc = cutoffs.get(elem[a], elem[b])
        if rc <= 0 or d > rc:
            continue
        # rij = p[a]-p[b]; vector a->b is -rij, vector b->a is +rij
        nbr[a].append((b, -v))
        nbr[b].append((a, v))
    return nbr


# ─────────────────────────────────────────────────────────────────────────────
# Coordination numbers
# ─────────────────────────────────────────────────────────────────────────────
def _cn_frame(fr, cutoffs, sp, cn_bins, max_cn):
    """Per-frame coordination kernel -> dict of picklable arrays/scalars."""
    nbr = _directed_neighbors(fr, cutoffs)
    elem = fr.elements
    out_cn = {A: {B: np.nan for B in sp} for A in sp}
    out_tot = {A: np.nan for A in sp}
    out_dist = {A: np.full(len(cn_bins), np.nan) for A in sp}
    for A in sp:
        idxA = np.where(elem == A)[0]
        if len(idxA) == 0:
            continue
        countsB = {B: 0 for B in sp}
        tot_per_atom = np.zeros(len(idxA))
        for n, k in enumerate(idxA):
            for (b, _v) in nbr[k]:
                countsB[elem[b]] += 1
            tot_per_atom[n] = len(nbr[k])
        for B in sp:
            out_cn[A][B] = countsB[B] / len(idxA)
        out_tot[A] = tot_per_atom.mean()
        hist = np.bincount(np.clip(tot_per_atom.astype(int), 0, max_cn),
                           minlength=len(cn_bins)).astype(float)
        out_dist[A] = hist / hist.sum()
    return out_cn, out_tot, out_dist


def coordination_numbers(traj, cutoffs: CutoffMatrix, n_blocks=5, max_cn=12, jobs=1):
    """Time-averaged coordination numbers.

    Returns
    -------
    dict
      "species"            : sorted species list
      "cn"                 : {A: {B: (mean, err)}}  partial CN A→B
      "cn_total"           : {A: (mean, err)}       total CN of species A
      "distribution"       : {A: (mean_hist, err_hist)}  P(CN=k), k=0..max_cn
      "cn_bins"            : array 0..max_cn
    """
    sp = species_of(traj)
    # accumulate per-frame values
    pf_cn = {A: {B: [] for B in sp} for A in sp}
    pf_tot = {A: [] for A in sp}
    pf_dist = {A: [] for A in sp}
    cn_bins = np.arange(0, max_cn + 1)

    worker = partial(_cn_frame, cutoffs=cutoffs, sp=sp, cn_bins=cn_bins, max_cn=max_cn)
    for out_cn, out_tot, out_dist in pmap(worker, traj, jobs=jobs):
        for A in sp:
            for B in sp:
                pf_cn[A][B].append(out_cn[A][B])
            pf_tot[A].append(out_tot[A])
            pf_dist[A].append(out_dist[A])

    cn = {A: {B: block_average(pf_cn[A][B], n_blocks) for B in sp} for A in sp}
    cn_total = {A: block_average(pf_tot[A], n_blocks) for A in sp}
    distribution = {A: block_average(pf_dist[A], n_blocks) for A in sp}
    return dict(species=sp, cn=cn, cn_total=cn_total,
                distribution=distribution, cn_bins=cn_bins)


# ─────────────────────────────────────────────────────────────────────────────
# Angular distribution function
# ─────────────────────────────────────────────────────────────────────────────
def _adf_frame(fr, cutoffs, triplets, edges, dtheta, nbins):
    """Per-frame ADF kernel -> {triplet: density array or nan array}."""
    nbr = _directed_neighbors(fr, cutoffs)
    elem = fr.elements
    out = {}
    for (B, A, C) in triplets:
        angles = []
        idxA = np.where(elem == A)[0]
        for k in idxA:
            bs = [v for (b, v) in nbr[k] if elem[b] == B]
            cs = [v for (c, v) in nbr[k] if elem[c] == C]
            if B == C:
                vs = bs
                for x in range(len(vs)):
                    for y in range(x + 1, len(vs)):
                        angles.append(_angle(vs[x], vs[y]))
            else:
                for vb in bs:
                    for vc in cs:
                        angles.append(_angle(vb, vc))
        if angles:
            h, _ = np.histogram(angles, bins=edges)
            area = h.sum() * dtheta
            out[(B, A, C)] = h / area if area > 0 else np.zeros(nbins)
        else:
            out[(B, A, C)] = np.full(nbins, np.nan)
    return out


def adf(traj, triplets, cutoffs: CutoffMatrix, nbins=180, n_blocks=5, jobs=1):
    """Bond-angle distribution for B–A–C triplets (centre = A).

    Parameters
    ----------
    triplets : list[(B, A, C)]
        Angle is measured at the central atom A between bonds A–B and A–C.
    cutoffs : CutoffMatrix
    nbins : int   over 0–180°
    n_blocks : int

    Returns
    -------
    dict
      "theta"        : (nbins,) bin centres in degrees
      (B,A,C)        : {"p": density, "err": 1σ}   normalised so ∫p dθ = 1
    """
    edges = np.linspace(0.0, 180.0, nbins + 1)
    theta = 0.5 * (edges[:-1] + edges[1:])
    dtheta = edges[1] - edges[0]

    pf = {t: [] for t in triplets}
    worker = partial(_adf_frame, cutoffs=cutoffs, triplets=triplets,
                     edges=edges, dtheta=dtheta, nbins=nbins)
    for out in pmap(worker, traj, jobs=jobs):
        for t in triplets:
            pf[t].append(out[t])

    result = {"theta": theta}
    for t in triplets:
        mean, err = block_average(pf[t], n_blocks)
        result[t] = {"p": mean, "err": err}
    return result


def _angle(v1, v2):
    """Angle (degrees) between two vectors."""
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return np.nan
    c = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(c))
