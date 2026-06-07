"""
csro.py — Warren-Cowley chemical short-range order parameter α_AB.

    α_AB = 1 − P(B|A) / x_B

P(B|A) = mean fraction of A's bonded neighbours that are B; x_B = mole fraction.
  α = 0  random mixing;  α < 0  A–B clustering (bond preference);
  α > 0  A–B avoidance.

Element-agnostic and time-averaged with block errors. Uses a
:class:`~amorph.core.neighbors.CutoffMatrix` for bonding.
"""
from __future__ import annotations
import numpy as np

from ..core.frame import species_of
from ..core.neighbors import neighbor_table, CutoffMatrix
from ..core.average import block_average


def _frame_PandX(frame, sp, cutoffs):
    idx_of = {e: k for k, e in enumerate(sp)}
    nbrs = neighbor_table(frame, cutoffs)
    elem = frame.elements
    ns = len(sp)
    P = np.full((ns, ns), np.nan)
    counts = np.zeros((frame.n_atoms, ns))
    for i in range(frame.n_atoms):
        for j in nbrs[i]:
            counts[i, idx_of[elem[j]]] += 1
    for a, A in enumerate(sp):
        mask = elem == A
        tot = counts[mask].sum(axis=1)
        valid = tot > 0
        if valid.sum() == 0:
            continue
        for b in range(ns):
            P[a, b] = np.mean(counts[mask, b][valid] / tot[valid])
    x = np.array([frame.count(e) / frame.n_atoms for e in sp])
    return P, x


def warren_cowley(traj, cutoffs: CutoffMatrix, n_blocks=5):
    """Time-averaged Warren-Cowley α_AB.

    Returns
    -------
    dict
      "species" : list
      "alpha"   : {"mean": (ns,ns), "err": (ns,ns)}
      "P"       : {"mean": (ns,ns), "err": (ns,ns)}  P(B|A)
      "x"       : (ns,) mean mole fractions
    """
    sp = species_of(traj)
    Ps, alphas, xs = [], [], []
    for fr in traj:
        P, x = _frame_PandX(fr, sp, cutoffs)
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = 1.0 - P / x[None, :]
        Ps.append(P); alphas.append(alpha); xs.append(x)
    aM, aE = block_average(alphas, n_blocks)
    pM, pE = block_average(Ps, n_blocks)
    x_mean = np.nanmean(xs, axis=0)
    return dict(species=sp,
                alpha={"mean": aM, "err": aE},
                P={"mean": pM, "err": pE},
                x=x_mean)
