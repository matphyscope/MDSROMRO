"""
tetrahedra.py — Tetrahedral order parameter q (Chau–Hardwick / Errington–Debenedetti).

For each atom and its 4 nearest neighbours:

    q = 1 − (3/8) Σ_{j<k} ( cos ψ_jk + 1/3 )²

q = 1 for a perfect tetrahedron, q ≈ 0 for an ideal gas. Computed per centre
species (4-nearest-neighbour geometric definition, independent of the bond
cutoff). Time-averaged with block errors; the final frame's per-atom q is
returned for the distribution plot, along with the AX₄ bond-angle list.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree

from ..core.frame import species_of
from ..core.average import block_average


def _knn_vectors(frame, k=4):
    """Vectors from each atom to its k nearest neighbours (PBC). Returns (N,k,3)."""
    L = frame.L
    p = frame.wrapped()
    tree = cKDTree(p, boxsize=L)
    # query k+1 (first is self at distance 0)
    dist, idx = tree.query(p, k=k + 1)
    vecs = np.empty((frame.n_atoms, k, 3))
    for a in range(frame.n_atoms):
        nbr = idx[a, 1:]            # drop self
        d = p[nbr] - p[a]
        d -= L * np.round(d / L)    # minimum image
        vecs[a] = d
    return vecs


def _q_from_vectors(vecs):
    """Tetrahedral order q for one atom given (4,3) neighbour vectors."""
    u = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    s = 0.0
    for j in range(4):
        for k in range(j + 1, 4):
            cospsi = np.clip(np.dot(u[j], u[k]), -1.0, 1.0)
            s += (cospsi + 1.0 / 3.0) ** 2
    return 1.0 - 3.0 / 8.0 * s


def tetrahedral_order(traj, n_blocks=5):
    """Time-averaged tetrahedral order parameter q per species.

    Returns
    -------
    dict
      "species" : list
      "q"       : {A: (mean, err)}   mean q per species
      "last_frame" : {"q": (N,), "elements": (N,)}
    """
    sp = species_of(traj)
    pf = {A: [] for A in sp}
    last = None
    for fr in traj:
        vecs = _knn_vectors(fr, k=4)
        q = np.array([_q_from_vectors(vecs[a]) for a in range(fr.n_atoms)])
        for A in sp:
            m = fr.mask(A)
            pf[A].append(q[m].mean() if m.any() else np.nan)
        last = {"q": q, "elements": fr.elements}
    qres = {A: block_average(pf[A], n_blocks) for A in sp}
    return dict(species=sp, q=qres, last_frame=last)
