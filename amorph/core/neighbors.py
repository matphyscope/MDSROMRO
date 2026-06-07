"""
neighbors.py — PBC neighbor search, element-aware cutoff matrix, bond lists.

All neighbor finding goes through scipy's ``cKDTree`` with ``boxsize`` for exact
minimum-image periodicity (orthogonal boxes). A :class:`CutoffMatrix` holds the
per-pair bond cutoffs (element-agnostic); a pair set to 0 means "never bonded"
(used e.g. for C–N / N–N in the SiCN Tersoff convention).

:class:`NeighborCache` computes the within-``r_max`` pair list once per frame
and reuses it, so RDF / CN / ADF / rings etc. don't each re-run the KDTree.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


# ─────────────────────────────────────────────────────────────────────────────
# Cutoff matrix
# ─────────────────────────────────────────────────────────────────────────────
class CutoffMatrix:
    """Symmetric per-element-pair bond cutoffs (Å).

    Parameters
    ----------
    cutoffs : dict
        {(A, B): rc}. Order of A, B does not matter. Missing pairs default to
        ``default`` (0.0 → treated as non-bonding).
    default : float
    """

    def __init__(self, cutoffs=None, default=0.0):
        self._c = {}
        self.default = float(default)
        if cutoffs:
            for (a, b), rc in cutoffs.items():
                self._c[self._key(a, b)] = float(rc)

    @staticmethod
    def _key(a, b):
        return (a, b) if a <= b else (b, a)

    def get(self, a, b) -> float:
        return self._c.get(self._key(a, b), self.default)

    def set(self, a, b, rc):
        self._c[self._key(a, b)] = float(rc)

    @property
    def rmax(self) -> float:
        return max([*self._c.values(), self.default]) if self._c else self.default

    def as_dict(self) -> dict:
        return dict(self._c)

    def __repr__(self):
        items = ", ".join(f"{a}-{b}:{rc:.3f}" for (a, b), rc in sorted(self._c.items()))
        return f"CutoffMatrix({items}, default={self.default})"


# ─────────────────────────────────────────────────────────────────────────────
# Low-level pair search
# ─────────────────────────────────────────────────────────────────────────────
def pair_list(frame, r_max):
    """All i<j atom pairs within ``r_max`` (PBC).

    Returns
    -------
    i, j : (M,) int   atom indices
    rij  : (M, 3) float   minimum-image displacement r_i - r_j
    r    : (M,) float   distance
    """
    L = frame.L
    p = frame.wrapped()
    tree = cKDTree(p, boxsize=L)
    pr = tree.query_pairs(r_max, output_type="ndarray")
    if len(pr) == 0:
        return (np.empty(0, int), np.empty(0, int),
                np.empty((0, 3)), np.empty(0))
    i, j = pr[:, 0], pr[:, 1]
    d = p[i] - p[j]
    d -= L * np.round(d / L)
    r = np.sqrt(np.einsum("ij,ij->i", d, d))
    return i, j, d, r


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame neighbor cache
# ─────────────────────────────────────────────────────────────────────────────
class NeighborCache:
    """Cache the within-``r_max`` pair list of a frame for reuse across analyses.

    Example
    -------
    >>> nc = NeighborCache(frame, r_max=10.0)
    >>> i, j, rij, r = nc.pairs()
    """

    def __init__(self, frame, r_max):
        self.frame = frame
        self.r_max = float(r_max)
        self._cache = None

    def pairs(self):
        if self._cache is None:
            self._cache = pair_list(self.frame, self.r_max)
        return self._cache


# ─────────────────────────────────────────────────────────────────────────────
# Bond list (per-pair cutoff)
# ─────────────────────────────────────────────────────────────────────────────
def bond_list(frame, cutoffs: CutoffMatrix, cache: NeighborCache | None = None):
    """Bonds (i<j) using the element-pair cutoff matrix.

    Parameters
    ----------
    frame : Frame
    cutoffs : CutoffMatrix
    cache : NeighborCache | None
        If given (and its r_max >= cutoffs.rmax) the cached pair list is reused.

    Returns
    -------
    bonds : (Nb, 2) int   atom indices
    r     : (Nb,) float   bond lengths
    """
    rmax = cutoffs.rmax
    if rmax <= 0:
        return np.empty((0, 2), int), np.empty(0)
    if cache is not None and cache.r_max >= rmax:
        i, j, _, r = cache.pairs()
    else:
        i, j, _, r = pair_list(frame, rmax)
    if len(i) == 0:
        return np.empty((0, 2), int), np.empty(0)

    elem = frame.elements
    # per-pair cutoff for each candidate
    rc = np.array([cutoffs.get(elem[a], elem[b]) for a, b in zip(i, j)])
    keep = (rc > 0) & (r <= rc)
    bonds = np.stack([i[keep], j[keep]], axis=1)
    return bonds, r[keep]


def neighbor_table(frame, cutoffs: CutoffMatrix, cache: NeighborCache | None = None):
    """Adjacency as a list-of-lists: ``nbrs[i]`` = list of bonded atom indices."""
    bonds, _ = bond_list(frame, cutoffs, cache)
    nbrs = [[] for _ in range(frame.n_atoms)]
    for a, b in bonds:
        nbrs[a].append(b)
        nbrs[b].append(a)
    return nbrs
