"""
hybridization.py — sp/sp²/sp³-style classification from coordination + planarity.

Element-agnostic generalisation of the SiCN sp²/sp³ analysis. Each atom is
classified by its bonded coordination number, and 3-coordinated atoms are
further split by the out-of-plane (improper) angle ω:

    ω ≈ 0°   → planar   (sp²-like, e.g. graphenic C)
    ω ≈ 35°  → pyramidal (sp³-like)

Per-species class fractions are time-averaged with block errors; the improper
angles of all 3-coordinated atoms (final frame) are returned for the ω
distribution plot.
"""
from __future__ import annotations
import numpy as np

from ..core.frame import species_of
from ..core.neighbors import neighbor_table, CutoffMatrix
from ..core.average import block_average

CLASSES = ["CN1-", "CN2", "CN3-planar", "CN3-pyramidal", "CN4", "CN5+"]


def _improper_angle(center, neighbor_coords, frame):
    """Out-of-plane angle ω (deg) of a 3-coordinated centre. 0=planar, ~35=tetra."""
    a, b, c = neighbor_coords
    va = frame.min_image(a - center)
    vb = frame.min_image(b - center)
    vc = frame.min_image(c - center)
    n = np.cross(vb - va, vc - va)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        return np.nan
    n /= nn
    d = abs(np.dot(n, va))
    bl = (np.linalg.norm(va) + np.linalg.norm(vb) + np.linalg.norm(vc)) / 3.0
    if bl < 1e-9:
        return np.nan
    return np.degrees(np.arcsin(min(1.0, d / bl)))


def _classify_frame(frame, cutoffs, planar_thresh, sp):
    nbrs = neighbor_table(frame, cutoffs)
    elem = frame.elements
    frac = {A: {c: 0 for c in CLASSES} for A in sp}
    counts = {A: 0 for A in sp}
    omegas = {A: [] for A in sp}
    for i in range(frame.n_atoms):
        A = elem[i]
        cn = len(nbrs[i])
        counts[A] += 1
        if cn <= 1:
            cls = "CN1-"
        elif cn == 2:
            cls = "CN2"
        elif cn == 3:
            w = _improper_angle(frame.coords[i],
                                [frame.coords[j] for j in nbrs[i]], frame)
            omegas[A].append(w)
            cls = "CN3-planar" if (w == w and w < planar_thresh) else "CN3-pyramidal"
        elif cn == 4:
            cls = "CN4"
        else:
            cls = "CN5+"
        frac[A][cls] += 1
    # convert to fractions
    out = {}
    for A in sp:
        n = counts[A]
        out[A] = np.array([frac[A][c] / n if n > 0 else np.nan for c in CLASSES])
    return out, omegas


def classify(traj, cutoffs: CutoffMatrix, planar_thresh=15.0, n_blocks=5):
    """Time-averaged hybridization classification.

    Returns
    -------
    dict
      "species" : list
      "classes" : list of class labels (column order)
      "fractions" : {A: (mean_vec, err_vec)} over CLASSES
      "omega_last" : {A: array of ω for CN=3 atoms in the final frame}
    """
    sp = species_of(traj)
    pf = {A: [] for A in sp}
    omega_last = {A: [] for A in sp}
    for fr in traj:
        out, omegas = _classify_frame(fr, cutoffs, planar_thresh, sp)
        for A in sp:
            pf[A].append(out[A])
            omega_last[A] = np.array([w for w in omegas[A] if w == w])
    fractions = {A: block_average(pf[A], n_blocks) for A in sp}
    return dict(species=sp, classes=CLASSES, fractions=fractions,
                omega_last=omega_last)
