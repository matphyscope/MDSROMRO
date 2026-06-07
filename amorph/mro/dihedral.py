"""
dihedral.py — Dihedral (torsion) angle distributions.

For each bonded central pair (j,k), every i∈N(j)\{k}, l∈N(k)\{j} forms a
torsion i–j–k–l. φ∈[0°,180°] is the angle between planes (i,j,k) and (j,k,l).
Flat distribution → no intermediate-range order; sharp peaks (≈60° gauche,
≈180° trans) → conformational order.

Distributions per element-typed quadruplet (canonical, reversal-symmetric) are
time-averaged with block errors. Quadruplet enumeration can be large; pass
``max_frames`` to subsample frames.
"""
from __future__ import annotations
import numpy as np

from ..core.average import block_average
from ._common import neighbor_sets


def _dihedral_angle(p1, p2, p3, p4, frame):
    b1 = frame.min_image(p2 - p1)
    b2 = frame.min_image(p3 - p2)
    b3 = frame.min_image(p4 - p3)
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    a, b = np.linalg.norm(n1), np.linalg.norm(n2)
    if a < 1e-9 or b < 1e-9:
        return np.nan
    c = np.clip(np.dot(n1, n2) / (a * b), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def _label(quad, elem):
    t = [elem[a] for a in quad]
    return min("-".join(t), "-".join(reversed(t)))


def _frame_dihedrals(frame, cutoffs, edges, types_wanted):
    nb = neighbor_sets(frame, cutoffs)
    elem = frame.elements
    coords = frame.coords
    hist = {lab: np.zeros(len(edges) - 1) for lab in types_wanted} if types_wanted else {}
    counts = {lab: 0 for lab in hist}
    bonds = set()
    for j in range(frame.n_atoms):
        for k in nb[j]:
            if j < k:
                bonds.add((j, k))
    raw = {} if types_wanted is None else None
    angle_by_label = {}
    for (j, k) in bonds:
        for i in nb[j]:
            if i == k:
                continue
            for l in nb[k]:
                if l == j or l == i:
                    continue
                phi = _dihedral_angle(coords[i], coords[j], coords[k], coords[l], frame)
                if np.isnan(phi):
                    continue
                lab = _label((i, j, k, l), elem)
                angle_by_label.setdefault(lab, []).append(phi)
    return angle_by_label


def dihedral_distribution(traj, cutoffs, labels=None, nbins=36,
                          max_frames=None, n_blocks=3):
    """Time-averaged torsion-angle distributions.

    Parameters
    ----------
    labels : list[str] | None
        Quadruplet labels to keep (e.g. 'Si-N-Si-N'). None → keep the most
        populated ones discovered in the first frame.
    nbins : int over 0–180°
    max_frames : int | None

    Returns
    -------
    dict
      "phi"    : (nbins,) bin centres (deg)
      "dist"   : {label: {"p": density, "err": 1σ}}   ∫p dφ = 1
      "counts" : {label: total angles (final frame)}
    """
    frames = traj
    if max_frames is not None and len(traj) > max_frames:
        idx = np.linspace(0, len(traj) - 1, max_frames).astype(int)
        frames = [traj[i] for i in idx]
    edges = np.linspace(0, 180, nbins + 1)
    phi = 0.5 * (edges[:-1] + edges[1:])
    dphi = edges[1] - edges[0]

    # discover labels from first frame if not given
    first = _frame_dihedrals(frames[0], cutoffs, edges, None)
    if labels is None:
        labels = sorted(first, key=lambda k: -len(first[k]))[:8]

    pf = {lab: [] for lab in labels}
    counts = {}
    for fr in frames:
        ang = _frame_dihedrals(fr, cutoffs, edges, None)
        for lab in labels:
            a = np.array(ang.get(lab, []))
            counts[lab] = len(a)
            if len(a):
                h, _ = np.histogram(a, bins=edges)
                area = h.sum() * dphi
                pf[lab].append(h / area if area > 0 else np.zeros(nbins))
            else:
                pf[lab].append(np.full(nbins, np.nan))
    dist = {lab: dict(zip(("p", "err"), block_average(pf[lab], n_blocks)))
            for lab in labels}
    return dict(phi=phi, dist=dist, counts=counts)
