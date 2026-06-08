"""
boo.py — Steinhardt bond-orientational order Q4, Q6, W4, W6 (via freud).

Per-atom local-symmetry fingerprints over the k nearest neighbours.
Reference (perfect crystals): tetrahedral/diamond Q4≈0.51 Q6≈0.63;
FCC Q4≈0.19 Q6≈0.58; BCC Q4≈0.04 Q6≈0.51; random → 0.

Time-averaged per species with block errors; the final frame's full per-atom
Q4/Q6/W4/W6 arrays are also returned for Q4–Q6 scatter maps.
"""
from __future__ import annotations
import numpy as np

from ..core.frame import species_of
from ..core.average import block_average

try:
    import freud
except ImportError:  # pragma: no cover
    freud = None

_KEYS = ("Q4", "Q6", "W4", "W6")


def _frame_boo(frame, num_neighbors):
    if freud is None:
        raise ImportError("freud is required: pip install freud-analysis")
    L = frame.L
    centered = frame.coords - (frame.box[:, 1] + frame.box[:, 0]) / 2
    fbox = freud.box.Box.from_box(L)
    aabb = freud.locality.AABBQuery(fbox, centered)
    nlist = aabb.query(centered, dict(num_neighbors=num_neighbors,
                                      exclude_ii=True)).toNeighborList()
    out = {}
    for key, (l, wl) in {"Q4": (4, False), "Q6": (6, False),
                         "W4": (4, True), "W6": (6, True)}.items():
        st = freud.order.Steinhardt(l=l, wl=wl)
        st.compute((fbox, centered), neighbors=nlist)
        out[key] = np.asarray(st.particle_order)
    return out


def steinhardt(traj, num_neighbors=4, n_blocks=5):
    """Time-averaged Steinhardt order parameters.

    Parameters
    ----------
    num_neighbors : int   k nearest neighbours per atom (4 = tetrahedral).

    Returns
    -------
    dict
      "species"    : list
      "num_neighbors" : int
      "Q4"/"Q6"/"W4"/"W6" : {A: (mean, err)}  per species
      "last_frame" : {"Q4":(N,), ..., "elements":(N,)}
    """
    sp = species_of(traj)
    pf = {k: {A: [] for A in sp} for k in _KEYS}
    last = None
    for fr in traj:
        vals = _frame_boo(fr, num_neighbors)
        for k in _KEYS:
            for A in sp:
                m = fr.mask(A)
                pf[k][A].append(vals[k][m].mean() if m.any() else np.nan)
        last = {**vals, "elements": fr.elements}
    result = {"species": sp, "num_neighbors": num_neighbors, "last_frame": last}
    for k in _KEYS:
        result[k] = {A: block_average(pf[k][A], n_blocks) for A in sp}
    return result
