"""
voronoi.py — Voronoi tessellation (geometric SRO) via freud.

Per-atom Voronoi cell volume (free volume) and face count (geometric
coordination, counts *all* nearest neighbours, not just bonded). Time-averaged
per species with block errors; also returns the full per-atom arrays of the
last frame for distribution plots.
"""
from __future__ import annotations
import numpy as np

from ..core.frame import species_of
from ..core.average import block_average

try:
    import freud
except ImportError:  # pragma: no cover
    freud = None


def _frame_voronoi(frame):
    if freud is None:
        raise ImportError("freud is required: pip install freud-analysis")
    L = frame.L
    centered = frame.coords - (frame.box[:, 1] + frame.box[:, 0]) / 2
    fbox = freud.box.Box.from_box(L)
    voro = freud.locality.Voronoi()
    voro.compute(system=(fbox, centered))
    volumes = np.asarray(voro.volumes)
    faces = np.asarray(voro.nlist.neighbor_counts)
    return volumes, faces


def voronoi(traj, n_blocks=5):
    """Time-averaged Voronoi volume & face count per species.

    Returns
    -------
    dict
      "species"     : list
      "volume"      : {A: (mean, err)}   mean cell volume per species (Å³)
      "faces"       : {A: (mean, err)}   mean face count per species
      "last_frame"  : {"volumes": (N,), "faces": (N,), "elements": (N,)}
                       full per-atom arrays of the final frame (for plotting)
    """
    sp = species_of(traj)
    pf_vol = {A: [] for A in sp}
    pf_fac = {A: [] for A in sp}
    last = None
    for fr in traj:
        vol, fac = _frame_voronoi(fr)
        for A in sp:
            m = fr.mask(A)
            pf_vol[A].append(vol[m].mean() if m.any() else np.nan)
            pf_fac[A].append(fac[m].mean() if m.any() else np.nan)
        last = {"volumes": vol, "faces": fac, "elements": fr.elements}
    volume = {A: block_average(pf_vol[A], n_blocks) for A in sp}
    faces = {A: block_average(pf_fac[A], n_blocks) for A in sp}
    return dict(species=sp, volume=volume, faces=faces, last_frame=last)
