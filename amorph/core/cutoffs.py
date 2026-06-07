"""
cutoffs.py — Bond cutoffs two ways, and a comparison helper.

Per the design decision, the package supports *both*:

1. **Fixed cutoffs** — values taken straight from the LAMMPS input deck
   (e.g. Si-C 2.25, Si-N 2.15 ...). These are not guesses: they are the same
   numbers the simulation's CN/ADF computes used, so analysis stays consistent
   with the run.
2. **g(r)-derived cutoffs** — the first minimum after the first peak of each
   partial g(r). Data-driven, but sensitive to noise, so always reported
   alongside the fixed values for the user to compare.

This module works on plain ``(r, g)`` arrays so it has no dependency on the SRO
RDF module (avoids a circular import). Feed it an RDF result computed by
:mod:`amorph.sro.rdf`.
"""
from __future__ import annotations
import numpy as np

from .neighbors import CutoffMatrix


def _smooth(y, window):
    """Simple centered moving average (odd window); window<=1 is a no-op."""
    if window is None or window <= 1:
        return y
    w = int(window) | 1  # force odd
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")


def first_minimum(r, g, search=(0.0, None), smooth_window=5):
    """Locate the first minimum of g(r) after its first maximum.

    Parameters
    ----------
    r, g : 1-D arrays
    search : (rlo, rhi)
        Restrict the search; rhi=None → end of array.
    smooth_window : int
        Moving-average window applied before peak/valley detection.

    Returns
    -------
    r_min : float | None   distance of the first minimum, or None if not found.
    """
    r = np.asarray(r, float)
    g = np.asarray(g, float)
    rlo, rhi = search
    rhi = r[-1] if rhi is None else rhi
    sel = (r >= rlo) & (r <= rhi)
    if not np.any(sel):
        return None
    rr, gg = r[sel], _smooth(g[sel], smooth_window)

    # first maximum: first index where g rises then falls
    dg = np.diff(gg)
    peak = None
    for k in range(1, len(dg)):
        if dg[k - 1] > 0 and dg[k] <= 0:
            peak = k
            break
    if peak is None:
        return None
    # first minimum after the peak: g falls then rises
    for k in range(peak + 1, len(dg)):
        if dg[k - 1] < 0 and dg[k] >= 0:
            return float(rr[k])
    return None


def derive_cutoffs(rdf_result, pairs, search=None, smooth_window=5, fallback=None):
    """Build a :class:`CutoffMatrix` from first minima of partial g(r).

    Parameters
    ----------
    rdf_result : dict
        Output of :func:`amorph.sro.rdf.partial_rdf`; must contain key ``"r"``
        and, per pair, ``rdf_result[(A, B)]["g"]``.
    pairs : list[(A, B)]
    search : dict | (rlo, rhi) | None
        Per-pair or global search window passed to :func:`first_minimum`.
    fallback : CutoffMatrix | None
        If a pair's minimum can't be found, use this matrix's value (else 0).

    Returns
    -------
    CutoffMatrix
    """
    r = rdf_result["r"]
    cm = CutoffMatrix()
    for p in pairs:
        if p not in rdf_result:
            continue
        win = (0.0, None)
        if isinstance(search, dict):
            win = search.get(p, (0.0, None))
        elif search is not None:
            win = search
        rmin = first_minimum(r, rdf_result[p]["g"], search=win,
                             smooth_window=smooth_window)
        if rmin is None:
            rmin = fallback.get(*p) if fallback is not None else 0.0
        cm.set(p[0], p[1], rmin)
    return cm


def compare(fixed: CutoffMatrix, derived: CutoffMatrix, pairs):
    """Return a printable comparison table (list of rows) of the two cutoff sets."""
    rows = [("pair", "fixed(Å)", "g(r)-min(Å)", "Δ(Å)")]
    for a, b in pairs:
        rf, rd = fixed.get(a, b), derived.get(a, b)
        rows.append((f"{a}-{b}", f"{rf:.3f}", f"{rd:.3f}", f"{rd - rf:+.3f}"))
    return rows


def format_table(rows):
    """Pretty-format rows (tuples of strings) into aligned text."""
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    lines = []
    for r in rows:
        lines.append("  ".join(str(v).ljust(widths[c]) for c, v in enumerate(r)))
    return "\n".join(lines)
