"""
rdf.py — Radial distribution functions and everything derived from them.

Time-averaged over a trajectory with block-averaged ±1σ error bands.

Provides
--------
* :func:`partial_rdf`   partial g_AB(r) for every species pair + total g(r)
* :func:`R_of_r`        R(r)=4πr²ρ·g(r)  (radial distribution function)
* :func:`G_of_r`        reduced G(r)=4πrρ·(g−1)
* :func:`coordination`  CN from shell integral of R(r) up to a cutoff
* :func:`measure_peaks` peak position / height / FWHM / Gaussian fit (R²,RMSE)

Normalisation (validated against the diamond/zincblende test crystals):
  unlike pair A≠B :  g = hist_AB / (N_A · ρ_B · shell)
  like  pair A=B  :  g = 2·hist_AA / (N_A · (N_A−1)/V · shell)
  total           :  g = 2·hist_all / (N · (N−1)/V · shell)
"""
from __future__ import annotations
import numpy as np

from functools import partial

from ..core.frame import species_of, unique_pairs
from ..core.neighbors import NeighborCache
from ..core.average import block_average
from ..core.parallel import pmap
from ..core._compat import trapezoid


# ─────────────────────────────────────────────────────────────────────────────
# Partial + total g(r)
# ─────────────────────────────────────────────────────────────────────────────
def _edges(r_max, nbins):
    edges = np.linspace(0.0, r_max, nbins + 1)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    return edges, r_mid, shell


def _frame_partials(frame, pairs, edges, shell, r_max):
    """Per-frame g(r) for each pair + total, returned as a dict pair->g array."""
    nc = NeighborCache(frame, r_max)
    i, j, _, r = nc.pairs()
    ea, eb = frame.elements[i], frame.elements[j]
    V = frame.volume
    out = {}
    for (A, B) in pairs:
        if A == B:
            mask = (ea == A) & (eb == A)
            Na = frame.count(A)
            h, _ = np.histogram(r[mask], bins=edges)
            h = h * 2.0
            ideal = Na * (Na - 1) / V if Na > 1 else 0.0
        else:
            mask = ((ea == A) & (eb == B)) | ((ea == B) & (eb == A))
            Na, Nb = frame.count(A), frame.count(B)
            h, _ = np.histogram(r[mask], bins=edges)
            ideal = Na * (Nb / V)
        out[(A, B)] = h / (ideal * shell) if ideal > 0 else np.full(len(shell), np.nan)
    # total
    N = frame.n_atoms
    h_all, _ = np.histogram(r, bins=edges)
    ideal_all = N * (N - 1) / V
    out["total"] = (2.0 * h_all) / (ideal_all * shell)
    return out


def partial_rdf(traj, pairs=None, r_max=10.0, nbins=500, n_blocks=5, jobs=1):
    """Time-averaged partial and total g(r) with block error bands.

    Parameters
    ----------
    traj : list[Frame]   production frames (already windowed/strided)
    pairs : list[(A,B)] | None   defaults to all unordered species pairs
    r_max : float
    nbins : int
    n_blocks : int

    Returns
    -------
    dict with:
      "r"        : (nbins,) bin centres
      "rho0"     : mean number density (atoms/Å³)
      "rho"      : {element: mean number density of that species}
      (A,B)      : {"g": mean, "err": 1σ}   for each pair
      "total"    : {"g": mean, "err": 1σ}
    """
    if pairs is None:
        pairs = unique_pairs(species_of(traj))
    edges, r_mid, shell = _edges(r_max, nbins)

    per_frame = {p: [] for p in pairs}
    per_frame["total"] = []
    worker = partial(_frame_partials, pairs=pairs, edges=edges, shell=shell, r_max=r_max)
    for g in pmap(worker, traj, jobs=jobs):
        for k in per_frame:
            per_frame[k].append(g[k])

    result = {"r": r_mid}
    for k in per_frame:
        mean, err = block_average(per_frame[k], n_blocks=n_blocks)
        result[k] = {"g": mean, "err": err}

    # densities (mean over frames)
    result["rho0"] = float(np.mean([fr.number_density() for fr in traj]))
    sp = species_of(traj)
    result["rho"] = {e: float(np.mean([fr.count(e) / fr.volume for fr in traj]))
                     for e in sp}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Derived curves
# ─────────────────────────────────────────────────────────────────────────────
def R_of_r(r, g, rho):
    """Radial distribution function R(r) = 4π r² ρ g(r)."""
    return 4.0 * np.pi * r ** 2 * rho * g


def G_of_r(r, g, rho):
    """Reduced/differential RDF G(r) = 4π r ρ (g(r) − 1)."""
    return 4.0 * np.pi * r * rho * (g - 1.0)


def coordination(r, g, rho_b, r_cut, r_min=0.0):
    """Coordination number = ∫ 4π r² ρ_b g(r) dr from r_min to r_cut.

    ``rho_b`` is the number density of the *neighbour* species (use total ρ for
    a total CN, or the partial ρ_B for a partial A→B CN).
    """
    sel = (r >= r_min) & (r <= r_cut)
    integrand = 4.0 * np.pi * r[sel] ** 2 * rho_b * g[sel]
    return float(trapezoid(integrand, r[sel]))


# ─────────────────────────────────────────────────────────────────────────────
# Peak measurement
# ─────────────────────────────────────────────────────────────────────────────
def _gaussian(x, a, mu, sigma, c):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c


def measure_peaks(r, g, search=(0.5, None), fit_halfwidth=0.4):
    """First-peak metrics of a g(r) curve.

    Returns dict: peak_r, height, fwhm, r2, rmse  (Gaussian fit around the peak).
    """
    from scipy.optimize import curve_fit
    rlo, rhi = search
    rhi = r[-1] if rhi is None else rhi
    sel = (r >= rlo) & (r <= rhi)
    rr, gg = r[sel], g[sel]
    if len(rr) < 5 or not np.any(np.isfinite(gg)):
        return dict(peak_r=np.nan, height=np.nan, fwhm=np.nan, r2=np.nan, rmse=np.nan)
    k = int(np.nanargmax(gg))
    peak_r, height = float(rr[k]), float(gg[k])

    # FWHM by half-max crossings around the peak
    half = 0.5 * height
    fwhm = np.nan
    left = right = None
    for m in range(k, 0, -1):
        if gg[m] <= half:
            left = np.interp(half, [gg[m], gg[m + 1]], [rr[m], rr[m + 1]])
            break
    for m in range(k, len(gg) - 1):
        if gg[m + 1] <= half:
            right = np.interp(half, [gg[m + 1], gg[m]], [rr[m + 1], rr[m]])
            break
    if left is not None and right is not None and right > left:
        fwhm = float(right - left)

    # Gaussian fit goodness in a window around the peak
    win = (rr >= peak_r - fit_halfwidth) & (rr <= peak_r + fit_halfwidth)
    r2 = rmse = np.nan
    if np.count_nonzero(win) >= 4:
        try:
            p0 = [height, peak_r, fwhm / 2.355 if fwhm == fwhm else 0.1, 0.0]
            popt, _ = curve_fit(_gaussian, rr[win], gg[win], p0=p0, maxfev=10000)
            pred = _gaussian(rr[win], *popt)
            resid = gg[win] - pred
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((gg[win] - np.mean(gg[win])) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            rmse = float(np.sqrt(np.mean(resid ** 2)))
        except (RuntimeError, ValueError):
            pass
    return dict(peak_r=peak_r, height=height, fwhm=fwhm, r2=r2, rmse=rmse)
