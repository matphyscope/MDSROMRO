"""
sweep.py — Temperature-series driver: property-vs-temperature curves.

The SiCN T-scan decks write one dump per temperature point. This module:
  1. discovers those dumps (both naming schemes used by the two decks),
  2. for each temperature, computes a set of time-averaged scalar observables
     (density, CN, tetrahedral order, rings, free-cluster size, network
     percolation, Bhatia–Thornton S_CC), each with a block-averaged error,
  3. assembles them into temperature-indexed columns ready to plot vs T.

Naming schemes auto-detected:
  large2 (custom dump, element col):  dump.300K.lammpstrj … dump.cool_300K.lammpstrj
  large1 (atom dump, scaled, no elem): dump_T0300.lammpstrj … dump_cool_T0300.lammpstrj
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np

from .core.io import load
from .core.average import select_frames, block_average
from .core.neighbors import CutoffMatrix
from .core.frame import species_of
from .core.parallel import pmap
from .sro import coordination, tetrahedra
from .mro import rings as mro_rings, clusters as mro_clusters, bhatia_thornton

# dump.300K / dump.cool_300K / dump_T0300 / dump_cool_T0300  (+ optional K)
_DUMP_RE = re.compile(r"dump[._](cool_)?T?0*(\d+)K?\.lammpstrj$")


def discover_dumps(directory):
    """Find temperature dumps in a directory (both naming schemes).

    Returns
    -------
    list[dict] sorted by (cool, T): each {"T": int, "cool": bool, "path": str,
                                          "label": "300K" / "300K(cool)"}
    """
    directory = Path(directory)
    out = []
    for p in directory.iterdir():
        m = _DUMP_RE.match(p.name)
        if not m:
            continue
        cool = m.group(1) is not None
        T = int(m.group(2))
        out.append({"T": T, "cool": cool, "path": str(p),
                    "label": f"{T}K{'(cool)' if cool else ''}"})
    out.sort(key=lambda d: (d["cool"], d["T"]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-trajectory scalar observables
# ─────────────────────────────────────────────────────────────────────────────
def trajectory_scalars(traj, cutoffs: CutoffMatrix, *, masses=None,
                       free_element=None, network_group=None,
                       bt_groups=None, tetra_center=None,
                       include_rings=False, max_ring_size=10,
                       max_frames_rings=3, n_blocks=5):
    """Compute named scalar observables (each as (mean, err)) for one trajectory.

    Only the analyses whose inputs are provided are run (free_element,
    network_group, bt_groups, tetra_center). Density and CN always run.
    """
    sp = species_of(traj)
    cols = {}

    # density & number density
    dens = [fr.mass_density(masses) for fr in traj]
    ndens = [fr.number_density() for fr in traj]
    cols["density_gcc"] = block_average(dens, n_blocks)
    cols["number_density"] = block_average(ndens, n_blocks)

    # coordination numbers (total per species)
    CN = coordination.coordination_numbers(traj, cutoffs, n_blocks=n_blocks)
    for A in sp:
        cols[f"CN_{A}"] = CN["cn_total"][A]

    # tetrahedral order per species
    if tetra_center is not None:
        TQ = tetrahedra.tetrahedral_order(traj, n_blocks=n_blocks)
        if tetra_center in TQ["q"]:
            cols[f"q_tetra_{tetra_center}"] = TQ["q"][tetra_center]

    # rings per atom (optional, expensive)
    if include_rings:
        RG = mro_rings.ring_statistics(traj, cutoffs, max_size=max_ring_size,
                                       max_frames=max_frames_rings,
                                       n_blocks=min(n_blocks, max_frames_rings))
        per_atom_total_mean = float(np.nansum(RG["per_atom"]["mean"]))
        per_atom_total_err = float(np.sqrt(np.nansum(RG["per_atom"]["err"] ** 2)))
        cols["rings_per_atom"] = (per_atom_total_mean, per_atom_total_err)

    # free-element clusters
    if free_element is not None and free_element in sp:
        FC = mro_clusters.free_clusters(traj, cutoffs, element=free_element,
                                        n_blocks=n_blocks)
        cols[f"free{free_element}_mean_La"] = FC["mean_La"]      # finite clusters only
        cols[f"free{free_element}_max_finite"] = FC["max_finite_size"]
        cols[f"free{free_element}_n_graphenic"] = FC["n_graphenic"]
        cols[f"free{free_element}_n_percolating"] = FC["n_percolating"]

    # network percolation
    if network_group is not None:
        NET = mro_clusters.network_percolation(traj, cutoffs, group=network_group,
                                               n_blocks=n_blocks)
        cols["network_largest_frac"] = NET["largest_fraction"]
        cols["network_percolation"] = NET["percolation_prob"]

    # Bhatia–Thornton S_CC at low Q (chemical (de)mixing)
    if bt_groups is not None:
        gA, gB = bt_groups
        try:
            BT = bhatia_thornton.bhatia_thornton(traj, gA, gB, n_blocks=n_blocks)
            # smallest physically accessible Q is 2π/L; don't claim below it
            qmin = 2.0 * np.pi / float(np.mean([fr.L.max() for fr in traj]))
            ilow = int(np.argmin(np.abs(BT["q"] - max(0.5, qmin))))
            cols["SCC_lowQ"] = (float(BT["S_CC"][ilow]), 0.0)
            cols["SCC_ideal"] = (float(BT["S_CC_ideal"]), 0.0)
        except ValueError:
            pass

    return cols


# ─────────────────────────────────────────────────────────────────────────────
# Full temperature series
# ─────────────────────────────────────────────────────────────────────────────
def _series_task(task):
    """Top-level worker (picklable): load one dump and compute its scalars."""
    d, path, type_map, prod_range, stride, cutoffs, scalar_kwargs = task
    traj = load(path, type_map=type_map, frames="all")
    traj = select_frames(traj, frame_range=prod_range, stride=stride)
    scalars = trajectory_scalars(traj, cutoffs, **scalar_kwargs)
    return d, scalars


def temperature_series(dumps, cutoffs: CutoffMatrix, *, type_map=None,
                       prod_range=None, stride=1, include_cool=True,
                       verbose=True, jobs=1, **scalar_kwargs):
    """Run :func:`trajectory_scalars` across a list of temperature dumps.

    Parameters
    ----------
    dumps : list[dict]
        Output of :func:`discover_dumps` (or hand-built dicts with T/cool/path/label).
    cutoffs : CutoffMatrix
    type_map : dict | None    for atom-style dumps without an element column.
    prod_range, stride : production window for every dump.
    include_cool : also process the post-cooling 300 K points.
    jobs : int   temperatures processed in parallel (<=0 = all CPUs). Each worker
        loads its own dump, so this is the most efficient axis to parallelise.
    scalar_kwargs : forwarded to :func:`trajectory_scalars`.

    Returns
    -------
    dict
      "T"        : (n,) temperatures
      "labels"   : list of labels (e.g. '300K', '300K(cool)')
      "cool"     : (n,) bool
      "columns"  : {name: {"mean": (n,), "err": (n,)}}
    """
    entries = [d for d in dumps if include_cool or not d["cool"]]
    T, labels, cool = [], [], []
    col_mean, col_err = {}, {}

    n_tot = len(entries)
    tasks = [(d, d["path"], type_map, prod_range, stride, cutoffs, scalar_kwargs)
             for d in entries]
    cb = (lambda k: print(f"  [{k}/{n_tot}] temperatures done", flush=True)) \
        if verbose else None
    for d, scalars in pmap(_series_task, tasks, jobs=jobs, on_done=cb):
        T.append(d["T"]); labels.append(d["label"]); cool.append(d["cool"])
        for name, (m, e) in scalars.items():
            col_mean.setdefault(name, []).append(m)
            col_err.setdefault(name, []).append(e)

    columns = {name: {"mean": np.array(col_mean[name]),
                      "err": np.array(col_err[name])}
               for name in col_mean}
    return dict(T=np.array(T), labels=labels, cool=np.array(cool),
                columns=columns)


def to_table(series):
    """Format a temperature series as aligned text (T + each column mean±err)."""
    names = list(series["columns"].keys())
    header = f"{'T(K)':>8} {'cool':>5}  " + "  ".join(f"{n:>20}" for n in names)
    lines = [header, "-" * len(header)]
    for i, (T, lab) in enumerate(zip(series["T"], series["labels"])):
        cool = "yes" if series["cool"][i] else ""
        row = f"{T:>8} {cool:>5}  "
        for n in names:
            m = series["columns"][n]["mean"][i]
            e = series["columns"][n]["err"][i]
            row += f"  {m:>9.4f}±{e:<9.4f}"
        lines.append(row)
    return "\n".join(lines)
