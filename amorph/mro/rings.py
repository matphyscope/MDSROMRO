"""
rings.py — Ring statistics (King's shortest-path criterion).

For each bond (i,j): remove it, find the shortest path i→j; ring size = path
length. Each ring of size n is found n times, deduplicated by atom set.

Element-agnostic: rings are also broken down by composition — "pure-X" if all
atoms are element X, otherwise "mixed". Counts per size are time-averaged over
the provided frames with block errors.

⚠ Cost: King's search is O(edges × BFS). For large cells use a modest number of
frames (``max_frames``) — the ring distribution is well-converged with a few
decorrelated snapshots.
"""
from __future__ import annotations
import numpy as np
from collections import Counter

from ..core.average import block_average
from ._common import build_graph, require_networkx


def _frame_rings(frame, cutoffs, max_size):
    nx = require_networkx()
    G = build_graph(frame, cutoffs)
    seen = {}
    for u, v in list(G.edges()):
        G.remove_edge(u, v)
        try:
            path = nx.shortest_path(G, u, v)
            size = len(path)
            if 3 <= size <= max_size:
                key = frozenset(path)
                if key not in seen:
                    seen[key] = tuple(sorted(path))
        except nx.NetworkXNoPath:
            pass
        G.add_edge(u, v)
    return list(seen.values())


def _classify(ring_atoms, elements):
    els = [elements[a] for a in ring_atoms]
    uniq = set(els)
    return f"pure-{els[0]}" if len(uniq) == 1 else "mixed"


def ring_statistics(traj, cutoffs, max_size=12, max_frames=None, n_blocks=3,
                    verbose=False):
    """Time-averaged ring-size distribution.

    Parameters
    ----------
    traj : list[Frame]
    cutoffs : CutoffMatrix
    max_size : int   largest ring to count
    max_frames : int | None   evenly subsample to at most this many frames
    n_blocks : int

    Returns
    -------
    dict
      "sizes"        : array 3..max_size
      "count"        : {"mean": (.,), "err": (.,)}   rings per frame by size
      "per_atom"     : {"mean": (.,), "err": (.,)}   rings per atom by size
      "composition"  : {size: {class: mean_count}}  (last-averaged breakdown)
      "n_frames"     : frames actually used
    """
    frames = traj
    if max_frames is not None and len(traj) > max_frames:
        idx = np.linspace(0, len(traj) - 1, max_frames).astype(int)
        frames = [traj[i] for i in idx]

    sizes = np.arange(3, max_size + 1)
    pf_count = []
    pf_peratom = []
    comp_acc = {s: Counter() for s in sizes}
    for n, fr in enumerate(frames):
        rings = _frame_rings(fr, cutoffs, max_size)
        by_size = Counter(len(r) for r in rings)
        cvec = np.array([by_size.get(s, 0) for s in sizes], dtype=float)
        pf_count.append(cvec)
        pf_peratom.append(cvec / fr.n_atoms)
        for r in rings:
            comp_acc[len(r)][_classify(r, fr.elements)] += 1
        if verbose:
            print(f"  [rings] frame {n+1}/{len(frames)}: {len(rings)} rings")

    cM, cE = block_average(pf_count, n_blocks)
    pM, pE = block_average(pf_peratom, n_blocks)
    composition = {int(s): {k: v / len(frames) for k, v in comp_acc[s].items()}
                   for s in sizes}
    return dict(sizes=sizes,
                count={"mean": cM, "err": cE},
                per_atom={"mean": pM, "err": pE},
                composition=composition,
                n_frames=len(frames))
