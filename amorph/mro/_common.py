"""Shared helpers for MRO analyses: bond graph + PBC cluster unwrapping."""
from __future__ import annotations
import numpy as np

from ..core.neighbors import bond_list, CutoffMatrix

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None


def require_networkx():
    if nx is None:
        raise ImportError("networkx is required: pip install networkx")
    return nx


def build_graph(frame, cutoffs: CutoffMatrix, nodes=None, pair_filter=None):
    """Build a networkx graph of the bond network.

    Parameters
    ----------
    nodes : iterable[int] | None   restrict to these atom indices (else all).
    pair_filter : callable(elemA, elemB)->bool | None
        Keep only bonds whose element pair passes this test (e.g. C–C only).
    """
    require_networkx()
    bonds, _ = bond_list(frame, cutoffs)
    G = nx.Graph()
    nodeset = set(range(frame.n_atoms)) if nodes is None else set(nodes)
    G.add_nodes_from(nodeset)
    elem = frame.elements
    for a, b in bonds:
        if a not in nodeset or b not in nodeset:
            continue
        if pair_filter is not None and not pair_filter(elem[a], elem[b]):
            continue
        G.add_edge(int(a), int(b))
    return G


def neighbor_sets(frame, cutoffs: CutoffMatrix):
    """Adjacency as a list of python sets (fast set-intersection for sharing)."""
    bonds, _ = bond_list(frame, cutoffs)
    nb = [set() for _ in range(frame.n_atoms)]
    for a, b in bonds:
        nb[int(a)].add(int(b))
        nb[int(b)].add(int(a))
    return nb


def unwrap_component(atoms, G, frame):
    """PBC-unwrap a connected component by walking its spanning tree.

    Returns (N,3) unwrapped coordinates in the order of ``list(atoms)``.
    """
    atoms = list(atoms)
    wrapped = frame.wrapped()
    L = frame.L
    pos = {atoms[0]: wrapped[atoms[0]].copy()}
    queue = [atoms[0]]
    seen = {atoms[0]}
    aset = set(atoms)
    while queue:
        u = queue.pop()
        for v in G.neighbors(u):
            if v not in aset or v in seen:
                continue
            dr = wrapped[v] - wrapped[u]
            dr -= L * np.round(dr / L)
            pos[v] = pos[u] + dr
            seen.add(v)
            queue.append(v)
    return np.array([pos[a] for a in atoms])


def gyration(coords):
    """Radius of gyration, principal extents L_a (2√λ_max), aspect λ_max/λ_min."""
    com = coords.mean(axis=0)
    diffs = coords - com
    Rg = float(np.sqrt(np.mean(np.sum(diffs ** 2, axis=1))))
    T = diffs.T @ diffs / len(coords)
    eig = np.sort(np.linalg.eigvalsh(T))[::-1]
    L_a = 2.0 * np.sqrt(max(eig[0], 0.0))
    aspect = eig[0] / max(eig[2], 1e-12)
    return Rg, L_a, aspect, eig
