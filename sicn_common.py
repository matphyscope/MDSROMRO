#!/usr/bin/env python3
"""
sicn_common.py — Shared utilities for all sicn_*.py analysis modules.

Provides:
  - Type constants (TYPE_SI, TYPE_C, TYPE_N)
  - Bond cutoffs (matched to v8 LAMMPS input)
  - LAMMPS data file parser (parse_lammps_data)
  - LAMMPS dump trajectory parser (parse_lammps_dump)
  - PBC distance computation
  - Bond list construction (KDTree-accelerated)
  - Mass density + composition helpers

All new modules (sicn_gr_extended, sicn_rings, sicn_bt, sicn_cluster, etc.)
import from this single source to avoid code duplication.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# NumPy version compatibility — np.trapezoid (NumPy 2.0+) vs np.trapz (1.x)
# ─────────────────────────────────────────────────────────────────────────────
if hasattr(np, "trapezoid"):
    np_trapezoid = np.trapezoid
else:
    np_trapezoid = np.trapz

# ─────────────────────────────────────────────────────────────────────────────
# Constants — matching LAMMPS data file convention (v8 unified)
# ─────────────────────────────────────────────────────────────────────────────
TYPE_SI = 1
TYPE_C  = 2
TYPE_N  = 3

TYPE_NAME = {TYPE_SI: "Si", TYPE_C: "C", TYPE_N: "N"}
TYPE_COLOR = {TYPE_SI: "#1f77b4", TYPE_C: "#2ca02c", TYPE_N: "#d62728"}
ATOMIC_MASS = {TYPE_SI: 28.09, TYPE_C: 12.01, TYPE_N: 14.01}

# Bond cutoffs (Å) — matched to v8 LAMMPS input file compute coord/atom
# IMPORTANT: C-N and N-N are 0 because Tersoff (chi_CN=0, N-N B=0) doesn't
# bond these pairs. LAMMPS CN computes also exclude them. Keep at 0 for
# consistency with LAMMPS definition.
BOND_CUTOFFS = {
    (TYPE_SI, TYPE_SI): 2.70,
    (TYPE_SI, TYPE_C ): 2.25,
    (TYPE_SI, TYPE_N ): 2.15,
    (TYPE_C , TYPE_C ): 1.85,
    (TYPE_C , TYPE_N ): 0.0,    # v7 intent: no C-N bond (chi_CN=0)
    (TYPE_N , TYPE_N ): 0.0,    # v7 intent: no N-N bond (Mota B=0)
}

# Pair labels for output (ordered)
PAIRS = [("Si","Si"), ("Si","C"), ("Si","N"), ("C","C"), ("C","N"), ("N","N")]
PAIR_TYPES = [(TYPE_SI,TYPE_SI),(TYPE_SI,TYPE_C),(TYPE_SI,TYPE_N),
              (TYPE_C,TYPE_C),(TYPE_C,TYPE_N),(TYPE_N,TYPE_N)]


def get_cutoff(t1: int, t2: int) -> float:
    """Symmetric bond cutoff lookup."""
    key = (min(t1, t2), max(t1, t2))
    return BOND_CUTOFFS.get(key, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# LAMMPS data file parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_lammps_data(filepath):
    """Parse a LAMMPS data file (orthogonal box). Returns dict:
        n_atoms : int
        types   : (N,) int  array (LAMMPS type IDs, 1-indexed)
        coords  : (N,3) float array (Å, unwrapped if dump-style image flags present,
                                     else as-written)
        box     : (3,2) float array [[xlo,xhi],[ylo,yhi],[zlo,zhi]] (Å)
    """
    types, coords = [], []
    box = [None, None, None]
    section = None
    SECTION_HEADERS = ("Masses", "Atoms", "Velocities", "Bonds", "Angles",
                       "Dihedrals", "Impropers", "PairIJ")

    with open(filepath) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Box bounds
            if "xlo xhi" in line:
                p = line.split(); box[0] = (float(p[0]), float(p[1])); continue
            if "ylo yhi" in line:
                p = line.split(); box[1] = (float(p[0]), float(p[1])); continue
            if "zlo zhi" in line:
                p = line.split(); box[2] = (float(p[0]), float(p[1])); continue
            # Section headers (may have trailing comments like "Atoms # atomic")
            first = line.split()[0]
            if first in SECTION_HEADERS:
                section = first
                continue
            # Atom lines: id type x y z [ix iy iz]
            if section == "Atoms":
                p = line.split()
                if len(p) >= 5:
                    try:
                        types.append(int(p[1]))
                        # apply image flags if present (cols 5,6,7) to unwrap
                        x = float(p[2]); y = float(p[3]); z = float(p[4])
                        if len(p) >= 8:
                            try:
                                ix, iy, iz = int(p[5]), int(p[6]), int(p[7])
                                Lx = box[0][1]-box[0][0] if box[0] else 0
                                Ly = box[1][1]-box[1][0] if box[1] else 0
                                Lz = box[2][1]-box[2][0] if box[2] else 0
                                x += ix * Lx
                                y += iy * Ly
                                z += iz * Lz
                            except (ValueError, IndexError):
                                pass
                        coords.append([x, y, z])
                    except ValueError:
                        pass

    types = np.array(types, dtype=int)
    coords = np.array(coords, dtype=float)
    box = np.array(box, dtype=float)
    if box.shape != (3, 2) or np.any(np.isnan(box)):
        raise ValueError(f"Could not parse box from {filepath}")
    return {
        "n_atoms": len(types),
        "types": types,
        "coords": coords,
        "box": box,
    }

    types = np.array(types, dtype=int)
    coords = np.array(coords, dtype=float)
    box = np.array(box, dtype=float)
    if box.shape != (3, 2) or np.any(np.isnan(box)):
        raise ValueError(f"Could not parse box from {filepath}")
    return {
        "n_atoms": len(types),
        "types": types,
        "coords": coords,
        "box": box,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LAMMPS dump trajectory parser  ('atom' or 'custom' style, orthogonal box)
# ─────────────────────────────────────────────────────────────────────────────
def parse_lammps_dump(filepath, frames="last"):
    """Parse LAMMPS dump trajectory. 'atom' or 'custom' style (scaled or unscaled).
    frames='last'  → single dict (last frame)
    frames='all'   → list of dicts (one per frame)
    frames=int     → first N frames
    """
    out = []
    cur, state, cols = None, None, []

    with open(filepath) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("ITEM: TIMESTEP"):
                if cur is not None:
                    out.append(_dump_finalize(cur, cols))
                cur = {"types": [], "coords": [], "box": []}
                state = "ts"
            elif line.startswith("ITEM: NUMBER OF ATOMS"):
                state = "natoms"
            elif line.startswith("ITEM: BOX BOUNDS"):
                state = "box"
            elif line.startswith("ITEM: ATOMS"):
                state = "atoms"
                cols = line.replace("ITEM: ATOMS", "").split()
            elif state == "box":
                p = line.split()
                cur["box"].append([float(p[0]), float(p[1])])
            elif state == "atoms":
                p = line.split()
                idx_t = cols.index("type") if "type" in cols else 1
                # x/y/z preferred; xs/ys/zs (scaled) fallback
                if "x" in cols:
                    ix, iy, iz = cols.index("x"), cols.index("y"), cols.index("z")
                    cur["types"].append(int(p[idx_t]))
                    cur["coords"].append([float(p[ix]), float(p[iy]), float(p[iz])])
                elif "xs" in cols:
                    ix, iy, iz = cols.index("xs"), cols.index("ys"), cols.index("zs")
                    cur["types"].append(int(p[idx_t]))
                    cur["coords"].append([float(p[ix]), float(p[iy]), float(p[iz])])
                    cur["_scaled"] = True
    if cur is not None:
        out.append(_dump_finalize(cur, cols))

    if not out:
        raise ValueError(f"No frames found in {filepath}")
    if frames == "last":
        return out[-1]
    elif frames == "all":
        return out
    elif isinstance(frames, int):
        return out[:frames]
    return out


def _dump_finalize(cur, cols):
    box = np.array(cur["box"], dtype=float)
    types = np.array(cur["types"], dtype=int)
    coords = np.array(cur["coords"], dtype=float)
    if cur.get("_scaled"):
        L = box[:, 1] - box[:, 0]
        coords = box[:, 0][None, :] + coords * L[None, :]
    # Sort by id if available (already sorted by LAMMPS dump_modify sort id, but safe)
    return {"n_atoms": len(types), "types": types, "coords": coords, "box": box}


# ─────────────────────────────────────────────────────────────────────────────
# PBC utilities
# ─────────────────────────────────────────────────────────────────────────────
def pbc_displacement(d, L):
    """Minimum-image displacement. d: (...,3), L: (3,) box lengths."""
    L = np.asarray(L)
    return d - L * np.round(d / L)


def pbc_distance(coords1, coords2, L):
    """Minimum-image distance(s) between pairs of coords."""
    return np.linalg.norm(pbc_displacement(coords1 - coords2, L), axis=-1)


def wrap_coords(coords, box):
    """Wrap coords into [box[:,0], box[:,1]) periodic cell."""
    L = box[:, 1] - box[:, 0]
    return box[:, 0][None, :] + (coords - box[:, 0][None, :]) % L[None, :]


# ─────────────────────────────────────────────────────────────────────────────
# Bond list (KDTree-accelerated, type-aware cutoff)
# ─────────────────────────────────────────────────────────────────────────────
def build_bond_list(struct, cutoffs=None):
    """Build list of (i, j) bond indices. PBC-correct.
    Returns:
        bonds      : (Nb, 2) int array of atom indices (i<j)
        coords_w   : (N, 3) coords wrapped into PBC box (origin at 0)
        L          : (3,) box lengths
    """
    from scipy.spatial import cKDTree

    if cutoffs is None:
        cutoffs = BOND_CUTOFFS

    coords = struct["coords"]
    types = struct["types"]
    box = struct["box"]
    L = box[:, 1] - box[:, 0]

    # Wrap into [0, L) for KDTree boxsize PBC
    coords_w = (coords - box[:, 0][None, :]) % L[None, :]

    max_cut = max(cutoffs.values())
    tree = cKDTree(coords_w, boxsize=L)
    pairs = tree.query_pairs(r=max_cut, output_type="ndarray")

    if pairs.size == 0:
        return np.empty((0, 2), dtype=int), coords_w, L

    bonds = []
    for i, j in pairs:
        ti, tj = types[i], types[j]
        rc = cutoffs.get((min(ti, tj), max(ti, tj)), 0.0)
        if rc <= 0:
            continue
        d = pbc_distance(coords_w[i], coords_w[j], L)
        if d <= rc:
            bonds.append((i, j))
    return np.asarray(bonds, dtype=int).reshape(-1, 2), coords_w, L


# ─────────────────────────────────────────────────────────────────────────────
# Density & composition
# ─────────────────────────────────────────────────────────────────────────────
def density_g_per_cc(struct):
    """Mass density in g/cc."""
    M = sum(ATOMIC_MASS[t] for t in struct["types"])
    V_A3 = float(np.prod(struct["box"][:, 1] - struct["box"][:, 0]))
    return (M / 6.022e23) / (V_A3 * 1e-24)


def composition(struct):
    """Return (x_Si, x_C, x_N) mole fractions."""
    N = len(struct["types"])
    return (
        float(np.sum(struct["types"] == TYPE_SI) / N),
        float(np.sum(struct["types"] == TYPE_C ) / N),
        float(np.sum(struct["types"] == TYPE_N ) / N),
    )


def number_density(struct):
    """Atoms per Å³."""
    V_A3 = float(np.prod(struct["box"][:, 1] - struct["box"][:, 0]))
    return struct["n_atoms"] / V_A3


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: detect file type & load
# ─────────────────────────────────────────────────────────────────────────────
def load_structure(filepath, frames="last"):
    """Auto-detect data vs dump and load. Returns struct dict (or list if frames='all')."""
    fp = Path(filepath)
    text_head = ""
    with open(fp) as f:
        for _ in range(3):
            text_head += f.readline()
    if "ITEM: TIMESTEP" in text_head:
        return parse_lammps_dump(fp, frames=frames)
    return parse_lammps_data(fp)


if __name__ == "__main__":
    # Quick self-test
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 sicn_common.py <data_or_dump_file>")
        sys.exit(0)
    s = load_structure(sys.argv[1])
    print(f"Loaded {s['n_atoms']} atoms")
    print(f"Box   : {s['box']}")
    print(f"Density: {density_g_per_cc(s):.4f} g/cc")
    x = composition(s)
    print(f"Comp  : Si={x[0]*100:.2f}%  C={x[1]*100:.2f}%  N={x[2]*100:.2f}%")
    print(f"Number density: {number_density(s):.5f} atoms/Å³")
    bonds, _, L = build_bond_list(s)
    print(f"Bonds : {len(bonds)} (avg CN ≈ {2*len(bonds)/s['n_atoms']:.2f})")
