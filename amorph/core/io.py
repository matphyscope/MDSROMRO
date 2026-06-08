"""
io.py — Element-agnostic LAMMPS readers (data files and dump trajectories).

Handles every variant produced by the SiCN input decks (and more):

* dump ``custom ... id type element x y z``   → element column present
* dump ``atom`` style                         → scaled ``xs ys zs``, no element
* dump with unwrapped ``xu yu zu``
* data files with/without image flags (``ix iy iz``)

When a dump has no ``element`` column the species is taken from ``type_map``
(``{type_id: symbol}``). If no map is given, the stringified type id is used as
the species label so the rest of the pipeline still works (fully generic).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from .frame import Frame


# ─────────────────────────────────────────────────────────────────────────────
# Dump trajectory
# ─────────────────────────────────────────────────────────────────────────────
def read_dump(path, type_map=None, frames="all"):
    """Read a LAMMPS dump file into a list of :class:`Frame`.

    Parameters
    ----------
    path : str | Path
    type_map : dict | None
        {lammps_type_id (int): element_symbol (str)}. Used only when the dump
        has no ``element`` column. If omitted, the type id (as a string) is the
        species label.
    frames : "all" | "last" | int
        "all" → list of every frame; "last" → list with the final frame;
        int N → first N frames.

    Returns
    -------
    list[Frame]
    """
    path = Path(path)
    out = []
    with open(path) as f:
        lines = f.readlines()

    i, n = 0, len(lines)
    warned_tilt = False
    while i < n:
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        timestep = int(lines[i + 1].split()[0])
        if not lines[i + 2].startswith("ITEM: NUMBER OF ATOMS"):
            raise ValueError(f"Malformed dump near line {i}: expected NUMBER OF ATOMS")
        natoms = int(lines[i + 3])

        if not lines[i + 4].startswith("ITEM: BOX BOUNDS"):
            raise ValueError(f"Malformed dump near line {i}: expected BOX BOUNDS")
        box = np.zeros((3, 2))
        for d in range(3):
            vals = list(map(float, lines[i + 5 + d].split()))
            box[d, 0], box[d, 1] = vals[0], vals[1]
            if len(vals) > 2 and abs(vals[2]) > 1e-8 and not warned_tilt:
                print("⚠️  triclinic box tilt detected — treated as orthogonal.")
                warned_tilt = True

        header = lines[i + 8]
        if not header.startswith("ITEM: ATOMS"):
            raise ValueError(f"Malformed dump near line {i}: expected ITEM: ATOMS")
        cols = header.split()[2:]
        idx = {c: k for k, c in enumerate(cols)}

        if {"x", "y", "z"} <= idx.keys():
            cx, cy, cz, scaled = idx["x"], idx["y"], idx["z"], False
        elif {"xu", "yu", "zu"} <= idx.keys():
            cx, cy, cz, scaled = idx["xu"], idx["yu"], idx["zu"], False
        elif {"xs", "ys", "zs"} <= idx.keys():
            cx, cy, cz, scaled = idx["xs"], idx["ys"], idx["zs"], True
        else:
            raise ValueError(f"No coordinate columns found in dump: {cols}")

        has_elem = "element" in idx
        ctype = idx.get("type")
        cid = idx.get("id")

        start = i + 9
        if start + natoms > n:
            print(f"⚠️  truncated final frame at timestep {timestep} — skipped.")
            break

        coords = np.empty((natoms, 3))
        elem = np.empty(natoms, dtype=object)
        ids = np.empty(natoms, dtype=np.int64) if cid is not None else None
        for a in range(natoms):
            parts = lines[start + a].split()
            coords[a, 0] = float(parts[cx])
            coords[a, 1] = float(parts[cy])
            coords[a, 2] = float(parts[cz])
            if has_elem:
                elem[a] = parts[idx["element"]]
            elif ctype is not None:
                t = int(parts[ctype])
                elem[a] = type_map[t] if type_map else str(t)
            else:
                raise ValueError("Dump has neither 'element' nor 'type' column.")
            if ids is not None:
                ids[a] = int(parts[cid])

        if scaled:
            coords = box[:, 0][None, :] + coords * (box[:, 1] - box[:, 0])[None, :]

        # Sort by atom id so atom index is consistent across frames (needed for
        # any per-atom time correlation). LAMMPS 'sort id' usually does this
        # already, but be safe.
        if ids is not None:
            order = np.argsort(ids)
            coords, elem = coords[order], elem[order]

        out.append(Frame(coords=coords, elements=elem.astype(str), box=box,
                         timestep=timestep))
        i = start + natoms

    if not out:
        raise ValueError(f"No frames parsed from {path}")
    if frames == "all":
        return out
    if frames == "last":
        return [out[-1]]
    if isinstance(frames, int):
        return out[:frames]
    raise ValueError(f"Invalid frames={frames!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Data file (single configuration)
# ─────────────────────────────────────────────────────────────────────────────
def read_data(path, type_map=None):
    """Read a LAMMPS *data* file → single-element list ``[Frame]``.

    ``type_map`` maps LAMMPS atom-type id → element symbol. If omitted the type
    id string is the species label.
    """
    path = Path(path)
    types, coords = [], []
    box = [None, None, None]
    section = None
    SECTIONS = ("Masses", "Atoms", "Velocities", "Bonds", "Angles",
                "Dihedrals", "Impropers", "PairIJ Coeffs", "Pair Coeffs")

    with open(path) as f:
        for raw in f:
            line = raw.split("#")[0].strip()
            if not line:
                continue
            if "xlo xhi" in raw:
                p = raw.split(); box[0] = (float(p[0]), float(p[1])); continue
            if "ylo yhi" in raw:
                p = raw.split(); box[1] = (float(p[0]), float(p[1])); continue
            if "zlo zhi" in raw:
                p = raw.split(); box[2] = (float(p[0]), float(p[1])); continue
            head = line.split()[0]
            if head in ("Masses", "Atoms", "Velocities", "Bonds", "Angles",
                        "Dihedrals", "Impropers") or line.startswith("Pair") \
                    or line.startswith("PairIJ"):
                section = "Atoms" if head == "Atoms" else head
                continue
            if section == "Atoms":
                p = line.split()
                if len(p) >= 5:
                    try:
                        t = int(p[1])
                        x, y, z = float(p[2]), float(p[3]), float(p[4])
                    except ValueError:
                        continue
                    if len(p) >= 8:
                        try:
                            ix, iy, iz = int(p[5]), int(p[6]), int(p[7])
                            x += ix * (box[0][1] - box[0][0])
                            y += iy * (box[1][1] - box[1][0])
                            z += iz * (box[2][1] - box[2][0])
                        except (ValueError, IndexError):
                            pass
                    types.append(t)
                    coords.append([x, y, z])

    box = np.array(box, dtype=float)
    if box.shape != (3, 2) or np.any(np.isnan(box)):
        raise ValueError(f"Could not parse box bounds from {path}")
    types = np.array(types, dtype=int)
    coords = np.array(coords, dtype=float)
    elem = np.array([type_map[t] if type_map else str(t) for t in types], dtype=str)
    return [Frame(coords=coords, elements=elem, box=box, timestep=None)]


# ─────────────────────────────────────────────────────────────────────────────
# Auto-detect
# ─────────────────────────────────────────────────────────────────────────────
def load(path, type_map=None, frames="all"):
    """Auto-detect dump vs data and load. Always returns ``list[Frame]``."""
    path = Path(path)
    head = ""
    with open(path) as f:
        for _ in range(3):
            head += f.readline()
    if "ITEM: TIMESTEP" in head:
        return read_dump(path, type_map=type_map, frames=frames)
    return read_data(path, type_map=type_map)
