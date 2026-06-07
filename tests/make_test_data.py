"""
make_test_data.py — Generate small synthetic structures with KNOWN answers so
the analysis code can be validated without real simulation output.

Structures
----------
* diamond cubic (single species "Si"): every atom has exactly CN=4, nearest-
  neighbour distance d = a*sqrt(3)/4. Perfect for validating neighbor search,
  RDF first peak, and coordination.
* zincblende (two species "Si","C"): CN=4 with only unlike neighbours — checks
  partial RDF / partial CN bookkeeping.

Outputs written next to this file:
  diamond.data, diamond_custom.lammpstrj (element col),
  diamond_atom.lammpstrj (atom-style scaled, no element),
  zincblende.data
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent


def diamond(a=5.43, reps=3, species=("Si",)):
    """Diamond-cubic lattice. If two species given → zincblende (sublattice A/B)."""
    basis_A = np.array([[0, 0, 0], [0.0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    basis_B = basis_A + 0.25
    coords, elems = [], []
    for i in range(reps):
        for j in range(reps):
            for k in range(reps):
                shift = np.array([i, j, k])
                for b in basis_A:
                    coords.append((b + shift) * a); elems.append(species[0])
                for b in basis_B:
                    coords.append((b + shift) * a)
                    elems.append(species[-1])  # same as A if single species
    coords = np.array(coords)
    box = np.array([[0, a * reps]] * 3, dtype=float)
    return coords, np.array(elems), box


def write_data(path, coords, elems, box):
    species = sorted(set(elems))
    tmap = {e: i + 1 for i, e in enumerate(species)}
    masses = {"Si": 28.085, "C": 12.011, "N": 14.007}
    with open(path, "w") as f:
        f.write("LAMMPS test data\n\n")
        f.write(f"{len(coords)} atoms\n{len(species)} atom types\n\n")
        for d, name in zip(range(3), ("x", "y", "z")):
            f.write(f"{box[d,0]:.6f} {box[d,1]:.6f} {name}lo {name}hi\n")
        f.write("\nMasses\n\n")
        for e in species:
            f.write(f"{tmap[e]} {masses.get(e, 1.0)}  # {e}\n")
        f.write("\nAtoms # atomic\n\n")
        for n, (c, e) in enumerate(zip(coords, elems), start=1):
            f.write(f"{n} {tmap[e]} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")


def write_dump_custom(path, frames):
    """frames: list of (coords, elems, box). Writes custom dump with element col."""
    with open(path, "w") as f:
        for t, (coords, elems, box) in enumerate(frames):
            tmap = {e: i + 1 for i, e in enumerate(sorted(set(elems)))}
            f.write("ITEM: TIMESTEP\n%d\n" % (t * 1000))
            f.write("ITEM: NUMBER OF ATOMS\n%d\n" % len(coords))
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            for d in range(3):
                f.write(f"{box[d,0]:.6f} {box[d,1]:.6f}\n")
            f.write("ITEM: ATOMS id type element x y z\n")
            for n, (c, e) in enumerate(zip(coords, elems), start=1):
                f.write(f"{n} {tmap[e]} {e} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")


def write_dump_atom(path, frames):
    """atom-style dump: id type xs ys zs (scaled), no element column."""
    with open(path, "w") as f:
        for t, (coords, elems, box) in enumerate(frames):
            tmap = {e: i + 1 for i, e in enumerate(sorted(set(elems)))}
            L = box[:, 1] - box[:, 0]
            sc = (coords - box[:, 0][None, :]) / L[None, :]
            f.write("ITEM: TIMESTEP\n%d\n" % (t * 1000))
            f.write("ITEM: NUMBER OF ATOMS\n%d\n" % len(coords))
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            for d in range(3):
                f.write(f"{box[d,0]:.6f} {box[d,1]:.6f}\n")
            f.write("ITEM: ATOMS id type xs ys zs\n")
            for n, (s, e) in enumerate(zip(sc, elems), start=1):
                f.write(f"{n} {tmap[e]} {s[0]:.6f} {s[1]:.6f} {s[2]:.6f}\n")


def main():
    # single-species diamond
    c, e, box = diamond(a=5.43, reps=3, species=("Si",))
    write_data(HERE / "diamond.data", c, e, box)

    # 3-frame trajectory with tiny thermal jitter → tests time-averaging
    rng = np.random.default_rng(0)
    frames = [(c + rng.normal(0, 0.03, c.shape), e, box) for _ in range(6)]
    write_dump_custom(HERE / "diamond_custom.lammpstrj", frames)
    write_dump_atom(HERE / "diamond_atom.lammpstrj", frames)

    # zincblende SiC: known nearest-neighbour distance, only Si-C bonds
    c2, e2, box2 = diamond(a=4.36, reps=3, species=("Si", "C"))
    write_data(HERE / "zincblende.data", c2, e2, box2)

    d_nn = 5.43 * np.sqrt(3) / 4
    print(f"diamond: {len(c)} atoms, nn distance = {d_nn:.4f} Å, expected CN=4")
    print(f"zincblende SiC: {len(c2)} atoms, nn = {4.36*np.sqrt(3)/4:.4f} Å (Si-C only)")
    print("wrote test files to", HERE)


if __name__ == "__main__":
    main()
