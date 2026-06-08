"""
frame.py — Element-agnostic snapshot/trajectory data model.

A :class:`Frame` is one orthogonal-box snapshot: per-atom species (element
symbols) and coordinates plus the periodic box. A trajectory is simply a
``list[Frame]``. Everything downstream (RDF, CN, rings, ...) consumes Frames,
so the package never hard-codes any particular element set.

Design choices
--------------
* Coordinates are stored *as read* (real-space Å). Wrapping into the cell and
  minimum-image displacements are provided as helpers — analyses wrap on demand
  (e.g. for KDTree neighbor search) but keep the originals for things that need
  continuity (cluster spanning, MSD, ...).
* Species are stored as string element symbols. If the dump lacks an
  ``element`` column the reader fills them from a ``type_map`` (or, as a last
  resort, the stringified LAMMPS type id) — see :mod:`amorph.core.io`.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .elements import mass_of


@dataclass
class Frame:
    """Single snapshot.

    Attributes
    ----------
    coords : (N, 3) float ndarray   real-space Å (as read)
    elements : (N,) object/str ndarray   element symbol per atom
    box : (3, 2) float ndarray   [[xlo, xhi], [ylo, yhi], [zlo, zhi]] (Å)
    timestep : int or None
    """
    coords: np.ndarray
    elements: np.ndarray
    box: np.ndarray
    timestep: int | None = None

    # ---- basic geometry ---------------------------------------------------
    @property
    def n_atoms(self) -> int:
        return len(self.coords)

    @property
    def L(self) -> np.ndarray:
        """Box edge lengths (3,)."""
        return self.box[:, 1] - self.box[:, 0]

    @property
    def volume(self) -> float:
        return float(np.prod(self.L))

    @property
    def species(self) -> list[str]:
        """Sorted unique element symbols present in this frame."""
        return sorted(set(self.elements.tolist()))

    def mask(self, element) -> np.ndarray:
        """Boolean mask selecting atoms of a given element."""
        return self.elements == element

    def relabeled(self, mapping) -> "Frame":
        """New Frame with elements remapped (shares coords/box).

        ``mapping`` is {old_element: new_label}; unmapped elements are kept.
        Useful for pseudo-binary grouping (e.g. {'Si':'A','N':'A','C':'B'}).
        """
        new_el = np.array([mapping.get(e, e) for e in self.elements], dtype=str)
        return Frame(coords=self.coords, elements=new_el, box=self.box,
                     timestep=self.timestep)

    def count(self, element) -> int:
        return int(np.count_nonzero(self.mask(element)))

    # ---- PBC helpers ------------------------------------------------------
    def wrapped(self) -> np.ndarray:
        """Coordinates wrapped into [0, L) (origin at box lo). Shape (N, 3)."""
        return (self.coords - self.box[:, 0][None, :]) % self.L[None, :]

    def min_image(self, dvec) -> np.ndarray:
        """Minimum-image displacement(s) for vector(s) ``dvec`` (..., 3)."""
        L = self.L
        return dvec - L * np.round(dvec / L)

    # ---- thermodynamic-ish scalars ---------------------------------------
    def composition(self) -> dict[str, float]:
        """Mole fractions {element: x}."""
        N = self.n_atoms
        return {e: self.count(e) / N for e in self.species}

    def number_density(self) -> float:
        """Atoms per Å³."""
        return self.n_atoms / self.volume

    def mass_density(self, masses=None) -> float:
        """Mass density in g/cc."""
        M = sum(mass_of(e, masses) for e in self.elements)
        return (M / 6.02214076e23) / (self.volume * 1e-24)


def species_of(traj) -> list[str]:
    """Union of element symbols across a trajectory (sorted)."""
    s = set()
    for fr in traj:
        s.update(fr.elements.tolist())
    return sorted(s)


def unique_pairs(species) -> list[tuple[str, str]]:
    """All unordered species pairs (A<=B) for a sorted species list."""
    s = sorted(species)
    return [(a, b) for i, a in enumerate(s) for b in s[i:]]


def as_trajectory(frames) -> list[Frame]:
    """Normalize a single Frame or list of Frames to a list."""
    if isinstance(frames, Frame):
        return [frames]
    return list(frames)
