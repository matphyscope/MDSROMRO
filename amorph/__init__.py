"""
amorph — element-agnostic structural analysis of amorphous materials from
LAMMPS trajectories (data files and dumps).

Two analysis tracks:
  * amorph.sro — short-range order  (g(r), CN, ADF, CSRO, Voronoi, BOO,
                 hybridization, tetrahedral order)
  * amorph.mro — medium-range order (rings, Bhatia–Thornton, clusters,
                 dihedral, tetrahedra connectivity)

Everything is driven from :mod:`amorph.core` (readers, neighbor search, cutoff
matrix, time-averaging). Analyses consume a list of frames (a trajectory) and
return time-averaged results with block-averaged error bars.
"""
from . import core
from . import sro
try:
    from . import mro
except ImportError:
    mro = None
from . import sweep
from . import presets

from .core import (Frame, load, read_dump, read_data, CutoffMatrix,
                   NeighborCache, select_frames, species_of, unique_pairs)

__version__ = "0.1.0"
__all__ = ["core", "sro", "mro", "Frame", "load", "read_dump", "read_data",
           "CutoffMatrix", "NeighborCache", "select_frames", "species_of",
           "unique_pairs"]
