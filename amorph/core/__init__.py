"""amorph.core — element-agnostic foundation shared by all SRO/MRO analyses."""
from .frame import Frame, species_of, unique_pairs, as_trajectory
from .io import load, read_dump, read_data
from .neighbors import (CutoffMatrix, NeighborCache, pair_list, bond_list,
                        neighbor_table)
from .average import select_frames, block_average, time_average
from . import cutoffs
from . import elements

__all__ = [
    "Frame", "species_of", "unique_pairs", "as_trajectory",
    "load", "read_dump", "read_data",
    "CutoffMatrix", "NeighborCache", "pair_list", "bond_list", "neighbor_table",
    "select_frames", "block_average", "time_average",
    "cutoffs", "elements",
]
