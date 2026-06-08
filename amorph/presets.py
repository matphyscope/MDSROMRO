"""
presets.py — Convenience presets for specific systems.

The package itself is element-agnostic; presets just bundle the conventional
inputs (type→element map and the fixed bond cutoffs taken from the LAMMPS input
deck) so notebooks can stay short. Nothing here is required — you can always
build a :class:`~amorph.core.neighbors.CutoffMatrix` by hand.
"""
from __future__ import annotations
from .core.neighbors import CutoffMatrix

# LAMMPS atom-type id → element, matching the SiCN.tersoff "Si C N" order used
# in every in_SiCN_*.lmp deck.
SICN_TYPE_MAP = {1: "Si", 2: "C", 3: "N"}

# Fixed bond cutoffs (Å) — copied verbatim from the v8/Tscan CN/ADF computes.
# C–N and N–N are 0 (non-bonding): the unified Tersoff sets chi_CN=0 and N–N
# B=0, so those pairs never bond. Using the same numbers keeps Python analysis
# consistent with the simulation's own CN/ADF.
SICN_CUTOFFS = {
    ("Si", "Si"): 2.70,
    ("Si", "C"): 2.25,
    ("Si", "N"): 2.15,
    ("C", "C"): 1.85,
    ("C", "N"): 0.0,
    ("N", "N"): 0.0,
}

# Canonical ADF triplets (B–A–C, centre = A) used by the LAMMPS decks.
SICN_ADF_TRIPLETS = [
    ("N", "Si", "N"),
    ("C", "Si", "C"),
    ("C", "Si", "N"),
    ("Si", "C", "Si"),
    ("Si", "N", "Si"),
    ("C", "C", "C"),
]


def sicn_cutoffs() -> CutoffMatrix:
    """Return the SiCN fixed-cutoff matrix (C–N, N–N non-bonding)."""
    return CutoffMatrix(SICN_CUTOFFS, default=0.0)
