"""
elements.py — Element symbols and atomic masses (element-agnostic).

The package is element-agnostic: it works with whatever species appear in the
trajectory. For mass-density we need atomic masses; a built-in periodic-table
table covers the common elements, and any missing/custom mass can be supplied
explicitly via ``masses=`` overrides on the functions that need them.
"""
from __future__ import annotations

# Standard atomic weights (g/mol), IUPAC conventional values. Covers the common
# elements; extend or override via the ``masses`` argument where needed.
ATOMIC_MASS = {
    "H": 1.008,  "He": 4.0026, "Li": 6.94,  "Be": 9.0122, "B": 10.81,
    "C": 12.011, "N": 14.007,  "O": 15.999, "F": 18.998,  "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06,  "Cl": 35.45,  "Ar": 39.948, "K": 39.098,  "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906,  "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95,  "Ag": 107.87, "Cd": 112.41, "In": 114.82,
    "Sn": 118.71, "Sb": 121.76, "Te": 127.60, "I": 126.90,  "Xe": 131.29,
    "Cs": 132.91, "Ba": 137.33, "La": 138.91, "Hf": 178.49, "Ta": 180.95,
    "W": 183.84,  "Re": 186.21, "Os": 190.23, "Ir": 192.22, "Pt": 195.08,
    "Au": 196.97, "Hg": 200.59, "Tl": 204.38, "Pb": 207.2,  "Bi": 208.98,
}

# A consistent, color-blind-friendly palette assigned deterministically to
# species in alphabetical order (so plots are stable across runs/notebooks).
_PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def element_colors(elements):
    """Return {element: hex_color} assigned deterministically (alphabetical)."""
    uniq = sorted(set(elements))
    return {e: _PALETTE[i % len(_PALETTE)] for i, e in enumerate(uniq)}


def mass_of(element, masses=None):
    """Atomic mass of ``element`` (g/mol). ``masses`` overrides/extends the table."""
    if masses and element in masses:
        return masses[element]
    if element in ATOMIC_MASS:
        return ATOMIC_MASS[element]
    raise KeyError(
        f"No atomic mass for element {element!r}. Pass masses={{'{element}': <g/mol>}} "
        f"to the function, or add it to amorph.core.elements.ATOMIC_MASS."
    )
