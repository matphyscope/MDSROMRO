#!/usr/bin/env python3
"""Size-sanity diagnostics for MRO analyses on a small cell.

Flags box-size limitations that produce silent artifacts (no exception):
  * g(r)/structure-factor: r_max must be <= L/2 (minimum-image)
  * Bhatia-Thornton: smallest accessible Q = 2pi/L  -> 'Q->0' is only an estimate
  * rings: a ring that needs >~ L/bond atoms to wrap is impossible to detect;
    conversely max_size must stay below the box-spanning length
  * clusters: a component that spans the box (percolates) has an ILL-DEFINED
    R_g / L_a (PBC self-connection) — must be excluded from size metrics
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from amorph import core, presets
from amorph.mro._common import build_graph, unwrap_component, gyration
import networkx as nx

path = sys.argv[1] if len(sys.argv) > 1 else "data/dump_T0300.lammpstrj"
CM = core.CutoffMatrix(presets.SICN_CUTOFFS, default=0.0)
traj = core.load(path, type_map=presets.SICN_TYPE_MAP, frames="all")
fr = traj[-1]                       # last (well-equilibrated) frame
L = fr.L
sp = core.species_of([fr])
print("=" * 70)
print(f"MRO size diagnostics — {Path(path).name}  (last frame)")
print("=" * 70)
print(f"atoms={fr.n_atoms}  box L = {L[0]:.2f} x {L[1]:.2f} x {L[2]:.2f} Å  "
      f"L/2 = {L.min()/2:.2f} Å")

# --- g(r) / S(Q) reach -------------------------------------------------------
print("\n[g(r)/S(Q)] minimum-image reach")
print(f"  max reliable r_max = L/2 = {L.min()/2:.2f} Å   (we use r_max=8.0 → "
      f"{'OK' if 8.0 <= L.min()/2 else 'TOO LARGE'})")
qmin = 2 * np.pi / L.max()
print(f"  smallest accessible Q = 2π/L = {qmin:.3f} Å⁻¹  → S(Q→0) below this is "
      f"EXTRAPOLATION, not measured (BT S_CC sampled at Q≈0.5 ≥ qmin: "
      f"{'OK' if 0.5 >= qmin else 'BELOW qmin'})")

# --- rings -------------------------------------------------------------------
bond_min = min(v for v in presets.SICN_CUTOFFS.values())
wrap_atoms = int(np.ceil(L.min() / 1.9))   # ~atoms to span box at ~bond length
print("\n[rings] box-spanning safety")
print(f"  a PBC-wrapping loop needs ≳ {wrap_atoms} atoms; max_size=9 < {wrap_atoms} "
      f"→ {'rings up to 9 are genuine (safe)' if 9 < wrap_atoms else 'RISK of wrap rings'}")

# --- free-C clusters: percolation & L_a validity -----------------------------
print("\n[free-C clusters] percolation check (R_g/L_a invalid if spanning)")
nodes = np.where(fr.mask("C"))[0]
G = build_graph(fr, CM, nodes=nodes, pair_filter=lambda a, b: a == "C" and b == "C")
comps = sorted(nx.connected_components(G), key=len, reverse=True)
print(f"  {len(nodes)} C atoms, {len(comps)} free-C clusters, "
      f"sizes top5 = {[len(c) for c in comps[:5]]}")
n_perc = 0
print(f"  {'size':>5} {'extent/L (x,y,z)':>22} {'maxfrac':>8} {'L_a(Å)':>8} {'percolates?':>11}")
for c in comps[:6]:
    atoms = list(c)
    if len(atoms) < 2:
        continue
    coords = unwrap_component(atoms, G, fr)
    extent = coords.max(0) - coords.min(0)
    frac = extent / L
    perc = bool(frac.max() >= 0.85)
    n_perc += perc
    Rg, L_a, aspect, _ = gyration(coords)
    print(f"  {len(atoms):>5} {str(np.round(frac,2)):>22} {frac.max():>8.2f} "
          f"{L_a:>8.2f} {str(perc):>11}")
print(f"  → {n_perc} of the shown clusters PERCOLATE: their L_a/R_g are PBC "
      f"artifacts and must be excluded from size statistics.")

print("\n" + "=" * 70)
