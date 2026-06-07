#!/usr/bin/env python3
"""
verify_real.py — Run a compact, paste-friendly analysis on a REAL structure.

Use this when the trajectory/data file is too large to upload: run it locally
(where the data already lives) and paste the TEXT output back into the chat.
No plots, no files written — just a numeric report we can sanity-check together.

Works on a single LAMMPS .data file (one frame) or a dump trajectory (time-
averaged over the production window). Element-agnostic; for atom-style dumps
without an `element` column pass --type-map.

USAGE
  python3 scripts/verify_real.py SiCN_300K_final.data
  python3 scripts/verify_real.py dump.300K.lammpstrj --prod 50: --stride 2
  python3 scripts/verify_real.py dump_T0300.lammpstrj --type-map 1:Si,2:C,3:N
  # generic (non-SiCN): supply your own cutoffs
  python3 scripts/verify_real.py foo.data --cutoffs Si-C:2.25,Si-N:2.15,C-C:1.85

Run from the repo root (so `import amorph` works), or set PYTHONPATH to it.
"""
from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")  # silence harmless nan-slice / DOF warnings

# Allow running as `python3 scripts/verify_real.py ...` from the repo root:
# put the repo root (parent of this script's dir) on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import amorph
    from amorph import core, presets
    from amorph.sro import rdf, coordination
    from amorph.core import cutoffs as cut
    from amorph.mro import rings as mro_rings, clusters as mro_clusters
    from amorph.mro import tetra_connectivity, bhatia_thornton
except ImportError as e:
    sys.exit(f"Cannot import amorph ({e}).\n"
             "Run from the repository root, or: PYTHONPATH=/path/to/MDSROMRO python3 ...")


def parse_type_map(s):
    if not s:
        return None
    out = {}
    for tok in s.split(","):
        k, v = tok.split(":")
        out[int(k)] = v.strip()
    return out


def parse_cutoffs(s):
    if not s:
        return None
    d = {}
    for tok in s.split(","):
        pair, rc = tok.split(":")
        a, b = pair.split("-")
        d[(a.strip(), b.strip())] = float(rc)
    return d


def parse_prod(s):
    if not s:
        return None
    lo, hi = s.split(":")
    return (int(lo) if lo else 0, int(hi) if hi else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="LAMMPS .data or dump trajectory")
    ap.add_argument("--type-map", default=None,
                    help="e.g. 1:Si,2:C,3:N  (only for dumps without element col)")
    ap.add_argument("--cutoffs", default=None,
                    help="override bond cutoffs e.g. Si-C:2.25,Si-N:2.15,C-C:1.85"
                         "  (default: SiCN preset)")
    ap.add_argument("--prod", default=None, help="production frame window lo:hi")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--rmax", type=float, default=10.0)
    ap.add_argument("--nbins", type=int, default=500)
    ap.add_argument("--nblocks", type=int, default=5)
    ap.add_argument("--max-ring", type=int, default=10)
    ap.add_argument("--ring-frames", type=int, default=3,
                    help="max frames used for ring statistics (expensive)")
    ap.add_argument("--tetra-center", default="Si")
    ap.add_argument("--free-element", default="C")
    ap.add_argument("--no-mro", action="store_true", help="skip rings/clusters/BT")
    args = ap.parse_args()

    tmap = parse_type_map(args.type_map) or presets.SICN_TYPE_MAP
    cdict = parse_cutoffs(args.cutoffs) or presets.SICN_CUTOFFS
    CM = core.CutoffMatrix(cdict, default=0.0)

    print("=" * 72)
    print(f"amorph {amorph.__version__}  —  verify_real on {args.file}")
    print("=" * 72)

    frames = core.load(args.file, type_map=tmap, frames="all")
    traj = core.select_frames(frames, frame_range=parse_prod(args.prod),
                              stride=args.stride)
    sp = core.species_of(traj)
    pairs = core.unique_pairs(sp)

    # ---- system info ----
    comp = np.mean([[fr.count(e) / fr.n_atoms for e in sp] for fr in traj], axis=0)
    rho = np.mean([fr.mass_density() for fr in traj])
    nrho = np.mean([fr.number_density() for fr in traj])
    L = np.mean([fr.L for fr in traj], axis=0)
    print(f"[system] frames(total/used) = {len(frames)}/{len(traj)}   "
          f"atoms = {traj[0].n_atoms}   species = {sp}")
    print(f"[system] box L (mean)      = {L[0]:.3f} x {L[1]:.3f} x {L[2]:.3f} Å")
    print(f"[system] composition mol%  = " +
          ", ".join(f"{e} {c*100:.2f}" for e, c in zip(sp, comp)))
    print(f"[system] mass density      = {rho:.4f} g/cc   "
          f"number density = {nrho:.5f} /Å³")
    print(f"[system] cutoffs           = {CM}")

    # ---- RDF peaks + cutoff comparison ----
    print("\n[g(r)] first-peak metrics (time-averaged):")
    R = rdf.partial_rdf(traj, pairs=pairs, r_max=args.rmax, nbins=args.nbins,
                        n_blocks=args.nblocks)
    print(f"  {'pair':<8}{'peak_r':>9}{'height':>9}{'FWHM':>9}{'R^2':>8}")
    for (A, B) in pairs:
        if CM.get(A, B) <= 0:
            continue
        m = rdf.measure_peaks(R["r"], R[(A, B)]["g"], search=(0.5, None))
        print(f"  {A+'-'+B:<8}{m['peak_r']:>9.3f}{m['height']:>9.3f}"
              f"{m['fwhm']:>9.3f}{m['r2']:>8.3f}")
    print("\n[cutoff] fixed vs g(r)-first-minimum:")
    derived = cut.derive_cutoffs(R, pairs, fallback=CM)
    print("  " + cut.format_table(
        cut.compare(CM, derived, [p for p in pairs if CM.get(*p) > 0])
    ).replace("\n", "\n  "))

    # ---- coordination + ADF ----
    print("\n[CN] coordination numbers (time-averaged ±1σ):")
    CN = coordination.coordination_numbers(traj, CM, n_blocks=args.nblocks)
    for A in sp:
        parts = ", ".join(f"{B}:{CN['cn'][A][B][0]:.3f}"
                          for B in sp if CM.get(A, B) > 0)
        tot = CN["cn_total"][A]
        print(f"  {A}: total {tot[0]:.3f}±{tot[1]:.3f}   [{parts}]")

    triplets = []
    for A in sp:
        bonded = [B for B in sp if CM.get(A, B) > 0]
        for i, B in enumerate(bonded):
            for C in bonded[i:]:
                triplets.append((B, A, C))
    if triplets:
        ADF = coordination.adf(traj, triplets, CM, nbins=180, n_blocks=args.nblocks)
        print("[ADF] bond-angle peak positions:")
        for t in triplets:
            p = ADF[t]["p"]
            if np.all(np.isnan(p)):
                continue
            print(f"  {t[0]}-{t[1]}-{t[2]:<3}: peak {ADF['theta'][np.nanargmax(p)]:.1f}°")

    if args.no_mro:
        print("\n(MRO skipped)")
        return

    # ---- rings ----
    print("\n[rings] King's ring statistics (rings per frame by size):")
    RG = mro_rings.ring_statistics(traj, CM, max_size=args.max_ring,
                                   max_frames=args.ring_frames,
                                   n_blocks=min(args.nblocks, args.ring_frames))
    rstr = ", ".join(f"{int(s)}:{m:.0f}"
                     for s, m in zip(RG["sizes"], RG["count"]["mean"]) if m > 0.5)
    print(f"  {rstr}   (frames used: {RG['n_frames']})")

    # ---- tetra connectivity ----
    if args.tetra_center in sp:
        TC = tetra_connectivity.tetra_connectivity(traj, CM,
                                                   center=args.tetra_center,
                                                   d_max=4.0, n_blocks=args.nblocks)
        fr = TC["fractions"]
        print(f"[tetra] {args.tetra_center} connectivity: " +
              ", ".join(f"{c} {fr[c]*100:.1f}%" for c in ['corner', 'edge', 'face']))
        br = {k: round(v[0], 1) for k, v in TC["corner_bridge"].items() if v[0] > 0}
        print(f"        corner bridge species (per frame): {br}")

    # ---- free clusters ----
    if args.free_element in sp:
        FC = mro_clusters.free_clusters(traj, CM, element=args.free_element,
                                        n_blocks=args.nblocks)
        print(f"[clusters] free-{args.free_element}: "
              f"n={FC['n_clusters'][0]:.0f}, isolated={FC['n_isolated'][0]:.0f}, "
              f"graphenic(≥{FC['min_graphenic']})={FC['n_graphenic'][0]:.0f}, "
              f"mean L_a={FC['mean_La'][0]:.2f} Å, max size={FC['max_size'][0]:.0f}")

    # ---- Bhatia-Thornton (SiCN default groups) ----
    if set(["Si", "N", "C"]).issubset(set(sp)):
        try:
            BT = bhatia_thornton.bhatia_thornton(traj, ["Si", "N"], ["C"],
                                                 r_max=args.rmax, nbins=args.nbins,
                                                 n_blocks=args.nblocks)
            ilow = int(np.argmin(np.abs(BT["q"] - 0.5)))
            tag = "CLUSTERING" if BT["S_CC"][ilow] > BT["S_CC_ideal"] else "ordering"
            print(f"[BT] (Si+N)-C  S_CC(Q≈0.5)={BT['S_CC'][ilow]:.4f} "
                  f"vs ideal {BT['S_CC_ideal']:.4f}  → {tag}")
        except ValueError:
            pass

    print("\n" + "=" * 72)
    print("DONE — copy everything above and paste it back into the chat.")
    print("=" * 72)


if __name__ == "__main__":
    main()
