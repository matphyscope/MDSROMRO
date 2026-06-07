#!/usr/bin/env python3
"""
run_batch4.py — Batch 4 종합 분석 일괄 실행 (11개 모듈)

Batch 1 + Batch 2 (10개) + Batch 4 신규 (1개) = 11개:

  Batch 1 — SRO + 거리 기반 분석:
    1. gr_extended  (Hybrid g(r) 0-10 LAMMPS + 10-20 .data)
    2. rings        (King's ring statistics — MRO)
    3. bt           (Bhatia-Thornton S_NN/S_CC/S_NC — MRO)
    4. cluster      (Free-C cluster, percolation, D_f — MRO)
    5. cn_adf       (LAMMPS time-averaged CN + ADF — SRO + IRO 가교각 일부)

  Batch 2 — 추가 SRO/IRO 분석:
    6. csro         (Warren-Cowley α_AB — SRO chemical)
    7. voronoi      (Voronoi tessellation — SRO geometric)
    8. boo          (Steinhardt Q4, Q6, W4, W6 — SRO symmetry)
    9. dihedral     (Torsion angle — IRO)
   10. sp23         (sp²/sp³ classification — SRO hybridization)

  Batch 4 — 추가 IRO 분석:
   11. tetra_connectivity ⭐ (Si tetrahedra corner/edge/face sharing — 진정한 IRO)

LAMMPS 시간평균 파일 자동 감지: rdf_300K.txt, CN_300K.txt, angles_300K.txt

사용법:
  python3 run_batch4.py SiCN_300K_final.data
  python3 run_batch4.py SiCN_300K_final.data --out my_results/

의존성:
  numpy, scipy, matplotlib, networkx, freud (=freud-analysis 3.x)
"""
from __future__ import annotations
import argparse
import sys
import time
import traceback
from pathlib import Path


def banner(title, char="=", width=78):
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def run_module(name, fn, *args, **kwargs):
    banner(f">>> {name}", char="-")
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"\n  ✓ {name} 완료 ({elapsed:.1f}초)")
        return True, result
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ✗ {name} 실패 ({elapsed:.1f}초)")
        print(f"    오류: {e}")
        traceback.print_exc()
        return False, None


def auto_detect_files(data_path):
    parent = Path(data_path).parent
    candidates = {
        "rdf": ["rdf_300K.txt", "rdf_final.txt"],
        "cn":  ["CN_300K.txt", "CN_final.txt"],
        "adf": ["angles_300K.txt", "angles_final.txt"],
    }
    found = {}
    for kind, names in candidates.items():
        for nm in names:
            p = parent / nm
            if p.exists():
                found[kind] = str(p)
                break
    return found


def write_summary(input_file, out_dir, results, files_used):
    out_dir = Path(out_dir)
    lines = []
    lines.append("=" * 78)
    lines.append(f"  Batch 4 종합 분석 결과 요약 (11개 모듈)")
    lines.append("=" * 78)
    lines.append(f"  입력 파일 : {input_file}")
    lines.append(f"  출력 위치 : {out_dir.resolve()}")
    lines.append(f"  실행 일시 : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("─ 사용된 파일 ─")
    lines.append(f"  .data    : {Path(input_file).name}")
    if "rdf" in files_used:
        lines.append(f"  rdf .txt : {Path(files_used['rdf']).name}  ⭐ LAMMPS 시간평균")
    if "cn" in files_used:
        lines.append(f"  CN .txt  : {Path(files_used['cn']).name}  ⭐ LAMMPS 시간평균")
    if "adf" in files_used:
        lines.append(f"  ADF .txt : {Path(files_used['adf']).name}  ⭐ LAMMPS 시간평균")
    lines.append("")

    # System info from g(r) result
    if "gr" in results and results["gr"][0]:
        ret = results["gr"][1]
        if ret is not None and len(ret) >= 4:
            r, g, n, struct = ret
            from sicn_common import density_g_per_cc, composition
            x = composition(struct)
            lines.append("─ 시스템 정보 ─")
            lines.append(f"  Atoms     : {struct['n_atoms']}")
            lines.append(f"  Box L     : {struct['box'][0,1]-struct['box'][0,0]:.3f} Å")
            lines.append(f"  Density   : {density_g_per_cc(struct):.4f} g/cc")
            lines.append(f"  Comp.     : Si {x[0]*100:.2f}%  "
                          f"C {x[1]*100:.2f}%  N {x[2]*100:.2f}%")
            lines.append("")

    # Append each module's summary text file (just the key lines)
    section_files = [
        ("Extended g(r) shells",     "01_gr_extended/shells.txt"),
        ("Ring statistics",           "02_rings/rings.txt"),
        ("Bhatia-Thornton",           "03_bt/bt_summary.txt"),
        ("Cluster / free-C",          "04_cluster/clusters.txt"),
        ("LAMMPS time-avg CN",        "05_cn_adf/cn_lammps.txt"),
        ("LAMMPS time-avg ADF peaks", "05_cn_adf/adf_peaks.txt"),
        ("Warren-Cowley CSRO",        "06_csro/csro_summary.txt"),
        ("Voronoi tessellation",      "07_voronoi/voronoi_summary.txt"),
        ("Steinhardt BOO",            "08_boo/boo_summary.txt"),
        ("Dihedral distribution",     "09_dihedral/dihedral_summary.txt"),
        ("sp²/sp³ classification",    "10_sp23/sp23_summary.txt"),
        ("Si tetra connectivity (IRO)", "11_tetra_conn/tetra_conn_summary.txt"),
    ]

    for section_name, rel_path in section_files:
        f = out_dir / rel_path
        if f.exists():
            lines.append(f"─ {section_name} ─")
            text = f.read_text().splitlines()
            for ln in text[2:]:  # skip first 2 (title + ===)
                if "---" in ln: continue
                if ln.strip() and not ln.startswith("="):
                    lines.append(f"  {ln}")
            lines.append("")

    lines.append("=" * 78)
    res_str = " ".join(
        f"{k}={'OK' if v[0] else ('SKIP' if v[0] is None else 'FAIL')}"
        for k, v in results.items())
    lines.append(f"  실행 결과: {res_str}")
    lines.append("=" * 78)

    summary_path = out_dir / "SUMMARY.txt"
    summary_path.write_text("\n".join(lines))
    return summary_path


def main():
    ap = argparse.ArgumentParser(
        description="Batch 4 종합 분석 일괄 실행 (11개 모듈)")
    ap.add_argument("input", help="LAMMPS data file")
    ap.add_argument("--rdf", default=None, help="rdf_300K.txt (auto-detect)")
    ap.add_argument("--cn",  default=None, help="CN_300K.txt (auto-detect)")
    ap.add_argument("--adf", default=None, help="angles_300K.txt (auto-detect)")
    ap.add_argument("--out", default="analysis_batch2", help="출력 디렉토리")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["gr", "rings", "bt", "cluster", "cn_adf",
                              "csro", "voronoi", "boo", "dihedral", "sp23",
                              "tetra_conn"])
    ap.add_argument("--r_max", type=float, default=None)
    ap.add_argument("--n_bins", type=int, default=800)
    ap.add_argument("--max_ring", type=int, default=12)
    ap.add_argument("--boo_k", type=int, default=4,
                    help="number of nearest neighbors for BOO (default 4)")
    ap.add_argument("--no-auto", action="store_true",
                    help="auto-detect 끄기")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"오류: 입력 파일 없음: {in_path}")
        sys.exit(1)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect
    files_used = {}
    if not args.no_auto:
        detected = auto_detect_files(in_path)
        for kind, path in detected.items():
            if getattr(args, kind) is None:
                setattr(args, kind, path)
    if args.rdf: files_used["rdf"] = args.rdf
    if args.cn:  files_used["cn"]  = args.cn
    if args.adf: files_used["adf"] = args.adf

    banner("Batch 4 종합 분석 (11개 모듈)")
    print(f"  Structure : {in_path}")
    print(f"  Output    : {out_dir.resolve()}")
    print(f"  LAMMPS 시간평균 파일 자동 감지:")
    print(f"    rdf : {files_used.get('rdf', '(없음)')}")
    print(f"    CN  : {files_used.get('cn',  '(없음)')}")
    print(f"    ADF : {files_used.get('adf', '(없음)')}")
    if args.skip:
        print(f"  Skip      : {', '.join(args.skip)}")
    print()

    # Import all modules
    try:
        from sicn_gr_extended import run as run_gr
        from sicn_rings       import run as run_rings
        from sicn_bt          import run as run_bt
        from sicn_cluster     import run as run_cluster
        from sicn_cn_adf      import run as run_cn_adf
        from sicn_csro        import run as run_csro
        from sicn_voronoi     import run as run_voronoi
        from sicn_boo         import run as run_boo
        from sicn_dihedral    import run as run_dihedral
        from sicn_sp23        import run as run_sp23
        from sicn_tetra_connectivity import run as run_tetra
    except ImportError as e:
        print(f"오류: import 실패 — sicn_*.py 파일이 같은 폴더에 있나요?\n  {e}")
        sys.exit(1)

    results = {}
    t_start = time.time()

    # ─── Batch 1 modules (1-5) ───
    if "gr" not in args.skip:
        kwargs = {"out_dir": str(out_dir / "01_gr_extended"),
                  "n_bins": args.n_bins}
        if args.r_max is not None: kwargs["r_max"] = args.r_max
        if args.rdf: kwargs["rdf_file"] = args.rdf
        results["gr"] = run_module("1/11: Extended g(r)", run_gr,
                                    str(in_path), **kwargs)
    else:
        results["gr"] = (None, None)

    if "rings" not in args.skip:
        results["rings"] = run_module("2/11: Ring statistics", run_rings,
                                       str(in_path),
                                       out_dir=str(out_dir / "02_rings"),
                                       max_size=args.max_ring)
    else:
        results["rings"] = (None, None)

    if "bt" not in args.skip:
        if args.rdf:
            results["bt"] = run_module("3/11: Bhatia-Thornton (LAMMPS rdf 사용)",
                                        run_bt, args.rdf,
                                        out_dir=str(out_dir / "03_bt"),
                                        data_file=str(in_path))
        else:
            results["bt"] = run_module("3/11: Bhatia-Thornton (.data)",
                                        run_bt, str(in_path),
                                        out_dir=str(out_dir / "03_bt"))
    else:
        results["bt"] = (None, None)

    if "cluster" not in args.skip:
        results["cluster"] = run_module("4/11: Cluster connectivity", run_cluster,
                                         str(in_path),
                                         out_dir=str(out_dir / "04_cluster"))
    else:
        results["cluster"] = (None, None)

    if "cn_adf" not in args.skip and (args.cn or args.adf):
        results["cn_adf"] = run_module("5/11: CN + ADF (LAMMPS 시간평균)",
                                        run_cn_adf, str(in_path),
                                        cn_file=args.cn, adf_file=args.adf,
                                        out_dir=str(out_dir / "05_cn_adf"))
    else:
        results["cn_adf"] = (None, None)
        if "cn_adf" not in args.skip:
            print("\n>>> 5/10: CN + ADF — 파일 없어 SKIP")

    # ─── Batch 2 신규 modules (6-10) ───
    if "csro" not in args.skip:
        results["csro"] = run_module("6/11: Warren-Cowley CSRO ⭐",
                                      run_csro, str(in_path),
                                      out_dir=str(out_dir / "06_csro"))
    else:
        results["csro"] = (None, None)

    if "voronoi" not in args.skip:
        results["voronoi"] = run_module("7/11: Voronoi tessellation ⭐",
                                         run_voronoi, str(in_path),
                                         out_dir=str(out_dir / "07_voronoi"))
    else:
        results["voronoi"] = (None, None)

    if "boo" not in args.skip:
        results["boo"] = run_module("8/11: Steinhardt BOO Q4/Q6/W4/W6 ⭐",
                                     run_boo, str(in_path),
                                     out_dir=str(out_dir / "08_boo"),
                                     num_neighbors=args.boo_k)
    else:
        results["boo"] = (None, None)

    if "dihedral" not in args.skip:
        results["dihedral"] = run_module("9/11: Dihedral / torsion ⭐",
                                          run_dihedral, str(in_path),
                                          out_dir=str(out_dir / "09_dihedral"))
    else:
        results["dihedral"] = (None, None)

    if "sp23" not in args.skip:
        results["sp23"] = run_module("10/11: sp²/sp³ classification ⭐",
                                      run_sp23, str(in_path),
                                      out_dir=str(out_dir / "10_sp23"))
    else:
        results["sp23"] = (None, None)

    if "tetra_conn" not in args.skip:
        results["tetra_conn"] = run_module(
            "11/11: Si tetra connectivity ⭐ (IRO)",
            run_tetra, str(in_path),
            out_dir=str(out_dir / "11_tetra_conn"))
    else:
        results["tetra_conn"] = (None, None)

    banner("종합 요약 생성")
    try:
        summary_path = write_summary(str(in_path), out_dir, results, files_used)
        print(f"  ✓ {summary_path}")
    except Exception as e:
        print(f"  ✗ Summary 실패: {e}")
        traceback.print_exc()

    elapsed = time.time() - t_start
    banner(f"전체 완료 (총 {elapsed:.1f}초)")
    print(f"\n  결과 위치: {out_dir.resolve()}")
    print(f"  메인 요약 파일: {out_dir.resolve()/'SUMMARY.txt'}")
    print(f"\n  각 분석 결과:")
    for name, (ok, _) in results.items():
        if ok is None:
            print(f"    {name:<10}: SKIP")
        elif ok:
            print(f"    {name:<10}: OK")
        else:
            print(f"    {name:<10}: FAIL")
    print()


if __name__ == "__main__":
    main()
