"""
sro_analysis.py
================
a-SiCN short-range order (SRO) 분석 — LAMMPS trajectory(.lammpstrj) 기반.

제공 기능
  1) read_lammpstrj   : self-describing dump 헤더를 읽어 frame 단위로 파싱
  2) partial_rdf      : partial g_ab(r), frame 평균 + block-averaged 1σ 에러밴드
  3) coordination     : shell-integral 방식 CN (bin-width 비의존)

설계 원칙
  - production window 프레임을 time-average 하는 것이 기본.
  - frame_range / stride 로 ramp·cooling 프레임 혼입을 차단.
  - CN 은 raw 이웃 수를 cutoff 까지 적분(평균)해서 bin-width 에 둔감하게.

⚠️ 의존성: numpy 만.
"""

from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------------
# ⚠️ 반드시 확인할 것
#   dump 에 'element' 컬럼이 있으면 이 map 은 무시되고 element 가 사용된다.
#   'type' 만 있으면 아래 map 으로 변환하므로, 네 LAMMPS data 파일의
#   atom type 순서(= potential 파일 element 순서)와 정확히 일치해야 한다.
#   아래는 placeholder 다 — 내가 가정한 게 아니라 네가 검증/수정할 자리.
# ----------------------------------------------------------------------------
TYPE_MAP = {1: "Si", 2: "C", 3: "N"}


# ============================================================================
# 1. Trajectory reader
# ============================================================================
class Frame:
    """단일 스냅샷. pos: (N,3) Å, elem: (N,) str, box: (3,) Å, box_lo: (3,) Å."""
    __slots__ = ("timestep", "pos", "elem", "box", "box_lo", "natoms")

    def __init__(self, timestep, pos, elem, box, box_lo):
        self.timestep = timestep
        self.pos = pos
        self.elem = elem
        self.box = box
        self.box_lo = box_lo
        self.natoms = len(pos)


def read_lammpstrj(path, type_map=TYPE_MAP):
    """
    LAMMPS dump(.lammpstrj)를 Frame 리스트로 읽는다. 헤더 기반 자동 파싱.

    지원 좌표 컬럼: x/y/z (unwrapped 포함 xu/yu/zu), xs/ys/zs (scaled).
    orthogonal box 만 지원 — tilt(xy xz yz)가 있으면 경고하고 무시한다.
    """
    frames = []
    with open(path, "r") as f:
        lines = f.readlines()

    i, n = 0, len(lines)
    warned_tilt = False
    while i < n:
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue

        timestep = int(lines[i + 1])

        # NUMBER OF ATOMS
        assert lines[i + 2].startswith("ITEM: NUMBER OF ATOMS")
        natoms = int(lines[i + 3])

        # BOX BOUNDS  (tilt 가 있으면 각 줄에 3개 값)
        assert lines[i + 4].startswith("ITEM: BOX BOUNDS")
        box = np.zeros(3)
        box_lo = np.zeros(3)
        for d in range(3):
            vals = list(map(float, lines[i + 5 + d].split()))
            box_lo[d] = vals[0]
            box[d] = vals[1] - vals[0]
            if len(vals) > 2 and abs(vals[2]) > 1e-8 and not warned_tilt:
                print("⚠️  box tilt(triclinic) 감지 — 현재 orthogonal 가정으로 처리함.")
                warned_tilt = True

        # ATOMS 헤더 (컬럼명)
        atoms_header = lines[i + 8]
        assert atoms_header.startswith("ITEM: ATOMS")
        cols = atoms_header.split()[2:]
        idx = {c: k for k, c in enumerate(cols)}

        # 좌표 컬럼 종류 판별
        if {"x", "y", "z"} <= idx.keys():
            cx, cy, cz, scaled = idx["x"], idx["y"], idx["z"], False
        elif {"xu", "yu", "zu"} <= idx.keys():
            cx, cy, cz, scaled = idx["xu"], idx["yu"], idx["zu"], False
        elif {"xs", "ys", "zs"} <= idx.keys():
            cx, cy, cz, scaled = idx["xs"], idx["ys"], idx["zs"], True
        else:
            raise ValueError(f"좌표 컬럼을 찾을 수 없음: {cols}")

        has_elem = "element" in idx
        ctype = idx.get("type")

        start = i + 9
        if start + natoms > n:           # 잘린 마지막 frame → 버리고 종료
            print(f"⚠️  truncated final frame at timestep {timestep} — 건너뜀.")
            break

        pos = np.empty((natoms, 3))
        elem = np.empty(natoms, dtype=object)
        for a in range(natoms):
            parts = lines[start + a].split()
            pos[a, 0] = float(parts[cx])
            pos[a, 1] = float(parts[cy])
            pos[a, 2] = float(parts[cz])
            if has_elem:
                elem[a] = parts[idx["element"]]
            else:
                elem[a] = type_map[int(parts[ctype])]

        if scaled:
            pos = box_lo + pos * box  # 0..1 → 실좌표

        frames.append(Frame(timestep, pos, elem.astype(str), box, box_lo))
        i = start + natoms

    return frames


def select_frames(frames, frame_range=None, stride=1):
    """production window 만 고르고 stride 로 decorrelate."""
    lo, hi = (0, len(frames)) if frame_range is None else frame_range
    return frames[lo:hi:stride]


# ============================================================================
# 2. Partial RDF  (frame 평균 + block-averaged 1σ band)
# ============================================================================
def _neighbor_pairs(frame, r_max):
    """주기경계 cKDTree 로 r_max 이내 모든 i<j 쌍과 거리 반환 (PBC minimum image)."""
    from scipy.spatial import cKDTree
    box = frame.box
    p = (frame.pos - frame.box_lo) % box        # [0, L) 로 wrap (cKDTree boxsize 요구)
    tree = cKDTree(p, boxsize=box)
    pr = tree.query_pairs(r_max, output_type="ndarray")
    if len(pr) == 0:
        return (np.empty(0, int), np.empty(0, int), np.empty(0))
    iu, ju = pr[:, 0], pr[:, 1]
    d = p[iu] - p[ju]
    d -= box * np.round(d / box)                # minimum image
    r = np.sqrt(np.einsum("ij,ij->i", d, d))
    return iu, ju, r


def partial_rdf(frames, pairs, r_max=6.0, nbins=200, n_blocks=5):
    """
    partial g_ab(r) 계산.

    pairs : [("Si","N"), ("Si","C"), ("C","C"), ...]  (순서 무관, 무방향)
    반환  : dict
        "r"          : (nbins,) 빈 중심 (Å)
        ("Si","N")   : {"g": (nbins,) 평균, "err": (nbins,) block 간 1σ}
        ...
    """
    dr = r_max / nbins
    edges = np.linspace(0.0, r_max, nbins + 1)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    shell_vol = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)

    nframes = len(frames)
    if nframes == 0:
        raise ValueError("frame 이 비어 있음 — frame_range/stride 확인.")
    n_blocks = min(n_blocks, nframes)
    block_ids = np.array_split(np.arange(nframes), n_blocks)

    # block 별 g 누적 → block 간 std 가 ±1σ 밴드
    g_blocks = {p: [] for p in pairs}

    for blk in block_ids:
        # 이 block 의 partial 히스토그램 합과, 정규화용 평균 밀도 축적
        hist = {p: np.zeros(nbins) for p in pairs}
        norm = {p: 0.0 for p in pairs}  # Σ_frames [ N_a * (N_b/V) ]  (a==b 보정 포함)
        for fi in blk:
            fr = frames[fi]
            V = float(np.prod(fr.box))
            iu, ju, r = _neighbor_pairs(fr, r_max)
            ea, eb = fr.elem[iu], fr.elem[ju]
            for (A, B) in pairs:
                if A == B:
                    mask = (ea == A) & (eb == A)
                    Na = int(np.count_nonzero(fr.elem == A))
                    # 같은 종: ideal 이웃 = (Na-1)/V * shell  → g→1
                    rho_b = (Na - 1) / V
                    norm[(A, B)] += Na * rho_b
                else:
                    mask = ((ea == A) & (eb == B)) | ((ea == B) & (eb == A))
                    Na = int(np.count_nonzero(fr.elem == A))
                    Nb = int(np.count_nonzero(fr.elem == B))
                    rho_b = Nb / V
                    norm[(A, B)] += Na * rho_b
                h, _ = np.histogram(r[mask], bins=edges)
                if A == B:
                    h = h * 2  # i<j 만 셌으므로 ordered(중심당) 카운트로 환산
                hist[(A, B)] += h

        for p in pairs:
            if norm[p] <= 0:
                g_blocks[p].append(np.full(nbins, np.nan))
                continue
            # <hist>_frame / (<Na*rho_b>_frame * shell_vol)
            g = (hist[p] / len(blk)) / ((norm[p] / len(blk)) * shell_vol)
            g_blocks[p].append(g)

    out = {"r": r_mid}
    for p in pairs:
        arr = np.array(g_blocks[p])              # (n_blocks, nbins)
        out[p] = {
            "g": np.nanmean(arr, axis=0),
            "err": np.nanstd(arr, axis=0, ddof=1) if n_blocks > 1
                   else np.zeros(nbins),
        }
    return out


# ============================================================================
# 3. Coordination number  (shell integral — bin-width 비의존)
# ============================================================================
def coordination(frames, cutoffs):
    """
    중심종 A 주변 종 B 의 평균 이웃 수(첫 최소 cutoff 까지).
    cutoffs : {("Si","N"): 2.20, ("Si","C"): 2.35, ...}  (Å, 무방향)
    반환    : {("Si","N"): (mean, std_over_frames), ...}
    """
    rc_max = max(cutoffs.values())
    per_frame = {p: [] for p in cutoffs}
    for fr in frames:
        iu, ju, r = _neighbor_pairs(fr, rc_max)
        ea, eb = fr.elem[iu], fr.elem[ju]
        for (A, B), rc in cutoffs.items():
            within = r < rc
            if A == B:
                cnt = np.count_nonzero(within & (ea == A) & (eb == A))
                Na = np.count_nonzero(fr.elem == A)
                # i<j 만 셌으므로 ×2 (양쪽 모두 이웃 보유), /Na
                per_frame[(A, B)].append(2 * cnt / max(Na, 1))
            else:
                m = within & (((ea == A) & (eb == B)) | ((ea == B) & (eb == A)))
                cnt = np.count_nonzero(m)
                Na = np.count_nonzero(fr.elem == A)  # A 중심 기준 CN_A->B
                per_frame[(A, B)].append(cnt / max(Na, 1))
    return {p: (float(np.mean(v)), float(np.std(v, ddof=1) if len(v) > 1 else 0.0))
            for p, v in per_frame.items()}


# ============================================================================
# 사용 예시
# ============================================================================
if __name__ == "__main__":
    frames = read_lammpstrj("dump.lammpstrj")
    print(f"{len(frames)} frames, {frames[0].natoms} atoms")

    # production window 만, 5 프레임 간격으로 decorrelate
    prod = select_frames(frames, frame_range=(20, len(frames)), stride=5)

    pairs = [("Si", "N"), ("Si", "C"), ("C", "C"), ("Si", "Si"), ("N", "N")]
    rdf = partial_rdf(prod, pairs, r_max=6.0, nbins=200, n_blocks=5)

    # CN 은 각 partial g(r) 첫 최소에서 cutoff 를 읽어 넣을 것
    cn = coordination(prod, cutoffs={
        ("Si", "N"): 2.20, ("Si", "C"): 2.35,
        ("C", "C"): 1.90, ("Si", "Si"): 2.80,
    })
    for p, (m, s) in cn.items():
        print(f"CN {p[0]}-{p[1]}: {m:.3f} ± {s:.3f}")
