"""
pdf_analysis.py
===============
비정질 구조의 실공간 PDF 분석 체인 (Jupyter 노트북에서 단계별 호출용).

    count histogram  →  g(r)  →  R(r)=4πr²ρ g  →  G(r)=4πrρ(g-1)

검증된 I/O·neighbor 코어(sro_analysis.py)를 재사용한다. 같은 폴더에 둘 것.

함수 그룹
  [setup]     system_info
  [chain]     pair_histogram → rdf_g → R_of_r → G_of_r
  [measure]   peak_min, gaussian, fit_gaussians, cn_from_R, running_cn,
              density_from_G
  [plot]      plot_curve, plot_partials
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit

from sro_analysis import read_lammpstrj, select_frames, _neighbor_pairs

# numpy 1.x: np.trapz / numpy 2.x: np.trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz

__all__ = [
    "read_lammpstrj", "select_frames", "system_info", "precompute_neighbors",
    "pair_histogram", "rdf_g", "R_of_r", "G_of_r",
    "peak_min", "gaussian", "fit_gaussians", "cn_from_R", "running_cn",
    "density_from_G", "plot_curve", "plot_partials",
]

MASS = {"Si": 28.09, "C": 12.01, "N": 14.01}   # amu (밀도 계산용)


# ============================================================================
# [setup] 시스템 정보 — 이후 처리에 필요한 N, V, ρ
# ============================================================================
def system_info(frames):
    """frame 리스트에서 조성·부피·수밀도(number density)·질량밀도를 요약."""
    f0 = frames[0]
    elems, cnt = np.unique(f0.elem, return_counts=True)
    counts = dict(zip(elems.tolist(), cnt.tolist()))
    N = f0.natoms
    V = float(np.mean([np.prod(f.box) for f in frames]))      # 평균 부피 (NPT)
    rho0 = N / V                                              # total 수밀도 (1/Å³)
    rho = {e: counts[e] / V for e in counts}                  # partial 수밀도
    M = sum(MASS.get(e, 0.0) * counts[e] for e in counts)     # amu
    dens = M / 6.02214076e23 / (V * 1e-24)                    # g/cc
    return {
        "n_frames": len(frames), "natoms": N, "counts": counts,
        "volume": V, "rho0": rho0, "rho": rho, "mass_density": dens,
    }


# ============================================================================
# 내부 헬퍼: bin, neighbor 캐시, frame 단위 count
# ============================================================================
def _bin_edges(r_max, nbins):
    edges = np.linspace(0.0, r_max, nbins + 1)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    shell_vol = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    return edges, r_mid, shell_vol


def precompute_neighbors(frames, r_max):
    """모든 frame 의 r_max 이내 이웃쌍 (iu,ju,r) 을 한 번에 캐시한다.
    노트북에서 한 번 만들어 모든 partial/total 계산에 재사용하면 빠르다."""
    return [_neighbor_pairs(fr, r_max) for fr in frames]


def _frame_counts(fr, neigh, pair, edges):
    """단일 frame 의 (히스토그램, N_a, rho_b, same) 반환."""
    iu, ju, r = neigh
    V = float(np.prod(fr.box))
    if pair is None:                                  # total
        sel = np.ones(len(r), dtype=bool)
        Na, rho_b, same = fr.natoms, (fr.natoms - 1) / V, True
    elif pair[0] == pair[1]:                          # same species
        A = pair[0]; ea, eb = fr.elem[iu], fr.elem[ju]
        sel = (ea == A) & (eb == A)
        Na = int(np.count_nonzero(fr.elem == A)); rho_b = (Na - 1) / V; same = True
    else:                                             # cross
        A, B = pair; ea, eb = fr.elem[iu], fr.elem[ju]
        sel = ((ea == A) & (eb == B)) | ((ea == B) & (eb == A))
        Na = int(np.count_nonzero(fr.elem == A))
        Nb = int(np.count_nonzero(fr.elem == B)); rho_b = Nb / V; same = False
    h, _ = np.histogram(r[sel], bins=edges)
    return h, Na, rho_b, same


# ============================================================================
# [chain-4] count histogram — 정규화 전 raw pair count (frame 평균)
# ============================================================================
def pair_histogram(frames, pair=None, r_max=8.0, nbins=300, cache=None):
    """
    거리별 pair 개수 히스토그램(unordered, frame 평균).
      pair=None → total / pair=("Si","N") → partial
    cache: precompute_neighbors() 결과를 넘기면 이웃 재계산 생략(빠름).
    반환: r, counts, shell_vol, N_a, rho_b, same
    """
    edges, r_mid, shell_vol = _bin_edges(r_max, nbins)
    if cache is None:
        cache = precompute_neighbors(frames, r_max)
    counts = np.zeros(nbins); Na_acc = rhob_acc = 0.0; same = False
    for fr, ng in zip(frames, cache):
        h, Na, rho_b, same = _frame_counts(fr, ng, pair, edges)
        counts += h; Na_acc += Na; rhob_acc += rho_b
    nf = len(frames)
    return {"r": r_mid, "counts": counts / nf, "shell_vol": shell_vol,
            "N_a": Na_acc / nf, "rho_b": rhob_acc / nf, "same": same}


# ============================================================================
# [chain-5] g(r) — pair correlation (+ block 1σ 에러밴드)
# ============================================================================
def rdf_g(frames, pair=None, r_max=8.0, nbins=300, n_blocks=10, cache=None):
    """
    g(r). frame 당 히스토그램을 한 번만 계산(single pass) 후 block 조립.
    same-species/total 은 ordered 환산(×2) 자동 처리 → g→1.
    반환: r, g, g_err(block 1σ), counts(raw, frame평균), rho_b(R/G용), rho0(total)
    """
    edges, r_mid, shell_vol = _bin_edges(r_max, nbins)
    if cache is None:
        cache = precompute_neighbors(frames, r_max)
    nf = len(frames)

    H = np.zeros((nf, nbins)); Na = np.zeros(nf); rhob = np.zeros(nf); same = False
    for k, (fr, ng) in enumerate(zip(frames, cache)):
        h, na, rb, same = _frame_counts(fr, ng, pair, edges)
        H[k] = h; Na[k] = na; rhob[k] = rb
    fac = 2.0 if same else 1.0

    def _g(idx):
        cnt = H[idx].mean(0) * fac
        denom = Na[idx].mean() * rhob[idx].mean() * shell_vol
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denom > 0, cnt / denom, 0.0)

    nb = max(1, min(n_blocks, nf))
    gblk = np.array([_g(b) for b in np.array_split(np.arange(nf), nb)])
    allidx = np.arange(nf)
    V0 = float(np.mean([np.prod(f.box) for f in frames]))
    return {"r": r_mid, "g": _g(allidx),
            "g_err": gblk.std(0, ddof=1) if nb > 1 else np.zeros(nbins),
            "counts": H.mean(0), "rho_b": rhob.mean(),
            "rho0": frames[0].natoms / V0}


# ============================================================================
# [chain-7] R(r) = 4πr²ρ g(r)     [chain-9] G(r) = 4πrρ (g-1)
# ============================================================================
def R_of_r(r, g, rho):
    """radial distribution. ∫R dr (껍질) = coordination number."""
    return 4.0 * np.pi * r ** 2 * rho * g


def G_of_r(r, g, rho):
    """reduced RDF (실험 PDF 비교형). r→0 기울기 = -4πρ."""
    return 4.0 * np.pi * r * rho * (g - 1.0)


# ============================================================================
# [measure] g(r) 로부터: 봉우리/최소/Gaussian
# ============================================================================
def peak_min(r, y, window):
    """[lo,hi] 안 첫 봉우리(위치,높이)와 그 뒤 첫 최소(위치,값)."""
    lo, hi = window
    m = (r >= lo) & (r <= hi)
    rr, yy = r[m], y[m]
    ip = int(np.argmax(yy))
    im = ip + int(np.argmin(yy[ip:]))
    return {"peak_r": rr[ip], "peak_h": yy[ip],
            "min_r": rr[im], "min_h": yy[im]}


def gaussian(x, A, mu, sig):
    return A * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def _multi_gauss(x, *p):
    y = np.zeros_like(x)
    for k in range(0, len(p), 3):
        y = y + gaussian(x, p[k], p[k + 1], p[k + 2])
    return y


def fit_gaussians(r, y, window, peaks):
    """
    [lo,hi] 구간을 n개 Gaussian 합으로 피팅 (merged peak deconvolution 등).
    peaks : [(A0,mu0,sig0), ...]  초기추정. 길이가 성분 수.
    반환  : {"params":[(A,mu,sig),...], "fit":(r_win, y_fit), "r2":...}
    """
    lo, hi = window
    m = (r >= lo) & (r <= hi)
    rr, yy = r[m], y[m]
    p0 = [v for trip in peaks for v in trip]
    lb = [0, lo, 1e-3] * len(peaks)
    ub = [np.inf, hi, (hi - lo)] * len(peaks)
    popt, _ = curve_fit(_multi_gauss, rr, yy, p0=p0, bounds=(lb, ub), maxfev=20000)
    yfit = _multi_gauss(rr, *popt)
    ss_res = np.sum((yy - yfit) ** 2)
    ss_tot = np.sum((yy - yy.mean()) ** 2)
    params = [tuple(popt[k:k + 3]) for k in range(0, len(popt), 3)]
    return {"params": params, "fit": (rr, yfit),
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan}


# ============================================================================
# [measure] R(r) 로부터: coordination number (적분)
# ============================================================================
def cn_from_R(r, R, r0, r1):
    """껍질 [r0,r1] 적분 → coordination number (trapezoid)."""
    m = (r >= r0) & (r <= r1)
    return float(_trapz(R[m], r[m]))


def running_cn(r, R):
    """누적 적분 N(r) = ∫₀ʳ R dr′ (첫최소까지 읽으면 CN)."""
    from scipy.integrate import cumulative_trapezoid
    return cumulative_trapezoid(R, r, initial=0.0)


# ============================================================================
# [measure] G(r) 로부터: number density (r→0 기울기)
# ============================================================================
def density_from_G(r, G, r_lo, r_hi):
    """
    첫 봉우리 이전(g≈0) 구간에서 G(r) ≈ -4πρ r 의 기울기로 ρ 추정.
    반환: (rho_est, slope).  실제 ρ0 와 비교하는 자가검증용.
    """
    m = (r >= r_lo) & (r <= r_hi)
    slope = np.polyfit(r[m], G[m], 1)[0]
    return -slope / (4.0 * np.pi), slope


# ============================================================================
# [plot]
# ============================================================================
def plot_curve(ax, r, y, err=None, label=None, color=None, baseline=None):
    if err is not None:
        ax.fill_between(r, y - err, y + err, color=color, alpha=0.25)
    ax.plot(r, y, color=color, lw=1.4, label=label)
    if baseline is not None:
        ax.axhline(baseline, color="k", lw=0.5, ls=":")
    ax.set_xlabel("r (Å)")
    ax.grid(alpha=0.2)
    if label:
        ax.legend(fontsize=9)
    return ax


def plot_partials(ax, r, curves, baseline=None):
    """curves: {"Si-N":(y,err or None), ...} 여러 partial 한 축에."""
    cyc = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    for (name, (y, err)), c in zip(curves.items(), cyc):
        plot_curve(ax, r, y, err=err, label=name, color=c)
    if baseline is not None:
        ax.axhline(baseline, color="k", lw=0.5, ls=":")
    return ax


# ============================================================================
# (확장) 자동 원소/쌍, 밀도 단위, FWHM Gaussian, ADF, 사면체 검증, 저장
# ============================================================================
import itertools

AMU_PER_A3_TO_G_CM3 = 1.6605390666   # 1 amu/Å³ = 1.6605 g/cm³
FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))   # σ→FWHM (2.3548)


# ---- [req6] 원소·쌍 자동 ----
def elements(frames):
    """trajectory에 존재하는 원소를 정렬해 반환."""
    return sorted(set(frames[0].elem.tolist()))


def all_pairs(frames):
    """모든 unordered 원소쌍 (동종 포함). 예: [(C,C),(C,N),...]"""
    return list(itertools.combinations_with_replacement(elements(frames), 2))


# ---- [req3] 밀도 단위 ----
def mean_atomic_mass(frames):
    el, cnt = np.unique(frames[0].elem, return_counts=True)
    tot = sum(MASS.get(e, 0.0) * c for e, c in zip(el, cnt))
    return tot / frames[0].natoms


def mass_density_gcc(rho_num, mean_mass):
    """number density(1/Å³) → 질량밀도(g/cm³)."""
    return rho_num * mean_mass * AMU_PER_A3_TO_G_CM3


# ---- [req5] Gaussian peak fit: center/height/FWHM + 적합도 ----
def fit_peak(r, y, peak_r, half_window=0.30, n_gauss=1):
    """
    peak_r 주변 ±half_window 를 n_gauss 개 Gaussian 합으로 피팅.
    반환: components[{height,center,sigma,fwhm,area}], r2, rmse, fit(r,yfit)
    """
    lo, hi = peak_r - half_window, peak_r + half_window
    m = (r >= lo) & (r <= hi)
    rr, yy = r[m], y[m]
    if n_gauss == 1:
        p0 = [max(yy.max(), 1e-6), peak_r, 0.05]
    else:
        p0 = []
        for k in range(n_gauss):
            p0 += [yy.max() / n_gauss, lo + (k + 0.5) * (hi - lo) / n_gauss, 0.05]
    lb = [0, lo, 1e-3] * n_gauss
    ub = [np.inf, hi, (hi - lo)] * n_gauss
    try:
        popt, _ = curve_fit(_multi_gauss, rr, yy, p0=p0, bounds=(lb, ub), maxfev=40000)
    except Exception as e:
        return {"components": [], "r2": np.nan, "rmse": np.nan,
                "fit": (rr, np.zeros_like(rr)), "error": str(e)}
    yfit = _multi_gauss(rr, *popt)
    ss_res = float(np.sum((yy - yfit) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    comps = []
    for k in range(0, len(popt), 3):
        A, mu, sig = popt[k:k + 3]
        comps.append({"height": A, "center": mu, "sigma": sig,
                      "fwhm": FWHM_FACTOR * sig, "area": A * sig * np.sqrt(2 * np.pi)})
    comps.sort(key=lambda c: c["center"])
    return {"components": comps,
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            "rmse": np.sqrt(ss_res / len(yy)), "fit": (rr, yfit)}


# ---- [req9] ADF: bond-angle distribution ----
def _adjacency(fr, neigh, cutoffs):
    """결합 cutoff 이내 이웃 인접리스트 + (min-image) 방향벡터."""
    iu, ju, r = neigh
    ei, ej = fr.elem[iu], fr.elem[ju]
    rc = np.array([cutoffs.get((a, b), cutoffs.get((b, a), 0.0))
                   for a, b in zip(ei, ej)])
    sel = r < rc
    bi, bj = iu[sel], ju[sel]
    d = fr.pos[bj] - fr.pos[bi]
    d -= fr.box * np.round(d / fr.box)
    adj = [[] for _ in range(fr.natoms)]
    el = fr.elem
    for k in range(len(bi)):
        i, j = int(bi[k]), int(bj[k])
        adj[i].append((el[j], d[k]))
        adj[j].append((el[i], -d[k]))
    return adj


def adf_all(frames, cutoffs, nbins=180, cache=None, r_max=None):
    """
    모든 (Xa-A-Xb) partial ADF + total. cutoffs={(elem,elem):bond_rc}.
    반환: theta(deg), {("Xa","A","Xb"): P(theta)}, total: key ("*","*","*")
    P 는 면적정규화(∫P dθ=1).
    """
    if r_max is None:
        r_max = max(cutoffs.values()) + 0.2
    if cache is None:
        cache = precompute_neighbors(frames, r_max)
    edges = np.linspace(0.0, 180.0, nbins + 1)
    th = 0.5 * (edges[:-1] + edges[1:])
    hist = {}
    tot = np.zeros(nbins)
    for fr, ng in zip(frames, cache):
        adj = _adjacency(fr, ng, cutoffs)
        for c in range(fr.natoms):
            nb = adj[c]
            if len(nb) < 2:
                continue
            A = fr.elem[c]
            els = [e for e, _ in nb]
            vecs = np.array([v for _, v in nb])
            norms = np.linalg.norm(vecs, axis=1)
            for a in range(len(nb)):
                for b in range(a + 1, len(nb)):
                    cosang = np.dot(vecs[a], vecs[b]) / (norms[a] * norms[b])
                    ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
                    bidx = min(int(ang / 180.0 * nbins), nbins - 1)
                    key = (min(els[a], els[b]), A, max(els[a], els[b]))
                    hist.setdefault(key, np.zeros(nbins))[bidx] += 1
                    tot[bidx] += 1
    out = {}
    for k, h in hist.items():
        area = _trapz(h, th)
        out[k] = h / area if area > 0 else h
    a = _trapz(tot, th)
    out[("*", "*", "*")] = tot / a if a > 0 else tot
    return th, out


def adf_peak_fwhm(theta, P, window=(80, 130)):
    """ADF 봉우리 위치와 FWHM. 반치점을 선형보간해 정확히 측정."""
    m = (theta >= window[0]) & (theta <= window[1])
    tt, pp = theta[m], P[m]
    ip = int(np.argmax(pp))
    peak = tt[ip]; hm = pp[ip] / 2.0

    li = ip
    while li > 0 and pp[li] > hm:
        li -= 1
    if li < ip and pp[li + 1] != pp[li]:
        tl = tt[li] + (hm - pp[li]) / (pp[li + 1] - pp[li]) * (tt[li + 1] - tt[li])
    else:
        tl = tt[li]

    ri = ip
    while ri < len(pp) - 1 and pp[ri] > hm:
        ri += 1
    if ri > ip and pp[ri] != pp[ri - 1]:
        tr = tt[ri - 1] + (hm - pp[ri - 1]) / (pp[ri] - pp[ri - 1]) * (tt[ri] - tt[ri - 1])
    else:
        tr = tt[ri]
    return peak, (tr - tl)


# ---- [req10] AX4 사면체 검증 (r2/r1 ↔ 결합각) ----
def tetra_geminal(r1, theta_deg=109.4712):
    """이상 사면체: geminal r2 = 2 r1 sin(θ/2), ratio = r2/r1 (109.47°→1.633)."""
    r2 = 2.0 * r1 * np.sin(np.radians(theta_deg) / 2.0)
    return r2, r2 / r1


def theta_from_r1r2(r1, r2):
    """관측 r1,r2 → 결합각 θ = 2 asin(r2/2r1)."""
    return 2.0 * np.degrees(np.arcsin(np.clip(r2 / (2.0 * r1), -1, 1)))


def r2_fwhm_from_angle(r1, theta_deg, fwhm_theta_deg):
    """ADF 각폭 → geminal r2 폭 예측: Δr2 ≈ r1 cos(θ/2) Δθ(rad)."""
    return r1 * np.cos(np.radians(theta_deg) / 2.0) * np.radians(fwhm_theta_deg)


# ---- [req8] 저장 ----
def save_curve(path, x, y, err=None, header="x y"):
    cols = [x, y]
    if err is not None:
        cols.append(err); header = header + " err"
    np.savetxt(path, np.column_stack(cols), header=header, fmt="%.6g")


def save_text(path, text):
    with open(path, "w") as f:
        f.write(text)


def first_peak(r, g, rmin=0.8, thresh=1.2):
    """
    첫 결합봉우리 자동 검출 = r>rmin 에서 g>thresh 인 첫 local maximum.
    (뒤 껍질이 더 높아도 '첫' local max 를 잡아 bridged 2차봉우리 함정을 피함)
    반환: {peak_r,peak_h,min_r,min_h,idx} 또는 None(봉우리 없음).
    """
    n = len(g)
    for i in range(1, n - 1):
        if r[i] < rmin:
            continue
        if g[i] > thresh and g[i] >= g[i - 1] and g[i] > g[i + 1]:
            j = i
            while j < n - 1 and g[j + 1] <= g[j]:
                j += 1
            return {"peak_r": float(r[i]), "peak_h": float(g[i]),
                    "min_r": float(r[j]), "min_h": float(g[j]), "idx": i}
    return None


# ============================================================================
# (확장) 앙상블 평균: seed 평균 vs frame(열) 평균
# ============================================================================
def ensemble_average_rdf(frame_sets, pair=None, r_max=6.0, nbins=200):
    """
    여러 구조(seed)에 대한 앙상블 평균 g(r).
    frame_sets: seed별 frame-list 의 리스트. 각 seed 내부는 frame(열) 평균,
                seed 간에는 평균 + seed-to-seed 표준편차(=구조적 에러바).
    반환: r, g(seed평균), g_seed_std(glass-to-glass 에러바), per_seed(배열)
    """
    gs = []
    r = None
    for frames in frame_sets:
        res = rdf_g(frames, pair, r_max, nbins, n_blocks=1)   # seed 내부: 열평균
        gs.append(res["g"]); r = res["r"]
    G = np.array(gs)
    return {"r": r, "g": G.mean(0),
            "g_seed_std": G.std(0, ddof=1) if len(gs) > 1 else np.zeros_like(r),
            "per_seed": G, "n_seed": len(gs)}


def ensemble_rdf_from_paths(paths, pair=None, r_max=6.0, nbins=200,
                            frame_range=(0, None), stride=1):
    """trajectory 파일 경로 리스트(seed들) → 앙상블 평균 g(r)."""
    sets = []
    for p in paths:
        fr = read_lammpstrj(p)
        lo, hi = frame_range; hi = len(fr) if hi is None else hi
        sets.append(select_frames(fr, (lo, hi), stride))
    return ensemble_average_rdf(sets, pair, r_max, nbins)
