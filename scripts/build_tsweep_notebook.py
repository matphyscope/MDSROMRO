"""Generate notebooks/Tsweep_analysis.ipynb (property-vs-temperature)."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "Tsweep_analysis.ipynb"


def md(*l): return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(l)}
def code(*l): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": "\n".join(l)}


cells = [
md("# 온도 스캔 분석 — 물성 vs 온도",
   "",
   "온도별 dump 파일을 자동으로 순회하며 SRO/MRO 물성을 시간평균으로 뽑아",
   "**온도 의존성 곡선**을 그립니다 (각 점 ±1σ 블록 에러).",
   "",
   "두 파일명 규칙 자동 인식:",
   "- `large2`: `dump.300K.lammpstrj` … `dump.3000K.lammpstrj`, `dump.cool_300K.lammpstrj` (element 포함)",
   "- `large1`: `dump_T0300.lammpstrj` … `dump_T1800.lammpstrj`, `dump_cool_T0300.lammpstrj` (element 없음 → TYPE_MAP 사용)"),

md("## 0. Import"),
code("import warnings; warnings.filterwarnings('ignore')",
     "import numpy as np",
     "import matplotlib.pyplot as plt",
     "%matplotlib inline",
     "import amorph",
     "from amorph import sweep, presets, core",
     "print('amorph', amorph.__version__)"),

md("## 1. Config",
   "",
   "- `DUMP_DIR` : 온도별 dump 들이 있는 폴더.",
   "- `TYPE_MAP` : element 컬럼 없는 atom-style dump(large1)일 때만 사용.",
   "- `INCLUDE_RINGS` : 고리 통계 포함 (느림). 프레임은 `MAX_FRAMES_RINGS`로 제한.",
   "- MD↔실험 온도 환산: T_exp = 300 + (T_MD−300)/1.5 (덱 주석 기준). 필요 시 사용."),
code("DUMP_DIR      = '.'                 # ← 온도별 dump 폴더",
     "TYPE_MAP      = presets.SICN_TYPE_MAP",
     "FIXED_CUTOFFS = presets.SICN_CUTOFFS",
     "",
     "FREE_ELEMENT  = 'C'",
     "NETWORK_GROUP = ['Si', 'N']",
     "BT_GROUPS     = (['Si', 'N'], ['C'])",
     "TETRA_CENTER  = 'Si'",
     "",
     "PROD_RANGE    = None      # 각 온도 dump의 production 구간",
     "STRIDE        = 1",
     "INCLUDE_RINGS = False     # True면 고리/atom 도 곡선에 포함 (느림)",
     "MAX_RING_SIZE = 10",
     "MAX_FRAMES_RINGS = 3",
     "INCLUDE_COOL  = True      # 냉각 후 300K 점 포함",
     "NBLOCKS       = 5",
     "ALPHA_EXP     = 1.50      # MD→실험 온도 환산 계수"),

md("## 2. dump 자동 탐색"),
code("dumps = sweep.discover_dumps(DUMP_DIR)",
     "print(f'발견한 온도점 {len(dumps)}개:')",
     "for d in dumps:",
     "    print(f\"  {d['label']:<14} {d['path'].split('/')[-1]}\")",
     "assert dumps, 'dump 파일을 못 찾음 — DUMP_DIR/파일명 확인 (dump.300K.lammpstrj 또는 dump_T0300.lammpstrj)'"),

md("## 3. 온도 시리즈 계산  (각 온도 시간평균 + 블록 에러)"),
code("CM = core.CutoffMatrix(FIXED_CUTOFFS, default=0.0)",
     "S = sweep.temperature_series(",
     "    dumps, CM, type_map=TYPE_MAP, prod_range=PROD_RANGE, stride=STRIDE,",
     "    include_cool=INCLUDE_COOL, tetra_center=TETRA_CENTER,",
     "    free_element=FREE_ELEMENT, network_group=NETWORK_GROUP,",
     "    bt_groups=BT_GROUPS, include_rings=INCLUDE_RINGS,",
     "    max_ring_size=MAX_RING_SIZE, max_frames_rings=MAX_FRAMES_RINGS,",
     "    n_blocks=NBLOCKS, verbose=True)",
     "print();  print(sweep.to_table(S))"),

md("## 4. 물성 vs 온도 곡선",
   "",
   "가열 구간(●)과 냉각 후 300K(★)를 구분 표시. 이력(hysteresis) 확인 가능."),
code("cols = list(S['columns'].keys())",
     "heat = ~S['cool']; cooled = S['cool']",
     "ncol = 3; nrow = int(np.ceil(len(cols)/ncol))",
     "fig, axes = plt.subplots(nrow, ncol, figsize=(5*ncol, 3.4*nrow))",
     "axes = np.atleast_1d(axes).ravel()",
     "for ax, name in zip(axes, cols):",
     "    m = S['columns'][name]['mean']; e = S['columns'][name]['err']",
     "    ax.errorbar(S['T'][heat], m[heat], yerr=e[heat], fmt='o-', capsize=3, color='#1f77b4', label='heating')",
     "    if cooled.any():",
     "        ax.scatter(S['T'][cooled], m[cooled], marker='*', s=160, color='#d62728', zorder=9, label='cooled 300K')",
     "    ax.set_xlabel('T_MD (K)'); ax.set_ylabel(name); ax.set_title(name, fontsize=10); ax.grid(alpha=0.3)",
     "    if name==cols[0]: ax.legend(fontsize=8)",
     "for ax in axes[len(cols):]: ax.set_visible(False)",
     "plt.tight_layout(); plt.show()"),

md("## 5. (옵션) 실험 온도축으로 환산해 보기",
   "",
   "T_exp = 300 + (T_MD − 300)/α (덱 주석의 α=1.5). 예시로 밀도·CN_Si를 표시."),
code("T_exp = 300 + (S['T'] - 300)/ALPHA_EXP",
     "fig, axes = plt.subplots(1, 2, figsize=(12,4.2))",
     "for ax, name in zip(axes, ['density_gcc', f'q_tetra_{TETRA_CENTER}']):",
     "    if name not in S['columns']: ",
     "        ax.set_visible(False); continue",
     "    m=S['columns'][name]['mean']; e=S['columns'][name]['err']",
     "    ax.errorbar(T_exp[~S['cool']], m[~S['cool']], yerr=e[~S['cool']], fmt='s-', capsize=3, color='#2ca02c')",
     "    ax.set_xlabel('T_exp (K)  [=300+(T_MD-300)/α]'); ax.set_ylabel(name); ax.set_title(name+' vs 실험온도'); ax.grid(alpha=0.3)",
     "plt.tight_layout(); plt.show()"),

md("---",
   "온도스캔 완료. 조성을 바꾸려면 `DUMP_DIR`만 해당 조성 폴더로 바꿔 다시 실행하세요.",
   "(large1은 atom-style이라 `TYPE_MAP`이 자동 사용됩니다.)"),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
NB.parent.mkdir(parents=True, exist_ok=True)
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("wrote", NB, "with", len(cells), "cells")
