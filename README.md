# amorph — 비정질 구조 분석 패키지 (SRO / MRO)

LAMMPS 트래젝토리(dump)·데이터 파일로부터 비정질 재료의 **단거리 규칙성(SRO)**과
**중거리 규칙성(MRO)**을 분석하는 **원소 비종속(element-agnostic)** Python 패키지입니다.
원래 a-SiCN melt-quench / 온도 스캔 분석을 위해 만들었지만, 임의의 다원소 비정질
시스템에 적용할 수 있습니다.

## 설계 원칙
- **정확성 우선**: 모든 관측량은 production 트래젝토리를 **시간평균**하고, 연속
  블록 간 표준편차로 **±1σ 에러바**를 함께 보고합니다.
- **원소 비종속**: dump의 `element`/`type` 컬럼에서 원소를 자동 감지. 어떤 조성·
  원자수·원소에도 동작합니다.
- **추측 최소화**: 결합 cutoff는 (1) LAMMPS 입력과 동일한 고정값과 (2) g(r) 첫
  최소에서 도출한 값을 **둘 다** 계산해 비교합니다. 기본 분석은 고정값을 사용.

## 구조
```
amorph/
  core/    io(리더) · frame · neighbors(cutoff행렬·이웃탐색) · cutoffs · average(시간평균)
  sro/     rdf · coordination(CN+ADF) · csro · voronoi · boo · hybridization · tetrahedra
  mro/     rings · bhatia_thornton · clusters · dihedral · tetra_connectivity
  sweep    온도별 dump 자동 순회 → 물성 vs 온도 곡선 (T-scan 드라이버)
  presets  SiCN 전용 type_map / cutoff / ADF 삼중항
notebooks/
  SRO_analysis.ipynb    ← 단거리 규칙성 (한 구조/한 온도)
  MRO_analysis.ipynb    ← 중거리 규칙성 (한 구조/한 온도)
  Tsweep_analysis.ipynb ← 온도 의존성 (온도별 dump 자동 순회, 물성 vs T)
```

## 빠른 사용
```python
import amorph
from amorph import core, presets
from amorph.sro import rdf, coordination

traj = core.load("dump.300K.lammpstrj", type_map=presets.SICN_TYPE_MAP)
prod = core.select_frames(traj, frame_range=(50, None), stride=1)
CM   = presets.sicn_cutoffs()

R  = rdf.partial_rdf(prod, r_max=10.0, nbins=500, n_blocks=5)
CN = coordination.coordination_numbers(prod, CM, n_blocks=5)
```
보통은 `notebooks/SRO_analysis.ipynb`의 *Config* 셀만 바꿔 실행하면 됩니다.

## 의존성
`numpy`, `scipy`, `matplotlib`, `freud-analysis`(Voronoi·BOO), `networkx`(MRO 고리/클러스터)

## 검증
`python3 tests/make_test_data.py`로 다이아몬드/징크블렌드 등 **정답이 알려진** 합성
구조를 만들어 코드를 검증합니다 (CN=4, 사면체각 109.5°, Q4≈0.51 등).

## 입력 데이터별 적용 (a-SiCN 4개 LAMMPS 덱 기준)
| 시뮬레이션 | dump 형식 | 비고 |
|---|---|---|
| `v8_large1` melt-quench (Si45C34N21, 7800) | dump 없음 | 최종 `.data` 단일 프레임 폴백 |
| `v8_large2` melt-quench (Si37C36N27, 8400) | custom(element) | 300K/1273K 트래젝토리 시간평균 |
| `Tscan_large1` (300→1800K) | atom(scaled, element 없음) | `type_map` 필요 |
| `Tscan_large2` (300→3000K) | custom(element) | 온도별 dump 각각 분석 |
