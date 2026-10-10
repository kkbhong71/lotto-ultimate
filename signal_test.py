# -*- coding: utf-8 -*-
"""
[검증 1] 과거 데이터에 예측력이 있는가 — 순위 검정

회차 t 마다 t 이전 데이터만으로 45개 번호에 순위를 매기고,
실제 t회차 당첨 6개 번호의 평균 순위를 본다.
  예측력 있음 → 23보다 작아야 함 / 없음 → 23.0

조합 생성 백테스트보다 표본이 수십 배 많아 검정력이 훨씬 높다.
500회차 × 6개 = 3,000 관측 → 평균순위 0.47 차이까지 탐지.
"""
import os, sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'engine'))
from bootstrap import load_engine, data            # noqa: E402

TEST_N = 500
v12, p13 = load_engine(patch_backtest=False)
hist = v12.safe_load_history(data('latest.csv'), verbose=False)
cfg = v12.Config()
comps = ['frequency','recency','gap','pair','momentum','zone','markov','assoc','FINAL']
ranks = {c: [] for c in comps}

for t in range(len(hist) - TEST_N, len(hist)):
    eng = v12.DataEngine.from_history(hist[:t], cfg)
    vecs = {'frequency': eng.frequency_analysis['weighted'],
            'gap': eng.gap_analysis['score'],
            'pair': eng.pair_analysis['number_pair_strength'],
            'momentum': eng.momentum_analysis['momentum'],
            'zone': eng.zone_analysis['zone_score'],
            'markov': eng.markov_analysis['score'],
            'assoc': eng.assoc_analysis['score'],
            'FINAL': eng.final_scores}
    r10 = eng.frequency_analysis['by_period'].get(10, {})
    vecs['recency'] = np.array([0] + [r10.get(n, 0) for n in range(1, 46)], dtype=float)
    for name, v in vecs.items():
        order = sorted(range(1, 46), key=lambda n: v[n], reverse=True)
        pos = {n: i + 1 for i, n in enumerate(order)}
        ranks[name].extend(pos[n] for n in hist[t])

print(f'=== 예측력 검정 ({TEST_N}회차 walk-forward, 관측 {TEST_N*6:,}개) ===\n')
print(f"  {'항목':<12}{'평균순위':>9}{'차이':>9}{'t':>8}{'p':>10}")
print('  ' + '-' * 48)
res = {}
for c in comps:
    a = np.array(ranks[c], float)
    t_st, p = stats.ttest_1samp(a, 23.0)
    res[c] = p
    print(f'  {c:<12}{a.mean():>9.3f}{a.mean()-23:>+9.3f}{t_st:>8.2f}{p:>10.4f}')

print('\n  [Holm-Bonferroni 보정]')
ps = sorted((res[c], c) for c in comps)
m, any_sig = len(ps), False
for i, (p, c) in enumerate(ps):
    thr = 0.05 / (m - i)
    sig = p < thr
    any_sig |= sig
    print(f'    {c:<12} p={p:.4f}  임계값={thr:.4f}  {"유의" if sig else "기각"}')
print(f'\n  → 예측력이 입증된 항목: {"있음" if any_sig else "없음 (0개)"}')
