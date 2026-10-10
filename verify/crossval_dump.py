# -*- coding: utf-8 -*-
"""
Python 기준값 생성 — JS 교차검증용.
json 은 repr 기반이라 부동소수점이 무손실 왕복한다.

  python3 crossval_dump.py     → pyref.json, pyclu.json
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'engine'))
from bootstrap import load_engine, data            # noqa: E402

v12, p13 = load_engine(patch_backtest=False)
hist = v12.safe_load_history(data('new_1245.csv'), verbose=False)
EngCls = p13.build_engine_class(v12.DataEngine)
eng = EngCls.from_history(hist, v12.Config())
vec = lambda a: [float(x) for x in np.asarray(a)[:46]]

out = {'history': hist, 'vectors': {
    'freq': vec(eng.frequency_analysis['weighted']),
    'gap': vec(eng.gap_analysis['score']),
    'pairStr': vec(eng.pair_analysis['number_pair_strength']),
    'mom': vec(eng.momentum_analysis['momentum']),
    'zone': vec(eng.zone_analysis['zone_score']),
    'markov': vec(eng.markov_analysis['score']),
    'markovRaw': vec(eng.markov_analysis['raw']),
    'assoc': vec(eng.assoc_analysis['score']),
    'assocRaw': vec(eng.assoc_analysis['raw']),
    'recency': vec(eng.recency_analysis['score']),
    'final': vec(eng.final_scores)}}

rng = np.random.default_rng(2026)
combos, pops, logs, filts = [], [], [], []
fs = v12.FilterSystem(eng)
for _ in range(3000):
    c = sorted(int(x) for x in rng.choice(np.arange(1, 46), 6, replace=False))
    combos.append(c)
    pops.append(float(p13.popularity_score(c, eng.last_draw)))
    logs.append(float(p13.popularity_log(c, eng.last_draw)))
    ok, why = fs.apply_all(c, track=False)
    filts.append(None if ok else why)
out.update(combos=combos, pops=pops, logs=logs, filters=filts)

units = []
for n in [1, 2, 7, 8, 9, 15, 16, 127, 128, 129, 200, 1000]:
    a = (rng.standard_normal(n) * 1e6).tolist()
    units.append({'n': n, 'a': a, 'sum': float(np.sum(a)), 'mean': float(np.mean(a)),
                  'p25': float(np.percentile(a, 25)), 'p50': float(np.percentile(a, 50)),
                  'p75': float(np.percentile(a, 75)), 'var': float(np.var(a))})
out['units'] = units
json.dump(out, open(os.path.join(HERE, 'pyref.json'), 'w'))

# 클러스터 기준값 — 웹앱은 sklearn 미설치 경로와 같은 알고리즘을 쓴다
v12.HAS_SKLEARN = False
geo = v12.GeometricAnalyzer(eng, 5)
probe = [sorted(int(x) for x in rng.choice(np.arange(1, 46), 6, replace=False))
         for _ in range(2000)]
json.dump({'top': [int(x) for x in geo.top_clusters],
           'probe': probe,
           'ids': [int(geo.get_cluster_id(c)) for c in probe],
           'bonus': [float(geo.get_cluster_bonus(c)) for c in probe]},
          open(os.path.join(HERE, 'pyclu.json'), 'w'))

print(f'기준값 생성 완료 — 회차 {len(hist)}, 벡터 11종, 조합 3000, 단위테스트 12')
print('  pyref.json, pyclu.json')
