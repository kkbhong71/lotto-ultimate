# -*- coding: utf-8 -*-
"""
[검증 2] 인기도 모형 재적합 + 음성 대조군 + 홀드아웃

  log λ_t = log(N_manual_t / 8,145,060) + β0 + x'β      quasi-Poisson

data/popularity_model_dataset.csv 를 입력으로 쓴다
(동행복권 회차별 1등 자동/수동 구분 및 총판매액에서 구축).
"""
import os, sys
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.special import gammaln

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
df = pd.read_csv(os.path.join(ROOT, 'data', 'popularity_model_dataset.csv'))
F = ['sum', 'rowmax', 'tailmax', 'prevov']
PHASE3 = {'sum': -0.049, 'rowmax': -0.074, 'tailmax': 0.059, 'prevov': -0.014}

def fit(cols, d, resp='manual', off='offset'):
    X = sm.add_constant(d[['z_' + c for c in cols]].values)
    return sm.GLM(d[resp].values, X, family=sm.families.Poisson(),
                  offset=d[off].values).fit(scale='X2')

print('=' * 70)
print('  인기도 모형 재적합 — 수동 1등 당첨자 기준')
print('=' * 70)
m = fit(F, df)
B, rs = 3000, np.random.default_rng(11)
boot = np.array([fit(F, df.iloc[rs.choice(len(df), len(df), replace=True)]).params
                 for _ in range(B)])
print(f'  표본 {len(df)}회차 ({df.draw_no.min()}~{df.draw_no.max()}) | '
      f'수동 당첨자 {int(df.manual.sum()):,}명 | 과산포 scale={m.scale:.2f}')
print(f"\n  {'피처':<9}{'계수':>10}{'1SD당':>9}{'95% CI':>22}{'기존값':>9}")
print('  ' + '-' * 60)
for i, c in enumerate(F):
    b = m.params[i+1]
    lo, hi = np.percentile(boot[:, i+1], [2.5, 97.5])
    ci = f'[{(np.exp(lo)-1)*100:+.1f}%,{(np.exp(hi)-1)*100:+.1f}%]'
    print(f'  {c:<9}{b:>10.4f}{(np.exp(b)-1)*100:>8.1f}%{ci:>22}{PHASE3[c]*100:>8.1f}%')

print('\n  [음성 대조군] 자동 당첨자 — 효과가 나오면 모형 결함')
df['n_auto'] = df.tickets * (1 - df.man_frac)
df['offset_auto'] = np.log(df.n_auto / 8_145_060)
mN = fit(F, df, resp='auto', off='offset_auto')
for i, c in enumerate(F):
    b, p = mN.params[i+1], mN.pvalues[i+1]
    print(f'    {c:<9}{(np.exp(b)-1)*100:>+7.1f}%  p={p:.4f}  '
          f'{"⚠️ 유의(문제)" if p < 0.05 else "✅ 무효과"}')

print('\n  [홀드아웃] 1045회차까지 적합 → 최근 200회차 예측')
tr, te = df[df.draw_no <= 1045], df[df.draw_no > 1045]
mT = fit(F, tr)
Xte = sm.add_constant(te[['z_' + c for c in F]].values)
pred = np.exp(Xte @ mT.params + te['offset'].values)
null = np.exp(mT.params[0] + te['offset'].values)
y = te['manual'].values
pll = lambda mu: float(np.sum(y*np.log(np.maximum(mu, 1e-12)) - mu - gammaln(y+1)))
print(f'    검증 {len(te)}회차 (수동 {y.sum()}명)')
print(f'    로그우도 개선 {pll(pred)-pll(null):+.2f} | 상관 {np.corrcoef(pred, y)[0,1]:+.3f}')

print('\n  [모형 없는 확인] 합계 5분위별 실제 수동 당첨률')
df['per_m'] = df.manual / (df.n_manual / 1e6)
df['q'] = pd.qcut(df['sum'], 5, labels=['최저','낮음','중간','높음','최고'])
g = df.groupby('q', observed=True).agg(회차=('manual','size'), 평균합=('sum','mean'),
                                       수동=('manual','sum'), 백만장당=('per_m','mean'))
base = g['백만장당'].mean()
print(f"    {'구간':<7}{'회차':>6}{'평균합':>8}{'수동':>8}{'100만장당':>11}{'상대인기':>10}")
for k, r in g.iterrows():
    print(f'    {k:<7}{int(r.회차):>6}{r.평균합:>8.0f}{int(r.수동):>8}'
          f'{r.백만장당:>11.3f}{r.백만장당/base:>9.2f}x')
