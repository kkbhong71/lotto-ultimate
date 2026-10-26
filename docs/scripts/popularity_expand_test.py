# -*- coding: utf-8 -*-
"""
[인기도 모형 확장] 4피처 → 5피처 (gap_var 추가)

  후보 13종을 홀드아웃 로그우도로 전진 선택하고, 음성 대조군(자동 당첨자)으로
  검증한다. small(1~9 개수)은 자동 당첨자에서도 유의해(p=0.049) 제외했다.

  필요 데이터: data/latest.csv, data/dhlottery_all.json
    curl -o data/dhlottery_all.json \
      https://raw.githubusercontent.com/smok95/lotto/master/results/all.json
"""

import os, json, importlib.util, sys, io, contextlib
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
import numpy as np, pandas as pd, statsmodels.api as sm
from collections import Counter
from scipy.special import gammaln
# -*- coding: utf-8 -*-
"""
[인기도 모형 확장] 사람의 선택 습관을 더 잘 포착할 수 있는가

 현재 4피처: 합계 · 용지 행 집중 · 끝수 반복 · 직전회차 중복
 후보 피처를 대거 추가해 홀드아웃 로그우도로 선택한다.
 (예측이 아니라 '사람이 뭘 고르는가'의 모형이므로 신호가 실재한다)
"""
import os, json, importlib.util, sys, io, contextlib
import numpy as np, pandas as pd, statsmodels.api as sm
from collections import Counter
from scipy.special import gammaln

def load(n,p):
    sp=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(sp)
    sys.modules[n]=m
    with contextlib.redirect_stdout(io.StringIO()): sp.loader.exec_module(m)
    return m
v12=load('v12', os.path.join(ROOT,'engine','lotto_integrated_v1_2.py'))
hist=v12.safe_load_history(os.path.join(ROOT,'data','latest.csv'), verbose=False)
d=json.load(open(os.path.join(ROOT,'data','dhlottery_all.json'),encoding='utf-8'))
d=sorted(d,key=lambda r:r['draw_no'])
C_TOTAL=8_145_060; TICKET=1000

def feats(nums, prev):
    n=sorted(nums)
    rows=Counter((x-1)//7 for x in n)       # 용지 행
    cols=Counter((x-1)%7 for x in n)        # 용지 열
    tails=Counter(x%10 for x in n)
    dec=Counter(min((x-1)//10,4) for x in n)
    gaps=[n[i+1]-n[i] for i in range(5)]
    consec=sum(1 for g in gaps if g==1)
    # 등차 부분수열(길이3 이상) 존재
    arith=0
    for i in range(4):
        for j in range(i+1,5):
            step=n[j]-n[i]
            if step>0 and (n[j]+step) in set(n): arith+=1
    return {
        'sum':sum(n),
        'rowmax':max(rows.values()),
        'tailmax':max(tails.values()),
        'prevov':len(set(n)&set(prev)) if prev else 0,
        'colmax':max(cols.values()),
        'decmax':max(dec.values()),
        'consec':consec,
        'arith':arith,
        'span':n[-1]-n[0],
        'small':sum(1 for x in n if x<=9),
        'mult7':sum(1 for x in n if x%7==0),
        'mult5':sum(1 for x in n if x%5==0),
        'uniqtail':len(set(tails)),
        'uniqrow':len(set(rows)),
        'gapvar':float(np.var(gaps)),
        'has7':int(7 in n),
        'odd':sum(1 for x in n if x%2),
    }
KEYS=list(feats([1,2,3,4,5,6],[]).keys())

rows=[]
for i,r in enumerate(d):
    wc=r.get('winners_combination')
    if not wc: continue
    a,s_,m_=wc.get('auto',0),wc.get('semi_auto',0),wc.get('manual',0)
    if a+s_+m_==0: continue
    prev=d[i-1]['numbers'] if i>0 else []
    f=feats(r['numbers'],prev)
    rows.append({'draw_no':r['draw_no'],'manual':m_,'total1':a+s_+m_,
                 'sales':r['total_sales_amount'],**f})
df=pd.DataFrame(rows)
df['man_frac']=(df.manual/df.total1).rolling(101,center=True,min_periods=15).mean()
df['tickets']=df.sales/TICKET
df['n_manual']=df.tickets*df.man_frac
df['offset']=np.log(df.n_manual/C_TOTAL)
df=df.dropna(subset=['offset'])

# 조합공간 기준 분포로 표준화
rng=np.random.default_rng(42)
ref={k:[] for k in KEYS}
pr=sorted(rng.choice(np.arange(1,46),6,replace=False).tolist())
for _ in range(200000):
    c=sorted(int(x) for x in rng.choice(np.arange(1,46),6,replace=False))
    p=sorted(int(x) for x in rng.choice(np.arange(1,46),6,replace=False))
    f=feats(c,p)
    for k in KEYS: ref[k].append(f[k])
REF={k:(float(np.mean(ref[k])),float(np.std(ref[k]))) for k in KEYS}
for k in KEYS:
    mu,sd=REF[k]; df['z_'+k]=(df[k]-mu)/max(sd,1e-9)

TR=df[df.draw_no<=1045]; TE=df[df.draw_no>1045]
def fit(cols,data=TR):
    X=sm.add_constant(data[['z_'+c for c in cols]].values)
    return sm.GLM(data['manual'].values,X,family=sm.families.Poisson(),
                  offset=data['offset'].values).fit(scale='X2')
def holdout_ll(cols):
    m=fit(cols)
    X=sm.add_constant(TE[['z_'+c for c in cols]].values)
    mu=np.exp(X@m.params+TE['offset'].values)
    y=TE['manual'].values
    return float(np.sum(y*np.log(np.maximum(mu,1e-12))-mu-gammaln(y+1)))



BASE=['sum','rowmax','tailmax','prevov']
FINAL=BASE+['gapvar']
TR=df[df.draw_no<=1045]; TE=df[df.draw_no>1045]
def fit(cols,data=TR,resp='manual',off='offset'):
    X=sm.add_constant(data[['z_'+c for c in cols]].values)
    return sm.GLM(data[resp].values,X,family=sm.families.Poisson(),
                  offset=data[off].values).fit(scale='X2')
def hll(cols):
    m=fit(cols); X=sm.add_constant(TE[['z_'+c for c in cols]].values)
    mu=np.exp(X@m.params+TE['offset'].values); y=TE['manual'].values
    return float(np.sum(y*np.log(np.maximum(mu,1e-12))-mu-gammaln(y+1)))
print(f'홀드아웃 로그우도  4피처 {hll(BASE):.3f} → 5피처 {hll(FINAL):.3f}  (Δ{hll(FINAL)-hll(BASE):+.3f})')

# 전체 데이터로 최종 적합
ALL=df
m=fit(FINAL,ALL)
B=3000; rs=np.random.default_rng(11); boot=[]
for _ in range(B):
    s=ALL.iloc[rs.choice(len(ALL),len(ALL),replace=True)]
    try: boot.append(fit(FINAL,s).params)
    except Exception: pass
boot=np.array(boot)
print(f'\n=== 최종 5피처 모형 (전체 {len(ALL)}회차, scale={m.scale:.2f}) ===')
print(f"  {'피처':<10}{'계수':>10}{'1SD당':>9}{'95% CI':>22}{'p':>9}")
print('  '+'-'*60)
coef={}
for i,c in enumerate(FINAL):
    b=m.params[i+1]; lo,hi=np.percentile(boot[:,i+1],[2.5,97.5])
    coef[c]=float(b)
    print(f'  {c:<10}{b:>10.4f}{(np.exp(b)-1)*100:>8.1f}%'
          f"{f'[{(np.exp(lo)-1)*100:+.1f}%,{(np.exp(hi)-1)*100:+.1f}%]':>22}{m.pvalues[i+1]:>9.4f}")

# 음성 대조군
ALL2=ALL.copy()
ALL2['nauto']=ALL2.total1-ALL2.manual
ALL2['offset_auto']=np.log(ALL2.tickets*(1-ALL2.man_frac)/C_TOTAL)
mn=fit(FINAL,ALL2,resp='nauto',off='offset_auto')
print(f'\n=== 음성 대조군 (자동 당첨자) ===')
bad=0
for i,c in enumerate(FINAL):
    p=mn.pvalues[i+1]; bad+= p<0.05
    print(f'  {c:<10}{(np.exp(mn.params[i+1])-1)*100:>+7.1f}%  p={p:.4f}  {"⚠ 유의" if p<0.05 else "✅ 무효과"}')
print(f'  → {"결함 의심" if bad else "정상 — 전 항목 무효과"}')

print(f'\n=== 코드에 넣을 값 ===')
MAP={'sum':'sum','rowmax':'same_row_max','tailmax':'tail_repeat',
     'prevov':'prev_overlap','gapvar':'gap_var'}
print('POP_COEF = {' + ', '.join(f"'{MAP[k]}': {coef[k]:.4f}" for k in FINAL) + '}')
print('POP_REF  = {' + ', '.join(f"'{MAP[k]}': ({REF[k][0]:.4f}, {REF[k][1]:.4f})" for k in FINAL) + '}')
