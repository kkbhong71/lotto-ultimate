# -*- coding: utf-8 -*-
"""
[전수 조사] 과거의 어떤 함수든, 미래의 어떤 속성이든 예측하는가

 과거 파생 피처 N개 × 미래 타깃 M개 = N×M개의 상관을 모두 계산한다.
 로또가 무작위라면 p값들이 균등분포 U(0,1)를 따라야 한다.
 어딘가에 신호가 있으면 작은 p값이 과잉 발생한다 → KS 검정으로 탐지.

 개별 검정 하나가 p<0.05인 것은 우연이지만, 분포 전체가 치우치면 진짜다.
"""
import os, importlib.util, sys, io, contextlib
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
import numpy as np
from scipy import stats
def load(n,p):
    sp=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(sp)
    sys.modules[n]=m
    with contextlib.redirect_stdout(io.StringIO()): sp.loader.exec_module(m)
    return m
v12=load('v12', os.path.join(ROOT,'engine','lotto_integrated_v1_2.py'))
hist=v12.safe_load_history(os.path.join(ROOT,'data','latest.csv'), verbose=False)
H=[sorted(r) for r in hist]; T=len(H)
S=[set(r) for r in H]
PRIME={2,3,5,7,11,13,17,19,23,29,31,37,41,43}

def props(r):
    """한 회차의 속성 추출"""
    n=sorted(r); d=set()
    for a in range(6):
        for b in range(a+1,6): d.add(abs(n[b]-n[a]))
    gaps=[n[i+1]-n[i] for i in range(5)]
    tails=[x%10 for x in n]
    rows=[(x-1)//7 for x in n]
    return {
        'sum':sum(n),'odd':sum(1 for x in n if x%2),'high':sum(1 for x in n if x>=23),
        'ac':len(d)-5,'max':n[-1],'min':n[0],'range':n[-1]-n[0],
        'maxgap':max(gaps),'mingap':min(gaps),'meangap':float(np.mean(gaps)),
        'prime':sum(1 for x in n if x in PRIME),'birth':sum(1 for x in n if x<=31),
        'uniqtail':len(set(tails)),'uniqrow':len(set(rows)),
        'consec':sum(1 for i in range(5) if n[i+1]==n[i]+1),
        'var':float(np.var(n)),'median':float(np.median(n)),
    }
P=[props(r) for r in H]
KEYS=list(P[0].keys())

# ── 과거 파생 피처 ───────────────────────────────────────────
feats={}
arr={k:np.array([p[k] for p in P],float) for k in KEYS}
for k in KEYS:
    a=arr[k]
    feats[f'{k}_lag1']=np.concatenate([[np.nan],a[:-1]])
    feats[f'{k}_lag2']=np.concatenate([[np.nan]*2,a[:-2]])
    feats[f'{k}_lag3']=np.concatenate([[np.nan]*3,a[:-3]])
    for w in [5,20,100]:
        mv=np.full(T,np.nan)
        for t in range(w,T): mv[t]=a[t-w:t].mean()
        feats[f'{k}_ma{w}']=mv
    sd=np.full(T,np.nan)
    for t in range(50,T): sd[t]=a[t-50:t].std()
    feats[f'{k}_sd50']=sd
# 번호 수준 파생
ov=np.full(T,np.nan); ov2=np.full(T,np.nan)
for t in range(1,T): ov[t]=len(S[t-1]&S[t-2]) if t>=2 else np.nan
for t in range(3,T): ov2[t]=len(S[t-1]&S[t-3])
feats['prev_overlap12']=ov; feats['prev_overlap13']=ov2
hot=np.full(T,np.nan)
for t in range(50,T):
    c=np.zeros(46)
    for r in H[t-50:t]:
        for n in r: c[n]+=1
    hot[t]=c.max()
feats['hot50_max']=hot
feats['round_mod7']=np.arange(T)%7
feats['round_mod13']=np.arange(T)%13
feats['round_mod52']=np.arange(T)%52

# ── 미래 타깃 ────────────────────────────────────────────────
targets={k:arr[k] for k in KEYS}
targets['overlap_prev']=np.array([len(S[t]&S[t-1]) if t>0 else np.nan for t in range(T)])

print(f'피처 {len(feats)}종 × 타깃 {len(targets)}종 = {len(feats)*len(targets)}개 검정')
pv=[]; best=[]
for fn,fv in feats.items():
    for tn,tvv in targets.items():
        m=~(np.isnan(fv)|np.isnan(tvv))
        m[:250]=False
        if m.sum()<200: continue
        r,p=stats.pearsonr(fv[m],tvv[m])
        pv.append(p); best.append((p,abs(r),fn,tn,m.sum()))
pv=np.array(pv)
print(f'유효 검정 {len(pv)}개\n')
print('=== p값 분포가 균등한가 (무작위라면 균등해야 함) ===')
ks,pks=stats.kstest(pv,'uniform')
print(f'  KS 검정  D={ks:.4f}  p={pks:.4f}  → {"⚠ 치우침 — 신호 가능성" if pks<0.05 else "균등 — 신호 없음"}')
print(f'  p<0.05 비율 {np.mean(pv<0.05)*100:.2f}%  (무작위 기대 5.00%)')
print(f'  p<0.01 비율 {np.mean(pv<0.01)*100:.2f}%  (무작위 기대 1.00%)')
print()
bon=0.05/len(pv)
sig=[b for b in best if b[0]<bon]
print(f'=== Bonferroni 보정 (임계값 {bon:.2e}) ===')
print(f'  통과한 검정: {len(sig)}개')
print()
print('=== 가장 작은 p값 상위 8개 (보정 전) ===')
for p,r,fn,tn,n in sorted(best)[:8]:
    print(f'  {fn:<18} → {tn:<14} r={r:+.3f}  p={p:.5f}  n={n}  '
          f'{"보정 통과" if p<bon else "우연 수준"}')
