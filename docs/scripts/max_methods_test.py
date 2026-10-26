# -*- coding: utf-8 -*-
"""
[최대 동원] 8종 점수 외에 적용 가능한 모든 예측 방법

 A. 시퀀스 신경망  — 최근 N회차를 45×N 이진 행렬로 보고 다음 회차를 예측
                     (torch 부재로 LSTM 대신 MLP, 입력은 동일한 시퀀스 윈도우)
 B. 고차 마르코프  — 1차뿐 아니라 2차·3차 전이, 쌍 조건부
 C. 네트워크 분석  — 동반출현 그래프의 차수·고유벡터 중심성·군집계수
 D. 디리클레 사후  — 베이즈 평활 출현확률 (여러 사전강도)
 E. 엔트로피/복잡도 — 최근 윈도우 분포의 Shannon 엔트로피, 번호별 기여
 F. 통합 앙상블    — A~E + 기존 49종을 전부 합친 GBM

 모든 검정은 동일 분할(학습 250~950 / 검증 951~1247)과 동일 지표
 (매 회차 상위 6개 예측의 실제 적중, 무작위 기대 0.8000)로 평가한다.
"""
import os, time, importlib.util, sys, io, contextlib
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats

def load(n,p):
    sp=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(sp)
    sys.modules[n]=m
    with contextlib.redirect_stdout(io.StringIO()): sp.loader.exec_module(m)
    return m
v12=load('v12', os.path.join(ROOT,'engine','lotto_integrated_v1_2.py'))
hist=v12.safe_load_history(os.path.join(ROOT,'data','latest.csv'), verbose=False)
H=[sorted(r) for r in hist]; S=[set(r) for r in H]; T=len(H)
M=np.zeros((T,45),np.float32)
for t,r in enumerate(H):
    for n in r: M[t,n-1]=1
SPLIT=950; START=250
se=0.7839/np.sqrt(T-1-SPLIT)

def report(name, scores_by_round, elapsed=None):
    """scores_by_round[t] = 길이45 점수벡터 → 상위6 적중 평가"""
    hits=[]
    for t in range(SPLIT+1,T):
        s=scores_by_round[t]
        top=np.argsort(-s)[:6]+1
        hits.append(len(set(int(x) for x in top)&S[t]))
    h=float(np.mean(hits)); z=(h-0.8)/se
    p=2*(1-stats.norm.cdf(abs(z)))
    tag='⚠ 신호' if (p<0.05 and z>0) else '무작위와 동일'
    ex=f'{elapsed:6.1f}s' if elapsed else ''
    print(f'  {name:<30}{h:>8.4f}{z:>+8.2f}{p:>9.4f}   {tag} {ex}')
    return h,z,p

print(f'검증 {T-1-SPLIT}회차 | 지표: 상위 6개 예측의 실제 적중 (무작위 0.8000, SE {se:.4f})\n')
print(f"  {'방법':<30}{'적중':>8}{'z':>8}{'p':>9}   판정")
print('  '+'-'*66)

# ── A. 시퀀스 신경망 ──────────────────────────────────────────
t0=time.time()
W=10
Xs=np.array([M[t-W:t].reshape(-1) for t in range(W,T)],np.float32)
Ys=M[W:]
tr=np.arange(len(Xs))<(SPLIT-W+1)
sc=StandardScaler().fit(Xs[tr])
A=np.zeros((T,45))
nn=MLPRegressor(hidden_layer_sizes=(256,128),max_iter=200,early_stopping=True,
                random_state=0).fit(sc.transform(Xs[tr]),Ys[tr])
pr=nn.predict(sc.transform(Xs))
for i,t in enumerate(range(W,T)): A[t]=pr[i]
report(f'A. 시퀀스 MLP (윈도우{W})', A, time.time()-t0)

# ── B. 고차 마르코프 ──────────────────────────────────────────
t0=time.time()
B=np.zeros((T,45))
for order in [2,3]:
    tr1=np.ones((45,45))  # 라플라스
    Bo=np.zeros((T,45))
    for t in range(T):
        if t>=order:
            src=set()
            for o in range(1,order+1): src|=S[t-o]
            v=np.zeros(45)
            for m in src: v+=tr1[m-1]
            Bo[t]=v/max(v.sum(),1e-9)
        if t>=order:
            for o in range(1,order+1):
                for m in S[t-o]:
                    for n in S[t]: tr1[m-1][n-1]+=1
    if order==2: B=Bo
    report(f'B. 마르코프 {order}차', Bo, time.time()-t0 if order==3 else None)

# ── C. 네트워크 중심성 ────────────────────────────────────────
t0=time.time()
C=np.zeros((T,45))
adj=np.zeros((45,45))
for t in range(T):
    if t>=START:
        d=adj.sum(axis=1)
        w,vec=np.linalg.eigh(adj+np.eye(45)*1e-9)
        ev=np.abs(vec[:,-1])
        deg=d/max(d.sum(),1e-9)
        C[t]=0.5*deg/max(deg.max(),1e-9)+0.5*ev/max(ev.max(),1e-9)
    for a in H[t]:
        for b in H[t]:
            if a!=b: adj[a-1][b-1]+=1
report('C. 네트워크 중심성', C, time.time()-t0)

# ── D. 디리클레 사후 ──────────────────────────────────────────
t0=time.time()
for alpha in [0.5, 5.0]:
    D=np.zeros((T,45)); cnt=np.zeros(45)
    for t in range(T):
        D[t]=(cnt+alpha)/(cnt.sum()+45*alpha)
        for n in H[t]: cnt[n-1]+=1
    report(f'D. 디리클레 사후 (α={alpha})', D, time.time()-t0 if alpha==5.0 else None)

# ── E. 엔트로피 기반 ──────────────────────────────────────────
t0=time.time()
E=np.zeros((T,45))
for t in range(T):
    if t>=100:
        c=M[t-100:t].sum(axis=0)
        p=(c+1)/(c.sum()+45)
        E[t]=-p*np.log(p)        # 번호별 엔트로피 기여 — 클수록 '덜 치우침'
report('E. 엔트로피 기여도', E, time.time()-t0)
