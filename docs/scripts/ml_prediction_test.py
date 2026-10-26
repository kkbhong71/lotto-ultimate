# -*- coding: utf-8 -*-
"""[강화 검정] 피처 50종 + 모델 3종으로 예측 가능성 재시도"""
import os, time, importlib.util, sys, io, contextlib
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
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
PRIME={2,3,5,7,11,13,17,19,23,29,31,37,41,43}
WIN=[3,5,10,20,30,50,100,200,500]

M=np.zeros((T,46),np.float32)
for t,r in enumerate(H):
    for n in r: M[t,n]=1
cs=np.vstack([np.zeros((1,46),np.float32),np.cumsum(M,axis=0)])

X=[];y=[];mt=[]
last=np.full(46,-1); gs=np.zeros(46); gn=np.zeros(46); gmax=np.zeros(46)
pair=np.zeros((46,46),np.float32); pair2=np.zeros((46,46),np.float32)
absent=np.zeros(46)
for t in range(T):
    if t>=250:
        p1,p2,p3=H[t-1],H[t-2],H[t-3]
        wc={w: cs[t]-cs[max(0,t-w)] for w in WIN}
        for k in range(1,46):
            g=t-1-last[k] if last[k]>=0 else t
            mg=gs[k]/gn[k] if gn[k]>0 else 7.5
            sdg=np.sqrt(max(gs[k]/max(gn[k],1)-mg**2,0)) if gn[k]>1 else 3.0
            ck=float(cs[t][k])
            f=[k,(k-1)//7,k%10,int(k<=31),int(k%2==1),int(k in PRIME),int(k>=23),
               ck/max(t,1), g, mg, g/max(mg,1e-9), sdg, gmax[k], absent[k],
               int(k in S[t-1]),int(k in S[t-2]),int(k in S[t-3]),
               int(k in S[t-4]) if t>=4 else 0, int(k in S[t-5]) if t>=5 else 0]
            # 직전 6개 번호 각각과의 동반출현
            f+= [float(pair[k][m])/max(ck,1) for m in p1]
            f+= [float(pair[k][m])/max(ck,1) for m in p2[:3]]
            f+= [float(pair2[k][m])/max(ck,1) for m in p1[:3]]
            # 다중 윈도우 빈도 + z스코어
            for w in WIN:
                v=float(wc[w][k]); e=w*6/45
                f.append(v/w); f.append((v-e)/max(np.sqrt(e*(1-6/45)),1e-9))
            X.append(f); y.append(int(k in S[t])); mt.append(t)
    for n in S[t]:
        if last[n]>=0:
            d=t-last[n]; gs[n]+=d; gn[n]+=1; gmax[n]=max(gmax[n],d)
        last[n]=t
    for n in range(1,46):
        absent[n]=0 if n in S[t] else absent[n]+1
    for a in H[t]:
        for b in H[t]:
            if a!=b: pair[a][b]+=1
    if t>=1:
        for a in H[t-1]:
            for b in H[t]:
                pair2[a][b]+=1

X=np.array(X,np.float32); y=np.array(y); mt=np.array(mt)
print(f'표본 {X.shape[0]:,}행 × {X.shape[1]}피처 (기존 20종 → {X.shape[1]}종)')
tr=mt<=950; te=mt>950
print(f'학습 {tr.sum():,} / 검증 {te.sum():,}\n')
se=0.7839/np.sqrt(len(np.unique(mt[te])))

def topk_hits(p):
    h=[]
    for t in np.unique(mt[te]):
        m=(mt==t)[te]
        h.append(y[te][m][np.argsort(-p[m])[:6]].sum())
    return float(np.mean(h))

sc=StandardScaler().fit(X[tr])
Xtr,Xte=sc.transform(X[tr]),sc.transform(X[te])
models=[
 ('HistGradientBoosting', HistGradientBoostingClassifier(max_iter=600,learning_rate=0.04,
     max_depth=8,l2_regularization=2.0,early_stopping=True,validation_fraction=0.15,random_state=0), False),
 ('RandomForest(400)', RandomForestClassifier(n_estimators=400,max_depth=12,
     min_samples_leaf=20,n_jobs=-1,random_state=0), False),
 ('MLP (128,64)', MLPClassifier(hidden_layer_sizes=(128,64),max_iter=120,
     early_stopping=True,random_state=0), True),
 ('LogisticRegression', LogisticRegression(max_iter=1000,C=0.5), True),
]
print(f"{'모델':<24}{'학습 AUC':>10}{'검증 AUC':>10}{'상위6 적중':>11}{'z':>8}   판정")
print('-'*72)
for name,clf,scaled in models:
    t0=time.time()
    A,B=(Xtr,Xte) if scaled else (X[tr],X[te])
    clf.fit(A,y[tr])
    ptr=clf.predict_proba(A)[:,1]; pte=clf.predict_proba(B)[:,1]
    h=topk_hits(pte); z=(h-0.8)/se
    print(f'{name:<24}{roc_auc_score(y[tr],ptr):>10.4f}{roc_auc_score(y[te],pte):>10.4f}'
          f'{h:>11.4f}{z:>+8.2f}   {"⚠ 신호" if z>1.96 else "무작위와 동일"}')
