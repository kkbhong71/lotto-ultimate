# -*- coding: utf-8 -*-
"""[양성 대조군 v2] 편향을 크게 키우고, 피처 생성을 누적합으로 가속"""
import os, numpy as np, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy import stats
WINDOWS=[5,10,20,50,100,200]

def make(T,bias,seed):
    rng=np.random.default_rng(seed); H=[]
    for t in range(T):
        w=np.ones(45)
        if bias>0 and t>=2:
            for n in H[t-2]: w[n-1]*=(1+bias)
        w/=w.sum()
        H.append(set(int(x) for x in rng.choice(np.arange(1,46),6,replace=False,p=w)))
    return H

def featurize(H):
    T=len(H)
    M=np.zeros((T,46),dtype=np.float32)
    for t,r in enumerate(H):
        for n in r: M[t,n]=1
    cs=np.vstack([np.zeros((1,46),np.float32),np.cumsum(M,axis=0)])  # cs[t] = 0..t-1 합
    last=np.full(46,-1); gs=np.zeros(46); gn=np.zeros(46)
    pair=np.zeros((46,46),np.float32)
    X=[];y=[];mt=[]
    for t in range(T):
        if t>=250:
            prev,prev2,prev3=H[t-1],H[t-2],H[t-3]
            wc={w: cs[t]-cs[max(0,t-w)] for w in WINDOWS}
            for k in range(1,46):
                g=t-1-last[k] if last[k]>=0 else t
                mg=gs[k]/gn[k] if gn[k]>0 else 7.5
                ps=float(sum(pair[k][m] for m in prev))
                ck=float(cs[t][k])
                f=[k,ck/max(t,1),g,g/max(mg,1e-9),mg,
                   int(k in prev),int(k in prev2),int(k in prev3),
                   ps/max(t,1),ps/max(ck,1),(k-1)//7,k%10,int(k<=31),int(k%2==1)]
                f+=[float(wc[w][k])/w for w in WINDOWS]
                X.append(f); y.append(int(k in H[t])); mt.append(t)
        for n in H[t]:
            if last[n]>=0: gs[n]+=t-last[n]; gn[n]+=1
            last[n]=t
        for a in H[t]:
            for b in H[t]:
                if a!=b: pair[a][b]+=1
    return np.array(X,np.float32),np.array(y),np.array(mt)

def evaluate(H):
    X,y,mt=featurize(H)
    tr=mt<=950; te=mt>950
    clf=HistGradientBoostingClassifier(max_iter=500,learning_rate=0.06,max_depth=6,
        l2_regularization=1.0,early_stopping=True,validation_fraction=0.15,random_state=0)
    clf.fit(X[tr],y[tr])
    p=clf.predict_proba(X[te])[:,1]
    auc=roc_auc_score(y[te],p)
    hits=[]
    for t in np.unique(mt[te]):
        m=(mt==t)[te]
        hits.append(y[te][m][np.argsort(-p[m])[:6]].sum())
    hits=np.array(hits,float)
    return auc,hits.mean(),len(hits)

print('각 조건 2회 반복, 검증 296회차')
print(f"{'조건':<24}{'검증 AUC':>10}{'상위6 적중':>11}   판정")
print('-'*54)
T=1247; se=0.7839/np.sqrt(296)
for bias,lab in [(0.0,'편향 없음 (음성)'),(0.25,'편향 25%'),(0.50,'편향 50%'),
                 (1.00,'편향 100%'),(2.00,'편향 200%')]:
    A=[];Hh=[]
    for sd in [42,7]:
        a,h,n=evaluate(make(T,bias,sd)); A.append(a); Hh.append(h)
    am=np.mean(A); hm=np.mean(Hh)
    z=(hm-0.8)/(se/np.sqrt(2))
    det = (am>0.52) or (z>1.96)
    print(f'{lab:<24}{am:>10.4f}{hm:>11.4f}   {"신호 검출 ✅" if det else "검출 안 됨"}')
