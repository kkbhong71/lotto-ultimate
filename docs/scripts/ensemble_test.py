# -*- coding: utf-8 -*-
"""[최대 동원 F] 모든 방법을 피처로 합친 통합 앙상블 + 양성 대조군"""
import os, time, importlib.util, sys, io, contextlib
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats
def load(n,p):
    sp=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(sp)
    sys.modules[n]=m
    with contextlib.redirect_stdout(io.StringIO()): sp.loader.exec_module(m)
    return m
v12=load('v12', os.path.join(ROOT,'engine','lotto_integrated_v1_2.py'))
PRIME={2,3,5,7,11,13,17,19,23,29,31,37,41,43}
WIN=[3,5,10,20,30,50,100,200,500]
SPLIT=950; START=250

def build(H):
    """모든 방법을 피처로 펼친다"""
    T=len(H); S=[set(r) for r in H]
    M=np.zeros((T,45),np.float32)
    for t,r in enumerate(H):
        for n in r: M[t,n-1]=1
    cs=np.vstack([np.zeros((1,46),np.float32),np.cumsum(
        np.hstack([np.zeros((T,1),np.float32),M]),axis=0)])
    # 시퀀스 MLP 예측(피처로 주입)
    W=10
    Xs=np.array([M[t-W:t].reshape(-1) for t in range(W,T)],np.float32); Ys=M[W:]
    trm=np.arange(len(Xs))<(SPLIT-W+1)
    sc0=StandardScaler().fit(Xs[trm])
    nn=MLPRegressor(hidden_layer_sizes=(256,128),max_iter=200,early_stopping=True,
                    random_state=0).fit(sc0.transform(Xs[trm]),Ys[trm])
    seq=np.zeros((T,45)); pr=nn.predict(sc0.transform(Xs))
    for i,t in enumerate(range(W,T)): seq[t]=pr[i]

    last=np.full(46,-1); gs=np.zeros(46); gn=np.zeros(46); gmax=np.zeros(46)
    pair=np.zeros((46,46),np.float32); pair2=np.zeros((46,46),np.float32)
    absent=np.zeros(46); adj=np.zeros((45,45)); cnt=np.zeros(45)
    mk2=np.ones((45,45)); mk3=np.ones((45,45))
    X=[];y=[];mt=[]
    for t in range(T):
        if t>=START:
            p1,p2,p3=H[t-1],H[t-2],H[t-3]
            wc={w: cs[t]-cs[max(0,t-w)] for w in WIN}
            # 네트워크
            d=adj.sum(axis=1); _,vec=np.linalg.eigh(adj+np.eye(45)*1e-9)
            ev=np.abs(vec[:,-1]); dn=d/max(d.max(),1e-9); evn=ev/max(ev.max(),1e-9)
            # 마르코프 2·3차
            src2=S[t-1]|S[t-2]; src3=src2|S[t-3]
            m2=np.zeros(45); m3=np.zeros(45)
            for m in src2: m2+=mk2[m-1]
            for m in src3: m3+=mk3[m-1]
            m2/=max(m2.sum(),1e-9); m3/=max(m3.sum(),1e-9)
            # 디리클레 · 엔트로피
            dir5=(cnt+5.0)/(cnt.sum()+225.0)
            c100=M[max(0,t-100):t].sum(axis=0); pp=(c100+1)/(c100.sum()+45)
            ent=-pp*np.log(pp)
            for k in range(1,46):
                g=t-1-last[k] if last[k]>=0 else t
                mg=gs[k]/gn[k] if gn[k]>0 else 7.5
                sdg=np.sqrt(max(gs[k]/max(gn[k],1)-mg**2,0)) if gn[k]>1 else 3.0
                ck=float(cs[t][k])
                f=[k,(k-1)//7,k%10,int(k<=31),int(k%2==1),int(k in PRIME),int(k>=23),
                   ck/max(t,1),g,mg,g/max(mg,1e-9),sdg,gmax[k],absent[k],
                   int(k in S[t-1]),int(k in S[t-2]),int(k in S[t-3]),
                   int(k in S[t-4]) if t>=4 else 0,int(k in S[t-5]) if t>=5 else 0]
                f+=[float(pair[k][m])/max(ck,1) for m in p1]
                f+=[float(pair[k][m])/max(ck,1) for m in p2[:3]]
                f+=[float(pair2[k][m])/max(ck,1) for m in p1[:3]]
                for w in WIN:
                    v=float(wc[w][k]); e=w*6/45
                    f.append(v/w); f.append((v-e)/max(np.sqrt(e*(1-6/45)),1e-9))
                # 신규 방법들
                f+=[seq[t][k-1], m2[k-1], m3[k-1], dn[k-1], evn[k-1], dir5[k-1], ent[k-1]]
                X.append(f); y.append(int(k in S[t])); mt.append(t)
        for n in S[t]:
            if last[n]>=0:
                dd=t-last[n]; gs[n]+=dd; gn[n]+=1; gmax[n]=max(gmax[n],dd)
            last[n]=t; cnt[n-1]+=1
        for n in range(1,46): absent[n]=0 if n in S[t] else absent[n]+1
        for a in H[t]:
            for b in H[t]:
                if a!=b: pair[a][b]+=1; adj[a-1][b-1]+=1
        if t>=1:
            for a in H[t-1]:
                for b in H[t]: pair2[a][b]+=1
        if t>=2:
            for m in (S[t-1]|S[t-2]):
                for n in S[t]: mk2[m-1][n-1]+=1
        if t>=3:
            for m in (S[t-1]|S[t-2]|S[t-3]):
                for n in S[t]: mk3[m-1][n-1]+=1
    return np.array(X,np.float32),np.array(y),np.array(mt)

def run(H,label):
    X,y,mt=build(H)
    tr=mt<=SPLIT; te=mt>SPLIT
    clf=HistGradientBoostingClassifier(max_iter=500,learning_rate=0.06,max_depth=6,
        l2_regularization=1.0,early_stopping=True,validation_fraction=0.15,random_state=0)
    clf.fit(X[tr],y[tr])
    ptr=clf.predict_proba(X[tr])[:,1]; pte=clf.predict_proba(X[te])[:,1]
    hits=[]
    for t in np.unique(mt[te]):
        m=(mt==t)[te]
        hits.append(y[te][m][np.argsort(-pte[m])[:6]].sum())
    h=float(np.mean(hits)); se=0.7839/np.sqrt(len(hits))
    z=(h-0.8)/se; p=2*(1-stats.norm.cdf(abs(z)))
    return X.shape[1],roc_auc_score(y[tr],ptr),roc_auc_score(y[te],pte),h,z,p,float(np.std(pte))

hist=v12.safe_load_history(os.path.join(ROOT,'data','latest.csv'), verbose=False)
H=[sorted(r) for r in hist]
t0=time.time()
nf,atr,ate,h,z,p,sd = run(H,'real')
print(f'=== F. 통합 앙상블 (피처 {nf}종: 기존 49 + 신규 7) ===')
print(f'  학습 AUC {atr:.4f} | 검증 AUC {ate:.4f}')
print(f'  상위6 적중 {h:.4f}  z={z:+.3f}  p={p:.4f}  → '
      f'{"⚠ 신호 검출" if p<0.05 and z>0 else "무작위와 동일"}   ({time.time()-t0:.0f}s)')

# 양성 대조군 — 같은 앙상블이 편향을 잡는가
def synth(T,bias,seed):
    rng=np.random.default_rng(seed); out=[]
    for t in range(T):
        w=np.ones(45)
        if bias>0 and t>=2:
            for n in out[t-2]: w[n-1]*=(1+bias)
        w/=w.sum()
        out.append(sorted(int(x) for x in rng.choice(np.arange(1,46),6,replace=False,p=w)))
    return out
print(f'\n=== 통합 앙상블의 검정력 (합성 데이터) ===')
for bias in [0.0,0.5,1.0]:
    _,a1,a2,hh,zz,pp,sd2 = run(synth(len(H),bias,42),'syn')
    print(f'  편향 {bias*100:>3.0f}%  검증 AUC {a2:.4f}  적중 {hh:.4f}  z={zz:+.2f}  '
          f'예측SD {sd2:.5f}  → {"신호 검출 ✅" if pp<0.05 and zz>0 else "검출 안 됨"}')
