# -*- coding: utf-8 -*-
"""
[보편 예측 검정] 당첨 수열은 압축되는가

 정보이론의 기본 결과: 어떤 수열이 예측 가능하면 반드시 압축 가능하다.
 역으로 압축되지 않으면 어떤 예측기도 작동할 수 없다. 모델에 의존하지 않는 검정이다.

 각 회차를 C(45,6) 조합의 사전식 순위(0 ~ 8,145,059)로 바꾸면 이론 하한은
 log2(8,145,060) = 22.96 비트다. 실제 수열을 여러 압축기로 압축해
 같은 길이의 균등난수 수열과 비교한다.
"""
import os, zlib, bz2, lzma, importlib.util, sys, io, contextlib
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
import numpy as np
from math import comb, log2
from scipy import stats

def load(n,p):
    sp=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(sp)
    sys.modules[n]=m
    with contextlib.redirect_stdout(io.StringIO()): sp.loader.exec_module(m)
    return m
v12=load('v12', os.path.join(ROOT,'engine','lotto_integrated_v1_2.py'))
hist=v12.safe_load_history(os.path.join(ROOT,'data','latest.csv'), verbose=False)
T=len(hist)

def rank(c):
    """조합 → 사전식 순위"""
    c=sorted(c); r=0; prev=0
    for i,x in enumerate(c):
        for v in range(prev+1,x):
            r+=comb(45-v,5-i)
        prev=x
    return r

def encode(rs):
    """순위 목록을 3바이트씩 빅엔디언으로"""
    b=bytearray()
    for r in rs: b+= int(r).to_bytes(3,'big')
    return bytes(b)

real=encode([rank(r) for r in hist])
H0=T*log2(comb(45,6))/8
print(f'회차 {T} | 원시 {len(real):,}바이트 | 이론 하한 {H0:,.0f}바이트 (회차당 22.96비트)\n')

def sizes(b):
    return (len(zlib.compress(b,9)), len(bz2.compress(b,9)), len(lzma.compress(b,preset=9)))

rz,rb,rl = sizes(real)
print(f"{'대상':<22}{'zlib':>9}{'bz2':>9}{'lzma':>9}{'최소/하한':>11}")
print('-'*62)
print(f"{'실제 당첨 수열':<22}{rz:>9,}{rb:>9,}{rl:>9,}{min(rz,rb,rl)/H0:>10.4f}")

rng=np.random.default_rng(0)
mins=[]
for i in range(12):
    syn=encode([rank(sorted(rng.choice(np.arange(1,46),6,replace=False).tolist())) for _ in range(T)])
    s=sizes(syn); mins.append(min(s))
    if i<2:
        print(f"{'합성 균등난수 '+str(i+1):<22}{s[0]:>9,}{s[1]:>9,}{s[2]:>9,}{min(s)/H0:>10.4f}")
mins=np.array(mins,float)
rmin=min(rz,rb,rl)
print(f"{'합성 12회 평균':<22}{'':>9}{'':>9}{mins.mean():>9,.0f}{mins.mean()/H0:>10.4f}")
print()
z=(rmin-mins.mean())/mins.std()
print(f'=== 판정 ===')
print(f'  실제 최소 압축 {rmin:,}바이트 / 합성 평균 {mins.mean():,.0f}바이트 (표준편차 {mins.std():.0f})')
print(f'  z = {z:+.3f}  → 실제가 합성보다 {"작다(구조 있음)" if z<-2 else "크다" if z>2 else "같다 — 구조 없음"}')
print(f'  실제 회차당 비트 {rmin*8/T:.3f}  (이론 하한 22.96)')
