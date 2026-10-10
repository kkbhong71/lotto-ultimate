# 이식 기록 — Python ↔ JavaScript 비트 동일성

목표는 "결과가 비슷하다"가 아니라 **판정에 쓰이는 값이 비트 단위로 같다**였습니다.
달성했지만, 가는 길에 두 번 막혔습니다. 둘 다 흔히 놓치는 함정이라 기록해 둡니다.

---

## 함정 1 — numpy의 pairwise summation

`np.sum()`과 `np.mean()`은 **좌→우 단순 누적이 아닙니다.** 8개씩 부분합을 만들어
트리 형태로 합칩니다(pairwise summation). 큰 배열에서 반올림 오차 누적을 줄이기
위한 설계인데, 결과적으로 **덧셈 순서가 달라집니다.**

JavaScript에서 이렇게 짜면 반드시 어긋납니다.

```javascript
const sum = a.reduce((x, y) => x + y, 0);   // ❌ Python과 다른 값
```

numpy 소스의 알고리즘을 그대로 복제해야 합니다.

```javascript
function npSum(a, start, n) {
  if (n < 8) {                       // 단순 루프
    let res = 0;
    for (let i = 0; i < n; i++) res += a[start + i];
    return res;
  }
  if (n <= 128) {                    // 8-way 언롤 후 트리 합산
    const r = [a[start],a[start+1],a[start+2],a[start+3],
               a[start+4],a[start+5],a[start+6],a[start+7]];
    const lim = n - (n % 8);
    for (let i = 8; i < lim; i += 8)
      for (let j = 0; j < 8; j++) r[j] += a[start + i + j];
    let res = ((r[0]+r[1])+(r[2]+r[3])) + ((r[4]+r[5])+(r[6]+r[7]));
    for (let i = lim; i < n; i++) res += a[start + i];
    return res;
  }
  let half = n / 2; half -= (half % 8);   // 재귀 분할
  return npSum(a, start, half) + npSum(a, start + half, n - half);
}
```

`np.percentile`도 주의가 필요합니다. 기본 `method='linear'`의 보간은 단순
`a + (b-a)*t`가 아니라 `t ≥ 0.5`일 때 `b - (b-a)*(1-t)`로 분기합니다.

```javascript
return t < 0.5 ? lo + diff * t : hi - diff * (1 - t);
```

---

## 함정 2 — pow는 구현마다 다르다

이쪽이 더 까다로웠습니다. 첫 교차검증에서 `markov`, `recency`, `final`이
1~4 ULP 어긋났는데, 지수 6개를 샘플링해 비교했을 때는 전부 일치해서
가설을 한 번 접었습니다. 전 범위를 돌려보고서야 드러났습니다.

```
0.97^k     1,245개 중 100개 불일치  (첫 불일치 k=6)
0.999^k    1,245개 중 131개 불일치  (첫 불일치 k=16)
```

```javascript
Math.pow(0.97, 6)  // 0.8329720049289999
0.97 ** 6          // 0.8329720049289998   (Python)
```

**IEEE-754는 `pow`의 정확한 반올림을 요구하지 않습니다.** 둘 다 규격 위반이
아니며, 플랫폼·런타임 버전에 따라 달라질 수도 있습니다.

### 해결 — 반복 곱셈

곱셈은 IEEE-754가 정확히 규정합니다. 감쇠 계수를 거듭제곱 대신 누적 곱으로
만들면 두 언어가 비트 단위로 같아집니다.

```python
def _decay_powers(base, n):
    out = np.empty(max(n, 1), dtype=float)
    p = 1.0
    for i in range(max(n, 1)):
        out[i] = p
        p *= base
    return out
```

```javascript
function decayPowers(base, n) {
  const m = Math.max(n, 1), out = new Float64Array(m);
  let p = 1;
  for (let i = 0; i < m; i++) { out[i] = p; p *= base; }
  return out;
}
```

검증: `0.97^k`, `0.999^k` 각 1,245개 전부 비트 동일.

주의할 점은 이것이 **Python 쪽 값도 바꾼다**는 것입니다. 반복 곱셈은 `pow`와
다른 값을 내므로, 양쪽을 함께 바꿔야 합니다. 한쪽만 바꾸면 문제가 옮겨갈 뿐입니다.

---

## 함정 3 — exp는 우회가 답이다

`pow`와 같은 이유로 `exp`도 1 ULP 어긋납니다(3,000조합 중 228건). `exp`는
반복 곱셈 같은 대체 경로가 없어서, 제거하는 대신 **판정 경로에서 빼냈습니다.**

인기도는 순위 비교와 임계값 판정에만 쓰이고, `exp`는 단조증가 변환입니다.
따라서 로그 스케일에서 비교하면 결과가 완전히 같으면서 `exp`가 사라집니다.

```
popularity_log(nums, prev)    판정용 — 순수 산술, 비트 동일
popularity_score(nums, prev)  표시용 — exp 적용, 1 ULP 차이 허용
```

게이트 임계값도 로그 스케일에서 계산하므로 판정 결과가 양 언어에서 동일합니다.

---

## 의도적으로 남긴 차이

| 항목 | Python | JavaScript | 이유 |
|---|---|---|---|
| 클러스터링 | KMeans + Silhouette (sklearn 있을 때) | 합계 3분위 | 브라우저에서 sklearn 사용 불가 |
| 난수원 | `numpy.random.Generator` (`os.urandom` 시드) | `crypto.getRandomValues` | 플랫폼 표준 |

클러스터링은 Python의 **sklearn 미설치 fallback 경로와 동일한 알고리즘**이므로
교차검증 대상에 포함했습니다. 웹앱 결과를 Python에서 재현하려면
`v12.HAS_SKLEARN = False`로 두고 실행하면 됩니다.

난수는 스트림이 다르므로 생성되는 조합 자체는 당연히 다릅니다. 교차검증은
**결정론적 계산 레이어**(점수·필터·인기도·클러스터)를 대상으로 합니다.
생성은 확률적이므로 비트 비교의 대상이 아닙니다.

---

## 검증 결과

```
[1] 수치 기본 함수      72/72     비트동일  ✅
[2] 점수 벡터 11종      495/495   비트동일  ✅
[3] popularityLog       3000/3000 비트동일  ✅
[4] 조합 필터 12종      불일치 0/3000      ✅
[5] 클러스터 레이어     불일치 0/2000      ✅
```

## 재실행

```bash
cd verify
python3 crossval_dump.py    # Python 기준값 생성
node verify_port.js         # index.html에서 엔진 추출 후 대조
```

`verify_port.js`는 엔진 사본을 두지 않고 `index.html`에서 직접 추출합니다.
검증한 코드와 배포되는 코드가 항상 같은 것임을 보장하기 위해서입니다.
