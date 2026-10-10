# LOTTO ULTIMATE — 인기 조합 회피 엔진

> **Creative Director** K.G.B. (열정맨) · **Oseong Vibe Coding EduHub**
> 웹앱: https://kkbhong71.github.io/lotto-ultimate/

로또 6/45 당첨번호 1,245회차(2002-12-07 ~ 2026-10-10)를 분석하는 도구입니다.
Python 엔진(Colab)과 단일 HTML 웹앱(서버 불필요) 두 가지로 제공됩니다.

> **이 저장소가 정본입니다.** 이전 버전은
> [`lotto-lab`](https://github.com/kkbhong71/lotto-lab)(웹앱 v2.1 / Python v1.1)에
> 보존되어 있습니다. 예측력 검정, 인기도 모형 실측 재적합, Python↔JS 비트 단위
> 교차검증은 모두 이 저장소에서 처음 수행되었습니다.

---

## 이 도구가 하지 못하는 것

**당첨 확률을 높이지 못합니다.** 1등 확률은 1/8,145,060이고 회차는 서로 독립이므로,
과거 데이터로 다음 번호를 예측하는 것은 수학적으로 불가능합니다.

이건 겸양이 아니라 이 저장소가 **직접 측정한 결과**입니다.

| 점수 항목 | 당첨번호 평균순위 | 기대값 | p |
|---|---|---|---|
| frequency | 23.092 | 23.0 | 0.695 |
| recency | 23.097 | 23.0 | 0.682 |
| gap | 22.635 | 23.0 | 0.123 |
| pair | 23.063 | 23.0 | 0.792 |
| momentum | 22.868 | 23.0 | 0.581 |
| zone | 23.205 | 23.0 | 0.378 |
| markov | 23.159 | 23.0 | 0.497 |
| assoc | 22.891 | 23.0 | 0.647 |
| **FINAL (8종 종합)** | **23.050** | 23.0 | 0.830 |

500회차 walk-forward, 관측 3,000개, 평균순위 0.47 차이까지 탐지 가능한 검정력.
Holm-Bonferroni 보정 후 **유의한 항목 0개**. 자세한 내용은
[docs/BACKTEST_REPORT.md](docs/BACKTEST_REPORT.md).

## 이 도구가 하는 것

**대중이 덜 고르는 조합을 선택합니다.** 당첨 확률은 그대로지만, 당첨됐을 때
나눠 가질 사람 수가 줄어 1인당 실수령액이 올라갑니다. 수학적으로 유일하게
남아 있는 레버입니다.

981회차(262~1245)의 **수동 1등 당첨자 2,878명**으로 인기도 모형을 적합했습니다.

| 조합 특성 | 인기도 변화(1SD당) | 95% CI |
|---|---|---|
| 합계가 클수록 | −18.4% | [−25.8%, −10.7%] |
| 용지 같은 행에 몰릴수록 | −10.0% | [−18.2%, −1.0%] |
| 끝수가 반복될수록 | +4.9% | [−1.8%, +11.8%] |
| 직전 회차와 겹칠수록 | −7.4% | [−13.6%, −0.9%] |

음성 대조군(자동 당첨자)에서 전 항목 무효과 확인, 홀드아웃 200회차 검증 통과.

**효과 크기** (1등 평균 8명 기준)

| 선택 전략 | 기대 공동당첨자 | 1인당 수령 배수 |
|---|---|---|
| 무작위 자동 | 7.94명 | 1.00x |
| 인기 하위 10% | 5.94명 | **1.35x** |
| 인기 하위 5% | 5.41명 | **1.48x** |
| 인기 하위 1% | 4.57명 | **1.75x** |

---

## 빠른 시작

### 웹앱 (설치 불필요)

https://kkbhong71.github.io/lotto-ultimate/ 에 접속하면 `data/latest.csv`를 자동으로
불러와 바로 사용할 수 있습니다. 더 최신 파일이 있으면 끌어다 놓아 교체하면 됩니다.

`index.html`을 받아 더블클릭해도 동작합니다(이 경우 브라우저 정책상 자동 로드가
막히므로 CSV를 직접 끌어다 놓으세요). 서버도 빌드도 필요 없고, 외부 리소스를
전혀 불러오지 않으며, 데이터는 브라우저 밖으로 나가지 않습니다.

### 주간 운영 — 매주 2분

CSV를 손으로 편집할 필요가 없습니다.

| | 할 일 |
|---|---|
| **토요일 추첨 전** | 번호 생성 → 대상 회차 확인 후 **예측 기록** |
| **추첨 후** | `1-B 새 회차 추가`에 당첨번호 6개만 붙여넣고 **추가** |
| | → 분석 갱신 · 예측 **자동 채점** · 다음 회차 자동 입력 준비 |
| **보관** | **CSV 내려받기** → `data/latest.csv`에 덮어 커밋 |

회차 번호와 추첨일은 자동으로 채워지고, 입력은 추가 전에 검증됩니다 —
1~45 범위, 중복 번호, 6개 미만, 회차 건너뜀, 과거와 동일한 조합을 모두 걸러냅니다.

내려받는 파일 이름은 회차와 무관하게 항상 `latest.csv`입니다. `data/latest.csv`에
그대로 덮어 커밋하면 다음 접속부터 자동 로드에 반영되므로, 파일을 따로 챙길 일이
없습니다. 과거 회차 데이터는 git 이력에 남습니다.

> 추가한 회차는 브라우저 메모리에만 있습니다. 커밋하거나 CSV를 내려받지 않고
> 새로고침하면 사라집니다. 예측 기록도 그 브라우저의 localStorage에만 저장되므로,
> 기기를 옮기기 전에 `CSV 내보내기`로 백업하세요.

### Python (Colab / 로컬)

```bash
pip install -r requirements.txt
python3 run.py                       # 5세트, 인기 하위 10%
python3 run.py --count 10 --pct 5    # 10세트, 인기 하위 5%
python3 run.py --seed 12345          # 재현 실행
```

코드에서 직접 쓰려면:

```python
import sys; sys.path.insert(0, 'engine')
from bootstrap import load_engine, data

v12, p13 = load_engine()
hist = v12.safe_load_history(data('latest.csv'))

out = p13.predict_v131(hist, count=5, pop_max_pct=10.0)
for s in out['sets']:
    print(', '.join(map(str, s)))
print('재현용 seed:', out['seed'])
```

`seed=None`(기본)이면 OS 엔트로피로 매번 다른 조합이 나오고, 정수를 주면 완전히
재현됩니다. 출력된 `seed`를 적어두면 나중에 같은 결과를 복원할 수 있습니다.

---

## 저장소 구성

```
index.html                              단일 파일 웹앱 (52KB, 외부 의존 없음)
run.py                                  CLI 진입점
engine/
  lotto_integrated_v1_2.py              기반 엔진 (분석·필터·백테스트)
  lotto_v13_patch.py                    v1.3.3 패치 (RNG·인기도·버그 수정)
data/
  latest.csv                            당첨번호 (최신본 — 이 이름 고정)
  popularity_model_dataset.csv          인기도 모형 적합용 981회차
  popularity_model_coefficients.json    적합된 계수
verify/
  crossval_dump.py                      Python 기준값 생성
  verify_port.js                        Python↔JS 교차검증
docs/
  ALGORITHM_GUIDE.md                    알고리즘 해설
  BACKTEST_REPORT.md                    검증 결과 전문
  PORTING_NOTES.md                      이식 기록 (부동소수점 함정)
  scripts/signal_test.py                예측력 검정 재현
  scripts/refit_popularity.py           인기도 모형 재적합 재현
```

## 교차검증 재실행

웹앱과 Python 엔진이 같은 숫자를 내는지 언제든 확인할 수 있습니다.

```bash
pip install -r requirements.txt
cd verify && python3 crossval_dump.py && node verify_port.js
```

`verify_port.js`는 엔진 사본을 따로 두지 않고 `index.html`에서 직접 추출해
대조하므로, 검증한 코드와 배포되는 코드가 항상 같습니다.

현재 상태: **판정 경로 전 항목 비트 단위 동일** (오차 0).

---

## 면책

이 소프트웨어는 당첨을 보장하지 않으며, 당첨 확률을 높이지도 않습니다.
인기도 모형은 기대 수령액에만 영향을 주며, 그 추정에도 불확실성이 있습니다.

복권은 여가로만 즐기고, 감당할 수 있는 범위를 넘지 마세요.
도박 문제로 어려움을 겪고 있다면 **도박문제상담 1336**
(한국도박문제예방치유원, 24시간 무료)으로 연락하실 수 있습니다.

## 라이선스

MIT — [LICENSE](LICENSE)

## 데이터 출처

- 당첨번호: 동행복권 (https://www.dhlottery.co.kr)
- 등수별 당첨자 수·판매액·자동/수동 구분: [smok95/lotto](https://github.com/smok95/lotto)
  (당첨번호 1,245회차를 동행복권 원본과 전수 대조해 불일치 0건 확인)
