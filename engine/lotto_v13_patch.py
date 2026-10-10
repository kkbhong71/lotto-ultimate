# -*- coding: utf-8 -*-
# =============================================================================
#  LOTTO v1.3 GENERATION PATCH  —  v1.2 위에 얹는 생성·선택 엔진 교체 모듈
# =============================================================================
#  [수정 대상 결함 — 실측으로 확정된 3건]
#   (D1) 전역 고정 시드(np.random.seed(42))
#        → 같은 CSV면 매 실행 100% 동일 출력. 재현성 도구를 실전 생성에 그대로 씀.
#   (D2) 점수 변별력 부족(CV 24%, 최대/최소 ~3배)
#        → np.random.choice가 가중치를 거의 무시하고 PRNG 스트림을 재생.
#          그 결과 '스트림 고정점' 조합이 서로 다른 데이터셋 13%에서 재등장.
#   (D3) algo_bonus 임의 편향(balanced 1.0 / 그 외 0.8)
#        → 4개 알고리즘 균등 투입인데 최종 선정은 balanced+pattern이 76.7% 독식.
#
#  [추가 개선]
#   (E1) 후보 풀 80개 → 수백 개 (5세트를 고르기엔 80개는 선택 여지가 없음)
#   (E2) 인기도 회피 감점 내장 — Phase 3 quasi-Poisson 계수 적용
#        (프로젝트 핵심 원칙: 유일한 수학적 레버 = 공동당첨 회피)
#   (E3) OS 엔트로피 시드 — 인간/소프트웨어 편향 제거 (방법론 4)
#   (E4) fallback 완화 순서 명시 + 어떤 제약이 풀렸는지 로그로 노출
#
#  [사용법]  v1.2 코드를 먼저 실행한 뒤, 이 파일을 같은 세션에서 실행하고
#            predict_v13(history) 또는 run_v13(csv_path) 호출
# =============================================================================

import os
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations


# =============================================================================
# 0. RNG 관리 — (D1) 해결
# =============================================================================
def make_rng(seed=None):
    """
    seed=None  → OS 엔트로피(os.urandom)로 매번 다른 스트림. 실전 생성용 기본값.
    seed=정수  → 완전 재현. 백테스트/교차검증 전용.

    전역 np.random.seed()를 쓰지 않고 Generator 인스턴스를 들고 다니는 이유:
    전역 시드는 모듈 어디서든 덮어써져 '어디서 난수가 고정됐는지' 추적이 불가능하다.
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(8), 'big')
    return np.random.default_rng(seed), seed


# =============================================================================
# 1. 인기도 회피 모델 — (E2)
# =============================================================================
# Phase 3 quasi-Poisson 역산 결과(1SD 변화당 인기도 변화율):
#   sum            -4.9%   합계가 클수록 덜 인기 (생일조합은 합이 작음)
#   same_row_max   -7.4%   OMR 같은 행에 몰릴수록 덜 인기
#   tail_repeat    +5.9%   끝수가 반복되면 더 인기
#   prev_overlap   -1.4%   직전 회차와 겹치면 덜 인기
# 점수가 높을수록 '대중이 많이 고르는 조합' = 당첨 시 분할 위험 → 감점 대상.
# [v1.3.2] 실측 재적합 결과 — 981회차(262~1245) 수동 1등 당첨자 2,878명 기준.
#   offset 포아송 회귀 log λ = log(N_manual/8145060) + β0 + x'β, quasi-Poisson(scale 3.16)
#   음성 대조군(자동 당첨자) 전 항목 무효과 확인, 홀드아웃 200회차 로그우도 +10.66
#   1SD당 인기도 변화: sum -18.4% / same_row_max -10.0% / tail_repeat +4.9% / prev_overlap -7.4%
POP_COEF = {'sum': -0.2033, 'same_row_max': -0.1055,
            'tail_repeat': 0.0474, 'prev_overlap': -0.0768}

# 조합공간 기준 평균/표준편차 — 랜덤 30만 조합으로 실측 (추정치 아님).
# [v1.3.2 정정] 기존 하드코딩값은 same_row_max 평균 1.75(실제 2.20),
#               tail_repeat 평균 1.35(실제 1.88)로 1SD 가까이 틀려 있었다.
#               표준화가 틀리면 계수가 맞아도 점수가 어긋난다.
POP_REF = {'sum': (137.9489, 29.9479), 'same_row_max': (2.2035, 0.5524),
           'tail_repeat': (1.8821, 0.5425), 'prev_overlap': (0.8002, 0.7836)}


def popularity_features(nums, prev_draw):
    """조합 1개의 인기도 피처 4종 추출"""
    s = sum(nums)
    rows = Counter((n - 1) // 7 for n in nums)        # 한국 로또 용지: 1행 7칸
    same_row_max = max(rows.values())
    tails = Counter(n % 10 for n in nums)
    tail_repeat = max(tails.values())
    prev_overlap = len(set(nums) & set(prev_draw)) if prev_draw else 0
    return {'sum': s, 'same_row_max': same_row_max,
            'tail_repeat': tail_repeat, 'prev_overlap': prev_overlap}


def popularity_log(nums, prev_draw):
    """
    인기도의 로그값(= 모형의 선형예측자). 판정 경로는 모두 이 함수를 쓴다.

    [v1.3.3] exp 를 거치지 않는 이유:
      exp 는 IEEE-754 가 정확한 반올림을 요구하지 않아 Python 과 JavaScript 가
      1 ULP 어긋난다(실측 3000조합 중 228건). 인기도는 '순위'와 '임계값 비교'
      에만 쓰이고 exp 는 단조증가 변환이므로, 로그 스케일에서 비교하면 결과가
      완전히 동일하면서 두 언어가 비트 단위로 일치한다.
    """
    f = popularity_features(nums, prev_draw)
    z = 0.0
    for k, coef in POP_COEF.items():
        mu, sd = POP_REF[k]
        z += coef * ((f[k] - mu) / sd)
    return float(z)


def popularity_score(nums, prev_draw):
    """표시용 인기도 배수. 1.0이 평균, >1이면 인기 조합. 판정에는 쓰지 않는다."""
    return float(np.exp(popularity_log(nums, prev_draw)))


# =============================================================================
# 2. 생성 엔진 — v1.2 EnsembleGenerator 교체본
# =============================================================================
def build_generator_class(base_cls):
    """v1.2의 EnsembleGenerator를 상속해 결함 부분만 덮어쓴다."""

    class GeneratorV13(base_cls):

        # 후보를 몇 배수로 뽑을지 — (E1). 80개 → count*CAND_MULT*4
        CAND_MULT = 40
        # 점수 선명화 지수 — (D2). p ∝ score^ALPHA
        SCORE_ALPHA = 2.2

        def __init__(self, engine, geometric=None, rng=None,
                     pop_weight=0.14):
            super().__init__(engine, geometric)
            self.rng = rng if rng is not None else make_rng()[0]
            self.pop_weight = pop_weight
            self.prev_draw = engine.last_draw or []
            self.relax_log = []

        # ── (D2) 가중치 선명화 ───────────────────────────────────────
        def _get_weights(self):
            """
            원본은 정규화된 점수를 그대로 p로 썼다. 그런데 45개 점수의 변동계수가
            24%에 불과해(최대/최소 3배) choice가 사실상 균등추출이 되고,
            고정 시드와 결합하면 '같은 번호가 계속 나오는' 고정점이 생긴다.
            지수 변환으로 모델의 신호를 실제 선택 확률에 반영시킨다.
            """
            w = self.scores[1:].astype(float).copy()
            w = np.clip(w, 1e-12, None)
            w = w ** self.SCORE_ALPHA
            return w / w.sum()

        # ── 공통: 안전 추출 ─────────────────────────────────────────
        def _draw(self, pool_nums, k, w):
            ws = w.sum()
            if ws <= 0 or k <= 0 or k > len(pool_nums):
                return None
            return self.rng.choice(pool_nums, k, replace=False, p=w / ws)

        # ── 알고리즘 1: 가중치 ──────────────────────────────────────
        def _algo_weighted(self, count):
            res, w = [], self._get_weights()
            nums_all = np.arange(1, 46)
            for _ in range(count * 25):
                if len(res) >= count:
                    break
                pick = self._draw(nums_all, 6, w)
                if pick is None:
                    continue
                self._try_add(sorted(int(x) for x in pick), 'weighted', res)
            return res

        # ── 알고리즘 2: 구간 밸런스 ─────────────────────────────────
        def _algo_balanced(self, count):
            res, w = [], self._get_weights()
            zones = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 45)]
            for _ in range(count * 25):
                if len(res) >= count:
                    break
                picks = [1] * 5
                picks[self.rng.integers(0, 5)] += 1
                nums, ok = [], True
                for i, (zs, ze) in enumerate(zones):
                    zn = np.arange(zs, ze + 1)
                    zw = w[zn - 1].copy()
                    got = self._draw(zn, picks[i], zw)
                    if got is None:
                        ok = False
                        break
                    nums.extend(int(x) for x in got)
                if ok and len(set(nums)) == 6:
                    self._try_add(sorted(set(nums)), 'balanced', res)
            return res

        # ── 알고리즘 3: 동반출현 패턴 ───────────────────────────────
        def _algo_pattern(self, count):
            res, w = [], self._get_weights()
            tp = self.engine.pair_analysis['top_pairs'][:30]
            nums_all = np.arange(1, 46)
            for _ in range(count * 25):
                if len(res) >= count:
                    break
                base = (list(tp[self.rng.integers(0, min(15, len(tp)))][0])
                        if tp and self.rng.random() < 0.7 else [])
                cur = set(base)
                tw = w.copy()
                for n in cur:
                    tw[n - 1] = 0.0
                got = self._draw(nums_all, 6 - len(cur), tw)
                if got is None:
                    continue
                cur.update(int(x) for x in got)
                if len(cur) == 6:
                    self._try_add(sorted(cur), 'pattern', res)
            return res

        # ── 알고리즘 4: 다중 클러스터 ───────────────────────────────
        def _algo_multi_cluster(self, count):
            res, w = [], self._get_weights()
            if not self.geometric:
                return res
            tg = [{'cluster': cid,
                   'target_sum': int(self.geometric.cluster_stats[cid]['avg_sum']),
                   'target_spread': int(self.geometric.cluster_stats[cid]['avg_spread'])}
                  for cid in self.geometric.top_clusters]
            if not tg:
                return res
            nums_all = np.arange(1, 46)
            for _ in range(count * 30):
                if len(res) >= count:
                    break
                ch = tg[self.rng.integers(0, len(tg))]      # 균등 — v3.4 수정 유지
                spread = max(5, ch['target_spread'])
                first = int(self.rng.integers(1, max(2, 46 - spread)))
                last = max(first + 5, min(45, first + spread + int(self.rng.integers(-5, 6))))
                seeds = list({first, min(45, last)})
                tw = w.copy()
                for n in seeds:
                    if 1 <= n <= 45:
                        tw[n - 1] = 0.0
                got = self._draw(nums_all, 6 - len(seeds), tw)
                if got is None:
                    continue
                cur = set(seeds) | {int(x) for x in got}
                if len(cur) != 6:
                    continue
                nums = sorted(cur)
                if abs(sum(nums) - ch['target_sum']) <= 35:
                    self._try_add(nums, f"cluster_{ch['cluster']}", res)
            return res

        # ── (D3) 스코어링: algo_bonus 제거 + 인기도 감점 ────────────
        def _score_candidates(self, candidates):
            seen, uniq, rn, rp = set(), [], [], []
            for c in candidates:
                k = tuple(c['nums'])
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(c)
                rn.append(sum(self.scores[n] for n in c['nums']))
                rp.append(sum(self.engine.pair_analysis['pair_score'][a][b]
                              for a, b in combinations(c['nums'], 2)))
            if not uniq:
                return []
            mx_n = max(rn) or 1.0
            mx_p = max(rp) or 1.0

            pops = [popularity_log(c['nums'], self.prev_draw) for c in uniq]
            p_lo, p_hi = min(pops), max(pops)
            p_rng = (p_hi - p_lo) or 1.0

            out = []
            for i, c in enumerate(uniq):
                nums = c['nums']
                s = sum(nums)
                diffs = {abs(nums[b] - nums[a])
                         for a in range(6) for b in range(a + 1, 6)}
                ac = len(diffs) - 5
                zones = [0] * 5
                for n in nums:
                    zones[min((n - 1) // 10, 4)] += 1

                sum_sc = 1.0 if 110 <= s <= 190 else 0.5
                ac_sc = float(np.clip(ac / 10.0, 0, 1))
                num_sc = float(np.clip(rn[i] / mx_n, 0, 1))
                pair_sc = float(np.clip(rp[i] / mx_p, 0, 1))
                bal_sc = float(np.clip(1.0 / (1.0 + np.var(zones)), 0, 1))
                clu_sc = (self.geometric.get_cluster_bonus(nums)
                          if self.geometric else 0.5)
                # 인기도는 낮을수록 좋다 → 1에서 뺀다
                avoid_sc = 1.0 - (pops[i] - p_lo) / p_rng

                total = (num_sc * 0.28 + pair_sc * 0.17 + sum_sc * 0.10
                         + ac_sc * 0.10 + bal_sc * 0.12 + clu_sc * 0.09
                         + avoid_sc * self.pop_weight)

                c['stat'] = {'sum': s, 'ac': ac,
                             'odd': sum(1 for n in nums if n % 2),
                             'high': sum(1 for n in nums if n >= 23),
                             'prime': sum(1 for n in nums if n in base_mod.PRIME_NUMBERS),
                             'cluster_bonus': clu_sc,
                             'cluster_id': (self.geometric.get_cluster_id(nums)
                                            if self.geometric else -1),
                             'pop_log': round(pops[i], 6),
                             'avoid_score': round(avoid_sc, 4)}
                c['total_score'] = round(float(total), 6)
                out.append(c)
            return out

        # ── 풀: 클러스터 + 알고리즘 이중 쿼터 ───────────────────────
        def _build_diverse_pool(self, candidates, count):
            pool_size = max(count * 30, 150)
            srt = sorted(candidates, key=lambda x: x['total_score'], reverse=True)
            by_c, by_a = defaultdict(list), defaultdict(list)
            for c in srt:
                by_c[c['stat'].get('cluster_id', -1)].append(c)
                by_a[c['algo'].split('_')[0]].append(c)
            pool, seen = [], set()
            for grp in (by_c, by_a):                 # 두 축 모두 최소 쿼터 보장
                for _, cs in grp.items():
                    for c in cs[:count * 3]:
                        if id(c) not in seen:
                            pool.append(c)
                            seen.add(id(c))
            for c in srt:
                if len(pool) >= pool_size:
                    break
                if id(c) not in seen:
                    pool.append(c)
                    seen.add(id(c))
            return pool

        # ── (E4) 선택: 제약 완화 순서를 명시하고 기록 ───────────────
        def _select_with_coverage(self, candidates, count):
            if not candidates:
                return []
            pool = self._build_diverse_pool(candidates, count)
            sel, used = [], set()
            c_cnt, a_cnt = Counter(), Counter()
            max_ov = self.engine.config.MAX_SET_OVERLAP
            max_cls = max(1, count // 2)
            max_alg = max(2, (count + 2) // 3)       # 한 알고리즘 독식 방지
            self.relax_log = []

            # 제약 단계: 0=전체 적용, 1=알고리즘 쿼터 해제, 2=클러스터 쿼터 해제,
            #            3=중복 제한까지 해제
            for slot in range(count):
                chosen, level = None, 0
                for level in range(4):
                    best, bv = None, -1.0
                    for c in pool:
                        if any(c is s for s in sel):
                            continue
                        ns = set(c['nums'])
                        cid = c['stat'].get('cluster_id', -1)
                        alg = c['algo'].split('_')[0]
                        if level < 3 and not all(len(ns & set(s['nums'])) <= max_ov
                                                 for s in sel):
                            continue
                        if level < 2 and c_cnt[cid] >= max_cls:
                            continue
                        if level < 1 and a_cnt[alg] >= max_alg:
                            continue
                        val = (c['total_score'] * 0.70
                               + (len(ns - used) / 6.0) * 0.30)
                        if val > bv:
                            bv, best = val, c
                    if best is not None:
                        chosen = best
                        break
                if chosen is None:
                    break
                if level > 0:
                    self.relax_log.append((slot + 1, level))
                sel.append(chosen)
                used.update(chosen['nums'])
                c_cnt[chosen['stat'].get('cluster_id', -1)] += 1
                a_cnt[chosen['algo'].split('_')[0]] += 1
            return sel

        # ── 메인 생성 ───────────────────────────────────────────────
        def generate(self, count=5, verbose=True):
            self.filter.reset()
            self._total_attempts = self._total_passed = 0
            m = count * self.CAND_MULT
            cands = (self._algo_weighted(m) + self._algo_balanced(m)
                     + self._algo_pattern(m) + self._algo_multi_cluster(m))
            if not cands:
                return []
            if verbose:
                pr = (self._total_passed / max(self._total_attempts, 1)) * 100
                print(f"  후보 {len(cands)}개 생성 (통과율 {pr:.1f}%)")
            return self._select_with_coverage(self._score_candidates(cands), count)

    return GeneratorV13


base_mod = None
GeneratorV13 = None


def install(v12_module):
    """v1.2 모듈을 받아 패치 클래스를 구성한다."""
    global base_mod, GeneratorV13
    base_mod = v12_module
    GeneratorV13 = build_generator_class(v12_module.EnsembleGenerator)
    return GeneratorV13


def predict_v13(history, count=5, seed=None, config=None, verbose=True):
    """v1.3 예측 — seed=None이면 매 실행 새로운 조합(실전), 정수면 재현(검증)"""
    rng, used_seed = make_rng(seed)
    cfg = config or base_mod.Config()
    eng = base_mod.DataEngine.from_history(history, cfg)
    if not eng.is_loaded:
        raise RuntimeError(eng.error_msg)
    geo = base_mod.GeometricAnalyzer(eng, cfg.N_CLUSTERS)
    gen = GeneratorV13(eng, geo, rng=rng)
    res = gen.generate(count=count, verbose=verbose)
    return {'sets': [r['nums'] for r in res], 'detail': res,
            'engine': eng, 'generator': gen, 'seed': used_seed}

# =============================================================================
# 3. [v1.3.1] DataEngine 보정 — recency 연속화
# =============================================================================
# 원본 recency는 '최근 10회차 출현 횟수'만 썼다. 10회차에 한 번호가 나올 수 있는
# 횟수는 사실상 0~3회뿐이라, 45개 번호가 고작 4단계 계단값으로 뭉개진다.
# → 동점이 대량 발생하고, 가중치 0.15가 변별에 거의 기여하지 못한다.
# 지수감쇠 누적으로 바꾸면 45개가 모두 다른 실수값을 갖는다.
RECENCY_DECAY = 0.97


def build_engine_class(base_cls):
    class DataEngineV13(base_cls):

        # ── [v1.3.3] 언어 간 비트 동일성을 위한 거듭제곱 ──────────────
        # Python 의 ** 와 JavaScript 의 Math.pow 는 IEEE-754 가 pow 의
        # 정확한 반올림을 요구하지 않아 지수의 약 10% 에서 1 ULP 어긋난다
        # (실측: 0.97^k 는 1245개 중 100개, 0.999^k 는 131개 불일치).
        # 곱셈은 IEEE-754 가 정확히 규정하므로, 반복 곱셈으로 바꾸면
        # 두 언어가 비트 단위로 같은 값을 낸다.
        @staticmethod
        def _decay_powers(base, n):
            out = np.empty(max(n, 1), dtype=float)
            p = 1.0
            for i in range(max(n, 1)):
                out[i] = p
                p *= base
            return out

        def _analyze_markov(self):
            cfg = self.config
            alpha = cfg.MARKOV_ALPHA
            T = self.total_rounds
            pw = self._decay_powers(cfg.MARKOV_DECAY, T)
            trans = np.zeros((46, 46), dtype=float)
            for t in range(1, T):
                w = pw[T - 1 - t]
                for m in self.history[t - 1]:
                    for n in self.history[t]:
                        trans[m][n] += w
            prob = np.zeros((46, 46), dtype=float)
            for m in range(1, 46):
                row_sum = trans[m][1:].sum()
                prob[m][1:] = (trans[m][1:] + alpha) / (row_sum + alpha * 45)
            scores = np.zeros(46)
            last = self.last_draw if self.last_draw else list(range(1, 7))
            for n in range(1, 46):
                scores[n] = float(np.mean([prob[m][n] for m in last]))
            return {'trans_matrix': trans, 'prob_matrix': prob,
                    'score': self._robust_normalize(scores),
                    'raw': scores, 'basis': sorted(last)}

        def _calculate_final_scores(self):
            cfg = self.config
            T = self.total_rounds

            # 연속 recency: 최근일수록 큰 가중치로 출현을 누적
            pw = self._decay_powers(RECENCY_DECAY, T)
            rec = np.zeros(46)
            for t, row in enumerate(self.history):
                w = pw[T - 1 - t]
                for n in row:
                    rec[n] += w
            rec = self._robust_normalize(rec)

            scores = np.zeros(46)
            for num in range(1, 46):
                scores[num] = (
                    self.frequency_analysis['weighted'][num]          * cfg.WEIGHT_FREQUENCY
                    + rec[num]                                        * cfg.WEIGHT_RECENCY
                    + self.gap_analysis['score'][num]                 * cfg.WEIGHT_GAP
                    + self.pair_analysis['number_pair_strength'][num] * cfg.WEIGHT_PAIR
                    + self.momentum_analysis['momentum'][num]         * cfg.WEIGHT_MOMENTUM
                    + self.zone_analysis['zone_score'][num]           * cfg.WEIGHT_ZONE
                    + self.markov_analysis['score'][num]              * cfg.WEIGHT_MARKOV
                    + self.assoc_analysis['score'][num]               * cfg.WEIGHT_ASSOC
                )
            self.recency_analysis = {'score': rec}
            s = scores[1:]
            if s.sum() > 0:
                scores[1:] = s / s.sum()
            return scores
    return DataEngineV13


# =============================================================================
# 4. [v1.3.1] BacktestEngine 안전로더 적용
# =============================================================================
# 원본 BacktestEngine.__init__ 은 csv_path 경로에서 df.iloc[:, 2:8] 로 번호를
# 꺼낸다. 프로젝트가 금지한 위치 기반 추출이며, 줄 끝 쉼표로 유령 열이 생기면
# 백테스트만 조용히 엉뚱한 열을 읽는다(예측은 멀쩡해서 발견이 늦다).
def patch_backtest_engine(v12_module):
    orig_init = v12_module.BacktestEngine.__init__

    def safe_init(self, csv_path=None, config=None, preloaded_history=None):
        self.config = config or v12_module.Config()
        if preloaded_history is not None:
            self.all_hist = preloaded_history
        elif csv_path is not None:
            self.all_hist = v12_module.safe_load_history(csv_path, verbose=False)
        else:
            raise ValueError("csv_path 또는 preloaded_history 필요")
        self.total = len(self.all_hist)

    v12_module.BacktestEngine.__init__ = safe_init
    return orig_init


# =============================================================================
# 5. [v1.3.1] 인기도 하드 필터 — 유일한 수학적 레버를 최대로
# =============================================================================
# 실데이터 측정 결과, 조합별 인기도는 최저~최고 2.59배까지 벌어진다.
# 예측력은 0이지만 이 축은 실재한다. 따라서 '맞히기'가 아니라
# '맞았을 때 덜 나눠 갖기'에 자원을 몰아준다.
class PopularityGate:
    """무작위 표본으로 인기도 분위수 기준선을 만들고, 상위 인기 조합을 차단"""

    def __init__(self, prev_draw, rng, n_sample=40000, max_pct=25.0):
        self.prev_draw = prev_draw
        self.max_pct = max_pct
        s = np.empty(n_sample)
        for i in range(n_sample):
            c = rng.choice(np.arange(1, 46), 6, replace=False)
            s[i] = popularity_log(sorted(int(x) for x in c), prev_draw)
        self.dist = np.sort(s)
        self.threshold = float(np.percentile(self.dist, max_pct))

    def percentile(self, nums):
        p = popularity_log(nums, self.prev_draw)
        return float(np.searchsorted(self.dist, p) / len(self.dist) * 100.0)

    def passes(self, nums):
        return popularity_log(nums, self.prev_draw) <= self.threshold



# =============================================================================
# 6. [v1.3.1] 통합 예측 진입점
# =============================================================================
def predict_v131(history, count=5, seed=None, config=None,
                 pop_max_pct=25.0, pop_weight=0.14, verbose=True):
    """
    v1.3.1 예측

      seed=None      실전(매 실행 새 조합) / 정수면 완전 재현
      pop_max_pct    인기도 상위 차단 기준. 25 = 하위 25% 인기 조합만 채택
                     (None 이면 하드 필터 해제, 점수 감점만 적용)
    """
    rng, used_seed = make_rng(seed)
    cfg = config or base_mod.Config()
    EngCls = build_engine_class(base_mod.DataEngine)
    eng = EngCls.from_history(history, cfg)
    if not eng.is_loaded:
        raise RuntimeError(eng.error_msg)
    geo = base_mod.GeometricAnalyzer(eng, cfg.N_CLUSTERS)

    gate = None
    if pop_max_pct is not None:
        gate = PopularityGate(eng.last_draw or [], np.random.default_rng(12345),
                              max_pct=pop_max_pct)

    gen = GeneratorV13(eng, geo, rng=rng, pop_weight=pop_weight)
    if gate is not None:
        # 조합 필터에 인기도 게이트를 한 단계 추가
        orig_apply = gen.filter.apply_all

        def apply_with_gate(nums, track=True):
            ok, why = orig_apply(nums, track=track)
            if not ok:
                return ok, why
            if not gate.passes(sorted(nums)):
                if track:
                    gen.filter.rejection_stats['인기도'] += 1
                    gen.filter.total_passed -= 1
                return False, '인기도'
            return True, ''
        gen.filter.apply_all = apply_with_gate

    res = gen.generate(count=count, verbose=verbose)
    sets = [r['nums'] for r in res]
    if gate is not None:
        for r in res:
            r['stat']['pop_pct'] = round(gate.percentile(r['nums']), 1)
    return {'sets': sets, 'detail': res, 'engine': eng,
            'generator': gen, 'gate': gate, 'seed': used_seed}
