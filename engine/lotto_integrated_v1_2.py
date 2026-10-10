# -*- coding: utf-8 -*-
# =============================================================================
#  LOTTO INTEGRATED v1.2  —  FINAL UNIFIED EDITION
# =============================================================================
#  [계보]
#    엔진 골격      : Dashboard v3.4        (4종 알고리즘 + KMeans + 학습/검증 분리 최적화)
#    데이터 로더    : Ultimate Hybrid v1.2  (6중 방어 로더)
#    랜덤 대조군    : Brother System v2.3   (실제 무작위 대조군 시뮬레이션)
#    신규 알고리즘  : Oracle v4.0 아이디어를 "연속 점수"로 재구현
#
#  [v1.1 -> v1.2 추가분]
#    (1) MarkovAnalyzer      : 직전 회차 조건부 전이확률 (라플라스 평활, 연속값)
#    (2) AssociationAnalyzer : Lift 기반 연관규칙          (연속값)
#    (3) MultiWindowAnalyzer : 30/50/100/전체 윈도우 병렬 비교 + 파라미터 민감도
#    (4) ValidationModule    : 부트스트랩 신뢰구간 + 필요 표본 크기 계산
#    (5) RandomControlBacktest : 모델 vs 실제 랜덤 대조군 동시 백테스트
#    (6) PredictionTracker   : 사전 예측 기록 및 사후 채점 (prediction_log.csv)
#
#  [사용법]
#    python lotto_integrated_v1_2.py
#    또는 Colab에서 CSV_PATH만 수정 후 전체 실행
# =============================================================================

import numpy as np
import random
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from copy import deepcopy

warnings.filterwarnings("ignore")

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ sklearn 미설치 — 간이 클러스터링 모드로 전환합니다.")


# =============================================================================
# 전역 상수
# =============================================================================
PRIME_NUMBERS = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}


# =============================================================================
# 0. 안전 데이터 로더  [출처: Ultimate Hybrid v1.2 — 6중 방어]
# =============================================================================
def safe_load_history(csv_path: str, verbose: bool = True):
    """
    CSV -> [[n1..n6], ...] (오름차순 정렬)

    v3.4 원본의 df.iloc[:, 2:8] 방식은 다음 상황에서 조용히 오작동한다.
      - 줄 끝 쉼표로 Unnamed 유령 열이 생긴 경우
      - 컬럼 순서가 다른 파일
      - 결측치/문자열이 섞인 행
    아래 6중 방어로 이 문제를 원천 차단한다.
    """
    df = pd.read_csv(csv_path)
    if verbose:
        print(f"  원본 데이터: {df.shape[0]}행 x {df.shape[1]}열")

    # [방어 1] 전부 비어 있는 '유령 열' 제거
    ghost_cols = [c for c in df.columns if df[c].isna().all()]
    if ghost_cols:
        if verbose:
            print(f"  [1] 빈 열 {len(ghost_cols)}개 제거: {ghost_cols}")
        df = df.drop(columns=ghost_cols)

    # [방어 2] 전부 비어 있는 행 제거
    before = len(df)
    df = df.dropna(how='all')
    if verbose and before != len(df):
        print(f"  [2] 빈 행 {before - len(df)}개 제거")

    # [방어 3] 번호 열 선택 — num1~num6 이름 우선, 없으면 뒤 6개
    num_cols = [c for c in df.columns if str(c).lower().startswith('num')]
    if len(num_cols) == 6:
        num_df = df[num_cols]
        if verbose:
            print(f"  [3] 번호 열(이름 기준): {num_cols}")
    else:
        num_df = df.iloc[:, -6:]
        if verbose:
            print(f"  [3] 번호 열(위치 기준): {list(num_df.columns)}")

    # [방어 4] 문자/공백 -> 숫자 강제 변환
    num_df = num_df.apply(pd.to_numeric, errors='coerce')

    # [방어 5] 결측 행 리포트 후 제외
    bad_rows = num_df[num_df.isna().any(axis=1)]
    if verbose and len(bad_rows) > 0:
        print(f"  [5] 결측치 {len(bad_rows)}개 행 제외")
    num_df = num_df.dropna().astype(int)

    # [방어 6] 로또 규칙 검증 — 1~45 범위, 중복 없는 6개
    valid_mask = num_df.apply(lambda r: r.between(1, 45).all() and r.nunique() == 6, axis=1)
    invalid_cnt = int((~valid_mask).sum())
    if verbose and invalid_cnt > 0:
        print(f"  [6] 규칙 위반 {invalid_cnt}개 행 제외")
    num_df = num_df[valid_mask]

    if len(num_df) == 0:
        raise ValueError("유효한 회차 데이터가 0건입니다. CSV 열 구조를 확인하세요.")

    return [sorted(int(n) for n in row) for row in num_df.values.tolist()]


class C:
    H  = '\033[95m'; B  = '\033[94m'; CN = '\033[96m'; G  = '\033[92m'
    Y  = '\033[93m'; R  = '\033[91m'; E  = '\033[0m';  BD = '\033[1m'
    DM = '\033[2m';  M  = '\033[35m'


# =============================================================================
# 1. Config
# =============================================================================
@dataclass
class Config:
    SUM_RANGE:            Tuple[int, int] = (95, 200)
    SUM_OPTIMAL:          Tuple[int, int] = (110, 190)
    MIN_AC:               int             = 7
    ODD_RATES:            List[int]       = field(default_factory=lambda: [2, 3, 4])
    HIGH_RATES:           List[int]       = field(default_factory=lambda: [2, 3, 4])
    MIN_ZONES:            int             = 3
    MAX_PER_ZONE:         int             = 3
    MIN_UNIQUE_ENDINGS:   int             = 3
    MAX_SAME_ENDING:      int             = 2
    PRIME_RANGE:          Tuple[int, int] = (1, 4)
    MAX_CONSECUTIVE:      int             = 2
    PREV_MATCH_RANGE:     Tuple[int, int] = (0, 3)
    # [v1.2] 기존 6종 + 신규 2종 = 8종 (합계 1.00)
    WEIGHT_FREQUENCY:     float           = 0.15
    WEIGHT_RECENCY:       float           = 0.15
    WEIGHT_GAP:           float           = 0.13
    WEIGHT_PAIR:          float           = 0.12
    WEIGHT_MOMENTUM:      float           = 0.16
    WEIGHT_ZONE:          float           = 0.11
    WEIGHT_MARKOV:        float           = 0.10   # [NEW] 전이확률
    WEIGHT_ASSOC:         float           = 0.08   # [NEW] Lift 연관규칙
    MARKOV_ALPHA:         float           = 1.0    # 라플라스 평활 계수
    MARKOV_DECAY:         float           = 0.999  # 오래된 전이 감쇠
    ASSOC_WINDOW:         int             = 300    # 연관규칙 산출 윈도우
    RECENT_PERIODS:       List[int]       = field(default_factory=lambda: [10, 30, 50, 100])
    N_CLUSTERS:           int             = 5
    CLUSTER_BONUS_WEIGHT: float           = 0.10
    MAX_SET_OVERLAP:      int             = 1

    def get_weights_dict(self) -> Dict[str, float]:
        return {
            'frequency': self.WEIGHT_FREQUENCY,
            'recency':   self.WEIGHT_RECENCY,
            'gap':       self.WEIGHT_GAP,
            'pair':      self.WEIGHT_PAIR,
            'momentum':  self.WEIGHT_MOMENTUM,
            'zone':      self.WEIGHT_ZONE,
            'markov':    self.WEIGHT_MARKOV,
            'assoc':     self.WEIGHT_ASSOC,
        }

    def set_weights_from_dict(self, w: Dict[str, float]):
        self.WEIGHT_FREQUENCY = w.get('frequency', self.WEIGHT_FREQUENCY)
        self.WEIGHT_RECENCY   = w.get('recency',   self.WEIGHT_RECENCY)
        self.WEIGHT_GAP       = w.get('gap',       self.WEIGHT_GAP)
        self.WEIGHT_PAIR      = w.get('pair',      self.WEIGHT_PAIR)
        self.WEIGHT_MOMENTUM  = w.get('momentum',  self.WEIGHT_MOMENTUM)
        self.WEIGHT_ZONE      = w.get('zone',      self.WEIGHT_ZONE)
        self.WEIGHT_MARKOV    = w.get('markov',    self.WEIGHT_MARKOV)
        self.WEIGHT_ASSOC     = w.get('assoc',     self.WEIGHT_ASSOC)


# =============================================================================
# 2. DataEngine
# =============================================================================
class DataEngine:

    def __init__(self, csv_path: str, config: Config = None,
                 exclude_last_n: int = 0):
        self.csv_path       = csv_path
        self.config         = config or Config()
        self.exclude_last_n = exclude_last_n
        self.is_loaded      = False
        self.error_msg      = ""
        try:
            self._load_data()
            self._analyze_all()
            self.is_loaded = True
        except Exception as e:
            self.error_msg = str(e)

    @classmethod
    def from_history(cls, history: List[List[int]],
                     config: Config = None) -> 'DataEngine':
        obj               = cls.__new__(cls)
        obj.csv_path      = None
        obj.config        = config or Config()
        obj.exclude_last_n = 0
        obj.is_loaded     = False
        obj.error_msg     = ""
        try:
            obj.history         = [row[:] for row in history]
            obj.hidden_draws    = []
            obj.total_rounds    = len(obj.history)
            obj.last_draw       = obj.history[-1] if obj.history else []
            obj.historical_sets = [frozenset(row) for row in obj.history]
            obj._analyze_all()
            obj.is_loaded = True
        except Exception as e:
            obj.error_msg = str(e)
        return obj

    def _load_data(self):
        # [v1.2] 6중 방어 로더 사용 (구 df.iloc[:, 2:8] 방식 폐기)
        all_history = safe_load_history(self.csv_path, verbose=False)
        if self.exclude_last_n > 0:
            self.history      = all_history[:-self.exclude_last_n]
            self.hidden_draws = all_history[-self.exclude_last_n:]
        else:
            self.history      = all_history
            self.hidden_draws = []
        self.total_rounds    = len(self.history)
        self.last_draw       = self.history[-1] if self.history else []
        self.historical_sets = [frozenset(row) for row in self.history]

    def _analyze_all(self):
        self.frequency_analysis = self._analyze_frequency()
        self.gap_analysis       = self._analyze_gaps()
        self.pair_analysis      = self._analyze_pairs()
        self.momentum_analysis  = self._analyze_momentum()
        self.zone_analysis      = self._analyze_zones()
        self.markov_analysis    = self._analyze_markov()      # [v1.2 NEW]
        self.assoc_analysis     = self._analyze_association() # [v1.2 NEW]
        self.pattern_stats      = self._analyze_patterns()
        self.final_scores       = self._calculate_final_scores()

    @staticmethod
    def _robust_normalize(arr: np.ndarray) -> np.ndarray:
        if len(arr) == 46:
            s = arr[1:]
        else:
            s = arr
        if len(s) == 0 or s.max() == s.min():
            return arr.copy()
        median   = np.median(s)
        q25, q75 = np.percentile(s, 25), np.percentile(s, 75)
        iqr      = q75 - q25
        if iqr > 0:
            normalized = (s - median) / iqr
            normalized = np.clip(normalized, -2, 2)
            normalized = (normalized + 2) / 4
        else:
            normalized = s / s.max() if s.max() > 0 else s.copy()
        if len(arr) == 46:
            result     = arr.copy()
            result[1:] = normalized
            return result
        return normalized.copy()

    def _analyze_frequency(self) -> Dict:
        result         = {'total': Counter(), 'by_period': {},
                          'weighted': np.zeros(46)}
        all_nums       = [n for row in self.history for n in row]
        result['total'] = Counter(all_nums)
        period_weights = {10: 4.0, 30: 2.5, 50: 1.5, 100: 1.0}
        for period in self.config.RECENT_PERIODS:
            if period <= self.total_rounds:
                result['by_period'][period] = Counter(
                    n for row in self.history[-period:] for n in row)
        for num in range(1, 46):
            score = 0
            for period, weight in period_weights.items():
                if period in result['by_period']:
                    expected = (period * 6) / 45
                    actual   = result['by_period'][period].get(num, 0)
                    score   += (actual / expected) * weight if expected > 0 else 0
            result['weighted'][num] = score
        result['weighted'] = self._robust_normalize(result['weighted'])
        return result

    def _analyze_gaps(self) -> Dict:
        result      = {'gap': {}, 'avg_gap': {}, 'score': np.zeros(46)}
        appearances = defaultdict(list)
        for idx, row in enumerate(self.history):
            for n in row:
                appearances[n].append(idx)
        for num in range(1, 46):
            if num in appearances and len(appearances[num]) > 1:
                gaps = [appearances[num][i+1] - appearances[num][i]
                        for i in range(len(appearances[num]) - 1)]
                result['avg_gap'][num] = np.mean(gaps)
                result['gap'][num]     = self.total_rounds - 1 - appearances[num][-1]
                ratio = (result['gap'][num] / result['avg_gap'][num]
                         if result['avg_gap'][num] > 0 else 1)
                if 0.8 <= ratio <= 1.5:
                    result['score'][num] = 1.0 - abs(ratio - 1.0) * 0.3
                elif ratio > 1.5:
                    result['score'][num] = max(0.15, 1.0 - (ratio - 1.5) * 0.2)
                else:
                    result['score'][num] = max(0.25, ratio * 0.6)
            else:
                result['gap'][num]     = self.total_rounds
                result['avg_gap'][num] = self.total_rounds / 2
                result['score'][num]   = 0.3
        result['score'] = self._robust_normalize(result['score'])
        return result

    def _analyze_pairs(self) -> Dict:
        result = {
            'pair_count':           Counter(),
            'pair_score':           defaultdict(lambda: defaultdict(float)),
            'top_pairs':            [],
            'number_pair_strength': np.zeros(46),
        }
        window = min(100, self.total_rounds)
        for row in self.history[-window:]:
            for pair in combinations(sorted(row), 2):
                result['pair_count'][pair] += 1
        result['top_pairs'] = result['pair_count'].most_common(50)
        for (a, b), count in result['pair_count'].items():
            strength = count / window
            result['pair_score'][a][b] = strength
            result['pair_score'][b][a] = strength
            result['number_pair_strength'][a] += strength
            result['number_pair_strength'][b] += strength
        result['number_pair_strength'] = self._robust_normalize(
            result['number_pair_strength'])
        return result

    def _analyze_momentum(self) -> Dict:
        result = {'momentum': np.zeros(46), 'trend': {}}
        if self.total_rounds >= 30:
            recent_10 = Counter(n for row in self.history[-10:] for n in row)
            recent_30 = Counter(n for row in self.history[-30:] for n in row)
            for num in range(1, 46):
                short = recent_10.get(num, 0) / 10
                mid   = recent_30.get(num, 0) / 30
                if mid > 0:
                    momentum = short / mid
                    result['momentum'][num] = momentum
                    result['trend'][num] = (
                        'UP' if momentum > 1.3 else
                        'DOWN' if momentum < 0.7 else 'STABLE')
                else:
                    result['momentum'][num] = 1.0 if short > 0 else 0.5
                    result['trend'][num]    = 'UP' if short > 0 else 'DORMANT'
        result['momentum'] = self._robust_normalize(result['momentum'])
        return result

    def _analyze_zones(self) -> Dict:
        zones  = [(1,10),(11,20),(21,30),(31,40),(41,45)]
        result = {'zone_freq': {z: Counter() for z in zones},
                  'zone_score': np.zeros(46)}
        recent = self.history[-50:] if self.total_rounds >= 50 else self.history
        for row in recent:
            for n in row:
                for z_s, z_e in zones:
                    if z_s <= n <= z_e:
                        result['zone_freq'][(z_s, z_e)][n] += 1
                        break
        for z_s, z_e in zones:
            nums   = list(range(z_s, z_e + 1))
            counts = [result['zone_freq'][(z_s, z_e)].get(n, 0) for n in nums]
            mx     = max(counts) if counts else 1
            for i, n in enumerate(nums):
                result['zone_score'][n] = counts[i] / mx if mx > 0 else 0
        return result

    # ─────────────────────────────────────────────────────────────────
    # [v1.2 NEW] 마르코프 전이확률  —  연속 점수 버전
    # ─────────────────────────────────────────────────────────────────
    # Oracle v4.0의 markov_chain은 계단식 처리로 동점이 대량 발생했다.
    # 여기서는 라플라스 평활을 적용한 조건부 전이확률을 그대로 쓴다.
    #
    #   P(n이 t회차에 등장 | m이 t-1회차에 등장)
    #     = (count[m][n] + alpha) / (rowsum[m] + alpha * 45)
    #
    # 최종 점수 = 직전 회차 6개 번호 m에 대한 P(n|m)의 평균.
    # 분모가 0이 되는 경우가 없고, 45개 번호가 모두 서로 다른 실수값을 갖는다.
    # ─────────────────────────────────────────────────────────────────
    def _analyze_markov(self) -> Dict:
        cfg    = self.config
        alpha  = cfg.MARKOV_ALPHA
        decay  = cfg.MARKOV_DECAY
        trans  = np.zeros((46, 46), dtype=float)

        T = self.total_rounds
        for t in range(1, T):
            # 최근 회차일수록 큰 가중치 (지수 감쇠)
            w = decay ** (T - 1 - t)
            for m in self.history[t - 1]:
                for n in self.history[t]:
                    trans[m][n] += w

        # 라플라스 평활 후 행 정규화 -> 조건부 확률
        prob = np.zeros((46, 46), dtype=float)
        for m in range(1, 46):
            row_sum = trans[m][1:].sum()
            prob[m][1:] = (trans[m][1:] + alpha) / (row_sum + alpha * 45)

        scores = np.zeros(46)
        last   = self.last_draw if self.last_draw else list(range(1, 7))
        for n in range(1, 46):
            scores[n] = float(np.mean([prob[m][n] for m in last]))

        norm = self._robust_normalize(scores)
        return {
            'trans_matrix': trans,
            'prob_matrix':  prob,
            'score':        norm,
            'raw':          scores,
            'basis':        sorted(last),
        }

    # ─────────────────────────────────────────────────────────────────
    # [v1.2 NEW] 연관규칙 (Lift)  —  연속 점수 버전
    # ─────────────────────────────────────────────────────────────────
    # pair_analysis 는 "얼마나 자주 같이 나왔나"(원시 동반출현 빈도)를 본다.
    # 이 알고리즘은 그것을 각 번호의 개별 출현률로 나눈 Lift 를 본다.
    #
    #   lift(a,b) = P(a,b) / (P(a) * P(b))
    #
    # lift > 1 이면 우연보다 자주 같이 등장, < 1 이면 서로 회피.
    # 자주 나오는 번호가 자동으로 높은 점수를 받는 편향이 제거되므로
    # pair_analysis 와 다른 축의 정보를 제공한다 (중복 아님).
    # ─────────────────────────────────────────────────────────────────
    def _analyze_association(self) -> Dict:
        cfg    = self.config
        window = min(cfg.ASSOC_WINDOW, self.total_rounds)
        recent = self.history[-window:]

        single = Counter()
        pair   = Counter()
        for row in recent:
            single.update(row)
            for p in combinations(sorted(row), 2):
                pair[p] += 1

        eps   = 1e-9
        p_one = {n: (single.get(n, 0) + 1) / (window + 45) for n in range(1, 46)}

        lift = defaultdict(dict)
        for (a, b), c in pair.items():
            p_ab = (c + 1) / (window + 45)
            val  = p_ab / (p_one[a] * p_one[b] + eps)
            lift[a][b] = val
            lift[b][a] = val

        default_lift = 1.0
        scores = np.zeros(46)
        last   = self.last_draw if self.last_draw else list(range(1, 7))
        for n in range(1, 46):
            vals = [lift[n].get(m, default_lift) for m in last if m != n]
            scores[n] = float(np.mean(vals)) if vals else default_lift

        top_rules = sorted(
            ((a, b, v) for a in lift for b, v in lift[a].items() if a < b),
            key=lambda x: x[2], reverse=True)[:15]

        norm = self._robust_normalize(scores)
        return {
            'lift':      lift,
            'score':     norm,
            'raw':       scores,
            'top_rules': top_rules,
            'window':    window,
        }

    def _analyze_patterns(self) -> Dict:
        result = {'sum_dist': [], 'ac_dist': []}
        for row in self.history:
            nums  = sorted(row)
            diffs = set(abs(nums[j]-nums[i])
                        for i in range(6) for j in range(i+1, 6))
            result['sum_dist'].append(sum(nums))
            result['ac_dist'].append(len(diffs) - 5)
        return result

    def _calculate_final_scores(self) -> np.ndarray:
        scores      = np.zeros(46)
        cfg         = self.config
        recent_10   = self.frequency_analysis['by_period'].get(10, Counter())
        expected_10 = (10 * 6) / 45
        for num in range(1, 46):
            actual  = recent_10.get(num, 0)
            recency = min(actual / expected_10, 1.5) / 1.5
            scores[num] = (
                self.frequency_analysis['weighted'][num]          * cfg.WEIGHT_FREQUENCY
                + recency                                          * cfg.WEIGHT_RECENCY
                + self.gap_analysis['score'][num]                 * cfg.WEIGHT_GAP
                + self.pair_analysis['number_pair_strength'][num] * cfg.WEIGHT_PAIR
                + self.momentum_analysis['momentum'][num]         * cfg.WEIGHT_MOMENTUM
                + self.zone_analysis['zone_score'][num]           * cfg.WEIGHT_ZONE
                + self.markov_analysis['score'][num]              * cfg.WEIGHT_MARKOV
                + self.assoc_analysis['score'][num]               * cfg.WEIGHT_ASSOC
            )
        s = scores[1:]
        if s.sum() > 0:
            scores[1:] = s / s.sum()
        return scores

    def get_hot_numbers(self, n=10):
        return sorted([(i, self.final_scores[i]) for i in range(1, 46)],
                      key=lambda x: x[1], reverse=True)[:n]

    def get_cold_numbers(self, n=10):
        return sorted([(i, self.final_scores[i]) for i in range(1, 46)],
                      key=lambda x: x[1])[:n]


# =============================================================================
# 3. GeometricAnalyzer
# =============================================================================
class GeometricAnalyzer:

    def __init__(self, engine: DataEngine, n_clusters: int = 5):
        self.engine        = engine
        self.n_clusters    = n_clusters
        self.cluster_stats = {}
        self.top_clusters  = []
        if HAS_SKLEARN and len(engine.history) > 30:
            self._build_clusters()
        else:
            self._build_simple_clusters()

    def _extract_features(self, nums: List[int]) -> np.ndarray:
        nums = sorted(nums)
        f    = []
        f.extend([sum(nums), np.mean(nums), np.std(nums), max(nums)-min(nums)])
        gaps = [nums[i+1]-nums[i] for i in range(5)]
        f.extend([np.mean(gaps), np.std(gaps), max(gaps), min(gaps)])
        f.extend([nums[0], nums[5], nums[2]+nums[3]])
        zones = [0]*5
        for n in nums:
            zones[min((n-1)//10, 4)] += 1
        f.extend(zones)
        diffs = set(abs(nums[j]-nums[i])
                    for i in range(6) for j in range(i+1, 6))
        f.extend([
            sum(1 for n in nums if n%2==1),
            sum(1 for n in nums if n>=23),
            len(diffs)-5,
            len(set(n%10 for n in nums)),
            sum(1 for n in nums if n in PRIME_NUMBERS),
            sum(1 for i in range(5) if nums[i+1]==nums[i]+1),
        ])
        return np.array(f, dtype=float)

    def _build_clusters(self):
        all_features        = [self._extract_features(r) for r in self.engine.history]
        self.feature_matrix = np.array(all_features)
        self.scaler         = StandardScaler()
        self.scaled         = self.scaler.fit_transform(self.feature_matrix)
        best_score, best_k  = -1, self.n_clusters
        max_k = max(3, min(10, len(self.engine.history) // 15))
        for k in range(3, max_k+1):
            km     = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(self.scaled)
            if len(set(labels)) < 2:
                continue
            sc = silhouette_score(self.scaled, labels)
            if sc > best_score:
                best_score, best_k = sc, k
        self.n_clusters = best_k
        self.kmeans     = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        self.labels     = self.kmeans.fit_predict(self.scaled)
        self._compute_cluster_stats()

    def _build_simple_clusters(self):
        self.kmeans     = None
        self.scaler     = None
        self.n_clusters = 3
        sums = [sum(row) for row in self.engine.history]
        q33  = np.percentile(sums, 33)
        q66  = np.percentile(sums, 66)
        self.labels = np.array([
            0 if s < q33 else (1 if s < q66 else 2) for s in sums])
        self._compute_cluster_stats()

    def _compute_cluster_stats(self):
        for cid in range(self.n_clusters):
            mask     = self.labels == cid
            draws    = [self.engine.history[i]
                        for i in range(len(self.engine.history)) if mask[i]]
            recent_n = min(50, len(self.engine.history))
            recent_c = sum(1 for i in range(-recent_n, 0)
                           if (i+len(self.labels)) >= 0
                           and self.labels[i] == cid)
            self.cluster_stats[cid] = {
                'count':        len(draws),
                'ratio':        len(draws) / max(len(self.engine.history), 1),
                'avg_sum':      np.mean([sum(r) for r in draws]) if draws else 0,
                'avg_spread':   np.mean([max(r)-min(r) for r in draws]) if draws else 0,
                'recent_ratio': recent_c / recent_n if recent_n > 0 else 0,
            }
        scored = sorted(
            [(cid, st['ratio']*0.5 + st['recent_ratio']*0.5)
             for cid, st in self.cluster_stats.items()],
            key=lambda x: x[1], reverse=True)
        self.top_clusters = [cid for cid, _ in scored[:3]]

    def get_cluster_id(self, nums: List[int]) -> int:
        if self.kmeans is not None:
            scaled = self.scaler.transform([self._extract_features(nums)])
            return int(self.kmeans.predict(scaled)[0])
        s    = sum(nums)
        sums = [sum(r) for r in self.engine.history]
        q33  = np.percentile(sums, 33)
        q66  = np.percentile(sums, 66)
        return 0 if s < q33 else (1 if s < q66 else 2)

    def get_cluster_bonus(self, nums: List[int]) -> float:
        cid = self.get_cluster_id(nums)
        if cid == self.top_clusters[0]:  return 1.0
        if cid in self.top_clusters[1:]: return 0.6
        return 0.2


# =============================================================================
# 4. FilterSystem
# =============================================================================
class FilterSystem:

    def __init__(self, engine: DataEngine, config: Config = None):
        self.engine          = engine
        self.config          = config or engine.config
        self.rejection_stats = Counter()
        self.total_checked   = 0
        self.total_passed    = 0

    def reset(self):
        self.rejection_stats = Counter()
        self.total_checked   = 0
        self.total_passed    = 0

    def apply_all(self, nums: List[int],
                  track: bool = True) -> Tuple[bool, str]:
        nums   = sorted(nums)
        checks = [
            (self._f_sum,     "합계"),
            (self._f_ac,      "AC값"),
            (self._f_odd,     "홀짝"),
            (self._f_high,    "고저"),
            (self._f_zones,   "구간"),
            (self._f_consec,  "연속"),
            (self._f_endings, "끝수"),
            (self._f_primes,  "소수"),
            (self._f_prev,    "이전회차"),
            (self._f_hist,    "역대중복"),
            (self._f_spread,  "간격"),
            (self._f_edge,    "경계"),
        ]
        if track:
            self.total_checked += 1
        for fn, name in checks:
            if not fn(nums):
                if track:
                    self.rejection_stats[name] += 1
                return False, name
        if track:
            self.total_passed += 1
        return True, ""

    def get_pass_rate(self) -> float:
        return (self.total_passed / self.total_checked
                if self.total_checked > 0 else 0.0)

    def get_rejection_summary(self, top_n: int = 6) -> List[Tuple[str, int, float]]:
        total_rej = sum(self.rejection_stats.values())
        return [
            (name, cnt, cnt/total_rej*100 if total_rej > 0 else 0.0)
            for name, cnt in self.rejection_stats.most_common(top_n)
        ]

    def _f_sum(self, n):
        return self.config.SUM_RANGE[0] <= sum(n) <= self.config.SUM_RANGE[1]

    def _f_ac(self, n):
        d = set(abs(n[j]-n[i]) for i in range(6) for j in range(i+1, 6))
        return (len(d)-5) >= self.config.MIN_AC

    def _f_odd(self, n):
        return sum(1 for x in n if x%2==1) in self.config.ODD_RATES

    def _f_high(self, n):
        return sum(1 for x in n if x>=23) in self.config.HIGH_RATES

    def _f_zones(self, n):
        zc = [0]*5
        for x in n:
            zc[min((x-1)//10, 4)] += 1
        return (sum(1 for c in zc if c>0) >= self.config.MIN_ZONES
                and max(zc) <= self.config.MAX_PER_ZONE)

    def _f_consec(self, n):
        run = 1
        for i in range(5):
            if n[i+1] == n[i]+1:
                run += 1
                if run > self.config.MAX_CONSECUTIVE:
                    return False
            else:
                run = 1
        return True

    def _f_endings(self, n):
        endings = [x%10 for x in n]
        return (len(set(endings)) >= self.config.MIN_UNIQUE_ENDINGS
                and max(Counter(endings).values()) <= self.config.MAX_SAME_ENDING)

    def _f_primes(self, n):
        pc = sum(1 for x in n if x in PRIME_NUMBERS)
        return self.config.PRIME_RANGE[0] <= pc <= self.config.PRIME_RANGE[1]

    def _f_prev(self, n):
        if not self.engine.last_draw:
            return True
        mc = len(set(n) & set(self.engine.last_draw))
        return self.config.PREV_MATCH_RANGE[0] <= mc <= self.config.PREV_MATCH_RANGE[1]

    def _f_hist(self, n):
        return frozenset(n) not in self.engine.historical_sets

    def _f_spread(self, n):
        return 3.5 <= np.mean([n[i+1]-n[i] for i in range(5)]) <= 13.0

    def _f_edge(self, n):
        if 1 in n and 45 in n:
            return sum(1 for x in n if 18<=x<=27) >= 1
        return True


# =============================================================================
# 5. EnsembleGenerator
# =============================================================================
class EnsembleGenerator:
    """
    [v3.4 핵심 수정]

    클러스터 다양성 문제를 2단계로 근본 해결

    [1단계] _algo_multi_cluster(): 비례 확률 → 균등 확률
      - 원본: top_clusters 비율(C1=46%)에 비례 → C1 후보 압도적 생성
      - 수정: 1/len(targets) 균등 확률 → 각 클러스터 동등 기회

    [2단계] _build_diverse_pool(): 클러스터 쿼터 pool 신규 추가
      - 원본: 점수 상위 50개 pool → C1이 점수도 높아 pool도 독점
      - 수정: 클러스터별 최소 count개 우선 확보 후 나머지 점수순 채움
              → pool 구성 단계부터 다양성 보장
    """

    def __init__(self, engine: DataEngine,
                 geometric: GeometricAnalyzer = None):
        self.engine    = engine
        self.geometric = geometric
        self.filter    = FilterSystem(engine)
        self.scores    = engine.final_scores.copy()

        self._total_attempts = 0
        self._total_passed   = 0

    def _get_weights(self) -> np.ndarray:
        w = self.scores[1:].copy()
        return w / w.sum() if w.sum() > 0 else np.ones(45)/45

    # ── _try_add ─────────────────────────────────────────────────────────
    def _try_add(self, nums: List[int], algo: str,
                 results: List[Dict]) -> bool:
        self._total_attempts += 1
        ok, _ = self.filter.apply_all(nums, track=True)
        if ok:
            self._total_passed += 1
            results.append({'nums': nums, 'algo': algo})
            return True
        return False

    # ── 메인 생성 ────────────────────────────────────────────────────────
    def generate(self, count: int = 5,
                 verbose: bool = True) -> List[Dict]:

        # 매 생성마다 필터 통계 초기화
        self.filter.reset()
        self._total_attempts = 0
        self._total_passed   = 0

        all_cands = []
        if verbose:
            print(f"  {C.DM}후보 생성 중...{C.E}")

        c1 = self._algo_weighted(count * 4)
        all_cands.extend(c1)
        if verbose:
            print(f"    Algo 1 (Weighted):     {len(c1):3d} 후보")

        c2 = self._algo_balanced(count * 4)
        all_cands.extend(c2)
        if verbose:
            print(f"    Algo 2 (Balanced):     {len(c2):3d} 후보")

        c3 = self._algo_pattern(count * 4)
        all_cands.extend(c3)
        if verbose:
            print(f"    Algo 3 (Pattern):      {len(c3):3d} 후보")

        if self.geometric:
            c4 = self._algo_multi_cluster(count * 4)
            all_cands.extend(c4)
            if verbose:
                print(f"    Algo 4 (MultiCluster): {len(c4):3d} 후보")

        pass_rate = (self._total_passed / self._total_attempts * 100
                     if self._total_attempts > 0 else 0.0)

        if verbose:
            print(f"  {C.DM}총 {len(all_cands)}개 후보 "
                  f"(시도 {self._total_attempts}회 / "
                  f"통과율 {pass_rate:.1f}%) → 스코어링 중...{C.E}")

        if not all_cands:
            print(f"  {C.R}[경고] 후보 없음. 필터 조건을 완화하세요.{C.E}")
            return []

        scored = self._score_candidates(all_cands)
        final  = self._select_with_coverage(scored, count)

        if verbose:
            print(f"  {C.G}✓ {len(final)}세트 최종 선정 "
                  f"(커버리지 최적화 / 생성 통과율: {pass_rate:.1f}%){C.E}")

        return final

    # ── 알고리즘 1: 가중치 기반 ──────────────────────────────────────────
    def _algo_weighted(self, count: int) -> List[Dict]:
        results  = []
        w        = self._get_weights()
        max_iter = min(count * 20, 80000)
        for _ in range(max_iter):
            if len(results) >= count:
                break
            try:
                nums = sorted(int(n) for n in
                              np.random.choice(range(1,46), 6,
                                               replace=False, p=w))
                self._try_add(nums, 'weighted', results)
            except Exception:
                continue
        return results

    # ── 알고리즘 2: 구간 밸런스 ──────────────────────────────────────────
    def _algo_balanced(self, count: int) -> List[Dict]:
        results  = []
        zones    = [(1,10),(11,20),(21,30),(31,40),(41,45)]
        w        = self._get_weights()
        max_iter = min(count * 20, 80000)
        for _ in range(max_iter):
            if len(results) >= count:
                break
            picks = [1,1,1,1,1]
            picks[np.random.randint(0,5)] += 1
            nums  = []
            valid = True
            for i, (z_s, z_e) in enumerate(zones):
                zone_nums = list(range(z_s, z_e+1))
                zone_w    = np.array([w[n-1] for n in zone_nums])
                zs        = zone_w.sum()
                zone_w    = (zone_w/zs if zs>0
                             else np.ones(len(zone_nums))/len(zone_nums))
                try:
                    picked = np.random.choice(zone_nums, picks[i],
                                              replace=False, p=zone_w)
                    nums.extend(int(n) for n in picked)
                except Exception:
                    valid = False
                    break
            if not valid:
                continue
            nums = sorted(set(nums))
            if len(nums) == 6:
                self._try_add(nums, 'balanced', results)
        return results

    # ── 알고리즘 3: 동반출현 패턴 ────────────────────────────────────────
    def _algo_pattern(self, count: int) -> List[Dict]:
        results   = []
        top_pairs = self.engine.pair_analysis['top_pairs'][:30]
        w         = self._get_weights()
        max_iter  = min(count * 20, 80000)
        for _ in range(max_iter):
            if len(results) >= count:
                break
            if top_pairs and np.random.random() < 0.7:
                idx  = np.random.randint(0, min(15, len(top_pairs)))
                base = list(top_pairs[idx][0])
            else:
                base = []
            nums_set = set(base)
            needed   = 6 - len(nums_set)
            temp_w   = w.copy()
            for n in nums_set:
                temp_w[n-1] = 0.0
            ts = temp_w.sum()
            if ts <= 0:
                continue
            temp_w /= ts
            try:
                extra = np.random.choice(range(1,46), needed,
                                         replace=False, p=temp_w)
                nums_set.update(int(n) for n in extra)
            except Exception:
                continue
            if len(nums_set) == 6:
                self._try_add(sorted(nums_set), 'pattern', results)
        return results

    # ── 알고리즘 4: 다중 클러스터 ────────────────────────────────────────
    def _algo_multi_cluster(self, count: int) -> List[Dict]:
        """
        [v3.4 핵심 수정 1단계] 비례 확률 → 균등 확률
        원본: probs = [ratio/total_r ...]  → C1(46%)이 절반 차지
        수정: probs = [1/n, 1/n, 1/n]     → C1/C2/C0 동등 기회
        → C2(저합계 108), C0(고합계 171) 후보가 pool에 충분히 생성됨
        """
        results  = []
        w        = self._get_weights()
        max_iter = min(count * 25, 80000)

        targets = [
            {
                'cluster':       cid,
                'target_sum':    int(self.geometric.cluster_stats[cid]['avg_sum']),
                'target_spread': int(self.geometric.cluster_stats[cid]['avg_spread']),
            }
            for cid in self.geometric.top_clusters
        ]
        if not targets:
            return results

        # [수정] 균등 확률 — 비율 무관하게 각 클러스터 동등 기회
        probs = [1.0 / len(targets)] * len(targets)

        for _ in range(max_iter):
            if len(results) >= count:
                break

            chosen        = targets[np.random.choice(len(targets), p=probs)]
            target_spread = max(5, chosen['target_spread'])
            first = np.random.randint(1, max(2, 46-target_spread))
            last  = min(45, first + target_spread + np.random.randint(-5, 6))
            last  = max(first+5, last)
            seeds = list({first, min(45, last)})

            temp_w = w.copy()
            for n in seeds:
                if 1 <= n <= 45:
                    temp_w[n-1] = 0.0
            ts = temp_w.sum()
            if ts <= 0:
                continue
            temp_w /= ts

            try:
                needed   = 6 - len(seeds)
                extra    = np.random.choice(range(1,46), needed,
                                            replace=False, p=temp_w)
                all_nums = set(seeds)
                all_nums.update(int(n) for n in extra)
            except Exception:
                continue

            if len(all_nums) < 6:
                continue
            if len(all_nums) > 6:
                all_nums = set(sorted(all_nums,
                                      key=lambda n: self.scores[n],
                                      reverse=True)[:6])
            nums = sorted(all_nums)
            if abs(sum(nums) - chosen['target_sum']) <= 35:
                self._try_add(nums, f"cluster_{chosen['cluster']}", results)

        return results

    # ── 스코어링 ─────────────────────────────────────────────────────────
    def _score_candidates(self, candidates: List[Dict]) -> List[Dict]:
        seen            = set()
        unique          = []
        raw_num_scores  = []
        raw_pair_scores = []

        for cand in candidates:
            key = tuple(cand['nums'])
            if key in seen:
                continue
            seen.add(key)
            nums = cand['nums']
            raw_num_scores.append(sum(self.scores[n] for n in nums))
            raw_pair_scores.append(
                sum(self.engine.pair_analysis['pair_score'][a][b]
                    for a, b in combinations(nums, 2)))
            unique.append(cand)

        if not unique:
            return []

        max_num  = max(raw_num_scores)  if max(raw_num_scores)  > 0 else 1.0
        max_pair = max(raw_pair_scores) if max(raw_pair_scores) > 0 else 1.0

        scored = []
        for i, cand in enumerate(unique):
            nums  = cand['nums']
            s     = sum(nums)
            diffs = set(abs(nums[j]-nums[i])
                        for i in range(6) for j in range(i+1, 6))
            ac    = len(diffs) - 5
            zones = [0]*5
            for n in nums:
                zones[min((n-1)//10, 4)] += 1

            sum_score     = 1.0 if 110 <= s <= 190 else 0.5
            ac_score      = np.clip(ac / 10.0,  0.0, 1.0)
            num_score     = np.clip(raw_num_scores[i]  / max_num,  0.0, 1.0)
            pair_score    = np.clip(raw_pair_scores[i] / max_pair, 0.0, 1.0)
            balance_score = np.clip(1.0 / (1.0 + np.var(zones)), 0.0, 1.0)
            cluster_bonus = (self.geometric.get_cluster_bonus(nums)
                             if self.geometric else 0.5)
            algo_bonus    = 1.0 if cand['algo'] == 'balanced' else 0.8

            total = (
                sum_score     * 0.12
                + ac_score    * 0.12
                + num_score   * 0.28
                + pair_score  * 0.18
                + balance_score * 0.13
                + cluster_bonus * self.engine.config.CLUSTER_BONUS_WEIGHT
                + algo_bonus  * 0.07
            )

            cand['stat'] = {
                'sum':           s,
                'ac':            ac,
                'odd':           sum(1 for n in nums if n%2==1),
                'high':          sum(1 for n in nums if n>=23),
                'prime':         sum(1 for n in nums if n in PRIME_NUMBERS),
                'cluster_bonus': cluster_bonus,
                'cluster_id':    (self.geometric.get_cluster_id(nums)
                                  if self.geometric else -1),
            }
            cand['total_score'] = round(total, 6)
            scored.append(cand)

        return scored

    # ── [v3.4 신규] 클러스터 쿼터 pool 구성 ─────────────────────────────
    def _build_diverse_pool(self, candidates: List[Dict],
                            count: int) -> List[Dict]:
        """
        [v3.4 핵심 수정 2단계] 클러스터 쿼터 pool
        원본: 점수 상위 50개 → C1이 점수도 높아 pool 독점
        수정:
          Step 1. 클러스터별로 점수순 정렬
          Step 2. 각 클러스터에서 최소 count개 우선 확보
          Step 3. 남은 슬롯은 전체 점수순으로 채움
        → pool 구성 단계부터 C2/C0 클러스터 후보 포함 보장
        """
        pool_size = max(count * 10, 50)

        # 클러스터별 점수순 분류
        by_cluster: Dict[int, List[Dict]] = defaultdict(list)
        for cand in sorted(candidates,
                           key=lambda x: x['total_score'], reverse=True):
            cid = cand['stat'].get('cluster_id', -1)
            by_cluster[cid].append(cand)

        pool     = []
        seen_ids = set()

        # Step 1: 각 클러스터에서 최소 count개 우선 확보
        for cid, cands in by_cluster.items():
            for c in cands[:count]:
                if id(c) not in seen_ids:
                    pool.append(c)
                    seen_ids.add(id(c))

        # Step 2: 나머지 슬롯을 전체 점수순으로 채움
        all_sorted = sorted(candidates,
                            key=lambda x: x['total_score'], reverse=True)
        for c in all_sorted:
            if len(pool) >= pool_size:
                break
            if id(c) not in seen_ids:
                pool.append(c)
                seen_ids.add(id(c))

        return pool

    # ── 커버리지 + 클러스터 다양성 최적화 선택 ───────────────────────────
    def _select_with_coverage(self, candidates: List[Dict],
                              count: int) -> List[Dict]:
        """
        [v3.4 수정]
        pool 구성: sorted[:50] → _build_diverse_pool() 호출
        max_same_cls: max(1, count//2) 동적 계산 유지
        """
        if not candidates:
            return []

        # [수정] 클러스터 쿼터 pool 사용
        pool = self._build_diverse_pool(candidates, count)

        selected        = []
        used_nums       = set()
        cluster_counter = Counter()
        max_overlap     = self.engine.config.MAX_SET_OVERLAP
        max_same_cls    = max(1, count // 2)  # count=5 → 2세트 제한

        for _ in range(count):
            best_idx   = -1
            best_value = -1.0

            for i, cand in enumerate(pool):
                if cand in selected:
                    continue
                nums_set = set(cand['nums'])
                cid      = cand['stat'].get('cluster_id', -1)

                # ① 세트 간 번호 중복 체크
                if not all(len(nums_set & set(s['nums'])) <= max_overlap
                           for s in selected):
                    continue

                # ② 클러스터 다양성 체크
                if cluster_counter[cid] >= max_same_cls:
                    continue

                new_coverage = len(nums_set - used_nums)
                value = cand['total_score'] * 0.7 + (new_coverage / 6.0) * 0.3

                if value > best_value:
                    best_value = value
                    best_idx   = i

            if best_idx >= 0:
                chosen = pool[best_idx]
                selected.append(chosen)
                used_nums.update(chosen['nums'])
                cluster_counter[chosen['stat'].get('cluster_id', -1)] += 1
            else:
                # fallback 1: 클러스터 제한만 해제
                remaining = [c for c in pool
                             if c not in selected
                             and all(len(set(c['nums']) & set(s['nums']))
                                     <= max_overlap for s in selected)]
                if remaining:
                    fb = max(remaining, key=lambda x: x['total_score'])
                    selected.append(fb)
                    used_nums.update(fb['nums'])
                    cluster_counter[fb['stat'].get('cluster_id', -1)] += 1
                elif pool:
                    # fallback 2: 모든 조건 해제
                    left = [c for c in pool if c not in selected]
                    if left:
                        fb = max(left, key=lambda x: x['total_score'])
                        selected.append(fb)
                        used_nums.update(fb['nums'])

        return selected


# =============================================================================
# 6. BacktestEngine
# =============================================================================
class BacktestEngine:

    def __init__(self, csv_path: str = None,
                 config: Config = None,
                 preloaded_history: List[List[int]] = None):
        self.config = config or Config()
        if preloaded_history is not None:
            self.all_hist = preloaded_history
        elif csv_path is not None:
            df            = pd.read_csv(csv_path)
            self.all_hist = [[int(n) for n in row]
                             for row in df.iloc[:, 2:8].values]
        else:
            raise ValueError("csv_path 또는 preloaded_history 필요")
        self.total = len(self.all_hist)

    def run(self, test_rounds: int = 20,
            preds_per_round: int = 5,
            start_from_end: int = None,
            verbose: bool = True) -> Dict:
        if verbose:
            print(f"\n{'='*65}")
            print(f"  🔬 BACKTEST  {test_rounds}회차 × {preds_per_round}세트")
            print(f"{'='*65}\n")

        if start_from_end is not None:
            end_idx   = min(self.total - start_from_end + test_rounds,
                            self.total)
            start_idx = end_idx - test_rounds
        else:
            start_idx = self.total - test_rounds
            end_idx   = self.total

        start_idx     = max(start_idx, 100)
        actual_rounds = end_idx - start_idx

        if actual_rounds <= 0:
            print(f"  {C.R}[오류] 유효 검증 구간 없음{C.E}")
            return {}

        match_dist  = Counter()
        all_matches = []
        highlights  = []

        for i in range(actual_rounds):
            round_idx = start_idx + i
            actual    = set(self.all_hist[round_idx])
            engine    = DataEngine.from_history(
                            self.all_hist[:round_idx], self.config)
            if not engine.is_loaded:
                continue
            geometric = GeometricAnalyzer(engine, self.config.N_CLUSTERS)
            generator = EnsembleGenerator(engine, geometric)
            preds     = generator.generate(count=preds_per_round, verbose=False)

            round_best = 0
            for pred in preds:
                mc = len(set(pred['nums']) & actual)
                match_dist[mc] += 1
                all_matches.append(mc)
                round_best = max(round_best, mc)
                if mc >= 3:
                    highlights.append((round_idx+1, pred['nums'],
                                       sorted(actual), mc))
            if verbose and round_best >= 2:
                sym = "🎯" if round_best >= 3 else "  "
                print(f"  {sym} 회차 {round_idx+1:4d}: "
                      f"최대 {round_best}개 일치 (실제: {sorted(actual)})")

        if not all_matches:
            return {}

        avg        = float(np.mean(all_matches))
        random_exp = 6 * 6 / 45
        improvement = ((avg / random_exp) - 1) * 100 if random_exp > 0 else 0.0
        composite  = (avg
                      + match_dist.get(3, 0) * 0.3
                      + match_dist.get(4, 0) * 1.0
                      + match_dist.get(5, 0) * 5.0)

        result = {
            'avg_match':          avg,
            'best_match':         max(all_matches),
            'total_predictions':  len(all_matches),
            'match_distribution': match_dist,
            'improvement_pct':    improvement,
            'composite_score':    composite,
            'highlights':         highlights,
        }
        if verbose:
            self._print_result(result, random_exp)
        return result

    @staticmethod
    def _print_result(result: Dict, random_exp: float):
        total = result['total_predictions']
        avg   = result['avg_match']
        imp   = result['improvement_pct']
        md    = result['match_distribution']
        print(f"\n  {'─'*55}")
        print(f"  📊 백테스트 요약  ({total}세트)")
        print(f"  {'─'*55}")
        print(f"  평균 일치 : {avg:.3f}  (랜덤 기대: {random_exp:.3f})")
        print(f"  최대 일치 : {result['best_match']}개")
        print(f"  복합 점수 : {result['composite_score']:.4f}")
        for mc in sorted(md.keys(), reverse=True):
            cnt = md[mc]
            pct = cnt/total*100
            print(f"    {mc}개: {cnt:4d}회 ({pct:5.1f}%) "
                  f"{'█'*max(1,int(pct/2))}")
        tag = f"+{imp:.1f}%" if imp >= 0 else f"{imp:.1f}%"
        print(f"\n  {'✅' if imp>0 else '⚠️'} 랜덤 대비 {tag}")
        if result['highlights']:
            print(f"\n  🎯 3개 이상 일치 하이라이트")
            for rnd, pred, actual, mc in result['highlights'][:5]:
                print(f"    회차 {rnd:4d}: {pred} → {mc}개 (실제:{actual})")


# =============================================================================
# 7. HyperparameterOptimizer
# =============================================================================
class HyperparameterOptimizer:

    def __init__(self, csv_path: str, config: Config = None):
        self.config = config or Config()
        self.all_hist = safe_load_history(csv_path, verbose=False)
        self.total    = len(self.all_hist)
        self.history: List[Dict] = []

    def optimize(self, n_trials: int = 20,
                 opt_rounds: int = 10,
                 valid_rounds: int = 10,
                 preds_per_round: int = 5,
                 verbose: bool = True) -> Config:
        if verbose:
            print(f"\n{'='*65}")
            print(f"  🔧 HYPERPARAMETER OPTIMIZATION  {n_trials} Trials")
            print(f"  최적화 구간: 뒤에서 {valid_rounds+opt_rounds}~{valid_rounds}회차")
            print(f"  검증   구간: 뒤에서 {valid_rounds}~0회차  (분리됨)")
            print(f"{'='*65}\n")

        weight_names = ['frequency','recency','gap','pair','momentum','zone','markov','assoc']
        best_score   = -1.0
        best_weights = self.config.get_weights_dict()

        for trial in range(n_trials):
            raw           = np.random.dirichlet(np.ones(len(weight_names)))
            trial_weights = {name: float(raw[idx])
                             for idx, name in enumerate(weight_names)}
            trial_config  = deepcopy(self.config)
            trial_config.set_weights_from_dict(trial_weights)

            bt = BacktestEngine(preloaded_history=self.all_hist,
                                config=trial_config)
            result = bt.run(
                test_rounds     = opt_rounds,
                preds_per_round = preds_per_round,
                start_from_end  = valid_rounds + opt_rounds,
                verbose         = False)
            if not result:
                continue

            score = result.get('composite_score', 0.0)
            self.history.append({
                'trial':   trial+1,
                'weights': trial_weights,
                'score':   score,
                'avg':     result.get('avg_match', 0.0),
            })
            if score > best_score:
                best_score   = score
                best_weights = trial_weights.copy()

            if verbose:
                marker = f"{C.Y}★{C.E}" if score == best_score else " "
                top_w  = max(trial_weights.items(), key=lambda x: x[1])
                print(f"  {marker} Trial {trial+1:3d}: "
                      f"score={score:.4f}  avg={result['avg_match']:.3f}  "
                      f"top={top_w[0]}({top_w[1]:.2f})")

        if verbose:
            print(f"\n  🏆 최적 가중치:")
            for name, val in best_weights.items():
                print(f"    {name:<12}: {val:.3f}  {'█'*int(val*30)}")
            print(f"  최적 복합 점수: {best_score:.4f}\n")

        optimal = deepcopy(self.config)
        optimal.set_weights_from_dict(best_weights)
        return optimal


# =============================================================================
# 7-A. [v1.2 NEW] 랜덤 대조군 백테스트   [출처: Brother System v2.3 개념]
# =============================================================================
class RandomControlBacktest:
    """
    모델과 '진짜 무작위 대조군'을 같은 회차에 동시 투입해 비교한다.

    v3.4 원본은 랜덤 기대값을 이론값(6*6/45 = 0.8)으로 고정 비교했다.
    그러나 표본이 작을 때는 이론값 자체보다 '실제로 굴려본 랜덤'과
    비교해야 공정하다. 두 집단의 차이(diff)에 부트스트랩 신뢰구간을
    씌워서, 그 차이가 0을 포함하는지로 판정한다.

      귀무가설 H0 : 모델 평균일치 - 랜덤 평균일치 = 0
      판정        : 95% 신뢰구간이 0을 포함하면 "무작위와 구분 불가"
    """

    def __init__(self, history: List[List[int]], config: Config = None):
        self.hist   = history
        self.config = config or Config()
        self.total  = len(history)

    @staticmethod
    def _random_set() -> List[int]:
        return sorted(np.random.choice(range(1, 46), size=6, replace=False).tolist())

    def run(self, test_rounds: int = 30,
            preds_per_round: int = 5,
            control_per_round: int = 100,
            verbose: bool = True) -> Dict:

        start_idx = max(self.total - test_rounds, 100)
        end_idx   = self.total
        n_rounds  = end_idx - start_idx
        if n_rounds <= 0:
            print(f"  {C.R}[오류] 검증 구간 부족{C.E}")
            return {}

        if verbose:
            print(f"\n{'='*65}")
            print(f"  🔬 RANDOM-CONTROL BACKTEST")
            print(f"  검증 {n_rounds}회차 | 모델 {preds_per_round}세트/회 "
                  f"| 대조군 {control_per_round}세트/회")
            print(f"{'='*65}\n")

        model_matches, ctrl_matches = [], []
        model_best,   ctrl_best     = [], []

        for i in range(n_rounds):
            idx    = start_idx + i
            actual = set(self.hist[idx])

            engine = DataEngine.from_history(self.hist[:idx], self.config)
            if not engine.is_loaded:
                continue
            geo  = GeometricAnalyzer(engine, self.config.N_CLUSTERS)
            gen  = EnsembleGenerator(engine, geo)
            sets = gen.generate(count=preds_per_round, verbose=False)

            m_round = [len(actual & set(s['nums'])) for s in sets]
            if not m_round:
                continue
            model_matches.extend(m_round)
            model_best.append(max(m_round))

            c_round = [len(actual & set(self._random_set()))
                       for _ in range(control_per_round)]
            ctrl_matches.extend(c_round)
            ctrl_best.append(max(c_round[:preds_per_round]))

            if verbose and (i + 1) % 5 == 0:
                print(f"    {i+1:3d}/{n_rounds} 회차 처리...")

        if not model_matches:
            return {}

        m_avg = float(np.mean(model_matches))
        c_avg = float(np.mean(ctrl_matches))
        diff  = m_avg - c_avg

        ci_low, ci_high = ValidationModule.bootstrap_diff_ci(
            model_matches, ctrl_matches, n_boot=3000)

        verdict = "무작위와 구분 불가 (통계적 동등)" if ci_low <= 0 <= ci_high \
                  else ("무작위보다 우수" if ci_low > 0 else "무작위보다 열등")

        result = {
            'rounds':         n_rounds,
            'model_avg':      m_avg,
            'control_avg':    c_avg,
            'theory_avg':     6 * 6 / 45,
            'diff':           diff,
            'ci':             (ci_low, ci_high),
            'verdict':        verdict,
            'model_best_avg': float(np.mean(model_best)) if model_best else 0.0,
            'ctrl_best_avg':  float(np.mean(ctrl_best))  if ctrl_best  else 0.0,
            'n_model':        len(model_matches),
            'n_control':      len(ctrl_matches),
        }
        if verbose:
            self._print(result)
        return result

    @staticmethod
    def _print(r: Dict):
        print(f"\n  {'─'*57}")
        print(f"  📊 모델 vs 실제 랜덤 대조군")
        print(f"  {'─'*57}")
        print(f"  모델   평균일치 : {r['model_avg']:.4f}  (표본 {r['n_model']}세트)")
        print(f"  대조군 평균일치 : {r['control_avg']:.4f}  (표본 {r['n_control']}세트)")
        print(f"  이론   기대값   : {r['theory_avg']:.4f}")
        print(f"  차이(모델-랜덤) : {r['diff']:+.4f}")
        print(f"  95% 신뢰구간    : [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
        mark = "⚖️" if "동등" in r['verdict'] else ("✅" if "우수" in r['verdict'] else "⚠️")
        print(f"\n  {mark} 판정: {r['verdict']}")
        if "동등" in r['verdict']:
            print(f"  {C.DM}  -> 신뢰구간이 0을 포함합니다. 현재 표본으로는"
                  f" 모델 우위를 주장할 수 없습니다.{C.E}")


# =============================================================================
# 7-B. [v1.2 NEW] 검증 모듈 — 부트스트랩 신뢰구간 + 필요 표본 크기
# =============================================================================
class ValidationModule:
    """
    '+12.5% 향상' 같은 숫자가 신호인지 잡음인지 판정하는 모듈.

    로또 1세트의 일치 개수는 초기하분포 Hypergeometric(N=45, K=6, n=6)를 따른다.
      평균  mu    = n*K/N                       = 0.8000
      분산  var   = n*(K/N)*(1-K/N)*(N-n)/(N-1) = 0.6148
      표준편차    = 0.7841

    표본이 100세트면 평균의 표준오차는 0.784/10 = 0.078이다.
    즉 '평균 0.9'가 나와도 랜덤에서 흔히 나오는 값이므로 우위 근거가 못 된다.
    """

    N, K, n = 45, 6, 6

    @classmethod
    def theory_mean(cls) -> float:
        return cls.n * cls.K / cls.N

    @classmethod
    def theory_var(cls) -> float:
        N, K, n = cls.N, cls.K, cls.n
        return n * (K / N) * (1 - K / N) * (N - n) / (N - 1)

    @staticmethod
    def bootstrap_ci(samples, n_boot: int = 3000, alpha: float = 0.05):
        """단일 표본 평균의 부트스트랩 신뢰구간"""
        arr = np.asarray(samples, dtype=float)
        if len(arr) == 0:
            return (0.0, 0.0)
        idx   = np.random.randint(0, len(arr), size=(n_boot, len(arr)))
        means = arr[idx].mean(axis=1)
        return (float(np.percentile(means, alpha / 2 * 100)),
                float(np.percentile(means, (1 - alpha / 2) * 100)))

    @staticmethod
    def bootstrap_diff_ci(a, b, n_boot: int = 3000, alpha: float = 0.05):
        """두 집단 평균 차이(a-b)의 부트스트랩 신뢰구간"""
        A, B = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if len(A) == 0 or len(B) == 0:
            return (0.0, 0.0)
        ia = np.random.randint(0, len(A), size=(n_boot, len(A)))
        ib = np.random.randint(0, len(B), size=(n_boot, len(B)))
        diffs = A[ia].mean(axis=1) - B[ib].mean(axis=1)
        return (float(np.percentile(diffs, alpha / 2 * 100)),
                float(np.percentile(diffs, (1 - alpha / 2) * 100)))

    @classmethod
    def required_sample_size(cls, effect_pct: float,
                             power: float = 0.80,
                             alpha: float = 0.05) -> int:
        """
        평균일치를 effect_pct(%) 만큼 끌어올리는 실력을
        검출하려면 몇 세트가 필요한가?

          n = (z_a/2 + z_b)^2 * sigma^2 / delta^2
        """
        z_a = 1.959964   # alpha = 0.05 양측
        z_b = 0.841621   # power = 0.80
        delta = cls.theory_mean() * (effect_pct / 100.0)
        if delta <= 0:
            return -1
        return int(np.ceil(((z_a + z_b) ** 2) * cls.theory_var() / (delta ** 2)))

    @classmethod
    def print_report(cls, observed=None, label: str = "관측 표본"):
        print(f"\n{C.BD}━━━ 🧪 통계 검증 리포트 ━━━{C.E}")
        print(f"  이론 평균일치 : {cls.theory_mean():.4f}")
        print(f"  이론 표준편차 : {np.sqrt(cls.theory_var()):.4f}")

        print(f"\n  {C.BD}[표본 크기별 평균의 표준오차]{C.E}")
        print(f"  {'세트 수':>10} | {'표준오차':>9} | {'랜덤에서 나올 수 있는 95% 범위':>30}")
        print(f"  {'-'*10}-+-{'-'*9}-+-{'-'*30}")
        sd = np.sqrt(cls.theory_var())
        for n_s in [50, 100, 300, 500, 1000, 3000]:
            se = sd / np.sqrt(n_s)
            lo, hi = cls.theory_mean() - 1.96 * se, cls.theory_mean() + 1.96 * se
            print(f"  {n_s:>10,} | {se:>9.4f} | {lo:>13.3f} ~ {hi:<13.3f}")

        print(f"\n  {C.BD}[검출하려는 향상폭별 필요 표본 크기]  (유의수준 5%, 검정력 80%){C.E}")
        print(f"  {'향상폭':>8} | {'필요 세트 수':>14} | {'주 5세트 기준 소요 기간':>22}")
        print(f"  {'-'*8}-+-{'-'*14}-+-{'-'*22}")
        for pct in [5, 10, 20, 30, 50]:
            need  = cls.required_sample_size(pct)
            weeks = need / 5
            years = weeks / 52
            span  = f"{weeks:,.0f}주 (약 {years:,.1f}년)"
            print(f"  {pct:>7}% | {need:>14,} | {span:>22}")

        if observed is not None and len(observed) > 0:
            lo, hi = cls.bootstrap_ci(observed)
            avg    = float(np.mean(observed))
            inside = lo <= cls.theory_mean() <= hi
            print(f"\n  {C.BD}[{label} 판정]{C.E}")
            print(f"  표본 크기     : {len(observed):,}세트")
            print(f"  관측 평균일치 : {avg:.4f}")
            print(f"  95% 신뢰구간  : [{lo:.4f}, {hi:.4f}]")
            if inside:
                print(f"  ⚖️  이론 랜덤값 {cls.theory_mean():.3f}이 신뢰구간 안에 있음"
                      f" → 무작위와 통계적으로 동등")
            else:
                side = "우수" if avg > cls.theory_mean() else "열등"
                print(f"  {'✅' if side=='우수' else '⚠️'}  이론 랜덤값이 신뢰구간 밖 → 무작위보다 {side}")


# =============================================================================
# 7-C. [v1.2 NEW] 다중 윈도우 분석   [출처: Brother System v2.3 개념]
# =============================================================================
class MultiWindowAnalyzer:
    """
    같은 데이터를 최근 30 / 50 / 100회 / 전체 로 잘라서 각각 예측을 돌린다.

    교육적 목적:
      "어느 구간을 보느냐"라는 단 하나의 파라미터만 바꿔도 추천 번호가
      얼마나 달라지는지를 눈으로 보여준다. 결과가 크게 흔들린다면
      그 모델은 데이터의 구조가 아니라 잡음을 읽고 있다는 뜻이다.
    """

    def __init__(self, history: List[List[int]], config: Config = None,
                 windows: List[int] = None):
        self.hist    = history
        self.config  = config or Config()
        self.windows = windows or [30, 50, 100, 0]   # 0 = 전체

    def run(self, count: int = 5, verbose: bool = True) -> Dict:
        out = {}
        for w in self.windows:
            label   = "전체" if w == 0 else f"최근 {w}회"
            subset  = self.hist if w == 0 else self.hist[-w:]
            if len(subset) < 20:
                continue

            cfg = deepcopy(self.config)
            cfg.N_CLUSTERS = min(5, max(2, len(subset) // 15))

            engine = DataEngine.from_history(subset, cfg)
            if not engine.is_loaded:
                continue
            geo  = GeometricAnalyzer(engine, cfg.N_CLUSTERS)
            gen  = EnsembleGenerator(engine, geo)
            sets = gen.generate(count=count, verbose=False)

            out[label] = {
                'window':  w,
                'rounds':  len(subset),
                'sets':    [s['nums'] for s in sets],
                'top10':   [n for n, _ in engine.get_hot_numbers(10)],
                'engine':  engine,
            }
            if verbose:
                print(f"    ✓ [{label}] {len(sets)}세트 생성 "
                      f"(분석 {len(subset)}회차)")
        self.result = out
        return out

    def print_report(self):
        if not getattr(self, 'result', None):
            return
        print(f"\n{C.BD}━━━ 🪟 다중 윈도우 비교 (파라미터 민감도) ━━━{C.E}")

        for label, d in self.result.items():
            print(f"\n  {C.CN}[{label}]{C.E}  분석 {d['rounds']}회차 "
                  f"| Top10: {d['top10']}")
            for i, nums in enumerate(d['sets'], 1):
                disp = "  ".join(f"{n:2d}" for n in nums)
                print(f"    Set {i}: [ {disp} ]   합:{sum(nums):3d}")

        # ── 민감도 지표: 윈도우 간 Top10 겹침률 ──────────────────
        labels = list(self.result.keys())
        print(f"\n  {C.BD}[Top10 번호 겹침률 — 낮을수록 결과가 불안정]{C.E}")
        header = "  " + " " * 10 + "".join(f"{l:>10}" for l in labels)
        print(header)
        overlaps = []
        for a in labels:
            row = f"  {a:<10}"
            for b in labels:
                ov = len(set(self.result[a]['top10']) & set(self.result[b]['top10']))
                row += f"{ov:>9}개"
                if a != b:
                    overlaps.append(ov)
            print(row)

        if overlaps:
            mean_ov = np.mean(overlaps)
            print(f"\n  평균 겹침: {mean_ov:.1f}/10개")
            if mean_ov < 5:
                print(f"  {C.Y}⚠️  구간만 바꿔도 상위 번호의 절반 이상이 교체됩니다.{C.E}")
                print(f"  {C.DM}   모델이 안정적 구조가 아니라 구간별 잡음을"
                      f" 읽고 있다는 신호입니다.{C.E}")
            else:
                print(f"  {C.G}   상위 번호가 비교적 안정적으로 유지됩니다.{C.E}")

        all_nums = set()
        for d in self.result.values():
            for s in d['sets']:
                all_nums.update(s)
        print(f"  전체 윈도우 통합 커버리지: {len(all_nums)}/45종 "
              f"({len(all_nums)/45*100:.0f}%)")


# =============================================================================
# 7-D. [v1.2] 예측 추적기 — 사전 기록 후 사후 채점
# =============================================================================
class PredictionTracker:
    """
    예측을 '먼저' 파일에 기록하고, 나중에 실제 당첨번호로 채점한다.
    사후에 잘 맞은 것만 골라 보는 편향을 원천 차단하기 위한 장치다.
    """

    COLUMNS = ['timestamp', 'target_round', 'source', 'numbers',
               'matched', 'actual', 'graded']

    def __init__(self, path: str = 'prediction_log.csv'):
        self.path = path
        try:
            # utf-8-sig 로 저장했으므로 읽을 때도 동일하게 (BOM 컬럼명 오염 방지)
            self.df = pd.read_csv(path, encoding='utf-8-sig')
            for c in self.COLUMNS:
                if c not in self.df.columns:
                    self.df[c] = np.nan
            self.df = self.df[self.COLUMNS]
        except Exception:
            self.df = pd.DataFrame(columns=self.COLUMNS)
        self._fix_dtypes()

    def _fix_dtypes(self):
        """빈 컬럼이 float64 로 잡혀 문자열 대입 시 터지는 문제 방지"""
        if len(self.df) == 0:
            return
        self.df['actual']  = self.df['actual'].astype('object')
        self.df['source']  = self.df['source'].astype('object')
        self.df['numbers'] = self.df['numbers'].astype('object')
        self.df['matched'] = pd.to_numeric(self.df['matched'], errors='coerce')
        self.df['graded']  = pd.to_numeric(self.df['graded'],
                                           errors='coerce').fillna(0).astype(int)

    def log(self, target_round: int, sets: List[List[int]],
            source: str = 'v1.2-main', verbose: bool = True) -> int:
        import datetime
        ts   = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        rows, dup = [], 0
        existing = set()
        if len(self.df) > 0:
            existing = {(int(r), str(s))
                        for r, s in zip(self.df['target_round'].fillna(-1),
                                        self.df['numbers'].fillna(''))}
        for nums in sets:
            key = (int(target_round), ",".join(map(str, sorted(nums))))
            if key in existing:
                dup += 1
                continue
            rows.append({'timestamp': ts, 'target_round': int(target_round),
                         'source': source, 'numbers': key[1],
                         'matched': np.nan, 'actual': '', 'graded': 0})
        if rows:
            self.df = pd.concat([self.df, pd.DataFrame(rows)], ignore_index=True)
            self._fix_dtypes()
            self.df.to_csv(self.path, index=False, encoding='utf-8-sig')
        if verbose:
            print(f"  📝 예측 기록: {len(rows)}건 저장"
                  f"{f' (중복 {dup}건 제외)' if dup else ''} → {self.path}")
        return len(rows)

    def grade(self, target_round: int, actual: List[int],
              verbose: bool = True) -> Dict:
        if len(self.df) == 0:
            return {}
        self._fix_dtypes()
        actual_set = set(actual)
        mask = (self.df['target_round'] == target_round) & (self.df['graded'] != 1)
        idxs = self.df.index[mask].tolist()
        for i in idxs:
            nums = [int(x) for x in str(self.df.at[i, 'numbers']).split(',')]
            self.df.at[i, 'matched'] = len(actual_set & set(nums))
            self.df.at[i, 'actual']  = ",".join(map(str, sorted(actual)))
            self.df.at[i, 'graded']  = 1
        self.df.to_csv(self.path, index=False, encoding='utf-8-sig')

        graded = self.df[self.df['graded'] == 1]['matched'].dropna().astype(float)
        res = {'newly_graded': len(idxs), 'cumulative_n': len(graded),
               'cumulative_avg': float(graded.mean()) if len(graded) else 0.0}
        if verbose:
            print(f"\n{C.BD}━━━ 📝 예측 추적 채점 ━━━{C.E}")
            print(f"  {target_round}회차 채점: {len(idxs)}건")
            print(f"  누적 표본 : {res['cumulative_n']}세트")
            print(f"  누적 평균 : {res['cumulative_avg']:.4f} "
                  f"(이론 랜덤 {ValidationModule.theory_mean():.4f})")
            if res['cumulative_n'] >= 30:
                ValidationModule.print_report(graded.tolist(), "누적 실전 예측")
            else:
                need = ValidationModule.required_sample_size(10)
                print(f"  {C.DM}  판정을 위해 최소 수백 세트 필요"
                      f" (10% 향상 검출 기준 {need:,}세트){C.E}")
        return res


# =============================================================================
# 8. 대시보드 출력
# =============================================================================
def print_dashboard(engine: DataEngine,
                    results: List[Dict],
                    geometric: GeometricAnalyzer = None,
                    generator: 'EnsembleGenerator' = None):

    print(f"\n{C.H}")
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║      🎰 LOTTO ANALYTICS ULTIMATE v3.4 — Enhanced Edition    ║")
    print("║   Soft Clustering + Coverage Opt + Cluster Diversity v2     ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print(f"{C.E}")

    print(f"{C.BD}━━━ 📊 DATA SUMMARY ━━━{C.E}")
    print(f"  총 {engine.total_rounds}회차  |  마지막 당첨: {sorted(engine.last_draw)}")

    if geometric:
        print(f"\n{C.BD}━━━ 🔮 CLUSTER ANALYSIS ━━━{C.E}")
        print(f"  {geometric.n_clusters}개 클러스터  |  "
              f"상위 타겟: {geometric.top_clusters}")
        for cid in geometric.top_clusters:
            st = geometric.cluster_stats[cid]
            print(f"    C{cid}: {st['count']:3d}회 ({st['ratio']*100:4.1f}%)  "
                  f"평균합:{st['avg_sum']:5.0f}  최근:{st['recent_ratio']*100:4.1f}%")

    print(f"\n{C.BD}━━━ 🔥 HOT / ❄️ COLD ━━━{C.E}")
    hot  = engine.get_hot_numbers(8)
    cold = engine.get_cold_numbers(5)
    print(f"  HOT : {' '.join(f'{C.R}{n:2d}{C.E}' for n,_ in hot)}")
    print(f"  COLD: {' '.join(f'{C.B}{n:2d}{C.E}' for n,_ in cold)}")

    print(f"\n{C.BD}━━━ 🎯 PREDICTION RESULTS ━━━{C.E}")
    print(f"  {'#':<3}│{'번호 조합':<26}│{'합':>4}│{'AC':>3}│"
          f"{'홀':>2}│{'소수':>3}│{'CID':>4}│{'Cls':>5}│{'점수':>8}│Algo")
    print(f"  {'─'*80}")
    for i, res in enumerate(results, 1):
        n   = res['nums']
        st  = res['stat']
        ns  = " ".join(f"{x:2d}" for x in n)
        cid = st.get('cluster_id', -1)
        cb  = st.get('cluster_bonus', 0.0)
        print(f"  {i:<3}│{C.G}{ns:<26}{C.E}│{st['sum']:4d}│"
              f"{st['ac']:3d}│{st['odd']:2d}│{st['prime']:3d}│"
              f"{cid:4d}│{cb:5.2f}│{res['total_score']:8.5f}│{res['algo']}")
    print(f"  {'─'*80}")

    all_nums = set(n for r in results for n in r['nums'])
    cov      = len(all_nums)
    print(f"\n  📈 커버리지: {cov}개 / 45개 ({cov/45*100:.1f}%)")

    if geometric:
        cls_dist = Counter(r['stat'].get('cluster_id', -1) for r in results)
        print(f"  🔀 클러스터 분포: "
              + "  ".join(f"C{cid}:{cnt}세트"
                          for cid, cnt in sorted(cls_dist.items())))

    print(f"\n  🔁 세트 간 중복 현황:")
    has_overlap = False
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            ov = set(results[i]['nums']) & set(results[j]['nums'])
            if ov:
                print(f"    세트 {i+1}↔{j+1}: {sorted(ov)} ({len(ov)}개)")
                has_overlap = True
    if not has_overlap:
        print(f"    (중복 없음)")

    if generator and generator.filter.rejection_stats:
        print(f"\n{C.DM}━━━ 🔍 Filter Rejection Stats ━━━{C.E}")
        total_chk  = generator.filter.total_checked
        total_pass = generator.filter.total_passed
        print(f"  전체 시도: {total_chk}회  통과: {total_pass}회  "
              f"통과율: {generator.filter.get_pass_rate()*100:.1f}%")
        for name, cnt, pct in generator.filter.get_rejection_summary(6):
            bar = "█" * max(1, int(pct/3))
            print(f"  {name:<8}: {cnt:5d}회 ({pct:5.1f}%) {bar}")

    print(f"\n{C.CN}━━━ 📋 COPY FORMAT ━━━{C.E}")
    for i, res in enumerate(results, 1):
        print(f"{', '.join(str(n) for n in res['nums'])}")
    print()


# =============================================================================
# 9. [v1.2] 통합 실행 오케스트레이션
# =============================================================================
def main(csv_path: str,
         count: int          = 5,
         run_optimize: bool  = False,
         run_backtest: bool  = False,
         run_multiwindow: bool = True,
         run_validation: bool  = True,
         track: bool           = True,
         seed: int             = 42):
    """
    통합 실행 파이프라인

      [0] 안전 로더로 데이터 적재 및 감사
      [1] (선택) 하이퍼파라미터 최적화  — 학습/검증 구간 분리
      [2] (선택) 랜덤 대조군 백테스트   — 부트스트랩 신뢰구간 판정
      [3] 메인 예측 (8종 가중 스코어 + 4종 생성 알고리즘 + 커버리지 최적화)
      [4] (선택) 다중 윈도우 비교       — 파라미터 민감도
      [5] (선택) 통계 검증 리포트       — 필요 표본 크기
      [6] (선택) 예측 사전 기록
    """
    np.random.seed(seed)
    random.seed(seed)

    print(f"\n{C.CN}╔════════════════════════════════════════════════════════════╗{C.E}")
    print(f"{C.CN}║   🎰 LOTTO INTEGRATED v1.2 — FINAL UNIFIED EDITION         ║{C.E}")
    print(f"{C.CN}║   v3.4 엔진 + v1.2 로더 + v2.3 대조군 + 마르코프/연관규칙  ║{C.E}")
    print(f"{C.CN}╚════════════════════════════════════════════════════════════╝{C.E}\n")

    # ── [0] 데이터 적재 및 감사 ─────────────────────────────────
    print(f"{C.Y}[0] 데이터 적재 및 무결성 감사{C.E}")
    try:
        history = safe_load_history(csv_path, verbose=True)
    except FileNotFoundError:
        print(f"{C.R}  ❌ 파일 없음: {csv_path}{C.E}")
        return None
    except Exception as e:
        print(f"{C.R}  ❌ 로드 실패: {e}{C.E}")
        return None
    print(f"{C.G}  ✓ 유효 회차 {len(history)}회 확보{C.E}")

    config = Config()

    # ── [1] 하이퍼파라미터 최적화 ───────────────────────────────
    if run_optimize:
        print(f"\n{C.Y}[1] 하이퍼파라미터 최적화 (8차원 탐색){C.E}")
        optimizer = HyperparameterOptimizer(csv_path, config)
        config    = optimizer.optimize(n_trials=20, opt_rounds=10,
                                       valid_rounds=10, preds_per_round=5)
    else:
        print(f"\n{C.DM}[1] 최적화 스킵 (기본 가중치 사용){C.E}")

    # ── [2] 랜덤 대조군 백테스트 ────────────────────────────────
    bt_result = None
    if run_backtest:
        print(f"\n{C.Y}[2] 랜덤 대조군 백테스트{C.E}")
        rcb       = RandomControlBacktest(history, config)
        bt_result = rcb.run(test_rounds=20, preds_per_round=5,
                            control_per_round=100, verbose=True)
    else:
        print(f"{C.DM}[2] 백테스트 스킵{C.E}")

    # ── [3] 메인 예측 ───────────────────────────────────────────
    print(f"\n{C.Y}[3] 메인 분석 및 예측{C.E}")
    engine = DataEngine.from_history(history, config)
    if not engine.is_loaded:
        print(f"{C.R}  ❌ {engine.error_msg}{C.E}")
        return None

    geometric = GeometricAnalyzer(engine, config.N_CLUSTERS)
    print(f"{C.G}  ✓ 클러스터링 완료 "
          f"({geometric.n_clusters}개, 상위: {geometric.top_clusters}){C.E}")

    generator = EnsembleGenerator(engine, geometric)
    results   = generator.generate(count=count, verbose=True)
    if not results:
        print(f"{C.R}  ❌ 예측 결과 없음{C.E}")
        return None

    print_dashboard(engine, results, geometric, generator)
    print_new_algo_panel(engine)

    # ── [4] 다중 윈도우 비교 ────────────────────────────────────
    mw = None
    if run_multiwindow:
        print(f"\n{C.Y}[4] 다중 윈도우 비교 분석{C.E}")
        mw = MultiWindowAnalyzer(history, config)
        mw.run(count=count, verbose=True)
        mw.print_report()
    else:
        print(f"{C.DM}[4] 다중 윈도우 스킵{C.E}")

    # ── [5] 통계 검증 리포트 ────────────────────────────────────
    if run_validation:
        print(f"\n{C.Y}[5] 통계 검증{C.E}")
        ValidationModule.print_report()

    # ── [6] 예측 사전 기록 ──────────────────────────────────────
    tracker = None
    if track:
        print(f"\n{C.Y}[6] 예측 사전 기록{C.E}")
        tracker = PredictionTracker()
        tracker.log(target_round=len(history) + 1,
                    sets=[r['nums'] for r in results],
                    source='v1.2-main')

    # ── 최종 복사용 출력 ────────────────────────────────────────
    print(f"\n{C.BD}━━━ 📋 최종 추천 (복사용) ━━━{C.E}")
    for r in results:
        print(", ".join(map(str, r['nums'])))

    print(f"\n{'='*65}")
    print("  ⚠️  로또 추첨은 회차 간 독립 시행입니다.")
    print("     이 도구는 당첨 확률을 높이지 못합니다.")
    print("     조합 품질 관리와 통계 학습을 위한 도구로 사용하세요.")
    print(f"{'='*65}\n")

    return {'engine': engine, 'geometric': geometric,
            'generator': generator, 'results': results,
            'backtest': bt_result, 'multiwindow': mw,
            'tracker': tracker, 'config': config, 'history': history}


# =============================================================================
# 9-A. [v1.2 NEW] 신규 알고리즘 패널 출력
# =============================================================================
def print_new_algo_panel(engine: DataEngine):
    mk = engine.markov_analysis
    ac = engine.assoc_analysis
    cfg = engine.config

    print(f"\n{C.BD}━━━ 🔗 신규 알고리즘 패널 (v1.2) ━━━{C.E}")

    print(f"\n  {C.CN}[마르코프 전이확률]{C.E}  가중치 {cfg.WEIGHT_MARKOV:.2f} "
          f"| 조건 = 직전 회차 {mk['basis']}")
    top_mk = sorted(range(1, 46), key=lambda n: mk['raw'][n], reverse=True)[:8]
    for n in top_mk:
        bar = '█' * max(1, int(mk['score'][n] * 22))
        print(f"    {n:2d}번  P={mk['raw'][n]:.5f}  {bar}")

    print(f"\n  {C.CN}[연관규칙 Lift]{C.E}  가중치 {cfg.WEIGHT_ASSOC:.2f} "
          f"| 산출 윈도우 최근 {ac['window']}회")
    print(f"    최강 동반 규칙 Top 5:")
    for a, b, v in ac['top_rules'][:5]:
        tag = "동반 경향" if v > 1 else "회피 경향"
        print(f"      {a:2d} ↔ {b:2d}   lift={v:.2f}  ({tag})")
    top_ac = sorted(range(1, 46), key=lambda n: ac['raw'][n], reverse=True)[:8]
    ac_txt = ", ".join("{}({:.2f})".format(n, ac['raw'][n]) for n in top_ac)
    print("    직전 회차와의 평균 lift 상위: " + ac_txt)

    print(f"\n  {C.DM}  * 두 알고리즘 모두 계단식 구간이 아닌 연속 실수값입니다."
          f"{C.E}")
    uniq_mk = len(set(np.round(mk['score'][1:], 9)))
    uniq_ac = len(set(np.round(ac['score'][1:], 9)))
    print(f"  {C.DM}    고유값 수 — 마르코프 {uniq_mk}/45, 연관규칙 {uniq_ac}/45"
          f" (동점 최소화 확인){C.E}")


# =============================================================================
# 10. 편의 함수 & Entry Point
# =============================================================================
def quick_predict(csv_path: str, count: int = 5, seed: int = 42):
    """빠른 예측 — 백테스트/최적화 없이 (수 초)"""
    return main(csv_path, count, run_optimize=False, run_backtest=False,
                run_multiwindow=False, run_validation=False,
                track=False, seed=seed)


def standard_run(csv_path: str, count: int = 5, seed: int = 42):
    """표준 실행 — 예측 + 다중 윈도우 + 검증 + 기록 (수십 초)"""
    return main(csv_path, count, run_optimize=False, run_backtest=False,
                run_multiwindow=True, run_validation=True,
                track=True, seed=seed)


def full_analysis(csv_path: str, count: int = 5, seed: int = 42):
    """전체 분석 — 최적화 + 랜덤 대조군 백테스트 포함 (수 분)"""
    return main(csv_path, count, run_optimize=True, run_backtest=True,
                run_multiwindow=True, run_validation=True,
                track=True, seed=seed)


def grade_last_round(target_round: int, actual: List[int],
                     path: str = 'prediction_log.csv'):
    """당첨 발표 후 채점:  grade_last_round(1236, [3,11,22,29,37,45])"""
    return PredictionTracker(path).grade(target_round, actual)


if __name__ == '__main__':
    CSV_PATH = '/content/new_1245.csv'   # ← Colab 경로. 로컬은 파일명만 두면 됩니다.

    # 표준 실행 (권장)
    out = standard_run(CSV_PATH, count=5)

    # 전체 분석이 필요하면 아래 주석 해제 (수 분 소요)
    # out = full_analysis(CSV_PATH, count=5)

    # 당첨 발표 후 채점 예시
    # grade_last_round(1236, [3, 11, 22, 29, 37, 45])