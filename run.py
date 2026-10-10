# -*- coding: utf-8 -*-
"""
LOTTO LAB — 번호 생성 진입점

  python3 run.py                       기본 5세트, 인기 하위 10%
  python3 run.py --count 10 --pct 5    10세트, 인기 하위 5%
  python3 run.py --seed 12345          재현 실행
  python3 run.py --csv data/new_1246.csv
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'engine'))
from bootstrap import load_engine, data            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--csv', default=None, help='당첨번호 CSV (기본 data/latest.csv)')
ap.add_argument('--count', type=int, default=5, help='생성할 세트 수')
ap.add_argument('--pct', default='10', help='인기도 상위 차단 기준 %% (off 면 감점만)')
ap.add_argument('--seed', type=int, default=None, help='재현용 시드 (생략 시 매번 다름)')
a = ap.parse_args()

v12, p13 = load_engine()
path = a.csv or data('latest.csv')
hist = v12.safe_load_history(path, verbose=True)
print(f'  유효 회차 {len(hist)}회 | 마지막 당첨 {hist[-1]}\n')

pct = None if str(a.pct).lower() == 'off' else float(a.pct)
out = p13.predict_v131(hist, count=a.count, seed=a.seed,
                       pop_max_pct=pct, verbose=False)

print(f"{'#':<4}{'번호':<26}{'합':>5}{'AC':>4}{'홀':>4}{'인기분위':>10}")
print('-' * 54)
for i, d in enumerate(out['detail'], 1):
    s = d['stat']
    pp = f"{s['pop_pct']:.1f}%" if 'pop_pct' in s else '—'
    print(f"{i:<4}{' '.join(f'{x:2d}' for x in d['nums']):<26}"
          f"{s['sum']:>5}{s['ac']:>4}{s['odd']:>4}{pp:>10}")

print(f"\n재현용 seed: {out['seed']}")
print('\n복사용')
for d in out['detail']:
    print(', '.join(map(str, d['nums'])))

print('\n' + '=' * 54)
print('  이 도구는 당첨 확률을 높이지 못합니다 (1/8,145,060).')
print('  인기 조합을 피해 당첨 시 분할 인원을 줄일 뿐입니다.')
print('  도박 문제 상담: 1336 (한국도박문제예방치유원)')
print('=' * 54)
