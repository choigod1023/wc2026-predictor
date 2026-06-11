"""
evaluate.py — 대회 후(또는 중간) 판정: 모델 vs 시장 Brier Score 대결
====================================================================
필요한 것:
  1. data/results.csv 에 월드컵 경기 결과가 채워져 있을 것 (--update 로 갱신)
  2. data/closing_odds.csv 에 사용자가 기록한 마감 배당 (소수 배당, 예: 1.85)

판정 로직:
  - 모델 확률: 개막 전 고정본 data/group_stage_predictions.csv (사후 수정 금지)
  - 시장 확률: 마감 배당 역수를 합으로 나눠 정규화(마진 제거)
      p_i = (1/odds_i) / Σ(1/odds_j)
  - 두 확률 세트의 멀티클래스 Brier를 같은 경기들에 대해 비교.
    모델 < 시장 이면 "시장보다 정확했다"는 실증.
"""
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import pandas as pd
import numpy as np

pred = pd.read_csv('data/group_stage_predictions.csv')
res = pd.read_csv('data/results.csv')
res = res[(res['date'] >= '2026-06-11') & (res['tournament'] == 'FIFA World Cup')]
res = res[res['home_score'].notna()].copy()

if len(res) == 0:
    raise SystemExit('아직 채점할 결과가 없습니다. python run_pipeline.py --update 먼저 실행.')

merged = pred.merge(res[['date', 'home_team', 'away_team', 'home_score', 'away_score']],
                    left_on=['date', 'home', 'away'],
                    right_on=['date', 'home_team', 'away_team'])
hs, as_ = merged['home_score'].astype(int), merged['away_score'].astype(int)
merged['outcome'] = np.where(hs > as_, 'H', np.where(hs == as_, 'D', 'A'))

def brier(p_h, p_d, p_a, outcome):
    onehot = np.stack([(outcome == 'H'), (outcome == 'D'), (outcome == 'A')], axis=1).astype(float)
    probs = np.stack([p_h, p_d, p_a], axis=1)
    return np.mean(np.sum((probs - onehot) ** 2, axis=1))

b_model = brier(merged['P_home'], merged['P_draw'], merged['P_away'], merged['outcome'])
print(f'채점 경기 수: {len(merged)}')
print(f'모델 Brier: {b_model:.4f}')

odds_path = 'data/closing_odds.csv'
if _os.path.exists(odds_path):
    odds = pd.read_csv(odds_path).dropna(subset=['odds_H', 'odds_D', 'odds_A'])
    m2 = merged.merge(odds[['date', 'home', 'away', 'odds_H', 'odds_D', 'odds_A']],
                      on=['date', 'home', 'away'])
    if len(m2) > 0:
        inv = 1 / m2[['odds_H', 'odds_D', 'odds_A']].values
        market = inv / inv.sum(axis=1, keepdims=True)  # 마진 제거 정규화
        b_market = brier(market[:, 0], market[:, 1], market[:, 2], m2['outcome'])
        b_model2 = brier(m2['P_home'], m2['P_draw'], m2['P_away'], m2['outcome'])
        print(f'\n배당 기록이 있는 {len(m2)}경기 직접 대결:')
        print(f'  모델 Brier: {b_model2:.4f}')
        print(f'  시장 Brier: {b_market:.4f}')
        print('  →', '모델 승리 (시장보다 정확)' if b_model2 < b_market else '시장 승리 (격차가 다음 버전의 개선 목표)')
    else:
        print('closing_odds.csv 에 채점 가능한 경기가 아직 없습니다.')
else:
    print('closing_odds.csv 없음 — 시장과의 직접 대결은 배당 기록 후 가능.')
