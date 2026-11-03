"""Compare frozen match probabilities with the stored odds on matched results.

The odds file has no per-row observation time: this is an exploratory comparison,
not verification that each price was observed immediately before kickoff.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KEYS = ['date', 'home', 'away']
ODDS = ['odds_H', 'odds_D', 'odds_A']


def brier(p_h, p_d, p_a, outcome):
    onehot = np.stack([(outcome == 'H'), (outcome == 'D'), (outcome == 'A')], axis=1).astype(float)
    probs = np.stack([p_h, p_d, p_a], axis=1)
    return np.mean(np.sum((probs - onehot) ** 2, axis=1))


def main():
    pred = pd.read_csv(ROOT / 'data/group_stage_predictions.csv')
    res = pd.read_csv(ROOT / 'data/results.csv')
    res = res[(res['date'] >= '2026-06-11') & (res['tournament'] == 'FIFA World Cup')]
    res = res.dropna(subset=['home_score', 'away_score'])

    if res.empty:
        raise SystemExit('아직 채점할 결과가 없습니다. python run_pipeline.py --update 먼저 실행.')

    merged = pred.merge(res[['date', 'home_team', 'away_team', 'home_score', 'away_score']],
                        left_on=KEYS, right_on=['date', 'home_team', 'away_team'],
                        validate='one_to_one')
    if merged.empty:
        raise SystemExit('예측과 일치하는 경기 결과가 없습니다. 날짜와 팀명 매핑을 확인하세요.')

    hs, away = merged['home_score'].astype(int), merged['away_score'].astype(int)
    merged['outcome'] = np.where(hs > away, 'H', np.where(hs == away, 'D', 'A'))
    b_model = brier(merged['P_home'], merged['P_draw'], merged['P_away'], merged['outcome'])
    print(f'채점 경기 수: {len(merged)}')
    print(f'모델 Brier: {b_model:.4f}')

    odds_path = ROOT / 'data/closing_odds.csv'
    if not odds_path.exists():
        print('closing_odds.csv 없음 — 시장과의 비교는 배당 기록 후 가능.')
        return

    odds = pd.read_csv(odds_path)
    odds[ODDS] = odds[ODDS].apply(pd.to_numeric, errors='coerce')
    odds = odds[np.isfinite(odds[ODDS]).all(axis=1) & (odds[ODDS] > 1).all(axis=1)]
    m2 = merged.merge(odds[KEYS + ODDS], on=KEYS, validate='one_to_one')
    if m2.empty:
        print('유효한 배당과 결과가 모두 일치하는 경기가 없습니다.')
        return

    inv = 1 / m2[ODDS].to_numpy()
    market = inv / inv.sum(axis=1, keepdims=True)
    b_market = brier(market[:, 0], market[:, 1], market[:, 2], m2['outcome'])
    b_model2 = brier(m2['P_home'], m2['P_draw'], m2['P_away'], m2['outcome'])
    print(f'\n배당 기록이 있는 {len(m2)}경기 탐색적 비교:')
    print(f'  모델 Brier: {b_model2:.4f}')
    print(f'  시장 Brier: {b_market:.4f}')
    print('  ※ 배당 관측 시각이 기록되지 않아 마감 배당 여부는 확인할 수 없습니다.')
    if np.isclose(b_model2, b_market, rtol=0, atol=1e-12):
        print('  → 두 점수가 같습니다.')
    elif b_model2 < b_market:
        print('  → 저장된 표본에서는 모델 점수가 더 낮습니다.')
    else:
        print('  → 저장된 표본에서는 시장 점수가 더 낮습니다.')


if __name__ == '__main__':
    main()
