"""
compare_models.py — 여러 모델을 동시에 검증·시뮬레이션·점수화
==============================================================
하나의 모델이 아니라 models.REGISTRY 의 모든 후보를 같은 기준으로 비교한다.

1) walk-forward Brier 리더보드
   - 학습 1990~2023 / 검증 2024~2026.6 경쟁경기(친선 제외)
   - 각 모델의 멀티클래스 Brier, log-loss, 단순 적중률
2) 모델별 우승 확률 시뮬레이션
   - 진짜 모델(SIM_MODELS)만 전체기간 재학습 후 72경기 확률 → 몬테카를로
   - 단계별(R32~결승) 도달 확률 + 우승 확률을 모델별로 비교
출력(JSON, 웹에서 그대로 소비):
   data/model_leaderboard.json
   data/champion_by_model.json
   data/stage_probs.json            (대표모델=Elo-로지스틱 기준 단계별 확률)
"""
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import numpy as np
import pandas as pd

from models import REGISTRY, SIM_MODELS, HDA
from sim import simulate, recover_groups

N_SIM = 10000
HOME_ADV = 100

# ── 데이터 ────────────────────────────────────────────────────
hist = pd.read_csv('data/elo_history.csv')
comp = hist[(hist['tournament'] != 'Friendly') & (hist['date'] >= '1990-01-01')].copy()
train = comp[comp['date'] <= '2023-12-31']
test = comp[comp['date'] >= '2024-01-01']

Xtr, ytr = train['elo_diff_pre'].values, train['outcome'].values
Xte, yte = test['elo_diff_pre'].values, test['outcome'].values
yte_idx = np.array([HDA.index(o) for o in yte])
onehot = np.zeros((len(yte), 3))
onehot[np.arange(len(yte)), yte_idx] = 1


def brier(p):
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def logloss(p):
    p = np.clip(p, 1e-12, 1)
    return float(-np.mean(np.log(p[np.arange(len(yte)), yte_idx])))


def accuracy(p):
    return float(np.mean(p.argmax(1) == yte_idx))


print(f'학습 {len(train)}경기 / 검증 {len(test)}경기\n')
print(f'{"모델":24s} {"Brier":>8s} {"LogLoss":>8s} {"적중률":>7s}')
print('-' * 52)
leaderboard = []
for cls in REGISTRY:
    m = cls().fit(Xtr, ytr)
    p = m.predict_proba(Xte)
    b, ll, ac = brier(p), logloss(p), accuracy(p)
    leaderboard.append({'name': m.name, 'short': m.short,
                        'brier': round(b, 4), 'logloss': round(ll, 4),
                        'accuracy': round(ac, 4)})
    print(f'{m.name:24s} {b:8.4f} {ll:8.4f} {ac:6.1%}')

leaderboard.sort(key=lambda r: r['brier'])
leaderboard[0]['best'] = True
json.dump(leaderboard, open('data/model_leaderboard.json', 'w'),
          ensure_ascii=False, indent=2)

# ── 모델별 우승/단계 시뮬레이션 ───────────────────────────────
ratings = pd.read_csv('data/elo_final.csv', index_col=0)['elo'].to_dict()
results = pd.read_csv('data/results.csv')
wc = results[(results['date'] >= '2026-06-11') &
             (results['tournament'] == 'FIFA World Cup')].copy()
groups = recover_groups(wc)

# 전체기간 재학습용
Xfull, yfull = comp['elo_diff_pre'].values, comp['outcome'].values
wc_sorted = list(wc.sort_values('date').itertuples())
diffs = np.array([ratings[r.home_team] + (0 if r.neutral else HOME_ADV)
                  - ratings[r.away_team] for r in wc_sorted])

champ_by_model = []
print('\n=== 모델별 우승 확률 Top 8 ===')
for cls in SIM_MODELS:
    m = cls().fit(Xfull, yfull)
    p72 = m.predict_proba(diffs)            # (72,3) [H,D,A]
    match_p = {(r.home_team, r.away_team): tuple(p72[i])
               for i, r in enumerate(wc_sorted)}
    out = simulate(match_p, groups, ratings, n_sim=N_SIM)
    champ_sorted = sorted(out['champion'].items(), key=lambda kv: kv[1], reverse=True)
    champ_by_model.append({
        'model': m.name, 'short': m.short,
        'champions': [{'team': t, 'p': round(v, 4)} for t, v in champ_sorted if v > 0][:24],
    })
    print(f'\n[{m.name}]')
    for t, v in champ_sorted[:8]:
        print(f'  {t:18s} {v:6.1%}')

# 주: 대표 단계별 확률(stage_probs.json)은 predict.py 의 스코어 기반 시뮬이 생성한다.
# 여기서는 W/D/A 모델들의 우승 확률 비교(champion_by_model)만 출력.
json.dump(champ_by_model, open('data/champion_by_model.json', 'w'),
          ensure_ascii=False, indent=2)
print('\n저장: model_leaderboard.json, champion_by_model.json')
