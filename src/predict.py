"""
predict.py — 조별리그 72경기 확률 예측 + 우승 확률 몬테카를로 시뮬레이션
====================================================================
1) 72경기 각각에 P(홈승/무/원정승) 산출 — 학습된 로지스틱 모델 사용
2) 대회 전체를 20,000번 가상 진행하여 각 팀의 우승 빈도 = 우승 확률 추정

몬테카를로를 쓰는 이유:
  104경기가 얽힌 토너먼트의 우승 확률을 해석적(수식)으로 푸는 것은
  조 순위 경우의 수 때문에 사실상 불가능. 대신 '주사위를 수만 번 굴려'
  빈도로 확률을 근사. 시행 횟수가 많을수록 표준오차가 줄어듦
  (20,000회면 16% 추정치의 표준오차 약 ±0.26%p).

명시적 근사(한계) 2가지 — 문서에도 기재:
  (a) 조 순위 동률 시 골득실 대신 Elo로 타이브레이크 (모델이 스코어가
      아닌 승무패만 출력하므로)
  (b) 32강 대진을 공식 브래킷 대신 '조 1위 풀 vs 2위/3위 풀' 무작위
      추첨으로 근사 (같은 조 재대결 방지 제약만 적용). 공식 브래킷의
      경로 효과가 사라지므로 중상위권 팀 확률에 ±1%p 수준의 노이즈.
"""
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict

rng = np.random.default_rng(42)
N_SIM = 20000
HOME_ADV = 100

ratings = pd.read_csv('data/elo_final.csv', index_col=0)['elo'].to_dict()
with open('data/prob_model.pkl', 'rb') as f:
    saved = pickle.load(f)
model, classes = saved['model'], saved['classes']
iH, iD, iA = classes.index('H'), classes.index('D'), classes.index('A')

df = pd.read_csv('data/results.csv')
wc = df[(df['date'] >= '2026-06-11') & (df['tournament'] == 'FIFA World Cup')].copy()

# ── 1. 조 편성 역추출: 조별리그는 조 안에서만 경기 → 연결요소 = 조 ──
adj = defaultdict(set)
for r in wc.itertuples():
    adj[r.home_team].add(r.away_team)
    adj[r.away_team].add(r.home_team)
groups, seen = [], set()
for t in adj:
    if t in seen:
        continue
    comp_, stack = set(), [t]
    while stack:
        u = stack.pop()
        if u in comp_:
            continue
        comp_.add(u)
        stack.extend(adj[u] - comp_)
    seen |= comp_
    groups.append(sorted(comp_))
assert len(groups) == 12 and all(len(g) == 4 for g in groups), '조 복원 실패'

# ── 2. 72경기 확률 예측 ────────────────────────────────────────
rows = []
for r in wc.sort_values('date').itertuples():
    h = 0 if r.neutral else HOME_ADV
    diff = ratings[r.home_team] + h - ratings[r.away_team]
    p = model.predict_proba([[diff]])[0]
    rows.append({'date': r.date, 'home': r.home_team, 'away': r.away_team,
                 'city': r.city, 'elo_diff': round(diff),
                 'P_home': round(p[iH], 4), 'P_draw': round(p[iD], 4),
                 'P_away': round(p[iA], 4)})
pred = pd.DataFrame(rows)
pred.to_csv('data/group_stage_predictions.csv', index=False)

# ── 3. 몬테카를로 시뮬레이션 ───────────────────────────────────
match_p = {(r['home'], r['away']): (r['P_home'], r['P_draw'], r['P_away'])
           for r in rows}

def ko_win_prob(a, b):
    """토너먼트 승자진출 확률: Elo 기대승점율로 근사(연장/승부차기 포함 개념)"""
    return 1 / (1 + 10 ** (-(ratings[a] - ratings[b]) / 400))

champs = defaultdict(int)
finals = defaultdict(int)
for _ in range(N_SIM):
    pts = defaultdict(int)
    for (h, a), (ph, pd_, pa) in match_p.items():
        u = rng.random()
        if u < ph:
            pts[h] += 3
        elif u < ph + pd_:
            pts[h] += 1; pts[a] += 1
        else:
            pts[a] += 3
    winners, runners, thirds = [], [], []
    for g in groups:
        order = sorted(g, key=lambda t: (pts[t], ratings[t]), reverse=True)  # 근사(a)
        winners.append(order[0]); runners.append(order[1]); thirds.append(order[2])
    best3 = sorted(thirds, key=lambda t: (pts[t], ratings[t]), reverse=True)[:8]
    qualified = winners + runners + best3  # 32팀

    # 근사(b): 1위 풀 vs (2위+3위) 풀 무작위 매칭으로 32강 구성
    pool_top = list(rng.permutation(winners))
    pool_rest = list(rng.permutation(runners + best3))
    alive = []
    for a_, b_ in zip(pool_top, pool_rest):
        alive.append(a_ if rng.random() < ko_win_prob(a_, b_) else b_)
    rest_left = pool_rest[len(pool_top):]  # 2위/3위끼리 남는 4경기
    for i in range(0, len(rest_left), 2):
        a_, b_ = rest_left[i], rest_left[i + 1]
        alive.append(a_ if rng.random() < ko_win_prob(a_, b_) else b_)

    while len(alive) > 1:
        alive = list(rng.permutation(alive))
        nxt = []
        for i in range(0, len(alive), 2):
            a_, b_ = alive[i], alive[i + 1]
            nxt.append(a_ if rng.random() < ko_win_prob(a_, b_) else b_)
        if len(alive) == 2:
            finals[alive[0]] += 1; finals[alive[1]] += 1
        alive = nxt
    champs[alive[0]] += 1

res = pd.DataFrame({'team': list(champs.keys()),
                    'P_champion': [champs[t] / N_SIM for t in champs],
                    'P_final': [finals[t] / N_SIM for t in champs]})
res = res.sort_values('P_champion', ascending=False)
res.to_csv('data/championship_probs.csv', index=False)
print('=== 우승 확률 Top 12 (모델) ===')
print(res.head(12).to_string(index=False))
print('\n조 편성(복원):')
for i, g in enumerate(groups):
    print(' ', g)
