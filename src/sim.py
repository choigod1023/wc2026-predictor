"""
sim.py — 토너먼트 몬테카를로 시뮬레이션 (공용 모듈)
====================================================
predict.py 와 compare_models.py 가 함께 쓰는 시뮬레이션 코어.
어떤 확률 모델이 만든 72경기 확률(match_p)이든 동일한 규칙으로 대회를 돌려
우승 확률과 단계별(R32~결승) 도달 확률을 추정한다.

근사 2가지는 문서/주석대로 유지:
  (a) 조 동률은 골득실 대신 Elo로 타이브레이크
  (b) 32강 대진은 공식 브래킷 대신 1위풀 vs 2·3위풀 무작위 추첨
"""
import numpy as np
from collections import defaultdict

STAGES = [32, 16, 8, 4, 2]            # 도달 라운드 크기
STAGE_NAME = {32: 'R32', 16: 'R16', 8: 'QF', 4: 'SF', 2: 'F'}


def recover_groups(wc_df):
    """조별리그는 조 안에서만 경기 → '서로 경기하는 팀들의 연결요소' = 조."""
    adj = defaultdict(set)
    for r in wc_df.itertuples():
        adj[r.home_team].add(r.away_team)
        adj[r.away_team].add(r.home_team)
    groups, seen = [], set()
    for t in adj:
        if t in seen:
            continue
        comp, stack = set(), [t]
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            stack.extend(adj[u] - comp)
        seen |= comp
        groups.append(sorted(comp))
    assert len(groups) == 12 and all(len(g) == 4 for g in groups), '조 복원 실패'
    return groups


def simulate(match_p, groups, ratings, n_sim=20000, seed=42):
    """
    match_p : dict {(home, away): (pH, pD, pA)}
    groups  : list[list[str]]  (12개 조)
    ratings : dict team -> elo  (타이브레이크/녹아웃 진출 확률용)
    반환: dict {
       'champion': {team: prob}, 'final': {team: prob},
       'stages':   {team: {'R32':p,'R16':p,'QF':p,'SF':p,'F':p}}
    }
    """
    rng = np.random.default_rng(seed)

    def ko(a, b):
        return 1 / (1 + 10 ** (-(ratings[a] - ratings[b]) / 400))

    champs = defaultdict(int)
    finals = defaultdict(int)
    reach = {s: defaultdict(int) for s in STAGES}

    items = list(match_p.items())
    for _ in range(n_sim):
        pts = defaultdict(int)
        for (h, a), (ph, pd_, pa) in items:
            u = rng.random()
            if u < ph:
                pts[h] += 3
            elif u < ph + pd_:
                pts[h] += 1
                pts[a] += 1
            else:
                pts[a] += 3
        winners, runners, thirds = [], [], []
        for g in groups:
            order = sorted(g, key=lambda t: (pts[t], ratings[t]), reverse=True)  # 근사(a)
            winners.append(order[0])
            runners.append(order[1])
            thirds.append(order[2])
        best3 = sorted(thirds, key=lambda t: (pts[t], ratings[t]), reverse=True)[:8]
        qualified = winners + runners + best3                       # 32강

        # 근사(b): 1위 풀 vs (2·3위) 풀 무작위 매칭
        pool_top = list(rng.permutation(winners))
        pool_rest = list(rng.permutation(runners + best3))
        alive = []
        for a_, b_ in zip(pool_top, pool_rest):
            alive.append(a_ if rng.random() < ko(a_, b_) else b_)
        rest_left = pool_rest[len(pool_top):]
        for i in range(0, len(rest_left), 2):
            a_, b_ = rest_left[i], rest_left[i + 1]
            alive.append(a_ if rng.random() < ko(a_, b_) else b_)
        # 이제 alive = 16강 진출팀

        for t in qualified:
            reach[32][t] += 1
        for t in alive:
            reach[16][t] += 1

        while len(alive) > 1:
            alive = list(rng.permutation(alive))
            nxt = []
            for i in range(0, len(alive), 2):
                a_, b_ = alive[i], alive[i + 1]
                nxt.append(a_ if rng.random() < ko(a_, b_) else b_)
            if len(alive) == 2:
                finals[alive[0]] += 1
                finals[alive[1]] += 1
            # nxt = 다음 라운드 진출팀; 라운드 크기로 단계 기록
            if len(nxt) in reach:
                for t in nxt:
                    reach[len(nxt)][t] += 1
            alive = nxt
        champs[alive[0]] += 1

    teams = set()
    for g in groups:
        teams.update(g)
    stages = {}
    for t in teams:
        stages[t] = {STAGE_NAME[s]: reach[s][t] / n_sim for s in STAGES}
    return {
        'champion': {t: champs[t] / n_sim for t in teams},
        'final': {t: finals[t] / n_sim for t in teams},
        'stages': stages,
    }
