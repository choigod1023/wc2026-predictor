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
from functools import cmp_to_key
from collections import defaultdict

STAGES = [32, 16, 8, 4, 2]            # 도달 라운드 크기
STAGE_NAME = {32: 'R32', 16: 'R16', 8: 'QF', 4: 'SF', 2: 'F'}
HOME_ADV = 100


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


# ══════════════════════════════════════════════════════════════════
# 스코어 기반 시뮬레이션 — 실제 스코어라인을 추첨해 2026 룰대로 순위 판정
#   · 조 동률: 승점 → 승자승(맞대결 승점·골득실·다득점) → 전체 골득실·다득점 → Elo
#   · 녹아웃: 스코어라인 추첨 → 동점 시 연장(λ×1/3) → 그래도 동점 시 승부차기
# 근사(b)만 유지: 32강 대진은 1위풀 vs 2·3위풀 무작위 추첨.
# ══════════════════════════════════════════════════════════════════

def make_lambda(params):
    """params(계수)로 (home_elo, away_elo, neutral) → (λ_home, λ_away) 함수 생성."""
    a0, a1 = params['a0'], params['a1']
    b0, b1 = params['b0'], params['b1']
    sc = params.get('scale', 100.0)

    def lam(eh, ea, neutral):
        d = eh + (0 if neutral else HOME_ADV) - ea
        return np.exp(a0 + a1 * d / sc), np.exp(b0 + b1 * d / sc)
    return lam


def _rank_group(teams, games, ratings):
    """games: [(home, away, hg, ag), ...] 6경기. 2026 타이브레이크로 정렬한 팀 순서 반환."""
    pts = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    ga = {t: 0 for t in teams}
    for h, a, hg, ag in games:
        gf[h] += hg; ga[h] += ag; gf[a] += ag; ga[a] += hg
        if hg > ag:
            pts[h] += 3
        elif hg < ag:
            pts[a] += 3
        else:
            pts[h] += 1; pts[a] += 1
    gd = {t: gf[t] - ga[t] for t in teams}

    def h2h(tied):
        s = set(tied)
        p = {t: 0 for t in tied}
        d = {t: 0 for t in tied}
        f = {t: 0 for t in tied}
        for h, a, hg, ag in games:
            if h in s and a in s:
                f[h] += hg; f[a] += ag
                d[h] += hg - ag; d[a] += ag - hg
                if hg > ag:
                    p[h] += 3
                elif hg < ag:
                    p[a] += 3
                else:
                    p[h] += 1; p[a] += 1
        return p, d, f

    def cmp(a, b):
        if pts[a] != pts[b]:
            return pts[b] - pts[a]
        tied = [t for t in teams if pts[t] == pts[a]]
        if len(tied) > 1:
            p, d, f = h2h(tied)
            if p[a] != p[b]:
                return p[b] - p[a]
            if d[a] != d[b]:
                return d[b] - d[a]
            if f[a] != f[b]:
                return f[b] - f[a]
        if gd[a] != gd[b]:
            return gd[b] - gd[a]
        if gf[a] != gf[b]:
            return gf[b] - gf[a]
        return -1 if ratings[a] > ratings[b] else (1 if ratings[a] < ratings[b] else 0)

    order = sorted(teams, key=cmp_to_key(cmp))
    return order, pts, gd, gf


def simulate_scores(fixtures, groups, ratings, params, n_sim=20000, seed=42,
                    played=None):
    """
    fixtures : [(home, away, neutral), ...] 조별리그 72경기 (일정·중립여부)
    groups   : list[list[str]]
    ratings  : dict team -> elo
    params   : score_model 계수 dict
    played   : {(home, away): (home_score, away_score)} 이미 끝난 조별 경기.
               주어지면 그 경기는 실제 스코어로 '고정'하고 남은 경기만 추첨한다
               → 대회가 진행될수록 예측이 실제 결과를 반영해 변동.
    스코어라인을 추첨해 2026 룰로 조 순위·녹아웃을 진행. 반환 형식은 simulate 와 동일.
    """
    rng = np.random.default_rng(seed)
    lam = make_lambda(params)
    played = played or {}

    def ko(a, b):
        """녹아웃 한 경기: 스코어 추첨 → 연장 → 승부차기. 승자 반환."""
        lh, la = lam(ratings[a], ratings[b], True)
        hg, ag = rng.poisson(lh), rng.poisson(la)
        if hg != ag:
            return a if hg > ag else b
        # 연장 (득점력 1/3)
        eh, ea = rng.poisson(lh / 3), rng.poisson(la / 3)
        if eh != ea:
            return a if eh > ea else b
        # 승부차기: Elo를 절반 스케일로 반영(거의 동전던지기)
        p = 1 / (1 + 10 ** (-(ratings[a] - ratings[b]) / 800))
        return a if rng.random() < p else b

    # 그룹별 경기 인덱스 미리 구성
    group_of = {}
    for gi, g in enumerate(groups):
        for t in g:
            group_of[t] = gi
    grp_fixtures = defaultdict(list)
    for h, a, neu in fixtures:
        grp_fixtures[group_of[h]].append((h, a, bool(neu)))

    champs = defaultdict(int)
    finals = defaultdict(int)
    reach = {s: defaultdict(int) for s in STAGES}

    for _ in range(n_sim):
        winners, runners, thirds = [], [], []
        third_keys = {}
        for gi, g in enumerate(groups):
            games = []
            for h, a, neu in grp_fixtures[gi]:
                if (h, a) in played:                     # 이미 끝난 경기는 실제 스코어 고정
                    hs, as_ = played[(h, a)]
                    games.append((h, a, int(hs), int(as_)))
                else:
                    lh, la = lam(ratings[h], ratings[a], neu)
                    games.append((h, a, int(rng.poisson(lh)), int(rng.poisson(la))))
            order, pts, gd, gf = _rank_group(g, games, ratings)
            winners.append(order[0]); runners.append(order[1]); thirds.append(order[2])
            t3 = order[2]
            third_keys[t3] = (pts[t3], gd[t3], gf[t3], ratings[t3])
        best3 = sorted(thirds, key=lambda t: third_keys[t], reverse=True)[:8]
        qualified = winners + runners + best3

        pool_top = list(rng.permutation(winners))
        pool_rest = list(rng.permutation(runners + best3))
        alive = []
        for a_, b_ in zip(pool_top, pool_rest):
            alive.append(ko(a_, b_))
        rest_left = pool_rest[len(pool_top):]
        for i in range(0, len(rest_left), 2):
            alive.append(ko(rest_left[i], rest_left[i + 1]))

        for t in qualified:
            reach[32][t] += 1
        for t in alive:
            reach[16][t] += 1

        while len(alive) > 1:
            alive = list(rng.permutation(alive))
            nxt = [ko(alive[i], alive[i + 1]) for i in range(0, len(alive), 2)]
            if len(alive) == 2:
                finals[alive[0]] += 1; finals[alive[1]] += 1
            if len(nxt) in reach:
                for t in nxt:
                    reach[len(nxt)][t] += 1
            alive = nxt
        champs[alive[0]] += 1

    teams = set()
    for g in groups:
        teams.update(g)
    stages = {t: {STAGE_NAME[s]: reach[s][t] / n_sim for s in STAGES} for t in teams}
    return {
        'champion': {t: champs[t] / n_sim for t in teams},
        'final': {t: finals[t] / n_sim for t in teams},
        'stages': stages,
    }
