"""
sim.py — 토너먼트 몬테카를로 시뮬레이션 (공용 모듈)
====================================================
predict.py 와 compare_models.py 가 함께 쓰는 시뮬레이션 코어.
어떤 확률 모델이 만든 72경기 확률(match_p)이든 동일한 규칙으로 대회를 돌려
우승 확률과 단계별(R32~결승) 도달 확률을 추정한다.

두 시뮬레이션(대표 simulate_scores, 모델비교 simulate) 모두 2026 공식 브래킷
(R32~결승)을 그대로 전개해 대진 경로 의존성을 보존한다(근사 b 제거). 3위 8팀은
공식 후보-조 제약을 지키는 완전매칭으로 슬롯 배정. 차이는 승부 판정뿐:
simulate_scores는 스코어라인 추첨(+연장·승부차기), simulate는 Elo 승리확률.
"""
import os as _os
import json as _json
import numpy as np
from functools import cmp_to_key, lru_cache
from collections import defaultdict

# FIFA 공식 3위 배정표 (Annex C, 495조합). 출처: 2026 WC 규정 Annex C
# (= Wikipedia Template:2026 FIFA World Cup third-place table 에서 전수 파싱·검증).
# 키: 진출한 8개 조 3위의 정렬된 조합(예 'ABDEGIKL') → {R32경기no(str): 3위 조라벨}.
_ALLOC_PATH = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                            'data', 'third_place_allocation.json')
with open(_ALLOC_PATH, encoding='utf-8') as _f:
    THIRD_ALLOC = _json.load(_f)

STAGES = [32, 16, 8, 4, 2]            # 도달 라운드 크기
STAGE_NAME = {32: 'R32', 16: 'R16', 8: 'QF', 4: 'SF', 2: 'F'}
HOME_ADV = 100

# ── 공식 2026 녹아웃 브래킷 (R32~결승) ──────────────────────────
# 출처: FIFA 공식 대진 / Wikipedia 2026 WC knockout stage. (= wc2026-web/lib/bracket.ts)
# 조 라벨 앵커: 각 조에 정확히 1팀씩 들어가는 시드/개최국 → 조 라벨(A~L) 복원용.
GROUP_ANCHOR = {
    'Mexico': 'A', 'Canada': 'B', 'Brazil': 'C', 'United States': 'D',
    'Germany': 'E', 'Netherlands': 'F', 'Belgium': 'G', 'Spain': 'H',
    'France': 'I', 'Argentina': 'J', 'Portugal': 'K', 'England': 'L',
}
# 슬롯: ('W',조) 1위 · ('R',조) 2위 · ('T',후보조들) 3위 · ('M',경기no) 앞 라운드 승자
BRACKET = {
    73: (('R', 'A'), ('R', 'B')), 74: (('W', 'E'), ('T', 'ABCDF')),
    75: (('W', 'F'), ('R', 'C')), 76: (('W', 'C'), ('R', 'F')),
    77: (('W', 'I'), ('T', 'CDFGH')), 78: (('R', 'E'), ('R', 'I')),
    79: (('W', 'A'), ('T', 'CEFHI')), 80: (('W', 'L'), ('T', 'EHIJK')),
    81: (('W', 'D'), ('T', 'BEFIJ')), 82: (('W', 'G'), ('T', 'AEHIJ')),
    83: (('R', 'K'), ('R', 'L')), 84: (('W', 'H'), ('R', 'J')),
    85: (('W', 'B'), ('T', 'EFGIJ')), 86: (('W', 'J'), ('R', 'H')),
    87: (('W', 'K'), ('T', 'DEIJL')), 88: (('R', 'D'), ('R', 'G')),
    89: (('M', 74), ('M', 77)), 90: (('M', 73), ('M', 75)),
    91: (('M', 76), ('M', 78)), 92: (('M', 79), ('M', 80)),
    93: (('M', 83), ('M', 84)), 94: (('M', 81), ('M', 82)),
    95: (('M', 86), ('M', 88)), 96: (('M', 85), ('M', 87)),
    97: (('M', 89), ('M', 90)), 98: (('M', 93), ('M', 94)),
    99: (('M', 91), ('M', 92)), 100: (('M', 95), ('M', 96)),
    101: (('M', 97), ('M', 98)), 102: (('M', 99), ('M', 100)),
    104: (('M', 101), ('M', 102)),
}
R32_NOS = list(range(73, 89))
R16_NOS = list(range(89, 97))
QF_NOS = list(range(97, 101))
SF_NOS = [101, 102]
FINAL_NO = 104
# 3위가 들어갈 수 있는 R32 슬롯별 후보 조 (위 BRACKET의 T슬롯)
TSLOTS = {74: 'ABCDF', 77: 'CDFGH', 79: 'CEFHI', 80: 'EHIJK',
          81: 'BEFIJ', 82: 'AEHIJ', 85: 'EFGIJ', 87: 'DEIJL'}
_TSLOT_NOS = list(TSLOTS)


@lru_cache(maxsize=None)
def assign_thirds(qual_letters):
    """진출한 8개 조 3위(정렬된 조 라벨 튜플) → {R32경기no(int): 조라벨} 슬롯 배정.
    FIFA 공식 Annex C 표(495조합 전수)를 그대로 사용. 표에 없는 경우(이론상 없음)만
    후보-조 제약 완전매칭으로 폴백. 결과 캐시(조합 ≤495)."""
    key = ''.join(qual_letters)          # qual_letters는 정렬된 튜플 → 키와 일치
    official = THIRD_ALLOC.get(key)
    if official:
        return {int(mno): g for mno, g in official.items()}
    # ── 폴백: 후보-조 제약 증대경로 완전매칭 (공식표 미존재 시) ──
    qual = set(qual_letters)
    mg = {}
    def aug(slot, seen):
        for g in sorted(set(TSLOTS[slot]) & qual):
            if g in seen:
                continue
            seen.add(g)
            if g not in mg or aug(mg[g], seen):
                mg[g] = slot
                return True
        return False
    for slot in _TSLOT_NOS:
        aug(slot, set())
    res = {slot: g for g, slot in mg.items()}
    if len(res) < 8:
        ls = [s for s in _TSLOT_NOS if s not in res]
        lg = [g for g in qual if g not in mg]
        for s, g in zip(ls, lg):
            res[s] = g
    return res


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


def split_wc(df):
    """WC 경기를 (조별리그 72경기, 녹아웃 경기) 로 분리.
    스테이지 컬럼이 없고, FIFA 일정상 조별 72경기가 모두 녹아웃보다 먼저 열리므로
    '날짜순 앞 72경기 = 조별리그'로 간주(데이터 의존 낮고 견고). 녹아웃 경기가
    추가돼도 recover_groups(조 안 경기만)·조별 산출이 깨지지 않게 한다."""
    wc = df[(df['date'] >= '2026-06-11') &
            (df['tournament'] == 'FIFA World Cup')].sort_values('date')
    return wc.head(72), wc.iloc[72:]


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
    # 공식 조 라벨(A~L) 복원 — 브래킷 슬롯 해소용
    idx_letter = {}
    for gi, g in enumerate(groups):
        lab = next((GROUP_ANCHOR[t] for t in g if t in GROUP_ANCHOR), None)
        if lab is None:
            raise ValueError(f'조 라벨 앵커(시드팀) 없음: {g}')
        idx_letter[gi] = lab

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
        # 조 순위(스코어 없으니 동률은 Elo로 근사 a) + 조 라벨별 매핑
        win_by, run_by, third_by, third_key = {}, {}, {}, {}
        for gi, g in enumerate(groups):
            order = sorted(g, key=lambda t: (pts[t], ratings[t]), reverse=True)
            L = idx_letter[gi]
            win_by[L], run_by[L], third_by[L] = order[0], order[1], order[2]
            third_key[L] = (pts[order[2]], ratings[order[2]])
        qual_thirds = sorted(third_key, key=lambda L: third_key[L], reverse=True)[:8]
        slot_third = assign_thirds(tuple(sorted(qual_thirds)))
        qualified = (list(win_by.values()) + list(run_by.values())
                     + [third_by[L] for L in qual_thirds])

        # 공식 브래킷 고정 전개 (simulate_scores 와 동일 구조, 승부는 Elo 확률)
        def slot_team(slot, no):
            k, v = slot
            if k == 'W':
                return win_by[v]
            if k == 'R':
                return run_by[v]
            return third_by[slot_third[no]]
        winner = {}
        for no in R32_NOS:
            h, a = BRACKET[no]
            x, y = slot_team(h, no), slot_team(a, no)
            winner[no] = x if rng.random() < ko(x, y) else y
        for no in R16_NOS + QF_NOS + SF_NOS + [FINAL_NO]:
            (_, vh), (_, va) = BRACKET[no]
            x, y = winner[vh], winner[va]
            winner[no] = x if rng.random() < ko(x, y) else y

        for t in qualified:
            reach[32][t] += 1
        for no in R32_NOS:
            reach[16][winner[no]] += 1
        for no in R16_NOS:
            reach[8][winner[no]] += 1
        for no in QF_NOS:
            reach[4][winner[no]] += 1
        for no in SF_NOS:
            reach[2][winner[no]] += 1
            finals[winner[no]] += 1
        champs[winner[FINAL_NO]] += 1

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
#   · 녹아웃: 공식 2026 브래킷(R32~결승) 고정 전개 → 스코어 추첨 → 연장(λ×1/3)
#            → 승부차기. 3위 8팀은 공식 후보-조 제약 완전매칭으로 슬롯 배정.
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
                    played=None, ko_played=None):
    """
    fixtures : [(home, away, neutral), ...] 조별리그 72경기 (일정·중립여부)
    groups   : list[list[str]]
    ratings  : dict team -> elo
    params   : score_model 계수 dict
    played   : {(home, away): (home_score, away_score)} 이미 끝난 조별 경기.
               주어지면 그 경기는 실제 스코어로 '고정'하고 남은 경기만 추첨한다.
    ko_played: {(home, away): (home_score, away_score)} 이미 끝난 녹아웃 경기.
               해당 대진의 진출팀을 '고정'하고 남은 녹아웃만 시뮬 → 토너먼트가
               진행될수록 우승/단계 확률이 실제 결과를 반영해 변동.
    스코어라인을 추첨해 2026 룰로 조 순위·녹아웃을 진행. 반환 형식은 simulate 와 동일.
    """
    rng = np.random.default_rng(seed)
    lam = make_lambda(params)
    played = played or {}
    ko_played = ko_played or {}

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

    # ── 실제 녹아웃 결과 → 진출팀(advancer) 사전계산 ──────────────
    # 점수가 갈리면 그 팀, 무승부(승부차기)면 '다음 라운드에 또 등장하는 팀'으로 추론.
    # (최신 라운드 무승부 등 추론 불가 시 None → 그 경기만 시뮬)
    _appear = defaultdict(int)
    for (h, a) in ko_played:
        _appear[h] += 1
        _appear[a] += 1
    ko_adv = {}
    for (h, a), (hs, as_) in ko_played.items():
        if hs > as_:
            adv = h
        elif as_ > hs:
            adv = a
        else:
            ah, aa = _appear[h] > 1, _appear[a] > 1
            adv = h if (ah and not aa) else a if (aa and not ah) else None
        ko_adv[frozenset((h, a))] = adv

    def play(A, B):
        """녹아웃 한 경기: 실제 결과가 있으면 진출팀 고정, 없으면 시뮬."""
        adv = ko_adv.get(frozenset((A, B)))
        return adv if adv is not None else ko(A, B)

    # 그룹별 경기 인덱스 + 공식 조 라벨(A~L) 복원(앵커 시드팀 기준)
    group_of = {}
    idx_letter = {}
    for gi, g in enumerate(groups):
        for t in g:
            group_of[t] = gi
        lab = next((GROUP_ANCHOR[t] for t in g if t in GROUP_ANCHOR), None)
        if lab is None:
            raise ValueError(f'조 라벨 앵커(시드팀) 없음: {g}')
        idx_letter[gi] = lab
    grp_fixtures = defaultdict(list)
    for h, a, neu in fixtures:
        grp_fixtures[group_of[h]].append((h, a, bool(neu)))

    champs = defaultdict(int)
    finals = defaultdict(int)
    reach = {s: defaultdict(int) for s in STAGES}

    for _ in range(n_sim):
        win_by, run_by, third_by = {}, {}, {}        # 조라벨 -> 팀
        third_keys = {}
        for gi, g in enumerate(groups):
            games = []
            for h, a, neu in grp_fixtures[gi]:
                if (h, a) in played:                  # 이미 끝난 경기는 실제 스코어 고정
                    hs, as_ = played[(h, a)]
                    games.append((h, a, int(hs), int(as_)))
                else:
                    lh, la = lam(ratings[h], ratings[a], neu)
                    games.append((h, a, int(rng.poisson(lh)), int(rng.poisson(la))))
            order, pts, gd, gf = _rank_group(g, games, ratings)
            L = idx_letter[gi]
            win_by[L], run_by[L], third_by[L] = order[0], order[1], order[2]
            t3 = order[2]
            third_keys[L] = (pts[t3], gd[t3], gf[t3], ratings[t3])

        # 3위 상위 8개 조 선발 → 공식 후보-제약 슬롯 배정
        qual_thirds = sorted(third_keys, key=lambda L: third_keys[L], reverse=True)[:8]
        slot_third = assign_thirds(tuple(sorted(qual_thirds)))   # {R32경기no: 조라벨}
        qualified = (list(win_by.values()) + list(run_by.values())
                     + [third_by[L] for L in qual_thirds])

        # 공식 브래킷 전개 — 고정 트리(라운드별 무작위 재추첨 없음). 경로 의존성 보존.
        def slot_team(slot, no):
            k, v = slot
            if k == 'W':
                return win_by[v]
            if k == 'R':
                return run_by[v]
            return third_by[slot_third[no]]          # 'T' (이 경기에 배정된 3위)
        winner = {}
        for no in R32_NOS:
            h, a = BRACKET[no]
            winner[no] = play(slot_team(h, no), slot_team(a, no))
        for no in R16_NOS + QF_NOS + SF_NOS + [FINAL_NO]:
            (_, vh), (_, va) = BRACKET[no]           # 앞 라운드 승자끼리
            winner[no] = play(winner[vh], winner[va])

        for t in qualified:
            reach[32][t] += 1
        for no in R32_NOS:                            # R32 승자 = R16 진출(16팀)
            reach[16][winner[no]] += 1
        for no in R16_NOS:                            # R16 승자 = 8강(8팀)
            reach[8][winner[no]] += 1
        for no in QF_NOS:                             # 8강 승자 = 4강(4팀)
            reach[4][winner[no]] += 1
        for no in SF_NOS:                             # 4강 승자 = 결승(2팀)
            reach[2][winner[no]] += 1
            finals[winner[no]] += 1
        champs[winner[FINAL_NO]] += 1

    teams = set()
    for g in groups:
        teams.update(g)
    stages = {t: {STAGE_NAME[s]: reach[s][t] / n_sim for s in STAGES} for t in teams}
    return {
        'champion': {t: champs[t] / n_sim for t in teams},
        'final': {t: finals[t] / n_sim for t in teams},
        'stages': stages,
    }
