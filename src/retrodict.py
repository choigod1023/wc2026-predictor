"""
retrodict.py — 104경기 전수 사전예측(누수 없음) + 성공/실패 채점
================================================================
문제:
  개막 전 고정본(group_stage_predictions.csv)은 조별리그 72경기뿐이다.
  녹아웃 32경기는 '누가 올라올지' 모르는 상태라 개막 전에 대진별 예측을 만들 수
  없었고, 그 결과 대회가 끝난 지금도 40%의 경기에 채점 가능한 예측이 없다.

해결(이 스크립트):
  각 경기를 '그 경기 직전 시점'의 정보만으로 예측한 뒤 실제 결과와 대조한다.
  104경기 전부에 예측·적중여부가 붙는다.

누수 차단 장치 (CLAUDE.md 절대원칙 2·3)
  1. 피처: elo_history.csv 의 elo_diff_pre — 정의상 '경기 전' Elo 차이.
     대회가 진행되며 갱신되지만, 각 경기 시점에서는 모두 과거 정보다.
  2. 계수: 로지스틱 회귀를 **개막 전(2026-06-11 미만) 경기만으로** 적합해
     고정(frozen)한다. 대회 결과는 계수에 단 한 경기도 들어가지 않는다.
     사양은 운영 모델과 동일(친선 제외·1990년 이후·시간가중 반감기 3년).
  → 즉 "개막 전날 확정한 모델 + 킥오프 직전 전력치"로 매 경기를 예측한 것과 같다.

채점 방식
  · 1X2 적중: argmax(P_home,P_draw,P_away) 가 실제 결과와 일치하는가.
  · Brier(멀티클래스): 확률의 질. 낮을수록 좋음. 기준선 2개와 함께 보고.
  · 진출팀 적중(녹아웃 전용): 무승부는 승부차기로 갈리므로 1X2만으로는
    '누가 올라갔나'를 못 채점한다. P(홈 진출)=P_home + P_draw·s 로 환산해
    (s = Elo 기대승점율, 연장·승부차기 국면의 근사) 실제 진출팀과 대조.
  · 시장 대결: closing_odds.csv 가 있는 경기(조별 72)에 한해 동일 Brier 비교.
  · 언/오버 2.5: 개막 전 고정본 score_predictions.json(조별 72)로 채점.

주의(사실/의견 구분 — 원칙 5)
  · 녹아웃 경기의 results.csv 스코어는 **연장 포함** 기록이다. 따라서 녹아웃의
    1X2 채점은 '90분'이 아니라 '연장까지의 결과' 기준이다(사실).
    승부차기로 갈린 경기는 1X2 상 무승부로 채점되고, 승패는 진출팀 채점이 맡는다.
  · s(연장·승부차기 승자 확률)를 Elo 기대승점율로 두는 것은 근사다(의견).

출력: data/retrodiction.json  (웹 '성적표' 탭이 그대로 읽는다)
실행: python src/retrodict.py
"""
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

TOURNAMENT_START = '2026-06-11'   # 개막일 — 이 날짜 이전 경기만 학습에 사용
HALFLIFE_YEARS = 3.0              # 운영 모델과 동일(experiments/time_decay_backtest.py 근거)
CLASSES = ['H', 'D', 'A']

# 라운드 구성: (키, 표시명, 경기수) — FIFA 일정상 날짜순으로 이 순서대로 열린다.
STAGE_PLAN = [('GROUP', '조별리그', 72), ('R32', '32강', 16), ('R16', '16강', 8),
              ('QF', '8강', 4), ('SF', '4강', 2), ('TP', '3·4위전', 1), ('F', '결승', 1)]
KO_KEYS = {'R32', 'R16', 'QF', 'SF', 'TP', 'F'}


# ── 1. 개막 전 데이터만으로 확률모델 적합(frozen) ────────────────
def fit_frozen_model():
    hist = pd.read_csv('data/elo_history.csv')
    comp = hist[(hist['tournament'] != 'Friendly') & (hist['date'] >= '1990-01-01')]
    train = comp[comp['date'] < TOURNAMENT_START].copy()

    age = (pd.to_datetime(TOURNAMENT_START) - pd.to_datetime(train['date'])).dt.days
    w = 0.5 ** (age.values.astype(float) / (HALFLIFE_YEARS * 365.25))

    m = LogisticRegression(max_iter=1000)
    m.fit(train[['elo_diff_pre']].values, train['outcome'].values, sample_weight=w)
    order = list(m.classes_)                      # sklearn은 알파벳순(A,D,H)
    base = {c: float(np.mean(train['outcome'].values == c)) for c in CLASSES}
    return m, order, len(train), base


# ── 2. 채점 유틸 ────────────────────────────────────────────────
def brier(p, actual):
    """멀티클래스 Brier. p=(pH,pD,pA), actual='H'|'D'|'A'. 완벽=0, 1/3찍기≈0.667"""
    onehot = [1.0 if c == actual else 0.0 for c in CLASSES]
    return float(sum((a - b) ** 2 for a, b in zip(p, onehot)))


def pick_of(p):
    return CLASSES[int(np.argmax(p))]


def agg(rows, key='hit'):
    """[{hit:bool, brier:float}] → 요약 dict"""
    n = len(rows)
    if n == 0:
        return {'n': 0, 'hits': 0, 'acc': None, 'brier': None}
    hits = sum(1 for r in rows if r[key])
    return {'n': n, 'hits': hits, 'acc': round(hits / n, 4),
            'brier': round(float(np.mean([r['brier'] for r in rows])), 4)}


def main():
    model, order, n_train, base_rate = fit_frozen_model()
    iH, iD, iA = order.index('H'), order.index('D'), order.index('A')

    # ── 3. 경기 목록 + 라운드 라벨 ──────────────────────────────
    res = pd.read_csv('data/results.csv')
    wc = res[(res['date'] >= TOURNAMENT_START) &
             (res['tournament'] == 'FIFA World Cup')].sort_values('date', kind='stable')
    wc = wc[wc['home_score'].notna()].reset_index(drop=True)
    total_planned = sum(n for _, _, n in STAGE_PLAN)
    if len(wc) != total_planned:
        print(f'주의: 결과가 있는 WC 경기 {len(wc)}건 (완주 시 {total_planned}건). '
              f'있는 만큼만 채점한다.')

    stage_key, stage_name = [], []
    for key, name, cnt in STAGE_PLAN:
        take = min(cnt, len(wc) - len(stage_key))
        stage_key += [key] * take
        stage_name += [name] * take
    wc = wc.iloc[:len(stage_key)].copy()
    wc['stage_key'] = stage_key
    wc['stage'] = stage_name

    # 경기 전 Elo 차이 — elo_history 에서 (date, home, away) 로 조인
    hist = pd.read_csv('data/elo_history.csv')
    diff_of = {(r.date, r.home_team, r.away_team): float(r.elo_diff_pre)
               for r in hist[hist['date'] >= TOURNAMENT_START].itertuples()}

    # 승부차기 승자 (무승부 경기의 진출팀 판정 근거)
    sh = pd.read_csv('data/shootouts.csv')
    pens_of = {(r.date, r.home_team, r.away_team): r.winner
               for r in sh[sh['date'] >= TOURNAMENT_START].itertuples()}

    # 마감 배당(시장 확률) — 기록된 경기만
    odds = pd.read_csv('data/closing_odds.csv').dropna(subset=['odds_H', 'odds_D', 'odds_A'])
    odds_of = {(r.date, r.home, r.away): (r.odds_H, r.odds_D, r.odds_A)
               for r in odds.itertuples()}

    # ── 4. 경기별 예측·채점 ────────────────────────────────────
    matches, market_rows = [], []
    for no, r in enumerate(wc.itertuples(), start=1):
        key = (r.date, r.home_team, r.away_team)
        if key not in diff_of:
            raise SystemExit(f'elo_history 에 없는 경기: {key} — elo.py 를 먼저 실행')
        d = diff_of[key]
        pv = model.predict_proba([[d]])[0]
        p = (float(pv[iH]), float(pv[iD]), float(pv[iA]))

        hs, as_ = int(r.home_score), int(r.away_score)
        actual = 'H' if hs > as_ else ('D' if hs == as_ else 'A')
        pick = pick_of(p)

        m = {'no': no, 'date': r.date, 'stageKey': r.stage_key, 'stage': r.stage,
             'home': r.home_team, 'away': r.away_team, 'city': r.city,
             'neutral': bool(r.neutral), 'eloDiffPre': round(d),
             'pHome': round(p[0], 4), 'pDraw': round(p[1], 4), 'pAway': round(p[2], 4),
             'pick': pick, 'actual': actual, 'hs': hs, 'as': as_,
             'hit': pick == actual, 'brier': round(brier(p, actual), 4)}

        # 녹아웃: 진출팀 예측·채점 (무승부 → 연장/승부차기 국면을 Elo로 근사)
        if r.stage_key in KO_KEYS:
            s = 1 / (1 + 10 ** (-d / 400))          # Elo 기대승점율 = 홈이 연장/PK 국면 승자일 확률(근사)
            p_adv_home = p[0] + p[1] * s
            adv_pred = r.home_team if p_adv_home >= 0.5 else r.away_team
            if hs != as_:
                adv_actual = r.home_team if hs > as_ else r.away_team
                decided = '정규(연장 포함)'
            else:
                adv_actual = pens_of.get(key)
                decided = '승부차기'
            m['pAdvHome'] = round(float(p_adv_home), 4)
            m['advPred'] = adv_pred
            m['advActual'] = adv_actual
            m['advHit'] = (adv_actual == adv_pred) if adv_actual else None
            m['decidedBy'] = decided

        # 시장(마감 배당) 비교 — 배당 기록이 있는 경기만
        if key in odds_of:
            oh, od, oa = odds_of[key]
            inv = np.array([1 / oh, 1 / od, 1 / oa])
            mp = inv / inv.sum()                     # 마진 제거 정규화
            mkt = {'oH': float(oh), 'oD': float(od), 'oA': float(oa),
                   'pHome': round(float(mp[0]), 4), 'pDraw': round(float(mp[1]), 4),
                   'pAway': round(float(mp[2]), 4)}
            mkt['pick'] = pick_of(tuple(mp))
            mkt['hit'] = mkt['pick'] == actual
            mkt['brier'] = round(brier(tuple(mp), actual), 4)
            m['market'] = mkt
            market_rows.append({'model_hit': m['hit'], 'model_brier': m['brier'],
                                'hit': mkt['hit'], 'brier': mkt['brier']})
        matches.append(m)

    # ── 5. 요약 지표 ───────────────────────────────────────────
    by_stage = []
    for key, name, _ in STAGE_PLAN:
        rows = [m for m in matches if m['stageKey'] == key]
        if rows:
            by_stage.append({'stageKey': key, 'stage': name, **agg(rows)})

    ko_rows = [m for m in matches if m['stageKey'] in KO_KEYS and m.get('advHit') is not None]
    adv_summary = {'n': len(ko_rows), 'hits': sum(1 for m in ko_rows if m['advHit']),
                   'acc': round(sum(1 for m in ko_rows if m['advHit']) / len(ko_rows), 4)
                   if ko_rows else None}

    # 기준선: 항상 1/3 찍기 / 학습기간 H·D·A 비율 찍기
    uni = float(np.mean([brier((1 / 3, 1 / 3, 1 / 3), m['actual']) for m in matches]))
    br_p = (base_rate['H'], base_rate['D'], base_rate['A'])
    baserate = float(np.mean([brier(br_p, m['actual']) for m in matches]))

    market = None
    if market_rows:
        market = {
            'n': len(market_rows),
            'modelAcc': round(sum(1 for r in market_rows if r['model_hit']) / len(market_rows), 4),
            'marketAcc': round(sum(1 for r in market_rows if r['hit']) / len(market_rows), 4),
            'modelBrier': round(float(np.mean([r['model_brier'] for r in market_rows])), 4),
            'marketBrier': round(float(np.mean([r['brier'] for r in market_rows])), 4),
            'note': '배당은 조별리그 72경기에만 기록되어 있어 녹아웃은 직접 비교 불가.',
        }
        market['winner'] = 'model' if market['modelBrier'] < market['marketBrier'] else 'market'

    # 언/오버 2.5 — 개막 전 고정 스코어 예측본으로 채점(조별 72)
    ou = None
    if _os.path.exists('data/score_predictions.json'):
        sp = json.load(open('data/score_predictions.json', encoding='utf-8'))
        by_key = {(s['date'], s['home'], s['away']): s for s in sp}
        rows = []
        for m in matches:
            s = by_key.get((m['date'], m['home'], m['away']))
            if not s:
                continue
            p_over = float(s['overUnder']['2.5'])
            over = (m['hs'] + m['as']) > 2.5
            rows.append({'hit': (p_over >= 0.5) == over,
                         'brier': (p_over - (1.0 if over else 0.0)) ** 2})
        if rows:
            ou = {'n': len(rows), 'hits': sum(1 for r in rows if r['hit']),
                  'acc': round(sum(1 for r in rows if r['hit']) / len(rows), 4),
                  'brier': round(float(np.mean([r['brier'] for r in rows])), 4),
                  'note': '개막 전 고정본 score_predictions.json(조별 72경기) 기준.'}

    # ── 6. 대회 단위 성패 (우승·결승·4강 예측) ──────────────────
    tournament = build_tournament_view(matches)

    out = {
        'generated': dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%MZ'),
        'method': {
            'feature': 'elo_diff_pre (경기 직전 Elo 차이, 홈 어드밴티지 반영)',
            'model': f'다항 로지스틱 · 개막 전({TOURNAMENT_START} 미만) 경기 {n_train:,}건으로 고정 적합'
                     f' · 시간가중 반감기 {HALFLIFE_YEARS}년',
            'leakage': '대회 결과는 계수에 미포함. 각 경기는 그 경기 직전 정보만 사용.',
            'koNote': '녹아웃 스코어는 연장 포함 기록 → 1X2는 연장까지 기준, '
                      '승부차기 경기는 진출팀 채점으로 승패 판정.',
        },
        'baseline': {'uniform': round(uni, 4), 'baseRate': round(baserate, 4),
                     'baseRateProbs': {k: round(v, 4) for k, v in base_rate.items()}},
        'summary': {'overall': agg(matches), 'byStage': by_stage,
                    'advance': adv_summary, 'market': market, 'ou25': ou},
        'tournament': tournament,
        'matches': matches,
    }
    json.dump(out, open('data/retrodiction.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    report(out)


def build_tournament_view(matches):
    """개막 전 우승확률(champion_history 첫 스냅샷) vs 실제 도달 라운드."""
    stage_teams = {}
    for m in matches:
        stage_teams.setdefault(m['stageKey'], set()).update([m['home'], m['away']])

    final = next((m for m in matches if m['stageKey'] == 'F'), None)
    champion = None
    if final:
        champion = (final['advActual'] if final.get('advActual')
                    else (final['home'] if final['hs'] > final['as'] else final['away']))

    pre = {}
    if _os.path.exists('data/champion_history.json'):
        hist = json.load(open('data/champion_history.json', encoding='utf-8'))
        if hist:
            pre = hist[0].get('p', {})              # 첫 스냅샷 = 개막 전(2026-06-10)
    ranked = sorted(pre.items(), key=lambda kv: -kv[1])
    rank_of = {t: i + 1 for i, (t, _) in enumerate(ranked)}

    # 시장 스냅샷(개막 전) — 구현확률 평균(과잉환급 미보정, 참고용)
    mkt = {}
    if _os.path.exists('data/odds_snapshot_2026-06-10.csv'):
        snap = pd.read_csv('data/odds_snapshot_2026-06-10.csv')
        for r in snap.itertuples():
            vals = [v for v in (r.implied_fd_pct, r.implied_dk_pct,
                                r.polymarket_pct, r.kalshi_pct) if pd.notna(v)]
            if vals:
                mkt[r.team] = round(float(np.mean(vals)) / 100, 4)
    mkt_rank = {t: i + 1 for i, (t, _) in
                enumerate(sorted(mkt.items(), key=lambda kv: -kv[1]))}

    def card(team):
        return {'team': team, 'modelP': pre.get(team), 'modelRank': rank_of.get(team),
                'marketP': mkt.get(team), 'marketRank': mkt_rank.get(team)}

    def hit_count(k, actual_set):
        """개막 전 우승확률 상위 k팀 중 실제로 그 라운드에 오른 팀 수."""
        top = [t for t, _ in ranked[:k]]
        return {'k': k, 'top': top, 'hits': len(set(top) & actual_set)}

    view = {
        'actualChampion': champion,
        'championCard': card(champion) if champion else None,
        'preTournamentTop': [{'team': t, 'p': p} for t, p in ranked[:10]],
        'finalists': [card(t) for t in sorted(stage_teams.get('F', []))],
        'semifinalists': [card(t) for t in sorted(stage_teams.get('SF', []))],
        'quarterfinalists': [card(t) for t in sorted(stage_teams.get('QF', []))],
        'topNvsActual': [
            {'stage': '결승 진출(2팀)', **hit_count(2, stage_teams.get('F', set()))},
            {'stage': '4강 진출(4팀)', **hit_count(4, stage_teams.get('SF', set()))},
            {'stage': '8강 진출(8팀)', **hit_count(8, stage_teams.get('QF', set()))},
        ],
        'note': '모델 확률은 개막 전(2026-06-10) 스냅샷. 시장 확률은 같은 날 북메이커·'
                '예측시장 구현확률의 단순평균(과잉환급 미보정)이라 합이 1을 넘는다.',
    }
    return view


def report(out):
    s = out['summary']
    o = s['overall']
    print('=== 104경기 전수 채점 (누수 없는 사전예측) ===')
    print(f"채점 경기: {o['n']}  적중 {o['hits']}  적중률 {o['acc']:.1%}  Brier {o['brier']:.4f}")
    print(f"  기준선 — 1/3 찍기 {out['baseline']['uniform']:.4f} / "
          f"비율 찍기 {out['baseline']['baseRate']:.4f}")
    print('\n[라운드별]')
    for r in s['byStage']:
        print(f"  {r['stage']:<8} {r['hits']:>3}/{r['n']:<3} = {r['acc']:.1%}   Brier {r['brier']:.4f}")
    a = s['advance']
    print(f"\n[녹아웃 진출팀 적중] {a['hits']}/{a['n']} = {a['acc']:.1%}")
    if s['market']:
        m = s['market']
        print(f"\n[모델 vs 시장] 배당 기록 {m['n']}경기")
        print(f"  모델 Brier {m['modelBrier']:.4f} (적중 {m['modelAcc']:.1%})")
        print(f"  시장 Brier {m['marketBrier']:.4f} (적중 {m['marketAcc']:.1%})")
        print(f"  → {'모델 승리' if m['winner'] == 'model' else '시장 승리'}")
    if s['ou25']:
        u = s['ou25']
        print(f"\n[언/오버 2.5] {u['hits']}/{u['n']} = {u['acc']:.1%}  Brier {u['brier']:.4f}")
    t = out['tournament']
    if t['actualChampion']:
        c = t['championCard']
        mp = f"{c['modelP']*100:.1f}% (모델 {c['modelRank']}순위)" if c['modelP'] else '기록 없음'
        kp = f"{c['marketP']*100:.1f}% (시장 {c['marketRank']}순위)" if c['marketP'] else '기록 없음'
        print(f"\n[대회 단위] 실제 우승: {t['actualChampion']} — 개막 전 {mp} / {kp}")
        for row in t['topNvsActual']:
            print(f"  {row['stage']}: 개막 전 상위 {row['k']}팀 중 {row['hits']}팀 적중")
    print('\n저장: data/retrodiction.json')


if __name__ == '__main__':
    main()
