"""
changes.py — 예측이 '왜' 변했는지 경기적 설명 생성
====================================================
champion_history(시점별 우승확률) 의 연속 스냅샷을 비교해, 크게 움직인 팀을 찾고
그 사이에 치러진 실제 경기로 이유를 붙인다(수치가 아니라 경기 내용으로).
예: "스페인 ↓ — 카보베르데전 0-0 무승부 (예상 밖)"

출력: data/prediction_changes.json
  [ { "date":"2026-06-15",
      "moves":[ {team, dir, delta, opp, sf, sa, result, upset} | {team, dir, delta, other:true} ] },
    ... 최신순 ]
"""
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import pandas as pd

THRESH = 0.02   # 2%p 이상 움직인 팀만
TOPN = 5        # 날짜별 최대 변동 항목


def build_changes():
    hpath = 'data/champion_history.json'
    if not _os.path.exists(hpath):
        return []
    hist = json.load(open(hpath))
    if len(hist) < 2:
        json.dump([], open('data/prediction_changes.json', 'w'))
        return []

    res = pd.read_csv('data/results.csv')
    res = res[(res['date'] >= '2026-06-11') &
              (res['tournament'] == 'FIFA World Cup') &
              (res['home_score'].notna())].copy()
    res['date'] = res['date'].astype(str)

    # 고정 예측(개막 전) — 이변 판정용
    pred = {}
    if _os.path.exists('data/group_stage_predictions.csv'):
        gp = pd.read_csv('data/group_stage_predictions.csv')
        for r in gp.itertuples():
            pred[(r.home, r.away)] = (r.P_home, r.P_draw, r.P_away)

    def team_match(team, lo, hi):
        """(lo, hi] 기간에 team이 치른 경기 한 건 반환(가장 최근)."""
        sub = res[(res['date'] > lo) & (res['date'] <= hi) &
                  ((res['home_team'] == team) | (res['away_team'] == team))]
        if len(sub) == 0:
            return None
        r = sub.sort_values('date').iloc[-1]
        home = r['home_team'] == team
        sf = int(r['home_score']) if home else int(r['away_score'])
        sa = int(r['away_score']) if home else int(r['home_score'])
        opp = r['away_team'] if home else r['home_team']
        result = 'win' if sf > sa else ('draw' if sf == sa else 'loss')
        # 이변: 고정 예측에서 team이 분명한 우세였는데 이기지 못함
        upset = False
        key = (r['home_team'], r['away_team'])
        if key in pred:
            pH, pD, pA = pred[key]
            fav_win = pH if home else pA
            if fav_win >= 0.55 and result != 'win':
                upset = True
            if (max(pH, pD, pA) == (pA if home else pH)) and result == 'win' and \
               (pA if home else pH) <= 0.30:
                upset = True   # 약체로 평가됐는데 이김
        return {'opp': opp, 'sf': sf, 'sa': sa, 'result': result, 'upset': upset}

    out = []
    for i in range(1, len(hist)):
        prev, cur = hist[i - 1], hist[i]
        lo, hi = prev['t'][:10], cur['t'][:10]
        teams = set(prev['p']) | set(cur['p'])
        moves = []
        for t in teams:
            d = cur['p'].get(t, 0) - prev['p'].get(t, 0)
            if abs(d) < THRESH:
                continue
            m = team_match(t, lo, hi)
            mv = {'team': t, 'dir': 'up' if d > 0 else 'down', 'delta': round(d, 4)}
            if m:
                mv.update(m)
            else:
                mv['other'] = True
            moves.append(mv)
        moves.sort(key=lambda x: -abs(x['delta']))
        moves = moves[:TOPN]
        if moves:
            out.append({'date': hi, 'moves': moves})
    out.reverse()   # 최신순
    json.dump(out, open('data/prediction_changes.json', 'w'),
              ensure_ascii=False, indent=1)
    return out


if __name__ == '__main__':
    c = build_changes()
    print(f'예측 변동 {len(c)}개 시점 생성')
    for e in c[:3]:
        print(e['date'], [f"{m['team']}{'↑' if m['dir']=='up' else '↓'}" for m in e['moves']])
