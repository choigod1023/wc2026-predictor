"""
elo.py — 국가대표 축구 Elo 레이팅 엔진
=====================================
참고 표준: World Football Elo Ratings (eloratings.net) 방식

Elo의 핵심 수식 (이해해야 할 것은 이 두 줄이 전부):

  기대 승률  E = 1 / (1 + 10^(-(R_home + H - R_away) / 400))
  업데이트   R_new = R_old + K * G * (실제결과 - E)

  - 실제결과: 승=1, 무=0.5, 패=0
  - 400이라는 분모: "레이팅 400점 차이 = 승률 약 10배 차이"가 되도록 하는
    스케일 상수. Elo 체계의 정의일 뿐 튜닝 대상이 아님.
  - (실제결과 - E)가 핵심: 이변일수록(예상과 다를수록) 점수 이동이 큼.
    강팀이 약팀을 이기면 거의 변동 없음. 약팀이 이기면 크게 이동.
"""
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import pandas as pd
import numpy as np

# ── 파라미터 1: K (경기 중요도 가중치) ─────────────────────────
# 친선전은 정보 가치가 낮다 → 레이팅 변동을 작게.
# 월드컵 본선은 양 팀 모두 전력 → 가장 신뢰할 수 있는 신호.
K_TABLE = {
    'FIFA World Cup': 60,
    'FIFA World Cup qualification': 40,
    'UEFA Euro': 50, 'Copa América': 50, 'African Cup of Nations': 50,
    'AFC Asian Cup': 50, 'CONCACAF Championship': 50, 'Gold Cup': 50,
    'Confederations Cup': 40,
    'UEFA Nations League': 30, 'CONCACAF Nations League': 30,
    'UEFA Euro qualification': 40, 'African Cup of Nations qualification': 40,
    'AFC Asian Cup qualification': 40, 'Copa América qualification': 40,
}
K_DEFAULT_COMPETITIVE = 30   # 그 외 군소 대회
K_FRIENDLY = 20

HOME_ADV = 100   # 파라미터 3: 홈 어드밴티지 (Elo 점수 환산)
INIT_RATING = 1500


def k_factor(tournament: str) -> float:
    if tournament == 'Friendly':
        return K_FRIENDLY
    return K_TABLE.get(tournament, K_DEFAULT_COMPETITIVE)


def goal_multiplier(score_diff: int) -> float:
    """파라미터 2: 골 차 배수. 1골차=1.0, 2골차=1.5, N골차(N>=3)=(11+N)/8"""
    d = abs(score_diff)
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return (11 + d) / 8


def expected_score(r_home: float, r_away: float, neutral: bool) -> float:
    """홈팀 기대 승점율. neutral이면 홈 어드밴티지 미적용."""
    h = 0 if neutral else HOME_ADV
    return 1 / (1 + 10 ** (-(r_home + h - r_away) / 400))


def run_elo(df: pd.DataFrame):
    """
    경기를 시간순으로 한 번 훑으며 레이팅을 갱신.
    반환:
      ratings — 최종 레이팅 dict
      history — 각 경기의 '경기 전' elo_diff 기록 (모델 학습용 피처)
                ※ 반드시 경기 '전' 값을 써야 데이터 누수(leakage)가 없음
    """
    ratings = {}
    hist_rows = []
    for row in df.itertuples(index=False):
        rh = ratings.get(row.home_team, INIT_RATING)
        ra = ratings.get(row.away_team, INIT_RATING)

        # 경기 전 스냅샷 저장 (학습 피처)
        neutral = bool(row.neutral)
        h = 0 if neutral else HOME_ADV
        elo_diff_pre = (rh + h) - ra

        hs, as_ = int(row.home_score), int(row.away_score)
        actual = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        e = expected_score(rh, ra, neutral)
        delta = k_factor(row.tournament) * goal_multiplier(hs - as_) * (actual - e)

        ratings[row.home_team] = rh + delta
        ratings[row.away_team] = ra - delta

        hist_rows.append({
            'date': row.date, 'home_team': row.home_team, 'away_team': row.away_team,
            'tournament': row.tournament, 'neutral': neutral,
            'elo_diff_pre': elo_diff_pre,
            'outcome': 'H' if actual == 1.0 else ('D' if actual == 0.5 else 'A'),
        })
    return ratings, pd.DataFrame(hist_rows)


if __name__ == '__main__':
    df = pd.read_csv('data/results.csv')
    played = df[df['home_score'].notna()].copy()
    played = played.sort_values('date')
    ratings, hist = run_elo(played)
    hist.to_csv('data/elo_history.csv', index=False)
    pd.Series(ratings).sort_values(ascending=False).to_csv('data/elo_final.csv',
                                                           header=['elo'])
    top = pd.Series(ratings).sort_values(ascending=False).head(15)
    print('=== 2026-06-08 기준 Elo Top 15 ===')
    for t, r in top.items():
        print(f'{t:20s} {r:7.0f}')
