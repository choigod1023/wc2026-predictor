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

진행 반영(대회/토너먼트 모두):
  · 끝난 조별 경기는 실제 스코어로, 끝난 녹아웃 경기는 진출팀으로 고정하고
    남은 경기만 시뮬 → 라운드가 진행될수록 우승·단계 확률이 실제 결과를 반영.
  · 녹아웃은 2026 공식 브래킷(Annex C 3위 배정 포함) 고정 전개.
  · split_wc 로 조별 72경기/녹아웃을 분리(녹아웃이 results.csv에 추가돼도 조
    복원이 깨지지 않음).
"""
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import pandas as pd
import numpy as np
import pickle

from sim import recover_groups, simulate_scores, split_wc

N_SIM = 20000
HOME_ADV = 100

ratings = pd.read_csv('data/elo_final.csv', index_col=0)['elo'].to_dict()
with open('data/prob_model.pkl', 'rb') as f:
    saved = pickle.load(f)
model, classes = saved['model'], saved['classes']
iH, iD, iA = classes.index('H'), classes.index('D'), classes.index('A')
score_params = json.load(open('data/score_model_params.json'))

df = pd.read_csv('data/results.csv')
# 조별리그 72경기 / 녹아웃 경기 분리 (녹아웃이 추가돼도 조 복원이 안 깨지게).
group_df, ko_df = split_wc(df)
wc = group_df                       # 조별리그(72) — 조 복원·조별 예측·프리즈 기준

# ── 1. 조 편성 역추출 (조별리그는 조 안에서만 경기 → 연결요소 = 조) ──
groups = recover_groups(wc)

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
# 채점 기준이 되는 '개막 전 고정본'은 한 번 만들어지면 사후 수정 금지(CLAUDE.md 원칙).
# 이미 있으면 덮어쓰지 않는다. 의도적 재생성은 FORCE_PREDICTIONS=1 로.
_FROZEN = 'data/group_stage_predictions.csv'
if _os.path.exists(_FROZEN) and _os.environ.get('FORCE_PREDICTIONS') != '1':
    print(f'(고정본 {_FROZEN} 유지 — 사후 수정 금지. 재생성은 FORCE_PREDICTIONS=1)')
else:
    pred.to_csv(_FROZEN, index=False)

# '현재(업데이트된 Elo)' 승무패 예측 — 항상 갱신. 조별 + 녹아웃 전체를 포함해
# 라이브 탭이 녹아웃 경기 예측도 보여줄 수 있게 한다. (고정본과 비교해 '예측 변화'용)
live_all = pd.concat([wc, ko_df]) if len(ko_df) else wc
live_rows = []
for r in live_all.sort_values('date').itertuples():
    h = 0 if r.neutral else HOME_ADV
    diff = ratings[r.home_team] + h - ratings[r.away_team]
    p = model.predict_proba([[diff]])[0]
    live_rows.append({'home': r.home_team, 'away': r.away_team,
                      'pHome': round(p[iH], 4), 'pDraw': round(p[iD], 4),
                      'pAway': round(p[iA], 4)})
json.dump(live_rows, open('data/live_predictions.json', 'w'), ensure_ascii=False)

# ── 3. 스코어 기반 몬테카를로 시뮬레이션 ───────────────────────
# 스코어라인을 추첨해 2026 룰(승자승→골득실)로 조 순위, 녹아웃은 공식 브래킷 전개.
# 이미 끝난 조별 경기는 실제 스코어로, 끝난 녹아웃 경기는 진출팀으로 고정 →
# 대회/토너먼트 진행에 따라 우승·단계 확률이 실제 결과를 반영해 변동.
fixtures = [(r.home_team, r.away_team, bool(r.neutral))
            for r in wc.sort_values('date').itertuples()]
played = {(r.home_team, r.away_team): (int(r.home_score), int(r.away_score))
          for r in wc.itertuples() if pd.notna(r.home_score)}
ko_played = {(r.home_team, r.away_team): (int(r.home_score), int(r.away_score))
             for r in ko_df.itertuples() if pd.notna(r.home_score)}
out = simulate_scores(fixtures, groups, ratings, score_params, n_sim=N_SIM,
                      played=played, ko_played=ko_played)
print(f'조별 {len(played)}경기 + 녹아웃 {len(ko_played)}경기 결과 고정, 남은 경기만 시뮬')

res = pd.DataFrame({'team': list(out['champion'].keys()),
                    'P_champion': [out['champion'][t] for t in out['champion']],
                    'P_final': [out['final'][t] for t in out['champion']]})
res = res.sort_values('P_champion', ascending=False)
res.to_csv('data/championship_probs.csv', index=False)

# 단계별(R32~결승) 도달 확률 + 우승 → stage_probs.json (웹 토너먼트 탭)
stage_rows = []
for t, st in out['stages'].items():
    stage_rows.append({'team': t, **{k: round(v, 4) for k, v in st.items()},
                       'champion': round(out['champion'][t], 4)})
stage_rows.sort(key=lambda r: r['champion'], reverse=True)
json.dump(stage_rows, open('data/stage_probs.json', 'w'), ensure_ascii=False, indent=2)

# 우승 확률 시계열 누적 (대회 진행에 따른 변화 추세 그래프용)
import datetime as _dt
HIST = 'data/champion_history.json'
hist = json.load(open(HIST)) if _os.path.exists(HIST) else []
snap = {t: round(float(out['champion'][t]), 4)
        for t in out['champion'] if out['champion'][t] >= 0.003}
# 직전 스냅샷과 동일하면(결과 변화 없음) 추가하지 않음
if not hist or hist[-1].get('p') != snap:
    hist.append({'t': _dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%MZ'), 'p': snap})
hist = hist[-400:]
json.dump(hist, open(HIST, 'w'), ensure_ascii=False)

# 예측 변동 '경기적' 이유 생성 (prediction_changes.json)
from changes import build_changes
build_changes()

print('=== 우승 확률 Top 12 (스코어 기반 시뮬) ===')
print(res.head(12).to_string(index=False))
print('\n저장: championship_probs.csv, stage_probs.json, prediction_changes.json')
