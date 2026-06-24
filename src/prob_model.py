"""
prob_model.py — Elo 점수차 → 승/무/패 확률 변환 + walk-forward 검증
==================================================================
왜 이 단계가 필요한가:
  Elo의 E값은 '기대 승점율'(승=1, 무=0.5 혼합)이라 무승부 확률을 따로 주지 않음.
  베팅 시장과 비교하려면 P(승), P(무), P(패) 세 개가 모두 필요.

방법: 다항 로지스틱 회귀 (multinomial logistic regression)
  입력 피처: elo_diff_pre (홈 어드밴티지 반영된 경기 전 Elo 차이) 단 1개
  출력: P(H), P(D), P(A)
  - 피처를 1개만 쓰는 이유: 단순할수록 과적합 위험이 낮고,
    Elo 차이 하나에 이미 팀 전력 정보 대부분이 압축되어 있음.
    (피처 추가는 반드시 검증 지표 개선을 확인한 뒤에만 — 백테스트 우선 원칙)

검증: walk-forward
  학습: 1990-01-01 ~ 2023-12-31 의 '경쟁 경기'(친선전 제외)
  평가: 2024-01-01 ~ 2026-06-08 의 경쟁 경기
  - 시간을 섞어서 나누면(random split) 미래 정보가 학습에 새어 들어가
    성능이 부풀려짐. 시간순 분리는 타협 불가 원칙.

지표: 멀티클래스 Brier Score = mean( ||예측확률벡터 - 원핫실제|| ^2 )
  완벽한 예측 = 0, 항상 1/3씩 찍기 = 0.667
  비교 기준선(baseline) 2개와 함께 보고:
    (a) uniform: 항상 (1/3, 1/3, 1/3)
    (b) base-rate: 학습기간의 전체 H/D/A 비율을 항상 출력
  모델이 기준선을 못 이기면 모델을 쓸 이유가 없음.
"""
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

hist = pd.read_csv('data/elo_history.csv')
comp = hist[hist['tournament'] != 'Friendly'].copy()   # 친선전 제외(노이즈)
comp = comp[comp['date'] >= '1990-01-01']

train = comp[comp['date'] <= '2023-12-31']
test = comp[comp['date'] >= '2024-01-01']
print(f'학습 경기 수: {len(train)},  검증 경기 수: {len(test)}')

CLASSES = ['H', 'D', 'A']
X_tr = train[['elo_diff_pre']].values
y_tr = train['outcome'].values
X_te = test[['elo_diff_pre']].values
y_te = test['outcome'].values

model = LogisticRegression(max_iter=1000)
model.fit(X_tr, y_tr)
# sklearn은 클래스를 알파벳순(A,D,H)으로 정렬하므로 순서 주의
order = list(model.classes_)

def brier(probs, actual):
    onehot = np.zeros_like(probs)
    for i, a in enumerate(actual):
        onehot[i, order.index(a)] = 1
    return np.mean(np.sum((probs - onehot) ** 2, axis=1))

p_model = model.predict_proba(X_te)
p_uniform = np.full((len(y_te), 3), 1 / 3)
rates = np.array([np.mean(y_tr == c) for c in order])
p_base = np.tile(rates, (len(y_te), 1))

print(f'Brier (uniform 1/3 찍기): {brier(p_uniform, y_te):.4f}')
print(f'Brier (학습기간 비율 찍기): {brier(p_base, y_te):.4f}')
print(f'Brier (Elo+로지스틱 모델): {brier(p_model, y_te):.4f}')
acc = np.mean(np.array(order)[p_model.argmax(1)] == y_te)
print(f'단순 적중률(가장 높은 확률 클래스 선택): {acc:.1%}')

# ── 캘리브레이션 점검: 예측확률 구간별 실제 적중률 ──────────────
print('\n캘리브레이션 (홈승 예측확률 구간 → 실제 홈승 비율):')
ph = p_model[:, order.index('H')]
actual_h = (y_te == 'H').astype(float)
for lo in np.arange(0, 1.0, 0.2):
    mask = (ph >= lo) & (ph < lo + 0.2)
    if mask.sum() > 10:
        print(f'  예측 {lo:.1f}~{lo+0.2:.1f}: 실제 {actual_h[mask].mean():.3f}  (n={mask.sum()})')

# ── 시간가중(time-decay) 검증: 균등 vs 반감기 3년 ────────────────
# 근거: Dixon&Coles(1997), Ley et al(2019). 오래된 경기를 지수 down-weight.
# experiments/time_decay_backtest.py 에서 두 독립 walk-forward 창 모두
# Brier 일관 개선 확인(2020~2023 -0.0015, 2024~현재 -0.0014, 최적 반감기 ≈3년).
HALFLIFE_YEARS = 3.0
def _decay_w(dates, ref):
    age = (pd.to_datetime(ref) - pd.to_datetime(dates)).dt.days.values.astype(float)
    return 0.5 ** (age / (HALFLIFE_YEARS * 365.25))

w_tr = _decay_w(train['date'], '2024-01-01')
m_decay = LogisticRegression(max_iter=1000)
m_decay.fit(X_tr, y_tr, sample_weight=w_tr)
p_decay = m_decay.predict_proba(X_te)
print(f'Brier (시간가중 {HALFLIFE_YEARS}년):  {brier(p_decay, y_te):.4f}'
      f'  (균등 대비 {brier(p_decay, y_te)-brier(p_model, y_te):+.4f})')

# 전체 데이터로 재학습한 최종(현재형) 모델 저장 — 시간가중 적용.
#   주의: 대회 후 '모델 vs 시장' 공정 채점용 개막 전 프리즈 스냅샷
#   (group_stage_predictions.csv·score_predictions.json)은 별도 가드로 보존됨.
#   여기서 만드는 prob_model.pkl 은 '현재 강도' 추정용(라이브/현재 예측).
import pickle
_latest = comp['date'].max()
final = LogisticRegression(max_iter=1000)
final.fit(comp[['elo_diff_pre']].values, comp['outcome'].values,
          sample_weight=_decay_w(comp['date'], _latest))
with open('data/prob_model.pkl', 'wb') as f:
    pickle.dump({'model': final, 'classes': list(final.classes_),
                 'halflife_years': HALFLIFE_YEARS}, f)
print(f'\n최종 모델 저장 완료 (전체 기간 재학습 · 시간가중 반감기 {HALFLIFE_YEARS}년)')
