# 실험 결과 기록

> 사실(측정값)과 의견(해석)을 구분 표기. 재현: `python3.11 experiments/<script>.py`

## 1. 시간가중(time-decay) — `time_decay_backtest.py` (2026-06-24)

**가설(문헌):** 오래된 경기를 지수적으로 down-weight하면 '현재 강도'를 더 잘
반영해 예측이 좋아질 수 있다. — Dixon & Coles (1997, JRSS-C); Ley, Van de Wiele
& Van Eetvelde (2019, Statistical Modelling).

**방법(사실):** prob_model.py와 동일한 walk-forward(시간순 분리)에서
LogisticRegression에 `sample_weight = 0.5**(age/halflife)`만 추가. 두 개의 독립
검증 창에서 반감기 스윕. 학습 피처는 elo_diff_pre 단일(경기 전 값)로 동일.

**측정값(사실):**

| 검증 창 | 균등(현행) Brier | 최적 반감기 | 최적 Brier | Δ |
|---|---|---|---|---|
| 2020~2023 (학습 ≤2019) | 0.5050 | 3년 | 0.5035 | **-0.0015** |
| 2024~현재 (학습 ≤2023) | 0.5075 | 2년 | 0.5061 | **-0.0014** |

- 두 창 모두 반감기 **1.5~5년** 구간에서 평탄하게 개선(단일 하이퍼파라미터
  과적합 아님). LogLoss·적중률도 같은 방향으로 소폭 개선.
- 두 창 공통으로 개선되는 가장 안정적 반감기 ≈ **3년**.

**판정(의견):** 개선은 **작지만 일관적·강건**하며 문헌과 부합. 프로젝트의
"walk-forward Brier 개선 시 채택" 기준을 통과. 단 Δ가 모델 간 차이(0.5053~0.5065)
수준으로 작아, 체감 변화는 미미할 것.

**채택 시 주의(의견):** 대회 후 '모델 vs 시장' 공정 채점용 **개막 전 프리즈
스냅샷**(score_predictions.json, group_stage_predictions.csv)은 재학습하면 안 됨.
시간가중은 **현재형 모델(prob_model.pkl·live 예측)** 에만 적용하는 것이 타당.

## 2. 스코어 모델 시간가중 — `score_decay_backtest.py` (2026-06-24)

**방법(사실):** score_model.py의 Dixon-Coles를 재현, Poisson GLM과 ρ 추정에
sample_weight만 추가. 두 독립 창에서 스코어 로그우도(↑)·O/U2.5 Brier(↓) 비교.

**측정값(사실, 반감기 3년):**

| 검증 창 | 균등 logLik→가중 | 균등 Brier→가중 |
|---|---|---|
| 2020~2023 | -2.8415 → **-2.8295** | 0.2469 → **0.2446** |
| 2024~현재 | -2.8883 → **-2.8824** | 0.2459 → **0.2445** |

- 두 창 모두 반감기 1~5년 전반에서 **두 지표 동시 개선**(강건).

**판정(의견):** prob 모델과 같은 방향으로 일관 개선 → 백테스트 기준 통과.

## 채택 결정 (2026-06-24)
사용자 선택: **현재형 모델에만** 반감기 3년 시간가중 적용.
- `src/prob_model.py`: prob_model.pkl 최종 적합에 시간가중(검증 리더보드는 균등).
- `src/score_model.py`: PRIMARY(현재형) Dixon-Coles 적합에 시간가중(검증 리더보드는 균등).
- 프리즈 스냅샷(group_stage_predictions.csv·score_predictions.json)은 가드로 보존.
