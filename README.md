# WC2026 Predictor

2026 월드컵 조별리그 72경기를 검증장으로 "내 모델 vs 베팅 시장" 정확도 대결을 하는 프로젝트.
실제 베팅용이 아님 (한국에서 해외 베팅은 불법). 상세 배경: docs/, 작업 원칙: CLAUDE.md

## 빠른 시작
    pip install -r requirements.txt
    python run_pipeline.py --update   # 최신 결과 반영 후 전체 재계산
    python src/evaluate.py            # 결과가 쌓이면 모델 vs 시장 채점

## 구조
    CLAUDE.md                  Claude Code 컨텍스트 (원칙·상태·다음 작업)
    run_pipeline.py            전체 파이프라인
    src/elo.py                 Elo 레이팅 엔진
    src/prob_model.py          확률 변환 + walk-forward 검증
    src/predict.py             72경기 예측 + 우승 시뮬레이션
    src/evaluate.py            대회 후 모델 vs 시장 Brier 판정
    data/closing_odds.csv      ★ 매 경기 킥오프 직전 배당을 여기에 기록 (소수 배당)
    docs/MATH.md               ★ 모든 수식 명세 (Elo·확률변환·Brier·몬테카를로)
    docs/                      알고리즘 해설 문서

## 수식
모든 계산식은 [docs/MATH.md](docs/MATH.md) 에 정리되어 있다. 핵심 두 줄:

    기대 승점율   E = 1 / (1 + 10^(-(R_home + H - R_away)/400))
    레이팅 갱신   R_new = R_old + K · G · (S − E)

## 웹 플랫폼
예측 결과를 누구나 볼 수 있는 웹 대시보드: **[wc2026-web](https://github.com/choigod1023/wc2026-web)**
(우승 확률·72경기 예측·모델 vs 시장 비교·수식 해설)
