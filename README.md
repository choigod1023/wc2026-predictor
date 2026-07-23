# WC2026 Predictor

2026 월드컵 조별리그 72경기를 검증장으로 "내 모델 vs 베팅 시장" 정확도 대결을 하는 프로젝트.
실제 베팅용이 아님 (한국에서 해외 베팅은 불법). 상세 배경: docs/, 작업 원칙: CLAUDE.md

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)

## ✨ 주요 기능
- **Elo 레이팅 엔진** — martj42/international_results 의 A매치 결과로 국가별 Elo를 산출(홈 어드밴티지 반영).
- **확률 변환 + walk-forward 검증** — Elo 차이를 다항 로지스틱으로 승/무/패 확률화하고, 미래 정보 누수 없이 walk-forward Brier로 검증.
- **멀티모델 리더보드** — Elo-로지스틱(기준)·+|diff|·Davidson·기준선을 동일 규약으로 등록해 Brier로 공정 비교(`src/models.py`, `src/compare_models.py`).
- **72경기 예측 + 몬테카를로** — 조별리그 예측과 토너먼트 시뮬레이션으로 단계별 진출·우승 확률 산출.
- **스코어 모델** — 포아송·Dixon-Coles 기반 스코어 예측 → 언더/오버·핸디캡 근거 제시.
- **모델 vs 시장 채점** — 개막 전 고정본 예측과 마감 배당(마진 제거 정규화)을 같은 경기에 대해 Brier로 비교(`src/evaluate.py`).
- **자동 갱신 파이프라인** — GitHub Actions로 6시간마다 결과·Elo·예측·마감배당을 갱신·커밋.

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
    src/sim.py                 토너먼트 몬테카를로 코어 (단계별 진출 확률 포함)
    src/models.py              확률 모델 레지스트리 (Elo-로지스틱·+|diff|·Davidson·기준선)
    src/compare_models.py      멀티모델 walk-forward Brier 리더보드 + 모델별 우승 시뮬
    src/score_model.py         스코어 예측(포아송·Dixon-Coles) → 언오버·핸디캡 + 근거
    src/capture_odds.py        named API 마감배당 → closing_odds.csv 자동 적재(멱등)
    src/export_web.py          파이프라인 산출물 → 웹(wc2026-web/data) JSON export
    src/evaluate.py            대회 후 모델 vs 시장 Brier 판정

## 자동 갱신 (서버에서 주기 실행)
모델은 요청마다가 아니라 **스케줄로** 돈다(입력=경기결과는 경기 끝날 때만 바뀜, GPU 불필요).
- 이 레포 `.github/workflows/refresh-data.yml`: 6시간마다 `run_pipeline.py --update`
  (결과·Elo·예측 갱신 + 마감배당 누적) → 변경분 자동 커밋.
- 웹 레포 `refresh-predictions.yml`: 6시간마다 이 레포를 clone해 파이프라인을 돌리고
  `export_web.py` 로 예측 JSON을 웹에 커밋 → Vercel 자동 재배포.
- 채점용 고정본 `group_stage_predictions.csv` 는 가드로 사후 수정 차단(재생성은 `FORCE_PREDICTIONS=1`).
- 수동 실행: GitHub Actions 탭의 "Run workflow" 버튼.
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
