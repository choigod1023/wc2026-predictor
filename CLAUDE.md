# CLAUDE.md — 2026 월드컵 예측 시스템

## 프로젝트 목적
실제 베팅이 아니라 **"내 모델이 베팅 시장 가격보다 정확한가"를 검증하는 프로젝트**다.
2026 월드컵(6/11~7/19) 조별리그 72경기를 검증장으로 사용한다.
사용자는 한국 거주자이며, 한국에서 해외 베팅 사이트 이용은 국민체육진흥법 위반이다.
실제 베팅을 돕는 기능(베팅 사이트 연동, 자동 베팅 등)은 추가하지 않는다.

## 절대 원칙 (위반 금지)
1. **백테스트 우선** — 어떤 피처/모델 변경도 walk-forward 검증에서 Brier Score
   개선을 확인하기 전에는 채택하지 않는다.
2. **시간 순서 엄수** — train/test는 반드시 시간 기준 분리. random split 금지.
3. **데이터 누수 금지** — 학습 피처는 반드시 경기 '전' 시점 값만 사용
   (elo_history.csv의 elo_diff_pre가 그 예).
4. **재현성** — 같은 입력이면 같은 출력. 시뮬레이션은 시드 고정(rng seed=42).
5. **사실/의견 구분** — 문서와 출력에서 측정된 수치(사실)와 해석(의견)을 구분 표기.

## 현재 상태 (2026-06-10, 개막 전날 기준)
- 데이터: data/results.csv (martj42, 2026-06-08 경기까지, 72경기 일정 포함)
- Elo 엔진(src/elo.py): eloratings.net 방식. K=60(WC)/50(대륙)/40(예선)/30(NL)/20(친선),
  골차 배수 1/1.5/(11+N)/8, 홈 어드밴티지 +100. 외부 스냅샷과 순위·격차 일치 확인됨.
- 확률 모델(src/prob_model.py): elo_diff_pre 단일 피처 다항 로지스틱.
  walk-forward(학습 1990~2023, 검증 2024~2026.6) Brier 0.5056
  (기준선: uniform 0.6667, base-rate 0.6385). 캘리브레이션 전 구간 양호.
- 예측(src/predict.py): 72경기 확률 + 몬테카를로 2만회 우승 확률.
  녹아웃은 2026 공식 브래킷 고정 전개(아래 '공식 브래킷 채택' 참조). 조 세부
  동률 일부만 Elo 근사 잔존. 상세는 docs/ 문서 6장.
- 시장 스냅샷: data/odds_snapshot_2026-06-10.csv (개막 전날 고정 기준선).
  모델 vs 시장 주요 괴리: 아르헨티나(모델 20.0% vs 시장 9~10%),
  프랑스(모델 10.7% vs 시장 16~17%), 미국(모델 0.07% vs 시장 1.6%).

## 시간가중 채택 (2026-06-24)
근거: Dixon&Coles(1997), Ley et al(2019). experiments/time_decay_backtest.py·
score_decay_backtest.py 가 두 독립 walk-forward 창 모두에서 Brier(prob)·
logLik/O/U Brier(score) 일관 개선 확인(소폭, 최적 반감기 ≈3년).
**현재형 모델에만** 반감기 3년 시간가중 적용: prob_model.pkl 최종 적합,
score_model PRIMARY 적합. 검증 리더보드 적합과 개막 전 프리즈 스냅샷
(group_stage_predictions.csv·score_predictions.json)은 균등/보존 유지.

## 진행 중인 숙제 (사용자 작업)
data/closing_odds.csv 에 각 경기 킥오프 직전 3-way 배당(odds_H/D/A, 소수 배당)을
기록하는 것. 템플릿 생성되어 있음. 이것이 대회 후 판정의 필수 데이터다.

## 공식 브래킷 채택 (2026-06-24)
대표 시뮬(simulate_scores)의 녹아웃을 **2026 공식 브래킷(R32~결승) 고정 전개**로
교체(근사 b 제거). 조 라벨은 앵커 시드팀으로 복원, 3위 8팀은 공식 후보-조 제약
완전매칭(495조합 전수 가능 확인)으로 슬롯 배정. 경로 의존성 보존 → 우승확률
변동(아르헨 25.8→27.8, 잉글 9.2→8.0 등). 단계합 정합(R32=32…F=2) 확인.
모델 비교용 simulate()는 단순성 위해 무작위 풀 유지(의도적).

## 다음 작업 후보 (우선순위순)
1. 경기 결과가 쌓이면 src/evaluate.py 로 모델 vs 시장 Brier 중간 점검
2. 피처 후보 실험: 스쿼드 시장가치, 최근 10경기 가중 폼
   — 반드시 한 번에 하나씩, walk-forward Brier 개선 확인 후 채택
3. 토너먼트 단계용 무승부→연장 모델 정교화 (현재는 Elo 기대승점율 근사)
4. (선택) compare_models의 simulate()도 공식 브래킷으로 통일

## 멀티모델 & 웹 (2026-06-11 추가)
- 멀티모델 비교: src/models.py(레지스트리: Elo-로지스틱 / +|diff| / Davidson / 기준선),
  src/compare_models.py 가 동일 walk-forward로 Brier 리더보드 + 모델별 우승 시뮬 출력.
  결과: data/model_leaderboard.json, champion_by_model.json, stage_probs.json.
  검증 Brier: +|diff| 0.5053 ≈ 기준 0.5056 ≈ Davidson 0.5065 (차이 미미 = Elo 한 피처가 충분).
- 시뮬 코어 src/sim.py: simulate(W/D/A, 모델비교용) + simulate_scores(스코어라인 추첨,
  대표). 대표 시뮬은 score_model 계수로 스코어를 추첨해 조 동률을 2026 승자승→골득실로
  판정(근사 a 제거), 녹아웃은 연장·승부차기. predict.py가 이걸로 championship_probs+
  stage_probs 생성. 파이프라인 순서: elo→prob_model→score_model→predict→compare_models.
- 마감배당 자동화 src/capture_odds.py: named 경기별 3-way 배당을 closing_odds.csv에
  비파괴 적재(멱등, --overwrite로 갱신). run_pipeline --update 시 자동 실행. evaluate.py가
  결과 쌓일 때 모델 vs 시장 Brier 자동 채점.
- 스코어 모델 src/score_model.py: Elo→득점 Poisson GLM + Dixon-Coles 보정,
  walk-forward(스코어 로그우도/O/U2.5 Brier)로 검증. 72경기 기대득점·Top스코어·
  언오버(1.5/2.5/3.5)·핸디캡·근거설명 → data/score_predictions.json, score_leaderboard.json.
  검증: DC logLik -2.887 > 독립포아송 -2.888 > 기준선 -3.255.
- 깃 레포 2개(public): 모델 github.com/choigod1023/wc2026-predictor,
  웹 github.com/choigod1023/wc2026-web (Next.js, Vercel 배포).
  웹 탭(카테고리 드롭다운 네비): 대시보드/라이브/분석(스코어·경우의수·모델비교·토너먼트)/
  정보(룰·수식). 라이브는 named.com API(키 불필요)를 Next 서버 라우트(app/api/live)로
  프록시(CORS 회피)해 스코어·경기별 3-way 배당·실시간 조별 순위 제공.
  스코어 탭은 score_predictions.json(언오버·핸디캡·근거), 경우의수 탭(app/api/scenarios+
  lib/scenarios.ts)은 잔여경기 조합 열거로 32강 직접진출 확정/탈락/확률을 2026 승자승 룰로 계산.
  룰 탭은 2026 변경(48팀·32강·승자승 최우선·FIFA랭킹 타이브레이크) 출처표기. 매핑 lib/teams.ts,
  조 복원/순위 lib/groups.ts.

## 실행 방법
- 전체 파이프라인(데이터 갱신 포함): `python run_pipeline.py --update`
- 데이터 갱신 없이 재계산만: `python run_pipeline.py` (elo→prob_model→predict→compare_models)
- 멀티모델 비교만: `python src/compare_models.py`
- 대회 후 평가: `python src/evaluate.py` (closing_odds.csv 와 결과 필요)
- 의존성: `pip install -r requirements.txt`
- 웹 JSON 갱신: 위 산출물(data/*.json)을 wc2026-web/data/ 로 복사

## 배경 문서
docs/WC2026_예측시스템_문서.md — 알고리즘 해설, 파라미터 근거, 검증 결과,
시장 비교, 한계. 설계 질문이 생기면 이 문서를 먼저 읽을 것.
