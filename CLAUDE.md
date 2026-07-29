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

## 대회 종료 · 104경기 전수 채점 (2026-07-29)
대회는 2026-07-19 종료(우승 스페인, 준우승 아르헨티나, 3위 잉글랜드).
개막 전 고정본은 조별 72경기뿐이라 녹아웃 32경기에 채점 가능한 예측이 없었다
→ **src/retrodict.py** 가 104경기 전부를 '누수 없는 사전예측'으로 채우고 채점한다.

누수 차단(원칙 2·3 준수)
- 피처: elo_history.csv 의 elo_diff_pre(정의상 경기 전 값). 대회 중 갱신되지만
  각 경기 시점에선 전부 과거 정보.
- 계수: 개막 전(2026-06-11 미만) 경기만으로 로지스틱 고정 적합(시간가중 3년).
  대회 결과는 계수에 한 경기도 들어가지 않는다.
- 교차검증: 조별 72경기 Brier 0.5359(사전예측) vs 0.5336(개막 전 고정본, evaluate.py)
  → 사실상 동일. 채점 파이프라인이 기존 결과를 재현함을 확인.

측정 결과(사실)
- 1X2 적중 69/104 = 66.3%, Brier 0.4957 (기준선: 1/3찍기 0.6667, 비율찍기 0.6365)
- 라운드별: 조별 61.1% / 32강 81.2% / 16강 62.5% / 8강 4/4 / 4강 2/2 / 3·4위전 0/1 / 결승 적중
- 녹아웃 진출팀 25/32 = 78.1%. 승부차기 4경기는 **전부** 모델 예측과 반대로 갈림(0/4).
- 모델 vs 시장(배당 기록 72경기): 모델 0.5359 vs 시장 0.4863 → **시장 승리**.
  녹아웃은 배당 미기록이라 직접 비교 불가.
- 언/오버 2.5(개막 전 고정본): 36/72 = 50.0%, Brier 0.2552.
- 대회 단위: 개막 전 1순위 스페인 26.6%(시장 17.1%)가 실제 우승. 상위 2팀=실제 결승 2팀,
  상위 4팀=실제 4강 4팀 전부 적중. 8강은 상위 8팀 중 4팀.

해석(의견): 경기 단위 확률의 질은 시장이 앞섰고(Brier), 대회 단위 상위권 판별은
모델이 잘 맞혔다. 두 지표는 서로 다른 것을 재므로 '모델이 이겼다'로 뭉뚱그리지 말 것.

산출물: data/retrodiction.json (웹 `/report` 성적표 탭이 그대로 읽음)

## 승부차기 데이터 (2026-07-29)
data/shootouts.csv(martj42 shootouts.csv) 추가. 무승부로 끝난 녹아웃의 진출팀을 확정.
기존엔 '다음 라운드 재등장' 휴리스틱으로 추론했는데, 양 팀 모두 앞선 라운드를 치른
경우(둘 다 등장 2회 이상) 판별에 실패해 그 대진을 계속 시뮬 → 대회가 끝났는데도
우승 확률이 스페인 94.9%에 머물렀다. simulate_scores(ko_pens=...) 로 구멍을 막아
현재 스페인 100%로 확정. run_pipeline --update 가 shootouts.csv 도 함께 내려받는다.

## 진행 중인 숙제 (사용자 작업)
data/closing_odds.csv 에 각 경기 킥오프 직전 3-way 배당(odds_H/D/A, 소수 배당)을
기록하는 것. 조별 72경기는 채워졌고 **녹아웃 32경기는 비어 있다** — 그래서 녹아웃
구간의 모델 vs 시장 비교가 불가능하다. 다음 대회에선 녹아웃도 함께 기록할 것.

## 공식 브래킷 채택 (2026-06-24)
대표 시뮬(simulate_scores)의 녹아웃을 **2026 공식 브래킷(R32~결승) 고정 전개**로
교체(근사 b 제거). 조 라벨은 앵커 시드팀으로 복원, 3위 8팀은 공식 후보-조 제약
완전매칭(495조합 전수 가능 확인)으로 슬롯 배정. 경로 의존성 보존 → 우승확률
변동(아르헨 25.8→27.8, 잉글 9.2→8.0 등). 단계합 정합(R32=32…F=2) 확인.
모델 비교용 simulate()는 단순성 위해 무작위 풀 유지(의도적).

## 다음 작업 후보 (우선순위순)
1. 시장에 진 0.05 격차(조별 Brier 0.5359 vs 0.4863) 좁히기 — 피처 후보 실험:
   스쿼드 시장가치, 최근 10경기 가중 폼. 반드시 한 번에 하나씩, walk-forward
   Brier 개선 확인 후 채택
2. 토너먼트 단계용 무승부→연장·승부차기 모델 정교화. 현재 retrodict.py 의
   P(진출)=P_home+P_draw·s 는 s를 Elo 기대승점율로 둔 근사이고, 실제 승부차기
   4경기를 0/4로 틀렸다(표본 4 = 우연 구분 불가, 개선 검증엔 과거 대회 데이터 필요)
3. 다음 대회에선 녹아웃 마감배당도 기록 — 지금은 녹아웃 시장 비교가 아예 불가

## 녹아웃 진행 반영 (2026-06-28)
results.csv 엔 스테이지 컬럼이 없어, sim.split_wc 가 '날짜순 앞 72경기=조별,
나머지=녹아웃'으로 분리(녹아웃이 추가돼도 recover_groups 안 깨짐). predict.py 는
조별 결과=스코어 고정 + 녹아웃 결과=진출팀 고정(simulate_scores 의 ko_played)으로
남은 경기만 시뮬 → 라운드 진행에 따라 우승·단계 확률 변동. 승부차기 진출자는
다음 라운드 등장으로 구조 추론(최신 라운드 무승부만 일시적으로 시뮬). live_predictions
.json 은 조별+녹아웃 전체 W/D/A 포함. score_model 은 녹아웃 포함(live_score.json),
compare_models 는 split_wc 로 조별만(모델비교=순수모델). 웹 computeStandings 는 조 내
경기만 집계(녹아웃 누수 차단), recoverGroups 는 정적 matches.json 기반이라 무영향.

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
- 대회 후 평가(조별·시장 대결): `python src/evaluate.py` (closing_odds.csv 와 결과 필요)
- 104경기 전수 채점: `python src/retrodict.py` → data/retrodiction.json
- 의존성: `pip install -r requirements.txt`
- 웹 JSON 갱신: 위 산출물(data/*.json)을 wc2026-web/data/ 로 복사

## 배경 문서
docs/WC2026_예측시스템_문서.md — 알고리즘 해설, 파라미터 근거, 검증 결과,
시장 비교, 한계. 설계 질문이 생기면 이 문서를 먼저 읽을 것.
