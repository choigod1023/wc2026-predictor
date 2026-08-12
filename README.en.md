# WC2026 Predictor

[한국어](README.md) · [日本語](README.ja.md) · **English**

A project that pits "my model vs. the betting market" on accuracy, using the 72 group-stage matches of the 2026 World Cup as the test bed.
Not intended for actual betting (offshore betting is illegal from Korea). Background: `docs/`; working principles: `CLAUDE.md`.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)

## ✨ Features
- **Elo rating engine** — computes per-nation Elo from martj42/international_results international match data (with home advantage).
- **Probability transform + walk-forward validation** — converts the Elo difference into win/draw/loss probabilities via multinomial logistic regression, validated with walk-forward Brier scores and no lookahead leakage.
- **Multi-model leaderboard** — registers Elo-logistic (baseline), +|diff|, Davidson, and naive baselines under one interface for a fair Brier comparison (`src/models.py`, `src/compare_models.py`).
- **72-match predictions + Monte Carlo** — group-stage predictions plus tournament simulation yield per-round advancement and championship probabilities.
- **Scoreline model** — Poisson / Dixon-Coles score prediction, backing under/over and handicap calls with explicit reasoning.
- **Model vs. market scoring** — compares the frozen pre-tournament predictions against closing odds (de-vigged and normalized) on the same matches via Brier (`src/evaluate.py`).
- **Automated refresh pipeline** — GitHub Actions updates and commits results, Elo, predictions, and closing odds every 6 hours.

## Quick start
    pip install -r requirements.txt
    python run_pipeline.py --update   # pull latest results, then recompute everything
    python src/evaluate.py            # score model vs. market once results accumulate

## Layout
    CLAUDE.md                  Claude Code context (principles, state, next tasks)
    run_pipeline.py            full pipeline
    src/elo.py                 Elo rating engine
    src/prob_model.py          probability transform + walk-forward validation
    src/predict.py             72-match predictions + championship simulation
    src/sim.py                 tournament Monte Carlo core (incl. per-round advancement probabilities)
    src/models.py              probability-model registry (Elo-logistic, +|diff|, Davidson, baselines)
    src/compare_models.py      multi-model walk-forward Brier leaderboard + per-model championship sim
    src/score_model.py         scoreline prediction (Poisson, Dixon-Coles) → under/over, handicap + reasoning
    src/capture_odds.py        closing odds from the named API → idempotent load into closing_odds.csv
    src/export_web.py          pipeline outputs → JSON export for the web app (wc2026-web/data)
    src/evaluate.py            post-tournament model vs. market Brier verdict

## Automated refresh (scheduled on a server)
The model runs **on a schedule**, not per request (its input — match results — only changes when a match ends, and no GPU is needed).
- This repo's `.github/workflows/refresh-data.yml`: runs `run_pipeline.py --update` every 6 hours
  (refresh results, Elo, predictions + accumulate closing odds) and auto-commits the diff.
- The web repo's `refresh-predictions.yml`: every 6 hours it clones this repo, runs the pipeline, and uses
  `export_web.py` to commit prediction JSON to the web app → Vercel redeploys automatically.
- The frozen scoring snapshot `group_stage_predictions.csv` is guarded against after-the-fact edits (regenerate with `FORCE_PREDICTIONS=1`).
- Manual run: the "Run workflow" button on the GitHub Actions tab.
    data/closing_odds.csv      ★ record each match's odds just before kickoff here (decimal odds)
    docs/MATH.md               ★ full specification of every formula (Elo, probability transform, Brier, Monte Carlo)
    docs/                      algorithm explanation documents

## The math
Every formula is written up in [docs/MATH.md](docs/MATH.md). The two core lines:

    expected score   E = 1 / (1 + 10^(-(R_home + H - R_away)/400))
    rating update    R_new = R_old + K · G · (S − E)

## Web platform
A public dashboard for the predictions: **[wc2026-web](https://github.com/choigod1023/wc2026-web)**
(championship probabilities, all 72 match predictions, model-vs-market comparison, and the math explained)

---

## 👤 Contribution & development environment

| Item | Detail |
|---|---|
| **Contribution share** | **100%** (solo development) |
| **Commits** | 24 / 24 (mine / all human commits) |
| **Contributors** | 1 |
| **AI coding tool** | Claude Code |
| **Automated commits** | 231 (GitHub Actions collection/refresh that I configured — excluded from the count) |

<sub>Counting basis: commits reachable from **every branch** on origin (merge commits and empty commits excluded), counted by commit author email with one person’s multiple addresses merged; bot and automation commits are excluded.</sub>
