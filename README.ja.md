# WC2026 Predictor

[한국어](README.md) · **日本語** · [English](README.en.md)

2026 ワールドカップのグループステージ 72 試合を検証の場として、「自分のモデル vs ベッティング市場」の精度で勝負するプロジェクト。
実際のベッティング用途ではありません（韓国からの海外ベッティングは違法です）。詳しい背景は docs/、作業の原則は CLAUDE.md を参照。

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)

## ✨ 主な機能
- **Elo レーティングエンジン** — martj42/international_results の A マッチ結果から国別 Elo を算出（ホームアドバンテージを反映）。
- **確率変換 + walk-forward 検証** — Elo 差を多項ロジスティックで勝/分/敗の確率に変換し、未来情報のリークなしに walk-forward Brier で検証。
- **マルチモデル・リーダーボード** — Elo-ロジスティック（基準）・+|diff|・Davidson・ベースラインを同一の規約で登録し、Brier で公平に比較（`src/models.py`, `src/compare_models.py`）。
- **72 試合の予測 + モンテカルロ** — グループステージ予測とトーナメントのシミュレーションで、ラウンド別の進出確率と優勝確率を算出。
- **スコアモデル** — ポアソン・Dixon-Coles ベースのスコア予測 → アンダー/オーバー・ハンディキャップの根拠を提示。
- **モデル vs 市場の採点** — 開幕前に固定した予測と締切オッズ（マージン除去で正規化）を同じ試合について Brier で比較（`src/evaluate.py`）。
- **自動更新パイプライン** — GitHub Actions により 6 時間ごとに結果・Elo・予測・締切オッズを更新・コミット。

## クイックスタート
    pip install -r requirements.txt
    python run_pipeline.py --update   # 最新結果を反映して全体を再計算
    python src/evaluate.py            # 結果が溜まったらモデル vs 市場を採点

## 構成
    CLAUDE.md                  Claude Code 用コンテキスト（原則・状態・次のタスク）
    run_pipeline.py            パイプライン全体
    src/elo.py                 Elo レーティングエンジン
    src/prob_model.py          確率変換 + walk-forward 検証
    src/predict.py             72 試合の予測 + 優勝シミュレーション
    src/sim.py                 トーナメント・モンテカルロのコア（ラウンド別進出確率を含む）
    src/models.py              確率モデルのレジストリ（Elo-ロジスティック・+|diff|・Davidson・ベースライン）
    src/compare_models.py      マルチモデル walk-forward Brier リーダーボード + モデル別優勝シミュレーション
    src/score_model.py         スコア予測（ポアソン・Dixon-Coles）→ アンダー/オーバー・ハンディキャップ + 根拠
    src/capture_odds.py        named API の締切オッズ → closing_odds.csv へ自動格納（冪等）
    src/export_web.py          パイプライン成果物 → Web（wc2026-web/data）へ JSON エクスポート
    src/evaluate.py            大会後のモデル vs 市場の Brier 判定

## 自動更新（サーバー上で定期実行）
モデルはリクエストごとではなく **スケジュール** で実行されます（入力である試合結果は試合終了時にしか変わらず、GPU も不要）。
- 本リポジトリの `.github/workflows/refresh-data.yml`: 6 時間ごとに `run_pipeline.py --update`
  （結果・Elo・予測の更新＋締切オッズの蓄積）→ 差分を自動コミット。
- Web リポジトリの `refresh-predictions.yml`: 6 時間ごとに本リポジトリを clone してパイプラインを実行し、
  `export_web.py` で予測 JSON を Web にコミット → Vercel が自動再デプロイ。
- 採点用の固定版 `group_stage_predictions.csv` はガードにより事後修正を防止（再生成は `FORCE_PREDICTIONS=1`）。
- 手動実行: GitHub Actions タブの "Run workflow" ボタン。
    data/closing_odds.csv      ★ 各試合のキックオフ直前のオッズをここに記録（デシマルオッズ）
    docs/MATH.md               ★ すべての数式仕様（Elo・確率変換・Brier・モンテカルロ）
    docs/                      アルゴリズム解説ドキュメント

## 数式
すべての計算式は [docs/MATH.md](docs/MATH.md) にまとめてあります。核心は次の 2 行です:

    期待勝点率   E = 1 / (1 + 10^(-(R_home + H - R_away)/400))
    レーティング更新   R_new = R_old + K · G · (S − E)

## Web プラットフォーム
予測結果を誰でも見られる Web ダッシュボード: **[wc2026-web](https://github.com/choigod1023/wc2026-web)**
（優勝確率・72 試合の予測・モデル vs 市場の比較・数式解説）

---

## 👤 コントリビューションと開発環境

| 項目 | 内容 |
|---|---|
| **貢献比率** | **100%**（単独開発） |
| **コミット** | 24 / 24（本人 / 全人力コミット） |
| **参加人数** | 1 名 |
| **AI コーディングツール** | Claude Code |
| **自動化コミット** | 231 件（本人が構成した GitHub Actions による自動収集・更新 — 集計対象外） |

<sub>集計基準（2026-08-12 時点のスナップショット）: origin の **すべてのブランチ** から到達可能なコミット（マージコミット・空コミットは除外）を対象とし、コミットの author メールアドレス基準で、同一人物の複数のメールアドレスは合算、ボット・自動化コミットは除外しています。</sub>
