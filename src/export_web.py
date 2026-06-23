"""
export_web.py — 파이프라인 산출물을 웹(wc2026-web)이 쓰는 JSON 형태로 export
==============================================================================
수동 복사를 대체한다. CI(깃헙 액션)와 로컬에서 동일하게 쓴다.
  python src/export_web.py --out /path/to/wc2026-web/data

갱신 대상(결과에 따라 바뀌는 전망치):
  championship.json, elo.json, stage_probs.json, score_predictions.json,
  score_leaderboard.json, model_leaderboard.json, champion_by_model.json

★ 갱신하지 않는 것(개막 전 고정본 — 채점 무결성):
  matches.json(=group_stage_predictions, 채점 기준), modelVsMarket.json(시장 스냅샷)
"""
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys
import csv
import json
import shutil

DEFAULT_OUT = _os.path.join('..', 'wc2026-web', 'data')
# 그대로 복사할 JSON (predictor/data → web/data)
COPY = [
    'stage_probs.json', 'score_predictions.json', 'score_leaderboard.json',
    'model_leaderboard.json', 'champion_by_model.json', 'champion_history.json',
    'prediction_changes.json', 'live_predictions.json', 'live_score.json',
    'odds_history.json',
]


def out_dir():
    if '--out' in sys.argv:
        return sys.argv[sys.argv.index('--out') + 1]
    return DEFAULT_OUT


def main():
    out = out_dir()
    if not _os.path.isdir(out):
        sys.exit(f'출력 폴더 없음: {out} (--out 으로 wc2026-web/data 경로 지정)')

    # championship.json (championship_probs.csv → [{team,champion,final}])
    with open('data/championship_probs.csv', encoding='utf-8') as f:
        champ = [{'team': r['team'],
                  'champion': float(r['P_champion']),
                  'final': float(r['P_final'])}
                 for r in csv.DictReader(f)]
    json.dump(champ, open(_os.path.join(out, 'championship.json'), 'w'),
              ensure_ascii=False, indent=0)

    # elo.json (elo_final.csv 상위 40 → [{team,elo}])
    with open('data/elo_final.csv', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    elo = [{'team': list(r.values())[0], 'elo': round(float(r['elo']), 1)}
           for r in rows][:40]
    json.dump(elo, open(_os.path.join(out, 'elo.json'), 'w'),
              ensure_ascii=False, indent=0)

    for name in COPY:
        src = _os.path.join('data', name)
        if _os.path.exists(src):
            shutil.copyfile(src, _os.path.join(out, name))

    # closing_odds.csv → closing_odds.json (기록된 행만; 종료 경기 배당 적중 표시용)
    with open('data/closing_odds.csv', encoding='utf-8') as f:
        co = []
        for r in csv.DictReader(f):
            if r.get('odds_H') and r.get('odds_D') and r.get('odds_A'):
                co.append({'date': r['date'], 'home': r['home'], 'away': r['away'],
                           'oH': float(r['odds_H']), 'oD': float(r['odds_D']),
                           'oA': float(r['odds_A'])})
    json.dump(co, open(_os.path.join(out, 'closing_odds.json'), 'w'),
              ensure_ascii=False, indent=0)

    print(f'export 완료 → {out}')
    print('  championship.json, elo.json, ' + ', '.join(COPY))
    print('  (고정본 matches.json·modelVsMarket.json 은 의도적으로 미갱신)')


if __name__ == '__main__':
    main()
