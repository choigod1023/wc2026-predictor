"""
run_pipeline.py — 전체 파이프라인 실행
  python run_pipeline.py            # 현재 데이터로 재계산
  python run_pipeline.py --update   # 최신 경기 결과 다운로드 후 재계산
대회 기간 중에는 매일 --update 로 실행하면 직전 경기까지 Elo에 반영된다.
"""
import os
import sys
import subprocess
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
DATA_URL = 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv'

scripts = ['src/elo.py', 'src/prob_model.py', 'src/score_model.py',
           'src/predict.py', 'src/compare_models.py']
n = len(scripts) + 1

if '--update' in sys.argv:
    print(f'[1/{n}] 데이터 갱신:', DATA_URL)
    urllib.request.urlretrieve(DATA_URL, 'data/results.csv')
else:
    print(f'[1/{n}] 데이터 갱신 생략 (--update 로 활성화)')

for i, script in enumerate(scripts, start=2):
    print(f'[{i}/{n}] {script}')
    r = subprocess.run([sys.executable, script])
    if r.returncode != 0:
        sys.exit(f'{script} 실패')

# --update 시 마감배당도 함께 스냅샷(비파괴: 빈 경기만). 네트워크 실패는 비치명.
if '--update' in sys.argv:
    print('[+] src/capture_odds.py (named 마감배당 적재)')
    subprocess.run([sys.executable, 'src/capture_odds.py'])

print('완료. 산출물: group_stage_predictions.csv, championship_probs.csv, '
      'model_leaderboard.json, champion_by_model.json, stage_probs.json, '
      'score_predictions.json, score_leaderboard.json, closing_odds.csv')
