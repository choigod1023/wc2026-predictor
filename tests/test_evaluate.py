import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src import evaluate


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'data').mkdir()
        self.patcher = patch.object(evaluate, 'ROOT', self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        self.pred = pd.DataFrame([dict(date='2026-06-11', home='A', away='B',
                                       P_home=.5, P_draw=.25, P_away=.25)])
        self.res = pd.DataFrame([dict(date='2026-06-11', home_team='A', away_team='B',
                                      home_score=1, away_score=0, tournament='FIFA World Cup')])

    def write(self, name, frame):
        frame.to_csv(self.root / 'data' / name, index=False)

    def run_evaluation(self):
        self.write('group_stage_predictions.csv', self.pred)
        self.write('results.csv', self.res)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            evaluate.main()
        return output.getvalue()

    def test_no_matching_results_stops_without_nan(self):
        self.res.loc[0, 'home_team'] = 'unmatched'
        with self.assertRaisesRegex(SystemExit, '일치하는 경기 결과가 없습니다'):
            self.run_evaluation()

    def test_invalid_odds_are_excluded(self):
        self.write('closing_odds.csv', pd.DataFrame([dict(date='2026-06-11', home='A', away='B',
                                                          odds_H=0, odds_D=3, odds_A=4)]))
        self.assertIn('유효한 배당', self.run_evaluation())

    def test_identical_probabilities_are_a_tie(self):
        self.write('closing_odds.csv', pd.DataFrame([dict(date='2026-06-11', home='A', away='B',
                                                          odds_H=2, odds_D=4, odds_A=4)]))
        output = self.run_evaluation()
        self.assertIn('두 점수가 같습니다', output)
        self.assertIn('관측 시각', output)

    def test_duplicate_odds_do_not_duplicate_scores(self):
        row = dict(date='2026-06-11', home='A', away='B', odds_H=2, odds_D=4, odds_A=4)
        self.write('closing_odds.csv', pd.DataFrame([row, row]))
        with self.assertRaises(pd.errors.MergeError):
            self.run_evaluation()


if __name__ == '__main__':
    unittest.main()
