import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from predict_match import PremierLeaguePredictor
from prepare_data import (
    add_elo_ratings, add_rest_days, build_team_history,
    calculate_current_elo_ratings, get_base_feature_columns,
)
import test_pipeline


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.predictor = PremierLeaguePredictor.__new__(PremierLeaguePredictor)
        p = self.predictor
        p.matches = add_elo_ratings(add_rest_days(test_pipeline.PipelineTests().make_matches()))
        p.team_history = build_team_history(p.matches)
        p.season = 'Season2025-2026'
        p.teams = ['Alpha', 'Beta']
        p.elos = calculate_current_elo_ratings(p.matches)
        p.features = get_base_feature_columns()
        p.trained_through = p.matches.Date.max()
        p._form_cache = {}
        p.temperature = 1.0
        p.model = Mock()
        p.model.predict_proba.return_value = np.array([[0.5, 0.3, 0.2]])
        p.home_goal_model = Mock()
        p.home_goal_model.predict.return_value = np.array([1.5])
        p.away_goal_model = Mock()
        p.away_goal_model.predict.return_value = np.array([1.0])
        self.future = p.matches.Date.max() + pd.Timedelta(days=7)

    def test_rejects_same_team(self):
        with self.assertRaisesRegex(ValueError, 'different'):
            self.predictor.predict('Alpha', 'Alpha', self.future)

    def test_rejects_dates_at_or_before_data_cutoff(self):
        for date in ['2025-01-01', self.predictor.matches.Date.max()]:
            with self.subTest(date=date), self.assertRaisesRegex(ValueError, 'after'):
                self.predictor.predict('Alpha', 'Beta', date)

    def test_rejects_dates_before_training_cutoff(self):
        self.predictor.trained_through = self.future + pd.Timedelta(days=2)
        with self.assertRaisesRegex(ValueError, 'after'):
            self.predictor.predict('Alpha', 'Beta', self.future)

    def test_rejects_invalid_dates(self):
        for date in ['not-a-date', pd.NaT]:
            with self.subTest(date=date), self.assertRaises(ValueError):
                self.predictor.predict('Alpha', 'Beta', date)

    def test_rejects_invalid_rest_overrides(self):
        for rest in [-1, np.nan, np.inf]:
            for side in ['home_rest_days', 'away_rest_days']:
                with self.subTest(rest=rest, side=side), self.assertRaises(ValueError):
                    self.predictor.predict('Alpha', 'Beta', self.future, **{side: rest})

    def test_rest_overrides_are_capped_like_training(self):
        self.predictor.predict('Alpha', 'Beta', self.future,
                               home_rest_days=20, away_rest_days=3)
        row = self.predictor.model.predict_proba.call_args.args[0].iloc[0]
        self.assertEqual(row.HomeRestDays, 14)
        self.assertEqual(row.RestDaysDiff, 11)

    def test_missing_recent_statistics_are_not_silently_averaged(self):
        history = self.predictor.team_history
        history.loc[history.Team.eq('Alpha') & history.Date.eq(history.Date.max()), 'ShotsFor'] = np.nan
        with self.assertRaisesRegex(ValueError, 'missing|finite'):
            self.predictor.predict('Alpha', 'Beta', self.future)

    def test_aliases_ignore_case_and_whitespace(self):
        self.predictor.teams = ['Man United', 'Tottenham']
        self.assertEqual(self.predictor.normalise_team('  man utd  '), 'Man United')
        self.assertEqual(self.predictor.normalise_team('SPURS'), 'Tottenham')

    def test_score_matrix_preserves_outcome_probabilities(self):
        result = self.predictor.predict('Alpha', 'Beta', self.future)
        matrix = result['score_matrix']
        self.assertTrue(np.isfinite(matrix).all())
        self.assertTrue((matrix >= 0).all())
        self.assertAlmostEqual(matrix.sum(), 1)
        self.assertAlmostEqual(np.tril(matrix, -1).sum(), result['home_prob'])
        self.assertAlmostEqual(np.trace(matrix), result['draw_prob'])
        self.assertAlmostEqual(np.triu(matrix, 1).sum(), result['away_prob'])


if __name__ == '__main__':
    unittest.main()
