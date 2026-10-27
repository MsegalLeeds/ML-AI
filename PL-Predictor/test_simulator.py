from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from league_simulator import load_remaining_fixtures, precompute_predictions


class RecordingPredictor:
    def __init__(self):
        self.calls = []
        self.matches = pd.DataFrame({
            'HomeTeam': ['Alpha'], 'AwayTeam': ['Beta'],
            'Date': pd.to_datetime(['2026-01-01']),
        })

    def predict(self, home, away, **kwargs):
        self.calls.append((home, away, kwargs))
        return {'home_team': home, 'away_team': away,
                'score_matrix': np.array([[0.2, 0.3], [0.4, 0.1]])}


class SimulatorTests(unittest.TestCase):
    def test_uses_fixture_dates_and_scheduled_rest_in_date_order(self):
        predictor = RecordingPredictor()
        fixtures = pd.DataFrame({
            'HomeTeam': ['Beta', 'Alpha'], 'AwayTeam': ['Alpha', 'Beta'],
            'Date': pd.to_datetime(['2026-01-11 12:30', '2026-01-08 20:00']),
        })
        predictions = precompute_predictions(predictor, fixtures)
        self.assertEqual(predictor.calls[0][0], 'Alpha')
        first = predictor.calls[0][2]
        second = predictor.calls[1][2]
        self.assertEqual(first['match_date'], pd.Timestamp('2026-01-08'))
        self.assertEqual(first['home_rest_days'], 7)
        self.assertEqual(first['away_rest_days'], 7)
        self.assertEqual(second['home_rest_days'], 3)
        self.assertEqual(second['away_rest_days'], 3)
        self.assertEqual(predictions[0]['cdf'][-1], 1)

    def load(self, rows):
        matches = pd.DataFrame({
            'HomeTeam': ['Alpha'], 'AwayTeam': ['Beta'],
            'Date': pd.to_datetime(['2026-01-01']), 'Season': ['S'],
        })
        predictor = SimpleNamespace(matches=matches, season='S', teams=['Alpha', 'Beta'],
                                    normalise_team=lambda name: name)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'fixtures.csv'
            pd.DataFrame(rows, columns=['Date', 'Home Team', 'Away Team']).to_csv(path, index=False)
            with patch('league_simulator.FIXTURE_FILE', path):
                return load_remaining_fixtures(predictor)

    def test_excludes_played_matches(self):
        remaining = self.load([('01/01/2026', 'Alpha', 'Beta'),
                               ('08/01/2026', 'Beta', 'Alpha')])
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining.iloc[0].HomeTeam, 'Beta')

    def test_rejects_duplicate_fixtures(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.load([('01/01/2026', 'Alpha', 'Beta'),
                       ('08/01/2026', 'Beta', 'Alpha'),
                       ('09/01/2026', 'Beta', 'Alpha')])

    def test_rejects_missing_fixtures(self):
        with self.assertRaisesRegex(ValueError, 'schedule'):
            self.load([('01/01/2026', 'Alpha', 'Beta')])

    def test_rejects_multiple_fixtures_for_one_team_on_one_date(self):
        with self.assertRaisesRegex(ValueError, 'same date'):
            self.load([('08/01/2026', 'Alpha', 'Beta'),
                       ('08/01/2026', 'Beta', 'Alpha')])

    def test_rejects_invalid_dates(self):
        with self.assertRaisesRegex(ValueError, 'date'):
            self.load([('01/01/2026', 'Alpha', 'Beta'),
                       ('bad-date', 'Beta', 'Alpha')])

    def test_rejects_unplayed_fixture_before_snapshot(self):
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            self.load([('01/01/2026', 'Alpha', 'Beta'),
                       ('31/12/2025', 'Beta', 'Alpha')])


if __name__ == '__main__':
    unittest.main()
