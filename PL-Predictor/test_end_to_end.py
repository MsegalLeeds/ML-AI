import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import league_simulator
import predict_match
import prepare_data
import train_model


class EndToEndTests(unittest.TestCase):
    def test_training_prediction_simulation_and_app(self):
        # Exercise the checked-in data without replacing a user's saved model.
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            data = root / 'model_data.csv'
            model = root / 'model.joblib'
            report = root / 'training_metrics.json'
            with patch.object(prepare_data, 'OUTPUT_FILE', data):
                prepare_data.main()
            with (patch.object(train_model, 'DATA_FILE', data),
                  patch.object(train_model, 'MODEL_FILE', model),
                  patch.object(train_model, 'METRICS_FILE', report)):
                train_model.main()
            metrics = json.loads(report.read_text())
            self.assertGreater(metrics['test']['evaluation_rows'], 0)
            for result in ['raw', 'calibrated', 'historical_frequency_baseline']:
                self.assertTrue(np.isfinite(list(metrics['test'][result].values())).all())
            with patch.object(predict_match, 'MODEL_FILE', model):
                predictor = predict_match.PremierLeaguePredictor()
                date = predictor.matches.Date.max() + pd.Timedelta(days=1)
                prediction = predictor.predict('Arsenal', 'Chelsea', date)
                self.assertAlmostEqual(prediction['score_matrix'].sum(), 1)
                with (patch.object(league_simulator, 'BASE_DIR', root),
                      patch.object(league_simulator, 'SIMULATIONS', 100)):
                    league_simulator.simulate()
                table = pd.read_csv(root / 'league_simulation.csv')
                self.assertEqual(len(table), 20)
                self.assertAlmostEqual(table.Title.sum(), 1)
                self.assertAlmostEqual(table.Top4.sum(), 4)
                self.assertAlmostEqual(table.Relegation.sum(), 3)
                self.assertTrue(table.AvgFinish.between(1, 20).all())
                # Run a copy next to the temporary simulation output.
                app_path = root / 'app.py'
                app_path.write_text(Path(__file__).with_name('app.py').read_text())
                app = AppTest.from_file(str(app_path)).run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                app.date_input[0].set_value(date.date())
                app.button[0].click().run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.error), 0)
                self.assertEqual(len(app.metric), 5)
                self.assertEqual(len(app.dataframe), 1)
                os.utime(root / 'league_simulation.csv', (1, 1))
                app.run(timeout=30)
                self.assertEqual(len(app.dataframe), 0)
                self.assertIn('older', app.warning[0].value)
                os.utime(root / 'league_simulation.csv', None)
                app.run(timeout=30)
                self.assertEqual(len(app.dataframe), 1)


if __name__ == '__main__':
    unittest.main()
