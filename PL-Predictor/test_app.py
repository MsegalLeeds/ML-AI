from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import predict_match


class AppTests(unittest.TestCase):
    def test_missing_model_shows_setup_instructions(self):
        with patch.object(predict_match, 'MODEL_FILE', Path('/nonexistent/model.joblib')):
            app = AppTest.from_file(str(Path(__file__).with_name('app.py'))).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.error)
        self.assertIn('train_model.py', app.error[0].value)


if __name__ == '__main__':
    unittest.main()
