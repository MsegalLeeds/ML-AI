# Premier League Predictor

## Files

- `prepare_data.py` - loads Football-Data season CSVs, creates leakage-safe rolling features, venue features and Elo.
- `train_model.py` - compares Rolling+Elo vs Rolling+Elo+Venue on validation data, selects the lower-log-loss model, tests it on 2025/26, and saves the production model.
- `benchmark.py` - compares both ML variants and Bet365 on exactly the same 2025/26 fixtures.
- `predict_match.py` - interactive prediction using the exact feature list saved with the trained model.
- `test_pipeline.py` - basic leakage/Elo/rolling-feature tests.
- `requirements.txt` - Python dependencies.

## Folder layout

```text
PL-Predictor/
├── data/
│   ├── Season2017-2018.csv
│   ├── ...
│   └── Season2026-2027.csv
├── prepare_data.py
├── train_model.py
├── benchmark.py
├── predict_match.py
├── test_pipeline.py
└── requirements.txt
```

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest test_pipeline.py
python prepare_data.py
python benchmark.py
python train_model.py
python predict_match.py
```

The model uses overall last-5 form, Elo, and optionally current-season home/away last-3 form. `train_model.py` chooses whether venue features actually improve validation log loss.
