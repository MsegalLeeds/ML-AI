# ML-AI

Hands-on machine-learning projects focused on building, evaluating and improving real predictive systems rather than only experimenting in notebooks.

## Projects

| Project | Status | What it does |
| --- | --- | --- |
| [Premier League predictor](PL-Predictor/README.md) | Working | Predicts match outcomes and expected goals, calibrates probabilities, and simulates the remaining league season in a Streamlit app |
| [Stock predictor](Stock-Predictor/README.md) | Working baseline | Trains a PyTorch LSTM on AAPL price/volume history and technical indicators, then compares next-day close predictions with a naive baseline |
| Drone ML project | Planned | Requirements and evaluation target still to be defined |

## Repository layout

```text
ML-AI/
├── PL-Predictor/
│   ├── app.py
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── predict_match.py
│   ├── league_simulator.py
│   ├── data/
│   └── fixtures/
├── Stock-Predictor/
│   ├── stock_predictor_basic.py
│   └── requirements.txt
├── ASSESSMENT.md
└── README.md
```

## Quick start

Create one virtual environment from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

On Windows use:

```text
.venv\Scripts\activate
```

### Premier League predictor

The Premier League project is tested with Python 3.12 and 3.13.

```bash
python -m pip install -r PL-Predictor/requirements.txt
python -m unittest discover -s PL-Predictor -v
python PL-Predictor/prepare_data.py
python PL-Predictor/train_model.py
python PL-Predictor/league_simulator.py
python -m streamlit run PL-Predictor/app.py
```

For an interactive terminal prediction:

```bash
python PL-Predictor/predict_match.py
```

See the [Premier League predictor guide](PL-Predictor/README.md) for data format, model assumptions, evaluation and update workflow.

### Stock predictor

The stock project downloads market data from Yahoo Finance when it runs.

```bash
python -m pip install -r Stock-Predictor/requirements.txt
python Stock-Predictor/stock_predictor_basic.py
```

The current baseline uses AAPL by default and trains an LSTM on 30-day sequences containing OHLCV data plus technical indicators. It uses chronological train/validation/test splits, fits scalers only on training data, applies early stopping, and compares the neural network against the simple baseline of predicting that tomorrow's close equals today's close.

See the [Stock Predictor guide](Stock-Predictor/README.md) for the feature set, architecture, metrics, limitations and next experiments.

## ML principles used in this repository

- Keep train, validation and test periods separated chronologically for time-series problems.
- Fit preprocessing only on training data to avoid leakage.
- Compare complex models against simple baselines.
- Prefer probability-quality metrics where probabilities are the actual output.
- Treat test data as a final evaluation set rather than a hyperparameter-tuning set.
- Document assumptions and known limitations alongside headline metrics.

## Notes

The Premier League [assessment](ASSESSMENT.md) records an earlier reliability review and the fixes made at that point in the project.

The stock predictor is an educational ML project and is **not financial advice**. A model that predicts historical test data well does not imply profitable live trading performance.
