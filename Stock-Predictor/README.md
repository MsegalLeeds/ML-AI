# Stock Predictor

A PyTorch LSTM baseline for next-day stock-price prediction using AAPL market history and technical indicators downloaded from Yahoo Finance.

This project is deliberately structured as an ML experiment rather than a trading claim: it uses chronological splits, train-only scaling, a validation set with early stopping, and a naive benchmark that predicts tomorrow's close will equal today's close.

## What the current model does

`stock_predictor_basic.py`:

1. Downloads adjusted AAPL OHLCV data from Yahoo Finance starting at `2020-01-01`.
2. Builds technical features from information available at or before each date.
3. Splits observations chronologically into 70% train, 15% validation and 15% test periods.
4. Fits feature and target scalers on the training period only.
5. Builds 30-trading-day sequences.
6. Trains a two-layer LSTM with mini-batches, dropout, gradient clipping and early stopping.
7. Predicts the next trading day's closing price.
8. Compares the LSTM with a naive baseline: `tomorrow's close = today's close`.
9. Reports price-error and direction metrics and plots predictions/errors and training history.

## Feature set

The current model uses 19 features per trading day:

### Raw market data

- Open
- High
- Low
- Close
- Volume

### Returns and trend

- 1-day percentage return
- Log return
- SMA 5
- SMA 10
- SMA 20
- EMA 10
- EMA 20

### Momentum and volatility

- 10-day rolling volatility
- RSI 14
- MACD
- MACD signal
- ATR 14

### Volume behaviour

- Volume percentage change
- Volume / 20-day average volume

With the default `SEQUENCE_LENGTH = 30`, each prediction is based on a `30 x 19` sequence.

## Model architecture

The current network is approximately:

```text
30 trading days x 19 features
            ↓
2-layer LSTM
hidden size = 64
            ↓
fully connected layer
            ↓
next-day close prediction
```

Training uses:

- Adam optimizer
- mini-batches of 64
- dropout of 0.2
- gradient clipping
- validation-based early stopping
- maximum 200 epochs

## Setup

From the repository root, activate your virtual environment and install the stock-project dependencies:

```bash
source .venv/bin/activate
python -m pip install -r Stock-Predictor/requirements.txt
```

On Windows:

```text
.venv\Scripts\activate
```

Run the model with:

```bash
python Stock-Predictor/stock_predictor_basic.py
```

The script automatically uses:

- Apple Metal Performance Shaders (`mps`) on supported Apple Silicon Macs
- CUDA when available
- CPU otherwise

## Configuration

The main experiment settings are near the top of `stock_predictor_basic.py`:

```python
TICKER = "AAPL"
START_DATE = "2020-01-01"
SEQUENCE_LENGTH = 30
BATCH_SIZE = 64
MAX_EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 20
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
```

Change `TICKER` to experiment with another Yahoo Finance symbol, but treat results for each symbol as a new experiment rather than assuming the same model settings generalise.

## Evaluation

The useful question is not whether the predicted line visually follows the real price. The useful question is whether the model improves on a simple benchmark on unseen future data.

The script compares:

```text
LSTM prediction
vs
Naive prediction: tomorrow = today
```

Metrics include:

- RMSE in dollars
- MAE in dollars
- directional accuracy
- training and validation loss

If the LSTM has a higher RMSE/MAE than the naive baseline, the neural network has not demonstrated useful predictive value even if the plotted curves look similar.

## Leakage controls

The current implementation avoids two common time-series mistakes:

- data is split chronologically rather than randomly;
- `StandardScaler` is fitted only on the training period.

Future changes should preserve these rules. Indicators and external data must also be aligned so that a prediction never uses information that would not have been known at prediction time.

## Current limitations

This is still a baseline experiment. Important limitations include:

- It predicts the next closing-price level, which can look artificially strong because tomorrow's price is highly correlated with today's price.
- It currently uses only AAPL-derived market features; it has no SPY, QQQ, sector, volatility-index or interest-rate context.
- It has one fixed train/validation/test split rather than walk-forward backtesting across multiple market regimes.
- It does not include transaction costs, slippage or a trading strategy.
- Directional accuracy alone is not enough to establish economic value.
- Yahoo Finance data is convenient for experimentation but should not be treated as institutional-quality market data.

## Recommended next experiments

1. Predict next-day or multi-day **returns** instead of the raw price level.
2. Add market context such as SPY, QQQ, XLK, VIX and Treasury-yield changes.
3. Compare 1-day, 3-day and 5-day prediction horizons.
4. Compare the LSTM against linear regression, logistic regression and tree-based models.
5. Add walk-forward / expanding-window backtests.
6. If a robust signal remains, evaluate a simple strategy with transaction costs and drawdown metrics.

## Disclaimer

This repository is for learning and experimentation. It is **not financial advice**, and historical predictive performance does not guarantee future returns.
