import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.preprocessing import StandardScaler


# ==========================================================
# CONFIG
# ==========================================================

TICKER = "AAPL"
START_DATE = "2020-01-01"

SEQUENCE_LENGTH = 30
BATCH_SIZE = 64
MAX_EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 20

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

torch.manual_seed(42)
np.random.seed(42)


# ==========================================================
# DEVICE
# ==========================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")

elif torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")

print("Using:", device)


# ==========================================================
# DOWNLOAD DATA
# ==========================================================

df = yf.download(
    TICKER,
    start=START_DATE,
    auto_adjust=True,
)

# yfinance can sometimes return MultiIndex columns.
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[
    [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]
].copy()


# ==========================================================
# TECHNICAL INDICATORS
# ==========================================================

# Daily returns
df["Return1"] = (
    df["Close"].pct_change()
)

df["LogReturn"] = (
    np.log(df["Close"]).diff()
)


# Moving averages
df["SMA5"] = (
    df["Close"]
    .rolling(5)
    .mean()
)

df["SMA10"] = (
    df["Close"]
    .rolling(10)
    .mean()
)

df["SMA20"] = (
    df["Close"]
    .rolling(20)
    .mean()
)


# Exponential moving averages
df["EMA10"] = (
    df["Close"]
    .ewm(
        span=10,
        adjust=False,
    )
    .mean()
)

df["EMA20"] = (
    df["Close"]
    .ewm(
        span=20,
        adjust=False,
    )
    .mean()
)


# Volatility
df["Volatility10"] = (
    df["Return1"]
    .rolling(10)
    .std()
)


# ==========================================================
# RSI
# ==========================================================

delta = df["Close"].diff()

gain = delta.clip(
    lower=0
)

loss = -delta.clip(
    upper=0
)

average_gain = (
    gain
    .rolling(14)
    .mean()
)

average_loss = (
    loss
    .rolling(14)
    .mean()
)

rs = (
    average_gain
    / average_loss.replace(
        0,
        np.nan,
    )
)

df["RSI14"] = (
    100
    - (
        100
        / (1 + rs)
    )
)


# ==========================================================
# MACD
# ==========================================================

ema12 = (
    df["Close"]
    .ewm(
        span=12,
        adjust=False,
    )
    .mean()
)

ema26 = (
    df["Close"]
    .ewm(
        span=26,
        adjust=False,
    )
    .mean()
)

df["MACD"] = (
    ema12 - ema26
)

df["MACDSignal"] = (
    df["MACD"]
    .ewm(
        span=9,
        adjust=False,
    )
    .mean()
)


# ==========================================================
# ATR
# ==========================================================

previous_close = (
    df["Close"].shift(1)
)

true_range = pd.concat(
    [
        df["High"] - df["Low"],

        (
            df["High"]
            - previous_close
        ).abs(),

        (
            df["Low"]
            - previous_close
        ).abs(),
    ],
    axis=1,
).max(axis=1)

df["ATR14"] = (
    true_range
    .rolling(14)
    .mean()
)


# ==========================================================
# VOLUME FEATURES
# ==========================================================

df["VolumeChange"] = (
    df["Volume"]
    .pct_change()
)

volume_average = (
    df["Volume"]
    .rolling(20)
    .mean()
)

df["VolumeRatio20"] = (
    df["Volume"]
    / volume_average
)


# ==========================================================
# TARGET
# ==========================================================

# Features from today / previous days
# predict NEXT trading day's closing price.

df["TargetClose"] = (
    df["Close"].shift(-1)
)


FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",

    "Return1",
    "LogReturn",

    "SMA5",
    "SMA10",
    "SMA20",

    "EMA10",
    "EMA20",

    "Volatility10",

    "RSI14",

    "MACD",
    "MACDSignal",

    "ATR14",

    "VolumeChange",
    "VolumeRatio20",
]


df = df.replace(
    [np.inf, -np.inf],
    np.nan,
)

df = df.dropna().copy()

print()
print("Rows:", len(df))
print("Features:", len(FEATURES))


# ==========================================================
# CHRONOLOGICAL SPLIT
# ==========================================================

n = len(df)

train_end = int(
    n * TRAIN_RATIO
)

validation_end = int(
    n
    * (
        TRAIN_RATIO
        + VALIDATION_RATIO
    )
)

print()
print(
    "Train:",
    df.index[0].date(),
    "→",
    df.index[
        train_end - 1
    ].date(),
)

print(
    "Validation:",
    df.index[
        train_end
    ].date(),
    "→",
    df.index[
        validation_end - 1
    ].date(),
)

print(
    "Test:",
    df.index[
        validation_end
    ].date(),
    "→",
    df.index[-1].date(),
)


# ==========================================================
# SCALING
# IMPORTANT: FIT ONLY ON TRAINING DATA
# ==========================================================

feature_scaler = StandardScaler()
target_scaler = StandardScaler()

feature_scaler.fit(
    df.iloc[
        :train_end
    ][FEATURES]
)

target_scaler.fit(
    df.iloc[
        :train_end
    ][["TargetClose"]]
)


X_scaled = feature_scaler.transform(
    df[FEATURES]
)

y_scaled = (
    target_scaler
    .transform(
        df[["TargetClose"]]
    )
    .ravel()
)


# ==========================================================
# SEQUENCES
# ==========================================================

X = []
y = []
target_rows = []

for end_index in range(
    SEQUENCE_LENGTH - 1,
    len(df),
):

    start_index = (
        end_index
        - SEQUENCE_LENGTH
        + 1
    )

    X.append(
        X_scaled[
            start_index:
            end_index + 1
        ]
    )

    y.append(
        y_scaled[end_index]
    )

    target_rows.append(
        end_index
    )


X = np.asarray(
    X,
    dtype=np.float32,
)

y = np.asarray(
    y,
    dtype=np.float32,
).reshape(-1, 1)

target_rows = np.asarray(
    target_rows
)


# ==========================================================
# SPLIT SEQUENCES
# ==========================================================

train_mask = (
    target_rows
    < train_end
)

validation_mask = (
    (target_rows >= train_end)
    &
    (
        target_rows
        < validation_end
    )
)

test_mask = (
    target_rows
    >= validation_end
)


X_train = X[
    train_mask
]

y_train = y[
    train_mask
]

X_validation = X[
    validation_mask
]

y_validation = y[
    validation_mask
]

X_test = X[
    test_mask
]

y_test = y[
    test_mask
]


print()
print(
    "X_train:",
    X_train.shape
)

print(
    "X_validation:",
    X_validation.shape
)

print(
    "X_test:",
    X_test.shape
)


# ==========================================================
# DATALOADERS
# ==========================================================

train_dataset = TensorDataset(
    torch.from_numpy(
        X_train
    ),
    torch.from_numpy(
        y_train
    ),
)

validation_dataset = TensorDataset(
    torch.from_numpy(
        X_validation
    ),
    torch.from_numpy(
        y_validation
    ),
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ==========================================================
# MODEL
# ==========================================================

class PredictionModel(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_layers=2,
        output_dim=1,
        dropout=0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.fc = nn.Sequential(
            nn.Linear(
                hidden_dim,
                32,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                32,
                output_dim,
            ),
        )

    def forward(
        self,
        x,
    ):

        output, _ = self.lstm(x)

        last_output = (
            output[:, -1, :]
        )

        return self.fc(
            last_output
        )


model = PredictionModel(
    input_dim=len(FEATURES),
).to(device)


criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ==========================================================
# TRAINING
# ==========================================================

best_validation_loss = (
    float("inf")
)

best_state = None

epochs_without_improvement = 0

train_history = []
validation_history = []


for epoch in range(
    MAX_EPOCHS
):

    # --------------------------
    # Training
    # --------------------------

    model.train()

    train_loss = 0.0

    for X_batch, y_batch in (
        train_loader
    ):

        X_batch = X_batch.to(
            device
        )

        y_batch = y_batch.to(
            device
        )

        optimizer.zero_grad()

        predictions = model(
            X_batch
        )

        loss = criterion(
            predictions,
            y_batch,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        train_loss += (
            loss.item()
            * X_batch.size(0)
        )

    train_loss /= len(
        train_dataset
    )


    # --------------------------
    # Validation
    # --------------------------

    model.eval()

    validation_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in (
            validation_loader
        ):

            X_batch = X_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )

            predictions = model(
                X_batch
            )

            loss = criterion(
                predictions,
                y_batch,
            )

            validation_loss += (
                loss.item()
                * X_batch.size(0)
            )

    validation_loss /= len(
        validation_dataset
    )


    train_history.append(
        train_loss
    )

    validation_history.append(
        validation_loss
    )


    if epoch % 10 == 0:

        print(
            f"Epoch {epoch:3d} | "
            f"Train {train_loss:.5f} | "
            f"Validation "
            f"{validation_loss:.5f}"
        )


    # --------------------------
    # Early stopping
    # --------------------------

    if (
        validation_loss
        < best_validation_loss
        - 1e-5
    ):

        best_validation_loss = (
            validation_loss
        )

        best_state = {
            key:
                value
                .detach()
                .cpu()
                .clone()

            for key, value
            in model.state_dict().items()
        }

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1


    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            f"\nEarly stopping at "
            f"epoch {epoch}"
        )

        break


# Restore best model
model.load_state_dict(
    best_state
)

model.to(device)


# ==========================================================
# TEST PREDICTIONS
# ==========================================================

model.eval()

X_test_tensor = (
    torch.from_numpy(
        X_test
    )
    .to(device)
)

with torch.no_grad():

    test_predictions_scaled = (
        model(
            X_test_tensor
        )
        .cpu()
        .numpy()
    )


actual_prices = (
    target_scaler
    .inverse_transform(
        y_test
    )
    .ravel()
)

predicted_prices = (
    target_scaler
    .inverse_transform(
        test_predictions_scaled
    )
    .ravel()
)


test_rows = (
    target_rows[
        test_mask
    ]
)

test_dates = (
    df.index[
        test_rows
    ]
)


# ==========================================================
# NAIVE BASELINE
# ==========================================================

# Predict tomorrow's close as today's close.

current_prices = (
    df["Close"]
    .iloc[
        test_rows
    ]
    .to_numpy()
)


# ==========================================================
# METRICS
# ==========================================================

model_rmse = (
    root_mean_squared_error(
        actual_prices,
        predicted_prices,
    )
)

baseline_rmse = (
    root_mean_squared_error(
        actual_prices,
        current_prices,
    )
)

model_mae = (
    mean_absolute_error(
        actual_prices,
        predicted_prices,
    )
)

baseline_mae = (
    mean_absolute_error(
        actual_prices,
        current_prices,
    )
)


actual_direction = (
    actual_prices
    > current_prices
)

predicted_direction = (
    predicted_prices
    > current_prices
)

direction_accuracy = np.mean(
    actual_direction
    == predicted_direction
)


print()
print("=" * 50)
print("TEST RESULTS")
print("=" * 50)

print(
    f"LSTM RMSE:      "
    f"${model_rmse:.2f}"
)

print(
    f"Naive RMSE:     "
    f"${baseline_rmse:.2f}"
)

print()

print(
    f"LSTM MAE:       "
    f"${model_mae:.2f}"
)

print(
    f"Naive MAE:      "
    f"${baseline_mae:.2f}"
)

print()

print(
    f"Direction accuracy: "
    f"{direction_accuracy:.1%}"
)


# ==========================================================
# PRICE PLOT
# ==========================================================

fig = plt.figure(
    figsize=(14, 10)
)

gs = fig.add_gridspec(
    3,
    1,
)


ax1 = fig.add_subplot(
    gs[:2, 0]
)

ax1.plot(
    test_dates,
    actual_prices,
    label="Actual",
)

ax1.plot(
    test_dates,
    predicted_prices,
    label="LSTM",
)

ax1.plot(
    test_dates,
    current_prices,
    linestyle="--",
    alpha=0.7,
    label="Naive baseline",
)

ax1.set_title(
    f"{TICKER} Next-Day Close Prediction"
)

ax1.set_ylabel(
    "Price ($)"
)

ax1.legend()


# ==========================================================
# ERROR PLOT
# ==========================================================

ax2 = fig.add_subplot(
    gs[2, 0]
)

model_error = np.abs(
    actual_prices
    - predicted_prices
)

baseline_error = np.abs(
    actual_prices
    - current_prices
)

ax2.plot(
    test_dates,
    model_error,
    label="LSTM error",
)

ax2.plot(
    test_dates,
    baseline_error,
    alpha=0.7,
    label="Naive error",
)

ax2.axhline(
    model_rmse,
    linestyle="--",
    label=(
        f"LSTM RMSE "
        f"${model_rmse:.2f}"
    ),
)

ax2.set_title(
    "Absolute Prediction Error"
)

ax2.set_xlabel(
    "Date"
)

ax2.set_ylabel(
    "Error ($)"
)

ax2.legend()

plt.tight_layout()

plt.show()


# ==========================================================
# TRAINING CURVES
# ==========================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    train_history,
    label="Training loss",
)

plt.plot(
    validation_history,
    label="Validation loss",
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "MSE"
)

plt.title(
    "Training History"
)

plt.legend()

plt.tight_layout()

plt.show()