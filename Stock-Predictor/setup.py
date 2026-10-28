import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using:", device)
ticker = 'AAPL'
df = yf.download(ticker, '2020-01-01')
df.Close.plot(figsize=(20,10))
scaler = StandardScaler()

df['Close'] = scaler.fit_transform(df['Close'])

sqe_length = 30
data = []

for i in range(len(df) - sqe_length):
    data.append(df.Close[i:i+sqe_length])

data = np.array(data)

train_size = int(len(data) * 0.8)

X_train = torch.from_numpy(
    data[:train_size, :-1, :]
).float().to(device)

y_train = torch.from_numpy(
    data[:train_size, -1, :]
).float().to(device)

X_test = torch.from_numpy(
    data[train_size:, :-1, :]
).float().to(device)

y_test = torch.from_numpy(
    data[train_size:, -1, :]
).float().to(device)

class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_dim,
            device=x.device,
        )

        c0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_dim,
            device=x.device,
        )
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

model = PredictionModel(
    input_dim=X_train.shape[2],
    hidden_dim=100,
    num_layers=2,
    output_dim=1,
).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


num_epochs = 200
for epoch in range(num_epochs):

    model.train()

    optimizer.zero_grad()

    y_train_pred = model(X_train)

    loss = criterion(
        y_train_pred,
        y_train
    )

    loss.backward()

    optimizer.step()

    if epoch % 10 == 0:
        print(
            epoch,
            loss.item()
        )

model.eval()

with torch.no_grad():
    y_test_pred = model(X_test)

test_loss = criterion(
    y_test_pred,
    y_test
)

print(
    "Test MSE:",
    test_loss.item()
)

y_test_pred =model(X_test)

y_train_pred = scaler.inverse_transform(y_train_pred.detach().cpu().numpy())
y_train = scaler.inverse_transform(y_train.detach().cpu().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().cpu().numpy())
y_test = scaler.inverse_transform(y_test.detach().cpu().numpy())

trian_rmse = root_mean_squared_error(y_train[:, 0], y_train_pred[:, 0])
test_rmse = root_mean_squared_error(y_test[:, 0], y_test_pred[:, 0])

fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(4, 1)
ax1 = fig.add_subplot(gs[:3, 0])
ax1.plot(df.iloc[-len(y_test):].index, y_test, color="blue", label="Actual")
ax1.plot(df.iloc[-len(y_test):].index, y_test, color="green", label="Predicted")
ax1.legend()
plt.title(f"{ticker}: Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")

ax2 = fig.add_subplot(gs[3, 0])
ax2.axhline(test_rmse, color="blue",linestyle="--", label="RMSE")
ax2.plot(df[-len(y_test):].index, abs(y_test - y_test_pred),"r", label="Prediction Error")
ax2.legend()
plt.title("Prediction Error")
plt.xlabel("Date")
plt.ylabel("Error")

plt.tight_layout()
plt.show()
