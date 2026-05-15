import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from config import (
    FCST_N_LAGS, FCST_N_FOURIER, FCST_ARIMA_ORDER,
    LSTM_SEQ_LEN, LSTM_HIDDEN, LSTM_LAYERS, LSTM_EPOCHS, LSTM_LR, LSTM_BATCH_SIZE,
)


def make_lag_features(series: np.ndarray, n_lags: int) -> np.ndarray:
    """Construit la matrice de lags temporels."""
    n = len(series)
    X = np.zeros((n - n_lags, n_lags))
    for i in range(n_lags):
        X[:, i] = series[n_lags - i - 1: n - i - 1]
    return X


def make_fourier_features(n: int, n_harmonics: int = 3, period: int = 48, offset: int = 0) -> np.ndarray:
    """Features sin/cos a periode fixe (defaut 48 = journalier). offset = indice absolu du premier point."""
    t = np.arange(offset, offset + n)
    cols = []
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(2 * np.pi * k * t / period))
        cols.append(np.cos(2 * np.pi * k * t / period))
    return np.column_stack(cols)



class RidgeForecaster:
    """Modele de prevision Ridge avec lags + features de Fourier."""

    def __init__(self, n_lags: int = FCST_N_LAGS, n_fourier: int = FCST_N_FOURIER):
        """Initialise le forecaster Ridge."""
        self._model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
        self.n_lags = n_lags
        self.n_fourier = n_fourier
        self._last_window: np.ndarray | None = None

    def _build_X(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Construit X et y depuis la serie."""
        lag_X = make_lag_features(series, self.n_lags)
        n = lag_X.shape[0]
        fourier_X = make_fourier_features(n, self.n_fourier, offset=self.n_lags)
        X = np.hstack([lag_X, fourier_X])
        y = series[self.n_lags:]
        return X, y

    def fit(self, series: np.ndarray, y=None):
        """Entraine Ridge sur la serie."""
        X, y_train = self._build_X(series)
        self._model.fit(X, y_train)
        self._last_window = series[-self.n_lags:].copy()
        self._series_len = len(series)
        return self

    def predict(self, h: int) -> np.ndarray:
        """Predit h pas en avant par recursion."""
        window = self._last_window.copy()
        preds = []
        n_start = self._series_len
        for step in range(h):
            f_row = make_fourier_features(1, self.n_fourier, offset=n_start + step)[0]
            x = np.hstack([window[::-1], f_row]).reshape(1, -1)
            y_hat = self._model.predict(x)[0]
            preds.append(y_hat)
            window = np.roll(window, -1)
            window[-1] = y_hat
        return np.array(preds)

    def coef_series(self) -> pd.Series:
        """Coefficients du modele Ridge."""
        lag_names = [f"lag_{i+1}" for i in range(self.n_lags)]
        fourier_names = []
        for k in range(1, self.n_fourier + 1):
            fourier_names += [f"sin_{k}", f"cos_{k}"]
        names = lag_names + fourier_names
        return pd.Series(self._model.coef_, index=names)


class ARIMAForecaster:
    """Modele ARIMA via statsmodels."""

    def __init__(self):
        """Initialise l'ARIMA."""
        self._result = None
        self._order = FCST_ARIMA_ORDER

    def fit(self, series: np.ndarray, order=FCST_ARIMA_ORDER):
        """Ajuste ARIMA(p,d,q), repli sur (1,1,1) si la convergence echoue."""
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        import warnings
        self._order = order
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                self._result = ARIMA(series, order=order).fit()
            except (ConvergenceWarning, Exception):
                self._order = (1, 1, 1)
                self._result = ARIMA(series, order=(1, 1, 1)).fit()
        return self

    def predict(self, h: int) -> np.ndarray:
        """Prevision h pas en avant."""
        forecast = self._result.forecast(steps=h)
        return np.asarray(forecast)

    def summary(self) -> str:
        """Resume du modele ARIMA."""
        if self._result is None:
            return "Modele non entraine."
        return str(self._result.summary())


class LSTMForecaster:
    """Prevision LSTM via PyTorch (optionnel)."""

    def __init__(self):
        """Initialise l'LSTM."""
        self._model = None
        self._scaler_mean = 0.0
        self._scaler_std = 1.0
        self.losses: list = []

    def fit(self, series: np.ndarray, callback=None):
        """Entraine le LSTM sur la serie normalisee. Leve RuntimeError si torch indisponible."""
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:
            raise RuntimeError("PyTorch non disponible sur cette instance.") from exc

        self._scaler_mean = series.mean()
        self._scaler_std = series.std() + 1e-8
        s = (series - self._scaler_mean) / self._scaler_std

        seq_len = LSTM_SEQ_LEN
        X_list, y_list = [], []
        for i in range(len(s) - seq_len):
            X_list.append(s[i: i + seq_len])
            y_list.append(s[i + seq_len])
        X = torch.tensor(np.array(X_list), dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(np.array(y_list), dtype=torch.float32)

        self._model = _LSTMNet(1, LSTM_HIDDEN, LSTM_LAYERS)
        opt = torch.optim.Adam(self._model.parameters(), lr=LSTM_LR)
        criterion = nn.MSELoss()
        self.losses = []
        n_samples = X.shape[0]

        for epoch in range(LSTM_EPOCHS):
            self._model.train()
            indices = torch.randperm(n_samples)
            epoch_loss = 0.0
            for start in range(0, n_samples, LSTM_BATCH_SIZE):
                batch_idx = indices[start:start + LSTM_BATCH_SIZE]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                opt.zero_grad()
                out = self._model(X_batch).squeeze(-1)
                loss = criterion(out, y_batch)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * len(batch_idx)
            avg_loss = epoch_loss / n_samples
            self.losses.append(avg_loss)
            if callback:
                callback(epoch, avg_loss)

        self._last_seq = torch.tensor(s[-seq_len:], dtype=torch.float32)
        return self

    def predict(self, h: int) -> np.ndarray:
        """Predit h pas en avant par recursion."""
        import torch
        self._model.eval()
        seq = self._last_seq.clone().unsqueeze(0).unsqueeze(-1)  # (1, seq, 1)
        preds = []
        with torch.no_grad():
            for _ in range(h):
                out = self._model(seq)
                val = out[0, 0].item()
                preds.append(val)
                new_step = torch.tensor([[[val]]])
                seq = torch.cat([seq[:, 1:, :], new_step], dim=1)
        arr = np.array(preds) * self._scaler_std + self._scaler_mean
        return arr


class _LSTMNet:
    """Reseau LSTM interne."""

    def __init__(self, input_size, hidden, n_layers):
        """Initialise le reseau."""
        import torch.nn as nn
        import torch

        class Net(nn.Module):
            def __init__(self):
                """Construit les couches LSTM et lineaire."""
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden, n_layers, batch_first=True)
                self.fc = nn.Linear(hidden, 1)

            def forward(self, x):
                """Passe avant : LSTM puis projection lineaire sur le dernier pas."""
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        self._net = Net()

    def __call__(self, x):
        """Appel du reseau."""
        return self._net(x)

    def train(self):
        """Mode entrainement."""
        self._net.train()

    def eval(self):
        """Mode evaluation."""
        self._net.eval()

    def parameters(self):
        """Parametres du reseau."""
        return self._net.parameters()

    def state_dict(self):
        """Etat du reseau."""
        return self._net.state_dict()
