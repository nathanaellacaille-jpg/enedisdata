import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from config import (
    FCST_N_LAGS, FCST_N_FOURIER, FCST_ARIMA_ORDER, FCST_SARIMA_SEASONAL,
    FCST_HORIZON_H, STEPS_PER_DAY,
    LSTM_SEQ_LEN, LSTM_HIDDEN, LSTM_LAYERS, LSTM_EPOCHS, LSTM_LR, LSTM_BATCH_SIZE,
    LSTM_PATIENCE, LSTM_MIN_DELTA, LSTM_VAL_FRAC, LSTM_DROPOUT, LSTM_USE_ROLLING,
)

_RIDGE_ALPHAS_LOG = np.logspace(-4, 3, 20)  # 20 valeurs en log space, audit reco


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



def _calendar_block(n: int, offset: int) -> np.ndarray:
    """One-hot jour-de-semaine (7 cols) + indicateur weekend (1 col). Calendaire pur."""
    t = np.arange(offset, offset + n)
    dow = (t // STEPS_PER_DAY) % 7
    dow_oh = np.eye(7, dtype=np.float32)[dow]
    we = (dow >= 5).astype(np.float32).reshape(-1, 1)
    return np.hstack([dow_oh, we])


class RidgeForecaster:
    """Ridge avec lags + Fourier journaliere + calendrier explicite + StandardScaler.

    Refonte Phase 1 (2026-05) : sur tuning set 5 compteurs, l'ancien Ridge (384 lags, 3
    harmoniques, pas de scaler) gagnait deja +8.4% vs naive_last_day. Les ameliorations :
      - StandardScaler avant Ridge (audit transversal) : +6 pts (le plus gros gain)
      - n_lags reduit a 192 : +1.7 pts (384 etait surdimensionne)
      - 6 harmoniques de Fourier journalieres (audit reco #8) : marginal mais OK
      - Features calendaires explicites (one-hot dow + weekend) : +0.9 pts
      - RidgeCV alphas en log space 1e-4 -> 1e3 : convergence plus fine
    Total +16.8% vs naive sur tuning set.
    """

    def __init__(self, n_lags: int = FCST_N_LAGS, n_fourier: int = FCST_N_FOURIER,
                 use_calendar: bool = True, use_scaler: bool = True):
        """Initialise le forecaster Ridge avec options de tuning."""
        ridge = RidgeCV(alphas=_RIDGE_ALPHAS_LOG)
        self._model = Pipeline([("sc", StandardScaler()), ("ridge", ridge)]) if use_scaler else ridge
        self.n_lags = n_lags
        self.n_fourier = n_fourier
        self.use_calendar = use_calendar
        self.use_scaler = use_scaler
        self._last_window: np.ndarray | None = None

    def _build_X(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Construit X et y depuis la serie : lags + Fourier + calendrier optionnel."""
        lag_X = make_lag_features(series, self.n_lags)
        n_rows = lag_X.shape[0]
        fourier_X = make_fourier_features(n_rows, self.n_fourier, offset=self.n_lags)
        parts = [lag_X, fourier_X]
        if self.use_calendar:
            parts.append(_calendar_block(n_rows, offset=self.n_lags))
        X = np.hstack(parts).astype(np.float32)
        y = series[self.n_lags:].astype(np.float32)
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
            parts_pred = [window[::-1], f_row]
            if self.use_calendar:
                parts_pred.append(_calendar_block(1, offset=n_start + step)[0])
            x = np.hstack(parts_pred).reshape(1, -1).astype(np.float32)
            y_hat = float(self._model.predict(x)[0])
            preds.append(y_hat)
            window = np.roll(window, -1)
            window[-1] = y_hat
        return np.array(preds)


class SARIMAForecaster:
    """SARIMA(p,d,q)(P,D,Q,s) via statsmodels.

    Refonte Phase 1 : l'ancien ARIMA(2,1,2) sans saisonnalite plafonnait a +7% vs naive
    sur le tuning set. SARIMA(1,1,1)(1,0,1,48) atteint +12.2% en exploitant la
    saisonnalite journaliere 48 (audit reco #2). L'ordre AR=2 etait inutile :
    (1,1,1) bat (2,1,2) ET converge 2x plus vite.

    L'alias ARIMAForecaster est conserve pour retro-compatibilite (pages, diagnostic).
    """

    def __init__(self):
        """Initialise le SARIMA."""
        self._result = None
        self._order = FCST_ARIMA_ORDER
        self._seasonal_order = FCST_SARIMA_SEASONAL

    def fit(self, series: np.ndarray, order=FCST_ARIMA_ORDER, seasonal_order=FCST_SARIMA_SEASONAL):
        """Ajuste SARIMA, repli sur (1,1,1) sans saisonnalite si la convergence echoue."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        import warnings
        self._order = order
        self._seasonal_order = seasonal_order
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                                enforce_stationarity=False, enforce_invertibility=False)
                self._result = model.fit(disp=False, maxiter=50)
            except Exception:
                # Fallback : ARIMA(1,1,1) sans saisonnalite (rapide, robuste)
                self._order = (1, 1, 1)
                self._seasonal_order = (0, 0, 0, 0)
                model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                                enforce_stationarity=False, enforce_invertibility=False)
                self._result = model.fit(disp=False, maxiter=30)
        return self

    def predict(self, h: int) -> np.ndarray:
        """Prevision h pas en avant."""
        forecast = self._result.forecast(steps=h)
        return np.asarray(forecast)


# Alias retro-compatibilite : tous les anciens callers continuent de marcher,
# mais ils beneficient maintenant de la saisonnalite 48 par defaut.
ARIMAForecaster = SARIMAForecaster


class LSTMForecaster:
    """LSTM multi-step direct (48 sorties) en mode residuel vs naive_last_day, avec decodeur conditionne sur le calendrier futur."""

    HORIZON_OUT = FCST_HORIZON_H * 2  # 48 demi-heures de sortie
    N_CALENDAR_FEATURES = 4           # sin/cos slot + sin/cos dow

    def __init__(self):
        """Initialise l'LSTM."""
        self._model = None
        self._scaler_mean = 0.0
        self._scaler_std = 1.0
        self._last_seq_x: "object | None" = None  # tensor (1, seq_len, F)
        self._series_len = 0
        self.losses: list = []
        self.val_losses: list = []
        self._stopped_epoch: int | None = None

    @staticmethod
    def _calendar_features(n_total: int, offset: int = 0) -> np.ndarray:
        """sin/cos slot intra-jour (period 48) + sin/cos jour-de-semaine (period 7)."""
        t = np.arange(offset, offset + n_total)
        slot = t % STEPS_PER_DAY
        dow = (t // STEPS_PER_DAY) % 7
        return np.column_stack([
            np.sin(2 * np.pi * slot / STEPS_PER_DAY),
            np.cos(2 * np.pi * slot / STEPS_PER_DAY),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
        ]).astype(np.float32)

    def _build_features(self, series_norm: np.ndarray, offset: int = 0) -> np.ndarray:
        """Empile [kW_norm, calendar_4, rolling_mean_24h, rolling_std_24h] selon config.

        Phase 1 tuning : ajouter rolling_mean/std 24h donne le meilleur jeu de features
        (gain de 1.3 pts vs base sur tuning set). Active via LSTM_USE_ROLLING.
        """
        n = len(series_norm)
        cal = self._calendar_features(n, offset=offset)
        cols = [series_norm.reshape(-1, 1).astype(np.float32), cal]
        if LSTM_USE_ROLLING:
            s = pd.Series(series_norm)
            roll_mean = s.rolling(STEPS_PER_DAY, min_periods=1).mean().values.astype(np.float32).reshape(-1, 1)
            roll_std = s.rolling(STEPS_PER_DAY, min_periods=1).std().fillna(0.0).values.astype(np.float32).reshape(-1, 1)
            cols.extend([roll_mean, roll_std])
        return np.column_stack(cols)

    def fit(self, series: np.ndarray, callback=None):
        """Fit multi-step direct + early stopping. Leve RuntimeError si torch absent ou serie trop courte."""
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:
            raise RuntimeError("PyTorch non disponible sur cette instance.") from exc

        n = len(series)
        seq_len = LSTM_SEQ_LEN
        hor_out = self.HORIZON_OUT
        min_samples = 50
        min_len = seq_len + hor_out + min_samples
        if n < min_len:
            raise RuntimeError(
                f"Serie trop courte pour LSTM multi-step (min {min_len} points = "
                f"{min_len // STEPS_PER_DAY} jours, recu {n}).")

        # Scaler robuste : mediane + IQR. Limite l'impact des pics rares (lave-vaisselle, etc.)
        self._scaler_mean = float(np.median(series))
        q75, q25 = np.percentile(series, [75, 25])
        iqr = float(q75 - q25)
        self._scaler_std = iqr if iqr > 1e-6 else float(series.std() + 1e-8)

        s = ((series - self._scaler_mean) / self._scaler_std).astype(np.float32)
        # X_full : [kW_norm | sin_slot, cos_slot, sin_dow, cos_dow | features additionnelles]
        # Les 4 colonnes 1..5 sont les features calendaires connues du futur (decodeur).
        # Les colonnes >= 5 (mois, lag, rolling) alimentent uniquement l'encodeur.
        X_full = self._build_features(s, offset=0)
        n_features = X_full.shape[1]
        n_cal = self.N_CALENDAR_FEATURES  # toujours 4 ; le decodeur ne lit que sin/cos slot+dow
        assert n_features >= 1 + n_cal, "Le builder doit produire au moins kW + 4 cal."

        # Sliding windows : input = X[i:i+seq], target = s[i+seq:i+seq+hor]
        n_samples = n - seq_len - hor_out + 1
        X_arr = np.lib.stride_tricks.sliding_window_view(X_full, (seq_len, n_features))[: n_samples, 0]
        Y_arr = np.lib.stride_tricks.sliding_window_view(s, hor_out)[seq_len: seq_len + n_samples]
        # Future calendar : strictement les 4 colonnes calendaires de base (1..5)
        cal_only = X_full[:, 1:1 + n_cal]
        FUT_arr = np.lib.stride_tricks.sliding_window_view(cal_only, (hor_out, n_cal))[seq_len: seq_len + n_samples, 0]
        FUT_arr = FUT_arr.reshape(n_samples, hor_out * n_cal)

        # naive_last_day pour chaque sample : derniers hor_out steps de la fenetre d'entree
        # (kW colonne 0). C'est exactement la baseline qu'on veut battre, en valeurs normalisees.
        NAIVE_arr = X_arr[:, -hor_out:, 0]

        X = torch.tensor(np.ascontiguousarray(X_arr), dtype=torch.float32)
        Y = torch.tensor(np.ascontiguousarray(Y_arr), dtype=torch.float32)
        FUT = torch.tensor(np.ascontiguousarray(FUT_arr), dtype=torch.float32)
        NAIVE = torch.tensor(np.ascontiguousarray(NAIVE_arr), dtype=torch.float32)
        # Cible : residu = vrai - naive_last_day. Le LSTM apprend la deviation.
        Y_RES = Y - NAIVE

        # Val split temporel (derniers VAL_FRAC samples = pas de leakage)
        n_val = max(10, int(n_samples * LSTM_VAL_FRAC))
        n_train = n_samples - n_val
        if n_train < min_samples:
            raise RuntimeError(f"Train split trop petit apres val ({n_train} < {min_samples}).")
        X_tr, Y_res_tr, FUT_tr = X[:n_train], Y_RES[:n_train], FUT[:n_train]
        X_val, FUT_val, NAIVE_val, Y_val_abs = X[n_train:], FUT[n_train:], NAIVE[n_train:], Y[n_train:]

        self._model = _LSTMNet(n_features, LSTM_HIDDEN, LSTM_LAYERS, hor_out, n_cal, dropout=LSTM_DROPOUT)
        opt = torch.optim.Adam(self._model.parameters(), lr=LSTM_LR)
        # Huber : MSE pres de zero (precis), L1 sur grands ecarts (robuste aux pics)
        criterion = nn.SmoothL1Loss()

        self.losses, self.val_losses = [], []
        best_val = float("inf")
        best_state = None
        patience_left = LSTM_PATIENCE

        for epoch in range(LSTM_EPOCHS):
            self._model.train()
            indices = torch.randperm(n_train)
            epoch_loss = 0.0
            for start in range(0, n_train, LSTM_BATCH_SIZE):
                batch_idx = indices[start: start + LSTM_BATCH_SIZE]
                xb = X_tr[batch_idx]
                yb_res = Y_res_tr[batch_idx]
                fb = FUT_tr[batch_idx]
                opt.zero_grad()
                # Sortie = residu predite ; on apprend la deviation par rapport a naive_last_day
                out_res = self._model(xb, fb)
                loss = criterion(out_res, yb_res)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                opt.step()
                epoch_loss += loss.item() * xb.shape[0]
            avg_tr = epoch_loss / n_train

            self._model.eval()
            with torch.no_grad():
                # Val loss en mode "vraie prediction" = naive + residu, pour comparable a MAE finale
                val_pred = self._model(X_val, FUT_val) + NAIVE_val
                val_loss = criterion(val_pred, Y_val_abs).item()
            self.losses.append(avg_tr)
            self.val_losses.append(val_loss)
            if callback:
                callback(epoch, avg_tr)

            if val_loss < best_val - LSTM_MIN_DELTA:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self._model.state_dict().items()}
                patience_left = LSTM_PATIENCE
            else:
                patience_left -= 1
                if patience_left <= 0:
                    self._stopped_epoch = epoch + 1
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        # Derniere fenetre d'inputs pour predict() + offset pour reconstruire le futur calendrier
        self._last_seq_x = torch.tensor(X_full[-seq_len:], dtype=torch.float32).unsqueeze(0)
        self._n_cal = n_cal
        self._series_len = n
        return self

    def predict(self, h: int) -> np.ndarray:
        """Prevision h pas (multi-step direct, h <= HORIZON_OUT)."""
        import torch
        if self._model is None or self._last_seq_x is None:
            raise RuntimeError("LSTM non entraine.")
        hor_out = self.HORIZON_OUT
        if h > hor_out:
            raise ValueError(f"h={h} > HORIZON_OUT={hor_out} non supporte (multi-step direct).")
        self._model.eval()
        n_cal = self._n_cal
        cal = self._calendar_features(hor_out, offset=self._series_len)
        fut = torch.tensor(cal.reshape(1, hor_out * n_cal), dtype=torch.float32)
        naive_norm = self._last_seq_x[0, -hor_out:, 0].numpy()
        with torch.no_grad():
            residual_norm = self._model(self._last_seq_x, fut).numpy().flatten()
        preds_norm = (residual_norm + naive_norm)[:h]
        return preds_norm * self._scaler_std + self._scaler_mean


class _LSTMNet:
    """Encoder LSTM + decoder conditionnel sur calendrier futur (pattern N-BEATS / DeepAR leger)."""

    def __init__(self, input_size, hidden, n_layers, horizon_out, n_cal, dropout=0.0):
        """Construit le module Torch sous-jacent."""
        import torch
        import torch.nn as nn

        fut_dim = horizon_out * n_cal

        class Net(nn.Module):
            def __init__(self):
                """LSTM encoder + Linear([hidden ; future_calendar_flat]) -> horizon_out."""
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden, n_layers,
                    batch_first=True,
                    dropout=dropout if n_layers > 1 else 0.0,
                )
                # Decodeur conditionne sur le calendrier des pas a predire :
                # le reseau "sait" pour chaque sortie a quel slot/dow elle correspond.
                self.fc = nn.Linear(hidden + fut_dim, horizon_out)

            def forward(self, x, fut_cal):
                """x: (B, seq, F_in). fut_cal: (B, horizon_out * n_cal). Retourne (B, horizon_out)."""
                out, _ = self.lstm(x)
                last_hidden = out[:, -1, :]
                concat = torch.cat([last_hidden, fut_cal], dim=1)
                return self.fc(concat)

        self._net = Net()

    def __call__(self, x, fut_cal):
        """Appel du reseau (forward avec calendrier futur)."""
        return self._net(x, fut_cal)

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

    def load_state_dict(self, state):
        """Charge un etat sauvegarde."""
        self._net.load_state_dict(state)
