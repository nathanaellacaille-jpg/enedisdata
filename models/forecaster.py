import numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from config import FCST_N_LAGS, FCST_N_FOURIER, STEPS_PER_DAY

_NLINEAR_L = 192
_NLINEAR_LAMBDA = 1e-3
_NLINEAR_WEIGHTS = Path(__file__).resolve().parent.parent / "assets" / "nlinear_global_weights.npy"
_nlinear_W_cache: np.ndarray | None = None


def _load_nlinear_W() -> np.ndarray:
    """Charge les poids NLinear global (cache module-level)."""
    global _nlinear_W_cache
    if _nlinear_W_cache is None:
        if not _NLINEAR_WEIGHTS.exists():
            raise FileNotFoundError(
                f"Poids NLinear introuvables : {_NLINEAR_WEIGHTS}. "
                "Lancer scripts/compute_nlinear_global_weights.py"
            )
        _nlinear_W_cache = np.load(str(_NLINEAR_WEIGHTS))
    return _nlinear_W_cache

LGBM_V2_LOOKBACK = 336   # 7 jours — horizon max de slot_J7
_LGBM_V2_PARAMS = dict(
    n_estimators=500,
    num_leaves=31,
    learning_rate=0.02,
    subsample=0.7,
    colsample_bytree=0.5,
    min_child_samples=50,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=1,
    verbose=-1,
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


class LGBMForecasterV2:
    """LightGBM DMSF v2 : 29 features domaine-metier au lieu de 192 lags bruts.

    Features par echantillon (origin t, horizon h) :
      Slot-level (cible t+h) : J-1/J-2/J-7 meme slot, moyenne 7j, std 7j, delta_slot
      Etat courant (en t)    : lag-1, lag-2, delta journalier (aujourd'hui - hier)
      Temporels (en t+h)     : Fourier journalier 6 harmoniques, DOW one-hot + weekend
    Target : residu vs J-1 (meme que V1), DMSF 48 modeles independants.
    Avantage vs V1 : pas de lags correles → moins d'overfitting, entrainement 3x plus rapide.
    """

    def __init__(self, n_fourier: int = FCST_N_FOURIER):
        """Initialise le forecaster LightGBM V2."""
        self.n_fourier = n_fourier
        self._models: list = []
        self._last_known: np.ndarray | None = None
        self._series_len: int = 0

    def fit(self, series: np.ndarray, y=None):
        """Entraine 48 LGBMRegressor avec features domaine-metier."""
        import lightgbm as lgb
        n = len(series)
        n_samp = n - LGBM_V2_LOOKBACK - STEPS_PER_DAY
        if n_samp <= 0:
            raise ValueError(
                f"Serie trop courte pour LGBMv2 ({n} pts, min {LGBM_V2_LOOKBACK + STEPS_PER_DAY + 1})"
            )
        origins = np.arange(LGBM_V2_LOOKBACK, LGBM_V2_LOOKBACK + n_samp)

        # Features independantes de h — calculees une seule fois
        cs = np.concatenate([[0.0], np.cumsum(series)])
        lag1 = series[origins - 1].astype(np.float32)
        lag2 = series[origins - 2].astype(np.float32)
        day_mean_curr = ((cs[origins] - cs[origins - STEPS_PER_DAY]) / STEPS_PER_DAY).astype(np.float32)
        day_mean_prev = ((cs[origins - STEPS_PER_DAY] - cs[origins - 2 * STEPS_PER_DAY]) / STEPS_PER_DAY).astype(np.float32)
        delta_day = (day_mean_curr - day_mean_prev).reshape(-1, 1)
        base_X = np.hstack([lag1.reshape(-1, 1), lag2.reshape(-1, 1), delta_day])

        self._models = []
        for h in range(STEPS_PER_DAY):
            # Features slot-level (pour le temps de prediction origin+h)
            slot_J1 = series[origins + h - STEPS_PER_DAY].astype(np.float32)
            slot_J2 = series[origins + h - 2 * STEPS_PER_DAY].astype(np.float32)
            slot_J7 = series[origins + h - 7 * STEPS_PER_DAY].astype(np.float32)
            slot_days = np.stack(
                [series[origins + h - k * STEPS_PER_DAY] for k in range(1, 8)], axis=1
            ).astype(np.float32)
            slot_mean_7d = slot_days.mean(axis=1)
            slot_std_7d = slot_days.std(axis=1)
            delta_slot = slot_J1 - slot_J2

            fourier_X = make_fourier_features(n_samp, self.n_fourier, offset=LGBM_V2_LOOKBACK + h)
            cal_X = _calendar_block(n_samp, offset=LGBM_V2_LOOKBACK + h)

            X = np.hstack([
                slot_J1.reshape(-1, 1), slot_J2.reshape(-1, 1), slot_J7.reshape(-1, 1),
                slot_mean_7d.reshape(-1, 1), slot_std_7d.reshape(-1, 1),
                delta_slot.reshape(-1, 1),
                base_X, fourier_X, cal_X,
            ]).astype(np.float32)

            y_res = (series[origins + h] - slot_J1).astype(np.float32)
            mdl = lgb.LGBMRegressor(**_LGBM_V2_PARAMS)
            mdl.fit(X, y_res)
            self._models.append(mdl)

        self._last_known = series[-LGBM_V2_LOOKBACK:].copy()
        self._series_len = len(series)
        return self

    def predict(self, h: int) -> np.ndarray:
        """Predit h pas : residu LGBMv2[step] + naive J-1[slot]."""
        import warnings
        s = self._last_known   # series[-336:]
        n = self._series_len

        # Features independantes du pas
        lag1 = float(s[-1])
        lag2 = float(s[-2])
        delta_day = float(s[-STEPS_PER_DAY:].mean() - s[-2 * STEPS_PER_DAY:-STEPS_PER_DAY].mean())

        preds = []
        for step in range(h):
            mdl = self._models[step % STEPS_PER_DAY]

            # Slot-level features pour le slot cible (n + step)
            slot_J1 = float(s[step - STEPS_PER_DAY])        # s[-48+step]
            slot_J2 = float(s[step - 2 * STEPS_PER_DAY])
            slot_J7 = float(s[step - 7 * STEPS_PER_DAY])
            slot_days = np.array([float(s[step - k * STEPS_PER_DAY]) for k in range(1, 8)])
            slot_mean_7d = float(slot_days.mean())
            slot_std_7d = float(slot_days.std())
            delta_slot = slot_J1 - slot_J2

            f_row = make_fourier_features(1, self.n_fourier, offset=n + step)[0]
            cal_row = _calendar_block(1, offset=n + step)[0]

            x = np.array(
                [slot_J1, slot_J2, slot_J7,
                 slot_mean_7d, slot_std_7d, delta_slot,
                 lag1, lag2, delta_day,
                 *f_row, *cal_row],
                dtype=np.float32,
            ).reshape(1, -1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                residual = float(mdl.predict(x)[0])
            preds.append(residual + slot_J1)

        return np.array(preds)


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
        self._last_day: np.ndarray | None = None

    def _build_X(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construit X, y_brut et y_naive (meme slot J-1) pour apprentissage residuel."""
        lag_X = make_lag_features(series, self.n_lags)
        n_rows = lag_X.shape[0]
        fourier_X = make_fourier_features(n_rows, self.n_fourier, offset=self.n_lags)
        parts = [lag_X, fourier_X]
        if self.use_calendar:
            parts.append(_calendar_block(n_rows, offset=self.n_lags))
        X = np.hstack(parts).astype(np.float32)
        y = series[self.n_lags:].astype(np.float32)
        # naive = meme slot J-1 pour chaque point cible
        naive = series[self.n_lags - STEPS_PER_DAY : len(series) - STEPS_PER_DAY].astype(np.float32)
        return X, y, naive

    def fit(self, series: np.ndarray, y=None):
        """Entraine Ridge sur le residu (y - naive_J-1)."""
        X, y_raw, y_naive = self._build_X(series)
        self._model.fit(X, y_raw - y_naive)
        self._last_window = series[-self.n_lags:].copy()
        self._last_day = series[-STEPS_PER_DAY:].copy()
        self._series_len = len(series)
        return self

    def predict(self, h: int) -> np.ndarray:
        """Predit h pas : residu Ridge + naive_J-1, fenetre mise a jour en brut."""
        window = self._last_window.copy()
        preds = []
        n_start = self._series_len
        for step in range(h):
            f_row = make_fourier_features(1, self.n_fourier, offset=n_start + step)[0]
            parts_pred = [window[::-1], f_row]
            if self.use_calendar:
                parts_pred.append(_calendar_block(1, offset=n_start + step)[0])
            x = np.hstack(parts_pred).reshape(1, -1).astype(np.float32)
            residual = float(self._model.predict(x)[0])
            y_hat = residual + float(self._last_day[step % STEPS_PER_DAY])
            preds.append(y_hat)
            window = np.roll(window, -1)
            window[-1] = y_hat
        return np.array(preds)


class NLinearGlobalForecaster:
    """NLinear MIMO pre-entraine sur le dataset complet (500 compteurs).

    fit() stocke uniquement la fenetre courante — aucun entrainement.
    predict() = W_global.T @ (window - window[-1]) + window[-1], O(L*STEPS).
    W_global (192x48) pre-calcule par scripts/compute_nlinear_global_weights.py.
    """

    def __init__(self):
        """Initialise le forecaster NLinear global."""
        self._window: np.ndarray | None = None

    def fit(self, series: np.ndarray, y=None):
        """Stocke la fenetre d'entree (pas de training)."""
        self._window = series[-_NLINEAR_L:].astype(float)
        return self

    def predict(self, h: int) -> np.ndarray:
        """Predit h pas via W_global pre-calcule."""
        W = _load_nlinear_W()
        x = self._window - self._window[-1]
        return (W.T @ x + self._window[-1])[:h]

