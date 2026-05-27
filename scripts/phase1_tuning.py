"""Phase 1 - Finetuning systematique des 3 modeles de prevision.

But : comparer 'a armes egales'. On donne a chacun sa meilleure chance avant le re-diagnostic
Phase 0 v2. Outputs :
  - assets/phase1_tuning.json : configs gagnantes + tableau comparatif
  - stdout : verdict de chaque pass

Methode : 5 compteurs stratifies, 1 fold (test = derniers 48 pas), seed deterministe.
Pas de re-tirage entre essais : meme split pour tous les modeles.

Usage : python scripts/phase1_tuning.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import (
    DEFAULT_LBL_PATH, DEFAULT_TS_PATH,
    FCST_HORIZON_H, STEPS_PER_DAY,
)
import models.forecaster as fc_mod
from utils.parser import parse_labels, parse_timeseries

HORIZON = FCST_HORIZON_H * 2
TRAIN_WINDOW_DAYS = 60
TUNING_SEED = 7  # different de Phase 0 (seed=42) pour eviter le data leakage tuning/eval


# === Utilities ============================================================

def _load_sample() -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    """Retourne [(meter_id, label, train_kw, true_kw)] pour 5 compteurs stratifies (3 RS + 2 RP)."""
    df = parse_timeseries(str(DEFAULT_TS_PATH), max_meters=None)
    labels = parse_labels(str(DEFAULT_LBL_PATH))
    rng = np.random.default_rng(TUNING_SEED)
    series_lens = df.groupby("meter_id", observed=True).size()
    min_len = TRAIN_WINDOW_DAYS * STEPS_PER_DAY + HORIZON
    eligible = series_lens[series_lens >= min_len].index.astype(str).tolist()
    rs = sorted(m for m in eligible if labels.get(m) == 1)
    rp = sorted(m for m in eligible if labels.get(m) == 0)
    rng.shuffle(rs)
    rng.shuffle(rp)
    selected = rs[:3] + rp[:2]
    print(f"  Echantillon tuning : {selected}")
    out = []
    for mid in selected:
        s = df[df["meter_id"].astype(str) == mid].sort_values("ts")["kw"].values.astype(float)
        train = s[-(TRAIN_WINDOW_DAYS * STEPS_PER_DAY):-HORIZON]
        true = s[-HORIZON:]
        out.append((mid, labels[mid], train, true))
    return out


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def _naive_last_day(train: np.ndarray) -> np.ndarray:
    """Reference : dernier jour repete."""
    return train[-HORIZON:]


# === LSTM : ablations de features =========================================

class _LSTMFeatExtended(fc_mod.LSTMForecaster):
    """LSTM dont _build_features est conditionne par self.feature_set."""

    def __init__(self, feature_set: str = "base"):
        """Stocke le jeu de features additionnelles a empiler avant fit."""
        super().__init__()
        self.feature_set = feature_set

    def _build_features(self, series_norm: np.ndarray, offset: int = 0) -> np.ndarray:
        """Empile [kW, cal_4, +extras selon feature_set]."""
        n = len(series_norm)
        cal = self._calendar_features(n, offset=offset)
        cols = [series_norm.reshape(-1, 1).astype(np.float32), cal]
        flags = self.feature_set.split("+")

        if "month" in flags or "all" in flags:
            # period annuelle = 365.25 jours x 48 slots
            t = np.arange(offset, offset + n)
            period = 365.25 * STEPS_PER_DAY
            cols.append(np.sin(2 * np.pi * t / period).astype(np.float32).reshape(-1, 1))
            cols.append(np.cos(2 * np.pi * t / period).astype(np.float32).reshape(-1, 1))

        if "lag" in flags or "all" in flags:
            # Lag j-1 (48 pas) et lag j-7 (336 pas). Pad zero pour les premieres positions.
            lag1 = np.zeros((n, 1), dtype=np.float32)
            if n > STEPS_PER_DAY:
                lag1[STEPS_PER_DAY:, 0] = series_norm[:-STEPS_PER_DAY]
            lag7 = np.zeros((n, 1), dtype=np.float32)
            if n > 7 * STEPS_PER_DAY:
                lag7[7 * STEPS_PER_DAY:, 0] = series_norm[:-7 * STEPS_PER_DAY]
            cols.extend([lag1, lag7])

        if "rolling" in flags or "all" in flags:
            s = pd.Series(series_norm)
            roll_mean = s.rolling(STEPS_PER_DAY, min_periods=1).mean().values.astype(np.float32).reshape(-1, 1)
            roll_std = s.rolling(STEPS_PER_DAY, min_periods=1).std().fillna(0.0).values.astype(np.float32).reshape(-1, 1)
            cols.extend([roll_mean, roll_std])

        return np.column_stack(cols)


def _patch_lstm(hidden: int = 32, layers: int = 1, seq_len: int = 336,
                dropout: float = 0.0, lr: float = 2e-3, batch: int = 64,
                epochs: int = 30, patience: int = 4, min_delta: float = 1e-3,
                val_frac: float = 0.10) -> None:
    """Monkey-patch des constantes LSTM dans le module forecaster."""
    fc_mod.LSTM_HIDDEN = hidden
    fc_mod.LSTM_LAYERS = layers
    fc_mod.LSTM_SEQ_LEN = seq_len
    fc_mod.LSTM_DROPOUT = dropout
    fc_mod.LSTM_LR = lr
    fc_mod.LSTM_BATCH_SIZE = batch
    fc_mod.LSTM_EPOCHS = epochs
    fc_mod.LSTM_PATIENCE = patience
    fc_mod.LSTM_MIN_DELTA = min_delta
    fc_mod.LSTM_VAL_FRAC = val_frac


def _eval_lstm(samples, feature_set: str, label: str) -> dict:
    """Entraine et evalue le LSTM (feature_set donne) sur les 5 compteurs. Retourne mae par compteur + mean."""
    rows = []
    t0 = time.time()
    for mid, lbl, train, true in samples:
        naive = _naive_last_day(train)
        naive_mae = _mae(true, naive)
        t = time.time()
        try:
            lstm = _LSTMFeatExtended(feature_set=feature_set)
            lstm.fit(train)
            pred = lstm.predict(HORIZON)
            mae = _mae(true, pred)
            dur = time.time() - t
            rows.append({"meter_id": mid, "label": lbl, "naive_mae": naive_mae, "lstm_mae": mae, "fit_s": dur,
                         "stopped_epoch": lstm._stopped_epoch, "n_epochs": len(lstm.losses)})
        except Exception as exc:
            print(f"    {label}/{mid} FAIL: {exc}")
            rows.append({"meter_id": mid, "label": lbl, "naive_mae": naive_mae, "lstm_mae": float("nan"),
                         "fit_s": float("nan"), "error": str(exc)})
    df = pd.DataFrame(rows)
    valid = df.dropna(subset=["lstm_mae"])
    mean_lstm = float(valid["lstm_mae"].mean()) if not valid.empty else float("nan")
    mean_naive = float(valid["naive_mae"].mean()) if not valid.empty else float("nan")
    n_win = int((valid["lstm_mae"] < valid["naive_mae"]).sum())
    print(f"  [{label:<22s}] mean_lstm={mean_lstm:.3f}  vs naive={mean_naive:.3f}  "
          f"gain={(mean_naive - mean_lstm) / (mean_naive + 1e-9) * 100:+.1f}%  "
          f"wins={n_win}/{len(valid)}  total={time.time() - t0:.0f}s")
    return {"label": label, "mean_lstm_mae": mean_lstm, "mean_naive_mae": mean_naive,
            "gain_pct": float((mean_naive - mean_lstm) / (mean_naive + 1e-9) * 100),
            "wins": n_win, "n": len(valid), "rows": rows}


# === Ridge tuning =========================================================

def _eval_ridge_variant(samples, n_lags: int, n_fourier: int, normalize: bool,
                       add_calendar: bool, alphas_log: bool, label: str) -> dict:
    """Eval une variante Ridge (lag, Fourier, scaler, calendrier explicite, grille alpha)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from models.forecaster import make_lag_features, make_fourier_features

    rows = []
    t0 = time.time()
    alphas = np.logspace(-4, 3, 20) if alphas_log else [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    for mid, lbl, train, true in samples:
        naive_mae = _mae(true, _naive_last_day(train))

        # Build X, y (regression : predire t depuis lags + Fourier + calendrier)
        lag_X = make_lag_features(train, n_lags)
        n = lag_X.shape[0]
        fourier_X = make_fourier_features(n, n_fourier, offset=n_lags)
        parts = [lag_X, fourier_X]

        if add_calendar:
            # offset = n_lags (premier point predit = train[n_lags])
            t_idx = np.arange(n_lags, n_lags + n)
            dow = (t_idx // STEPS_PER_DAY) % 7
            # one-hot dow
            dow_oh = np.eye(7, dtype=np.float32)[dow]
            # weekend binary
            we = (dow >= 5).astype(np.float32).reshape(-1, 1)
            parts.append(dow_oh)
            parts.append(we)

        X = np.hstack(parts)
        y = train[n_lags:]

        if normalize:
            pipe = Pipeline([("sc", StandardScaler()), ("ridge", RidgeCV(alphas=alphas))])
        else:
            pipe = RidgeCV(alphas=alphas)

        pipe.fit(X, y)

        # Prediction recursive sur 48 pas
        window = train[-n_lags:].copy()
        preds = []
        for step in range(HORIZON):
            f_row = make_fourier_features(1, n_fourier, offset=len(train) + step)[0]
            parts_pred = [window[::-1], f_row]
            if add_calendar:
                t_pred = len(train) + step
                dow_p = (t_pred // STEPS_PER_DAY) % 7
                dow_oh_p = np.zeros(7, dtype=np.float32)
                dow_oh_p[dow_p] = 1.0
                we_p = np.array([1.0 if dow_p >= 5 else 0.0], dtype=np.float32)
                parts_pred.extend([dow_oh_p, we_p])
            x = np.hstack(parts_pred).reshape(1, -1)
            preds.append(float(pipe.predict(x)[0]))
            window = np.roll(window, -1)
            window[-1] = preds[-1]
        pred = np.array(preds)
        mae = _mae(true, pred)
        rows.append({"meter_id": mid, "label": lbl, "naive_mae": naive_mae, "ridge_mae": mae})

    df = pd.DataFrame(rows)
    mean_ridge = float(df["ridge_mae"].mean())
    mean_naive = float(df["naive_mae"].mean())
    n_win = int((df["ridge_mae"] < df["naive_mae"]).sum())
    print(f"  [{label:<35s}] mean_ridge={mean_ridge:.3f}  vs naive={mean_naive:.3f}  "
          f"gain={(mean_naive - mean_ridge) / (mean_naive + 1e-9) * 100:+.1f}%  wins={n_win}/{len(df)}  "
          f"t={time.time() - t0:.0f}s")
    return {"label": label, "mean_ridge_mae": mean_ridge, "mean_naive_mae": mean_naive,
            "gain_pct": float((mean_naive - mean_ridge) / (mean_naive + 1e-9) * 100),
            "wins": n_win, "n": len(df),
            "config": {"n_lags": n_lags, "n_fourier": n_fourier, "normalize": normalize,
                       "add_calendar": add_calendar, "alphas_log": alphas_log},
            "rows": rows}


# === SARIMA ===============================================================

def _eval_sarima(samples, order, seasonal_order, label: str) -> dict:
    """SARIMA(p,d,q)(P,D,Q,s) avec fallback en cas de non-convergence."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    rows = []
    t0 = time.time()
    for mid, lbl, train, true in samples:
        naive_mae = _mae(true, _naive_last_day(train))
        t = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                                enforce_stationarity=False, enforce_invertibility=False)
                fit = model.fit(disp=False, maxiter=80)
                pred = np.asarray(fit.forecast(steps=HORIZON))
                mae = _mae(true, pred)
                dur = time.time() - t
                rows.append({"meter_id": mid, "label": lbl, "naive_mae": naive_mae,
                             "sarima_mae": mae, "fit_s": dur})
            except Exception as exc:
                print(f"    {mid} SARIMA fail: {exc}")
                rows.append({"meter_id": mid, "label": lbl, "naive_mae": naive_mae,
                             "sarima_mae": float("nan"), "error": str(exc)})
    df = pd.DataFrame(rows)
    valid = df.dropna(subset=["sarima_mae"])
    mean_s = float(valid["sarima_mae"].mean()) if not valid.empty else float("nan")
    mean_n = float(valid["naive_mae"].mean()) if not valid.empty else float("nan")
    n_win = int((valid["sarima_mae"] < valid["naive_mae"]).sum())
    print(f"  [{label:<30s}] mean_sarima={mean_s:.3f}  vs naive={mean_n:.3f}  "
          f"gain={(mean_n - mean_s) / (mean_n + 1e-9) * 100:+.1f}%  wins={n_win}/{len(valid)}  "
          f"t={time.time() - t0:.0f}s")
    return {"label": label, "order": list(order), "seasonal_order": list(seasonal_order),
            "mean_sarima_mae": mean_s, "mean_naive_mae": mean_n,
            "gain_pct": float((mean_n - mean_s) / (mean_n + 1e-9) * 100),
            "wins": n_win, "n": len(valid), "rows": rows}


# === Main =================================================================

def main() -> None:
    """Execute les 3 passes de tuning et ecrit assets/phase1_tuning.json."""
    print("=" * 72)
    print("PHASE 1 - FINETUNING SYSTEMATIQUE (5 compteurs, 1 fold)")
    print("=" * 72)
    print("\n[1/4] Chargement de l'echantillon...")
    samples = _load_sample()

    out = {
        "computed_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {"n_meters": 5, "train_window_days": TRAIN_WINDOW_DAYS,
                   "horizon": HORIZON, "seed": TUNING_SEED},
        "lstm_features": [],
        "lstm_hparams": [],
        "ridge_variants": [],
        "sarima_variants": [],
        "best": {},
    }

    # --- 2. LSTM features ablation ---
    print("\n[2a/4] LSTM - ablation features (hparams par defaut)")
    _patch_lstm()  # defaults
    feature_sets = ["base", "base+month", "base+lag", "base+rolling", "base+all"]
    for fs in feature_sets:
        r = _eval_lstm(samples, fs, f"LSTM_{fs}")
        out["lstm_features"].append({"feature_set": fs, **r})

    best_feat = max(out["lstm_features"], key=lambda x: x["gain_pct"])
    print(f"\n  -> Best feature set : {best_feat['feature_set']} (gain {best_feat['gain_pct']:+.1f}%)")
    out["best"]["lstm_feature_set"] = best_feat["feature_set"]

    # --- 2b. LSTM hparams sur best feature set ---
    print(f"\n[2b/4] LSTM - hparam grid sur feature_set='{best_feat['feature_set']}'")
    hparam_grid = [
        # (HIDDEN, LAYERS, SEQ_LEN, dropout)
        (32, 1, 336, 0.0),   # baseline current
        (64, 1, 336, 0.0),
        (32, 2, 336, 0.1),
        (64, 2, 336, 0.1),
        (32, 1, 192, 0.0),
        (64, 1, 192, 0.0),
        (64, 1, 672, 0.0),
    ]
    for (h, l, sl, dr) in hparam_grid:
        _patch_lstm(hidden=h, layers=l, seq_len=sl, dropout=dr)
        r = _eval_lstm(samples, best_feat["feature_set"], f"H{h}L{l}S{sl}D{dr}")
        out["lstm_hparams"].append({"hidden": h, "layers": l, "seq_len": sl, "dropout": dr, **r})

    best_hp = max(out["lstm_hparams"], key=lambda x: x["gain_pct"])
    print(f"\n  -> Best hparam : H{best_hp['hidden']} L{best_hp['layers']} "
          f"S{best_hp['seq_len']} D{best_hp['dropout']} (gain {best_hp['gain_pct']:+.1f}%)")
    out["best"]["lstm_hparams"] = {k: best_hp[k] for k in ["hidden", "layers", "seq_len", "dropout"]}
    out["best"]["lstm_gain_pct"] = best_hp["gain_pct"]

    # Reset
    _patch_lstm()

    # --- 3. Ridge tuning ---
    print("\n[3/4] Ridge - ablation features + grid")
    ridge_variants = [
        # (n_lags, n_fourier, normalize, calendar, alphas_log, label)
        (384, 3, False, False, False, "current_prod (384/3/no_norm/no_cal)"),
        (384, 3, True,  False, True,  "+normalize +alphas_log"),
        (384, 6, True,  False, True,  "+fourier_6"),
        (384, 6, True,  True,  True,  "+calendar (dow_oh + we)"),
        (192, 6, True,  True,  True,  "n_lags=192"),
        (96,  6, True,  True,  True,  "n_lags=96"),
        (336, 6, True,  True,  True,  "n_lags=336"),
    ]
    for cfg in ridge_variants:
        r = _eval_ridge_variant(samples, *cfg[:5], label=cfg[5])
        out["ridge_variants"].append(r)
    best_r = max(out["ridge_variants"], key=lambda x: x["gain_pct"])
    print(f"\n  -> Best Ridge : {best_r['label']} (gain {best_r['gain_pct']:+.1f}%)")
    out["best"]["ridge"] = best_r["config"]
    out["best"]["ridge_gain_pct"] = best_r["gain_pct"]

    # --- 4. SARIMA ---
    print("\n[4/4] SARIMA - test saisonnalite 48")
    sarima_configs = [
        ((2, 1, 2), (0, 0, 0, 0), "ARIMA(2,1,2) current"),
        ((2, 1, 2), (1, 0, 1, 48), "SARIMA(2,1,2)(1,0,1,48)"),
        ((1, 1, 1), (1, 0, 1, 48), "SARIMA(1,1,1)(1,0,1,48)"),
        ((0, 1, 1), (0, 1, 1, 48), "SARIMA(0,1,1)(0,1,1,48)"),  # airline season
    ]
    for (o, so, lbl) in sarima_configs:
        r = _eval_sarima(samples, o, so, lbl)
        out["sarima_variants"].append(r)
    best_s = max(out["sarima_variants"], key=lambda x: x["gain_pct"])
    print(f"\n  -> Best SARIMA : {best_s['label']} (gain {best_s['gain_pct']:+.1f}%)")
    out["best"]["sarima"] = {"order": best_s["order"], "seasonal_order": best_s["seasonal_order"]}
    out["best"]["sarima_gain_pct"] = best_s["gain_pct"]

    # --- Dump ---
    dest = ROOT / "assets" / "phase1_tuning.json"
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=str)
    print(f"\nTuning ecrit dans {dest}")

    # --- Verdict synthese ---
    print("\n" + "=" * 72)
    print("VERDICT TUNING")
    print("=" * 72)
    print(f"  LSTM  best : feat='{out['best']['lstm_feature_set']}' "
          f"hp={out['best']['lstm_hparams']} gain={out['best']['lstm_gain_pct']:+.1f}%")
    print(f"  Ridge best : {out['best']['ridge']} gain={out['best']['ridge_gain_pct']:+.1f}%")
    print(f"  SARIMA best: order={out['best']['sarima']['order']} "
          f"seasonal={out['best']['sarima']['seasonal_order']} gain={out['best']['sarima_gain_pct']:+.1f}%")


if __name__ == "__main__":
    main()
