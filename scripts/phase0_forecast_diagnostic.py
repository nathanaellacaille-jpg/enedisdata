"""Phase 0 - Diagnostic chiffre de la prevision 24h.

Backtest rolling sur un echantillon stratifie RS/RP, sur 3 folds walk-forward.
Compare Ridge, ARIMA, LSTM contre 4 baselines naives :
  - naive_persistence : h+i = derniere valeur connue
  - naive_last_day    : repete le dernier jour
  - naive_weekly      : meme creneau la semaine derniere
  - seasonal_mean     : moyenne historique (dow, slot)

Sorties :
  - assets/forecast_baseline_metrics.json : dump chiffre lu par la page 2
  - stdout : verdict synthetique

Usage : python scripts/phase0_forecast_diagnostic.py [--n-meters N] [--n-folds K]
                                                    [--train-window-days D] [--lstm-epochs E]
"""

from __future__ import annotations

import argparse
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
    DEFAULT_LBL_PATH,
    DEFAULT_TS_PATH,
    FCST_ARIMA_ORDER,
    FCST_HORIZON_H,
    FCST_N_LAGS,
    STEPS_PER_DAY,
)
from models.forecaster import ARIMAForecaster, LSTMForecaster, RidgeForecaster
from utils.features import extract_features
from utils.parser import parse_labels, parse_timeseries

HORIZON = FCST_HORIZON_H * 2  # 48 demi-heures
WEEK_STEPS = STEPS_PER_DAY * 7


# === Metriques ============================================================

def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """WAPE = sum|err| / sum|y|, plus stable que MAPE quand y proche de zero."""
    denom = float(np.sum(np.abs(y_true)))
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100) if denom > 1e-9 else float("nan")


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sMAPE symetrique, defini meme quand y_true contient des zeros."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 1e-9
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE classique, exclut les y_true == 0."""
    nonzero = np.abs(y_true) > 1e-9
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)


def _metrics_per_step(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Retourne MAE, RMSE, MAPE, sMAPE, WAPE, R2 + vecteur d'erreur absolue par pas."""
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-9 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": _mape(y_true, y_pred),
        "smape": _smape(y_true, y_pred),
        "wape": _wape(y_true, y_pred),
        "r2": r2,
        "abs_err_per_step": np.abs(err),
    }


# === Baselines naives =====================================================

def _baseline_persistence(train_kw: np.ndarray, h: int, _train_keys: np.ndarray, _test_keys: np.ndarray) -> np.ndarray:
    """h+i = derniere valeur connue."""
    return np.full(h, train_kw[-1], dtype=float)


def _baseline_last_day(train_kw: np.ndarray, h: int, _train_keys: np.ndarray, _test_keys: np.ndarray) -> np.ndarray:
    """Repete le dernier jour connu."""
    day = train_kw[-STEPS_PER_DAY:]
    reps = (h // STEPS_PER_DAY) + 1
    return np.tile(day, reps)[:h].astype(float)


def _baseline_weekly(train_kw: np.ndarray, h: int, _train_keys: np.ndarray, _test_keys: np.ndarray) -> np.ndarray:
    """Reutilise les valeurs observees a la meme heure 7 jours plus tot."""
    if len(train_kw) >= WEEK_STEPS:
        return train_kw[-WEEK_STEPS:-WEEK_STEPS + h].astype(float)
    return _baseline_last_day(train_kw, h, _train_keys, _test_keys)


def _baseline_seasonal_mean(train_kw: np.ndarray, h: int, train_keys: np.ndarray, test_keys: np.ndarray) -> np.ndarray:
    """Moyenne historique par (jour-de-semaine, slot). Fallback : moyenne globale."""
    global_mean = float(train_kw.mean()) if len(train_kw) > 0 else 0.0
    means = pd.Series(train_kw).groupby(train_keys).mean()
    return np.array([means.get(int(k), global_mean) for k in test_keys], dtype=float)


BASELINES = {
    "naive_persistence": _baseline_persistence,
    "naive_last_day": _baseline_last_day,
    "naive_weekly": _baseline_weekly,
    "seasonal_mean": _baseline_seasonal_mean,
}


# === Echantillonnage compteurs ============================================

def _stratified_sample(labels: dict, available_ids: set, n_per_class: int, rng: np.random.Generator) -> list:
    """Tire un echantillon stratifie {n RS, n RP} parmi les ids disponibles."""
    rs = [m for m in available_ids if labels.get(m) == 1]
    rp = [m for m in available_ids if labels.get(m) == 0]
    rng.shuffle(rs)
    rng.shuffle(rp)
    selected = rs[:n_per_class] + rp[:n_per_class]
    rng.shuffle(selected)
    return selected


# === Backtest par compteur =================================================

def _fold_split(n_points: int, n_folds: int, horizon: int, train_window: int | None) -> list[tuple[int, int, int]]:
    """Retourne la liste (train_start, train_end, test_end) pour chaque fold walk-forward.

    Fold k (k=0..K-1) : test = points [test_end-h .. test_end[, train = points [train_start .. test_end-h[.
    test_end pour fold k = n - (K-1-k)*h, fold final fini sur la derniere observation.
    """
    folds = []
    for k in range(n_folds):
        test_end = n_points - (n_folds - 1 - k) * horizon
        test_start = test_end - horizon
        train_end = test_start
        if train_window is not None:
            train_start = max(0, train_end - train_window)
        else:
            train_start = 0
        if train_end - train_start < FCST_N_LAGS + STEPS_PER_DAY:
            continue
        folds.append((train_start, train_end, test_end))
    return folds


def _run_one_fold(
    meter_df: pd.DataFrame,
    train_start: int,
    train_end: int,
    test_end: int,
    horizon: int,
    lstm_epochs: int | None,
) -> dict:
    """Entraine les 3 modeles + 4 baselines sur un fold, retourne les metriques par modele."""
    train_df = meter_df.iloc[train_start:train_end]
    test_df = meter_df.iloc[train_end:test_end]
    train_kw = train_df["kw"].values.astype(float)
    test_kw = test_df["kw"].values.astype(float)

    train_dow = train_df["ts"].dt.dayofweek.values
    train_slot = (train_df["ts"].dt.hour * 2 + train_df["ts"].dt.minute // 30).values
    test_dow = test_df["ts"].dt.dayofweek.values
    test_slot = (test_df["ts"].dt.hour * 2 + test_df["ts"].dt.minute // 30).values
    train_keys = train_dow * 48 + train_slot
    test_keys = test_dow * 48 + test_slot

    preds: dict[str, np.ndarray] = {}
    train_mae: dict[str, float] = {}

    # Baselines
    for name, fn in BASELINES.items():
        preds[name] = fn(train_kw, horizon, train_keys, test_keys)

    # Ridge
    try:
        ridge = RidgeForecaster()
        ridge.fit(train_kw)
        preds["ridge"] = ridge.predict(horizon)
        # MAE in-sample pour mesurer le sur-apprentissage
        X_tr, y_tr = ridge._build_X(train_kw)
        train_mae["ridge"] = float(np.mean(np.abs(ridge._model.predict(X_tr) - y_tr)))
    except Exception as exc:  # noqa: BLE001
        preds["ridge"] = np.full(horizon, np.nan)
        train_mae["ridge"] = float("nan")
        print(f"    Ridge fail: {exc}")

    # ARIMA (silencieux : convergence warnings frequents)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            arima = ARIMAForecaster()
            arima.fit(train_kw, order=FCST_ARIMA_ORDER)
            preds["arima"] = arima.predict(horizon)
        except Exception as exc:  # noqa: BLE001
            preds["arima"] = np.full(horizon, np.nan)
            print(f"    ARIMA fail: {exc}")

    # LSTM
    try:
        lstm = LSTMForecaster()
        if lstm_epochs is not None:
            from config import LSTM_EPOCHS  # noqa: F401
            import models.forecaster as fc_mod
            old_eps = fc_mod.LSTM_EPOCHS
            fc_mod.LSTM_EPOCHS = lstm_epochs
            try:
                lstm.fit(train_kw)
            finally:
                fc_mod.LSTM_EPOCHS = old_eps
        else:
            lstm.fit(train_kw)
        preds["lstm"] = lstm.predict(horizon)
        # Approx train MAE : derniere loss (MSE sur serie normalisee) reconvertie en MAE kW
        if lstm.losses:
            last_mse_norm = lstm.losses[-1]
            train_mae["lstm"] = float(np.sqrt(last_mse_norm) * lstm._scaler_std)
        else:
            train_mae["lstm"] = float("nan")
    except Exception as exc:  # noqa: BLE001
        preds["lstm"] = np.full(horizon, np.nan)
        train_mae["lstm"] = float("nan")
        print(f"    LSTM fail: {exc}")

    out = {"test_kw": test_kw, "test_mean": float(np.mean(np.abs(test_kw))), "preds": {}, "train_mae": train_mae}
    for name, pred in preds.items():
        if np.isnan(pred).any():
            out["preds"][name] = None
            continue
        out["preds"][name] = _metrics_per_step(test_kw, pred)
    return out


# === Main ==================================================================

def main() -> None:
    """Execute le diagnostic Phase 0 et ecrit assets/forecast_baseline_metrics.json."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-meters-per-class", type=int, default=25,
                        help="Nb compteurs RS et RP a echantillonner (max ~72 RS dispos).")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--train-window-days", type=int, default=60,
                        help="Fenetre d'entrainement (jours) avant chaque fold. 0 = tout l'historique. "
                             "Production utilise 0 ; on plafonne ici pour tenir en <30 min sur CPU.")
    parser.add_argument("--lstm-epochs", type=int, default=15,
                        help="Epochs LSTM (prod = 40). Reduit pour la duree du diagnostic.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ts-path", type=str, default=str(DEFAULT_TS_PATH))
    parser.add_argument("--lbl-path", type=str, default=str(DEFAULT_LBL_PATH))
    args = parser.parse_args()

    print("=" * 72)
    print("PHASE 0 - DIAGNOSTIC PREVISION 24H")
    print("=" * 72)
    print(f"  n_meters_per_class : {args.n_meters_per_class}")
    print(f"  n_folds            : {args.n_folds}")
    print(f"  train_window_days  : {args.train_window_days} (0 = full)")
    print(f"  lstm_epochs        : {args.lstm_epochs}")
    print(f"  seed               : {args.seed}")

    # 1. Chargement
    print("\n[1/5] Chargement des donnees...")
    t0 = time.time()
    ts_path = Path(args.ts_path)
    lbl_path = Path(args.lbl_path)
    if not ts_path.exists():
        print(f"  ERREUR : {ts_path} introuvable.")
        sys.exit(1)
    df = parse_timeseries(str(ts_path), max_meters=None)
    labels = parse_labels(str(lbl_path))
    print(f"  Timeseries : {df['meter_id'].nunique()} compteurs, {len(df):,} points "
          f"(parse en {time.time() - t0:.1f}s)")

    # 2. Echantillonnage
    rng = np.random.default_rng(args.seed)
    series_lens = df.groupby("meter_id", observed=True).size()
    min_len = FCST_N_LAGS + HORIZON * (args.n_folds + 1) + STEPS_PER_DAY
    eligible_ids = set(series_lens[series_lens >= min_len].index.astype(str))
    eligible_ids &= set(labels.keys())
    print(f"\n[2/5] Echantillonnage stratifie")
    print(f"  Compteurs avec >= {min_len} points et label : {len(eligible_ids)}")
    selected = _stratified_sample(labels, eligible_ids, args.n_meters_per_class, rng)
    n_rs = sum(1 for m in selected if labels[m] == 1)
    n_rp = sum(1 for m in selected if labels[m] == 0)
    print(f"  Echantillon retenu : {len(selected)} compteurs ({n_rs} RS, {n_rp} RP)")

    # 3. Features (pour caracteriser les modes d'echec)
    print("\n[3/5] Extraction features (pour caracterisation des erreurs)...")
    t0 = time.time()
    sample_df = df[df["meter_id"].astype(str).isin(selected)].copy()
    features = extract_features(sample_df)
    features.index = features.index.astype(str)
    print(f"  Features : {features.shape[0]} compteurs x {features.shape[1]} colonnes "
          f"({time.time() - t0:.1f}s)")

    # 4. Backtest
    print("\n[4/5] Backtest rolling...")
    train_window_pts = args.train_window_days * STEPS_PER_DAY if args.train_window_days > 0 else None

    model_names = ["ridge", "arima", "lstm"] + list(BASELINES.keys())
    # Aggregats globaux et par classe : liste de (mae, rmse, ...) par fold
    rows: list[dict] = []
    horizon_err: dict[str, list[np.ndarray]] = {n: [] for n in model_names}

    t_global = time.time()
    for i, meter_id in enumerate(selected):
        meter_df = sample_df[sample_df["meter_id"].astype(str) == meter_id].sort_values("ts").reset_index(drop=True)
        n_pts = len(meter_df)
        folds = _fold_split(n_pts, args.n_folds, HORIZON, train_window_pts)
        if not folds:
            continue
        label = labels.get(meter_id, -1)
        t_meter = time.time()
        for fold_idx, (tr_s, tr_e, te_e) in enumerate(folds):
            fold_res = _run_one_fold(meter_df, tr_s, tr_e, te_e, HORIZON, args.lstm_epochs)
            for name in model_names:
                m = fold_res["preds"].get(name)
                if m is None:
                    continue
                rows.append({
                    "meter_id": meter_id,
                    "label": label,
                    "fold": fold_idx,
                    "model": name,
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "mape": m["mape"],
                    "smape": m["smape"],
                    "wape": m["wape"],
                    "r2": m["r2"],
                    "test_mean": fold_res["test_mean"],
                    "train_mae": fold_res["train_mae"].get(name, float("nan")),
                })
                horizon_err[name].append(m["abs_err_per_step"])
        print(f"  [{i + 1:3d}/{len(selected)}] {meter_id} (label={label}) "
              f"folds={len(folds)}  {time.time() - t_meter:.1f}s  "
              f"(total {time.time() - t_global:.0f}s)")
    print(f"\n  Backtest termine en {time.time() - t_global:.0f}s")

    # 5. Synthese
    print("\n[5/5] Synthese...")
    df_res = pd.DataFrame(rows)
    if df_res.empty:
        print("  Aucun resultat exploitable.")
        sys.exit(1)

    # Agregats par modele
    agg_models = {}
    for name in model_names:
        sub = df_res[df_res["model"] == name]
        if sub.empty:
            continue
        agg_models[name] = {
            "mae_mean": float(sub["mae"].mean()),
            "mae_std": float(sub["mae"].std(ddof=0)),
            "rmse_mean": float(sub["rmse"].mean()),
            "wape_mean": float(sub["wape"].mean()),
            "smape_mean": float(sub["smape"].mean()),
            "r2_mean": float(sub["r2"].mean()),
            "n_obs": int(len(sub)),
        }
        # MAE par pas d'horizon
        if horizon_err[name]:
            arr = np.vstack(horizon_err[name])
            agg_models[name]["mae_per_horizon"] = arr.mean(axis=0).round(4).tolist()

    # Agregats par classe
    per_class = {}
    for cls_key, cls_label in [("rs", 1), ("rp", 0)]:
        per_class[cls_key] = {}
        for name in model_names:
            sub = df_res[(df_res["model"] == name) & (df_res["label"] == cls_label)]
            if sub.empty:
                continue
            per_class[cls_key][name] = {
                "mae_mean": float(sub["mae"].mean()),
                "wape_mean": float(sub["wape"].mean()),
                "smape_mean": float(sub["smape"].mean()),
                "n_obs": int(len(sub)),
            }

    # Taux de battement vs naive_weekly (par fold, on regarde si MAE < MAE_naive_weekly)
    win_rates = {}
    pivot = df_res.pivot_table(index=["meter_id", "fold"], columns="model", values="mae")
    if "naive_weekly" in pivot.columns:
        for name in model_names:
            if name == "naive_weekly" or name not in pivot.columns:
                continue
            both = pivot[[name, "naive_weekly"]].dropna()
            if len(both) == 0:
                continue
            win_rates[name] = {
                "vs_naive_weekly": float((both[name] < both["naive_weekly"]).mean()),
                "n_compared": int(len(both)),
                "median_gain_pct": float(((both["naive_weekly"] - both[name]) / both["naive_weekly"] * 100).median()),
            }

    # Worst cases : par modele, top 5 meter_ids ayant le plus grand MAE / test_mean
    worst_cases: dict[str, list] = {}
    for name in model_names:
        sub = df_res[df_res["model"] == name].copy()
        if sub.empty:
            continue
        sub["norm_mae"] = sub["mae"] / sub["test_mean"].replace(0, np.nan)
        worst = (sub.groupby("meter_id", observed=True).agg({"norm_mae": "mean", "mae": "mean", "test_mean": "mean", "label": "first"})
                    .sort_values("norm_mae", ascending=False).head(5))
        worst_cases[name] = []
        for mid, row in worst.iterrows():
            feat = features.loc[mid] if mid in features.index else None
            worst_cases[name].append({
                "meter_id": mid,
                "label": int(row["label"]),
                "norm_mae": float(row["norm_mae"]),
                "mae_kw": float(row["mae"]),
                "test_mean_kw": float(row["test_mean"]),
                "zero_ratio": float(feat["zero_ratio"]) if feat is not None else None,
                "max_gap_days": float(feat["max_gap_days"]) if feat is not None else None,
                "ratio_we_wd": float(feat["ratio_we_wd"]) if feat is not None else None,
                "active_days_ratio": float(feat["active_days_ratio"]) if feat is not None else None,
            })

    # Indicateur d'overfit LSTM
    lstm_overfit = None
    sub_lstm = df_res[df_res["model"] == "lstm"]
    if not sub_lstm.empty and sub_lstm["train_mae"].notna().any():
        tr = float(sub_lstm["train_mae"].mean())
        te = float(sub_lstm["mae"].mean())
        lstm_overfit = {
            "train_mae_mean": tr,
            "test_mae_mean": te,
            "ratio_test_over_train": float(te / tr) if tr > 1e-9 else None,
        }

    # --- Verdict imprime ---
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"\n  {'Modele':<22s} {'MAE':>8s} {'WAPE%':>8s} {'sMAPE%':>8s} {'R2':>7s}")
    print(f"  {'-' * 22} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 7}")
    sorted_models = sorted(agg_models.items(), key=lambda kv: kv[1]["mae_mean"])
    for name, m in sorted_models:
        print(f"  {name:<22s} {m['mae_mean']:>8.3f} {m['wape_mean']:>8.2f} {m['smape_mean']:>8.2f} {m['r2_mean']:>7.3f}")

    best_overall = sorted_models[0][0] if sorted_models else None
    print(f"\n  Meilleur modele (MAE global) : {best_overall}")

    if win_rates:
        print(f"\n  Taux de victoire vs naive_weekly (gain MAE > 0) :")
        for name, w in win_rates.items():
            print(f"    {name:<22s} {w['vs_naive_weekly'] * 100:5.1f}%  "
                  f"(gain median {w['median_gain_pct']:+.1f}%, n={w['n_compared']})")

    if lstm_overfit and lstm_overfit["ratio_test_over_train"] is not None:
        ratio = lstm_overfit["ratio_test_over_train"]
        flag = "OVERFIT" if ratio > 2.0 else "ok"
        print(f"\n  LSTM train MAE = {lstm_overfit['train_mae_mean']:.3f}  "
              f"test MAE = {lstm_overfit['test_mae_mean']:.3f}  ratio = {ratio:.2f}  -> {flag}")

    print(f"\n  Erreur par classe (MAE moyenne) :")
    print(f"    {'Modele':<22s} {'RS':>8s} {'RP':>8s} {'ecart':>8s}")
    for name in [m for m, _ in sorted_models]:
        rs_m = per_class.get("rs", {}).get(name, {}).get("mae_mean")
        rp_m = per_class.get("rp", {}).get(name, {}).get("mae_mean")
        if rs_m is None or rp_m is None:
            continue
        print(f"    {name:<22s} {rs_m:>8.3f} {rp_m:>8.3f} {rs_m - rp_m:>+8.3f}")

    # --- Dump JSON ---
    out = ROOT / "assets" / "forecast_baseline_metrics.json"
    out.parent.mkdir(exist_ok=True)
    payload = {
        "computed_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {
            "n_meters_per_class": args.n_meters_per_class,
            "n_folds_per_meter": args.n_folds,
            "train_window_days": args.train_window_days,
            "horizon_steps": HORIZON,
            "lstm_epochs": args.lstm_epochs,
            "fcst_n_lags": FCST_N_LAGS,
            "fcst_arima_order": list(FCST_ARIMA_ORDER),
        },
        "sample": {
            "n_meters": len(selected),
            "n_rs": n_rs,
            "n_rp": n_rp,
            "n_folds_per_meter": args.n_folds,
            "total_evaluations": int(len(df_res) / max(len(model_names), 1)),
        },
        "models": agg_models,
        "per_class": per_class,
        "win_rates_vs_naive_weekly": win_rates,
        "worst_cases": worst_cases,
        "lstm_overfit": lstm_overfit,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\nMetriques baseline ecrites dans {out}")


if __name__ == "__main__":
    main()
