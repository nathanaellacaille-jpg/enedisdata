import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.parser import parse_timeseries, parse_labels
from utils.metrics import compute_metrics
from models.forecaster import RidgeForecaster, NLinearGlobalForecaster, LGBMForecasterV2, LGBM_V2_LOOKBACK

TS = ROOT / "RES2-6-9.csv"
LBL = ROOT / "RES2-6-9-labels.csv"
OUT = ROOT / "assets" / "forecast_baseline_metrics.json"

N_PER_CLASS = 25
N_FOLDS = 3
TRAIN_DAYS = 60
HORIZON = 48
STEPS = 48
LGBM_METERS = 8
LGBM_TRAIN_DAYS = 90
SEED = 42


def _folds(series, train_pts):
    """Genere (train, test) pour N_FOLDS origines reculant de HORIZON."""
    for f in range(N_FOLDS):
        end = len(series) - f * HORIZON
        test = series[end - HORIZON:end]
        train_full = series[:end - HORIZON]
        if len(train_full) < train_pts:
            continue
        yield train_full[-train_pts:], test


def _naive_last_day(train):
    return train[-STEPS:]


def _naive_weekly(train):
    return train[-7 * STEPS:-6 * STEPS]


def _agg(maes, rmses, per_h):
    """Agrege les metriques d'un modele."""
    per_h_mean = (per_h["sum"] / np.maximum(per_h["cnt"], 1)).round(4).tolist()
    return {
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "rmse_mean": float(np.mean(rmses)),
        "n_obs": len(maes),
        "mae_per_horizon": per_h_mean,
    }


def main():
    print("Chargement...", flush=True)
    t0 = time.time()
    df = parse_timeseries(str(TS), max_meters=None)
    labels = parse_labels(str(LBL))
    present = set(df["meter_id"].astype(str).unique())
    print(f"  {len(present)} compteurs  t={time.time() - t0:.0f}s", flush=True)

    min_pts = TRAIN_DAYS * STEPS + N_FOLDS * HORIZON
    lengths = df.groupby("meter_id", observed=True).size()
    lengths.index = lengths.index.astype(str)
    rng = np.random.default_rng(SEED)

    def _pick(label_val):
        pool = [m for m in present if labels.get(m) == label_val and lengths.get(m, 0) >= min_pts]
        return list(rng.choice(sorted(pool), size=min(N_PER_CLASS, len(pool)), replace=False))

    sel_rs, sel_rp = _pick(1), _pick(0)
    selected = set(sel_rs) | set(sel_rp)
    print(f"  selection : {len(sel_rs)} RS + {len(sel_rp)} RP", flush=True)

    series_by = {}
    for mid, g in df.groupby("meter_id", observed=True):
        m = str(mid)
        if m in selected:
            series_by[m] = g.sort_values("ts")["kw"].values.astype(float)

    models = ["ridge", "nlinear", "naive_last_day"]
    maes = {k: [] for k in models}
    rmses = {k: [] for k in models}
    per_h = {k: {"sum": np.zeros(HORIZON), "cnt": np.zeros(HORIZON)} for k in models}
    weekly_mae = []
    eval_mae = {k: [] for k in models}

    print("Backtest Ridge / NLinear / naive...", flush=True)
    t0 = time.time()
    for m in selected:
        for train, test in _folds(series_by[m], TRAIN_DAYS * STEPS):
            preds = {
                "ridge": RidgeForecaster().fit(train).predict(HORIZON),
                "nlinear": NLinearGlobalForecaster().fit(train).predict(HORIZON),
                "naive_last_day": _naive_last_day(train),
            }
            w_mae = float(np.mean(np.abs(test - _naive_weekly(train))))
            weekly_mae.append(w_mae)
            for k, pred in preds.items():
                mt = compute_metrics(test, pred)
                maes[k].append(mt["MAE"])
                rmses[k].append(mt["RMSE"])
                eval_mae[k].append(mt["MAE"])
                err = np.abs(test - pred)
                per_h[k]["sum"] += err
                per_h[k]["cnt"] += 1
    print(f"  {len(weekly_mae)} evaluations  t={time.time() - t0:.0f}s", flush=True)

    out_models = {k: _agg(maes[k], rmses[k], per_h[k]) for k in models}

    win_rates = {}
    weekly_arr = np.array(weekly_mae)
    for k in models:
        marr = np.array(eval_mae[k])
        gains = (weekly_arr - marr) / np.maximum(weekly_arr, 1e-9) * 100
        win_rates[k] = {
            "vs_naive_weekly": float(np.mean(marr < weekly_arr)),
            "n_compared": len(marr),
            "median_gain_pct": float(np.median(gains)),
        }

    print("Backtest LightGBM (sous-ensemble)...", flush=True)
    t0 = time.time()
    lgbm_meters = sel_rp[:LGBM_METERS]
    l_maes, l_rmses, l_weekly = [], [], []
    l_per_h = {"sum": np.zeros(HORIZON), "cnt": np.zeros(HORIZON)}
    min_lgbm = LGBM_V2_LOOKBACK + STEPS
    for m in lgbm_meters:
        s = series_by[m]
        for f in range(N_FOLDS):
            end = len(s) - f * HORIZON
            test = s[end - HORIZON:end]
            train_full = s[:end - HORIZON]
            train = train_full[-LGBM_TRAIN_DAYS * STEPS:] if len(train_full) >= LGBM_TRAIN_DAYS * STEPS else train_full
            if len(train) < min_lgbm:
                continue
            pred = LGBMForecasterV2().fit(train).predict(HORIZON)
            mt = compute_metrics(test, pred)
            l_maes.append(mt["MAE"])
            l_rmses.append(mt["RMSE"])
            l_weekly.append(float(np.mean(np.abs(test - _naive_weekly(train)))))
            l_per_h["sum"] += np.abs(test - pred)
            l_per_h["cnt"] += 1
    print(f"  {len(l_maes)} evaluations  t={time.time() - t0:.0f}s", flush=True)

    lgbm_block = _agg(l_maes, l_rmses, l_per_h)
    lgbm_block["_note"] = f"LGBMv2 DMSF, {LGBM_METERS} RP x {N_FOLDS} folds, train={LGBM_TRAIN_DAYS}j (cout eleve)"
    out_models["lgbm"] = lgbm_block
    l_marr, l_warr = np.array(l_maes), np.array(l_weekly)
    l_gains = (l_warr - l_marr) / np.maximum(l_warr, 1e-9) * 100
    win_rates["lgbm"] = {
        "vs_naive_weekly": float(np.mean(l_marr < l_warr)),
        "n_compared": len(l_marr),
        "median_gain_pct": float(np.median(l_gains)),
    }

    payload = {
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {
            "n_meters_per_class": N_PER_CLASS,
            "n_folds_per_meter": N_FOLDS,
            "train_window_days": TRAIN_DAYS,
            "horizon_steps": HORIZON,
            "fcst_n_lags": 192,
            "lgbm_meters": LGBM_METERS,
            "lgbm_train_days": LGBM_TRAIN_DAYS,
        },
        "sample": {
            "n_meters": len(selected),
            "n_rs": len(sel_rs),
            "n_rp": len(sel_rp),
            "n_folds_per_meter": N_FOLDS,
            "total_evaluations": len(weekly_mae),
        },
        "models": out_models,
        "win_rates_vs_naive_weekly": win_rates,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Ecrit : {OUT}", flush=True)
    for k, v in sorted(out_models.items(), key=lambda kv: kv[1]["mae_mean"]):
        print(f"  {k:16} MAE={v['mae_mean']:.3f}  (n={v['n_obs']})", flush=True)


if __name__ == "__main__":
    main()
