#!/usr/bin/env python3
"""
scripts/experiment_lgbm_quick.py — Mini-backtest LightGBM 5 compteurs x 2 folds
(estimation rapide pour calibrer le JSON)

Usage : python scripts/experiment_lgbm_quick.py
"""
import sys, time, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

STEPS = 48
N_METERS = 5
TRAIN_DAYS = 60
N_FOLDS = 2


def mae(a, b):
    """MAE scalaire."""
    return float(np.mean(np.abs(np.asarray(a).ravel() - np.asarray(b).ravel())))


def rolling_backtest(series, fn, train_days=TRAIN_DAYS, n_folds=N_FOLDS):
    """fn(train) -> np.ndarray (STEPS,). Retourne liste de MAE."""
    n, pts = len(series), train_days * STEPS
    gap = max(1, (n - pts - STEPS * 2) // max(n_folds - 1, 1))
    maes = []
    for fold in range(n_folds):
        ts = pts + fold * gap
        if ts + STEPS > n:
            break
        train, test = series[max(0, ts - pts):ts], series[ts:ts + STEPS]
        if len(train) < STEPS * 5:
            continue
        try:
            pred = fn(train)
            maes.append(mae(test, pred))
        except Exception as e:
            print(f"    [fold {fold}] {e}", flush=True)
    return maes


def lgbm_fn(train):
    """LightGBM DMSF."""
    from models.forecaster import LGBMForecaster
    m = LGBMForecaster()
    m.fit(train.astype(float))
    return m.predict(STEPS)


def ridge_fn(train):
    """Ridge autoregressif."""
    from models.forecaster import RidgeForecaster
    m = RidgeForecaster()
    m.fit(train.astype(float))
    return m.predict(STEPS)


def naive_fn(train):
    """Naive J-1."""
    return train[-STEPS:].astype(float)


def run():
    """Mini-backtest et affichage des resultats."""
    from utils.parser import parse_timeseries, parse_labels
    print(f"Chargement ({N_METERS} compteurs, {N_FOLDS} folds)...", flush=True)
    df = parse_timeseries(str(ROOT / "RES2-6-9.csv"), max_meters=N_METERS)

    lbl_path = ROOT / "RES2-6-9-labels.csv"
    labels = parse_labels(str(lbl_path)) if lbl_path.exists() else {}

    configs = {"lgbm": lgbm_fn, "ridge": ridge_fn, "naive_last_day": naive_fn}
    all_maes = {k: [] for k in configs}

    meters = [m for m in df["meter_id"].unique()
              if (df["meter_id"] == m).sum() > TRAIN_DAYS * STEPS + STEPS * 6][:N_METERS]

    for mi, mid in enumerate(meters):
        s = df[df["meter_id"] == mid].sort_values("ts")["kw"].values.astype(float)
        lbl = labels.get(str(mid))
        cls = None if lbl is None else ("rp" if lbl == 0 else "rs")
        print(f"  [{mi+1}/{len(meters)}] {mid}  n={len(s)}  cls={cls or '?'}", flush=True)
        for name, fn in configs.items():
            t0 = time.time()
            fold_maes = rolling_backtest(s, fn)
            dt = time.time() - t0
            if fold_maes:
                all_maes[name].extend(fold_maes)
                print(f"    {name}: MAE={np.mean(fold_maes):.4f}  t={dt:.1f}s", flush=True)

    print("\n=== Resultats ===", flush=True)
    ref = float(np.mean(all_maes["naive_last_day"])) if all_maes["naive_last_day"] else 1.0
    for name, vals in all_maes.items():
        if not vals:
            continue
        m = float(np.mean(vals))
        gain = (ref - m) / ref * 100
        print(f"  {name:<18}  MAE={m:.4f}  vs_naive={gain:+.1f}%", flush=True)

    return all_maes


if __name__ == "__main__":
    t0 = time.time()
    run()
    print(f"\nTemps total : {time.time()-t0:.1f}s", flush=True)
