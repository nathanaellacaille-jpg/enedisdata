#!/usr/bin/env python3
"""
scripts/experiment_lgbm_v2.py — Backtest rapide LGBMv2 vs V1 vs Ridge vs naive

5 compteurs x 3 folds, meme protocole que forecast_baseline_metrics.json.
Usage : python scripts/experiment_lgbm_v2.py
"""
import sys, time, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

STEPS = 48
N_METERS = 5
TRAIN_DAYS = 90
N_FOLDS = 3


def mae(a, b):
    """MAE scalaire."""
    return float(np.mean(np.abs(np.asarray(a).ravel() - np.asarray(b).ravel())))


def rolling_backtest(series, fn, train_days=TRAIN_DAYS, n_folds=N_FOLDS):
    """fn(train) -> np.ndarray (STEPS,). Retourne liste de mae."""
    n, pts = len(series), train_days * STEPS
    gap = max(1, (n - pts - STEPS * 2) // max(n_folds - 1, 1))
    results = []
    for fold in range(n_folds):
        ts = pts + fold * gap
        if ts + STEPS > n:
            break
        train, test = series[max(0, ts - pts):ts], series[ts:ts + STEPS]
        if len(train) < STEPS * 8:
            continue
        try:
            pred = fn(train)
            results.append(mae(test, pred))
        except Exception as e:
            print(f"    [fold {fold}] ERREUR: {e}", flush=True)
    return results


def naive_j1(train):
    """Naive J-1."""
    return train[-STEPS:].astype(float)


def lgbm_v1(train):
    """LGBMForecaster V1 (192 lags)."""
    from models.forecaster import LGBMForecaster
    m = LGBMForecaster()
    m.fit(train.astype(float))
    return m.predict(STEPS)


def lgbm_v2(train):
    """LGBMForecasterV2 (29 features domaine)."""
    from models.forecaster import LGBMForecasterV2
    m = LGBMForecasterV2()
    m.fit(train.astype(float))
    return m.predict(STEPS)


def ridge(train):
    """RidgeForecaster."""
    from models.forecaster import RidgeForecaster
    m = RidgeForecaster()
    m.fit(train.astype(float))
    return m.predict(STEPS)


def main():
    """Execute le backtest de validation V2."""
    from utils.parser import parse_timeseries, parse_labels

    print(f"Chargement ({N_METERS} compteurs, {N_FOLDS} folds, train={TRAIN_DAYS}j)...", flush=True)
    df = parse_timeseries(str(ROOT / "RES2-6-9.csv"), max_meters=N_METERS)

    lbl_path = ROOT / "RES2-6-9-labels.csv"
    labels = {}
    if lbl_path.exists():
        try:
            labels = parse_labels(str(lbl_path))
        except Exception:
            pass

    configs = [
        ("lgbm_v2", lgbm_v2),
        ("lgbm_v1", lgbm_v1),
        ("ridge", ridge),
        ("naive_last_day", naive_j1),
    ]

    all_maes = {k: [] for k, _ in configs}

    meters = [m for m in df["meter_id"].unique()
              if (df["meter_id"] == m).sum() > TRAIN_DAYS * STEPS + STEPS * 8]
    meters = meters[:N_METERS]

    for mi, mid in enumerate(meters):
        s = df[df["meter_id"] == mid].sort_values("ts")["kw"].values.astype(float)
        lbl = labels.get(str(mid))
        cls = None if lbl is None else ("rp" if lbl == 0 else "rs")
        print(f"  [{mi+1}/{len(meters)}] {mid}  n={len(s)}  cls={cls or '?'}", flush=True)

        for name, fn in configs:
            t0 = time.time()
            vals = rolling_backtest(s, fn)
            dt = time.time() - t0
            if vals:
                all_maes[name].extend(vals)
                print(f"    {name:<16}: MAE={np.mean(vals):.4f}  t={dt:.1f}s", flush=True)

    print("\n=== Resultats ===", flush=True)
    ref = float(np.mean(all_maes["naive_last_day"])) if all_maes["naive_last_day"] else 1.0
    for name, _ in configs:
        vals = all_maes[name]
        if not vals:
            continue
        m = float(np.mean(vals))
        gain = (ref - m) / ref * 100
        print(f"  {name:<18} MAE={m:.4f}  vs_naive={gain:+.1f}%", flush=True)

    out = {
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"train_days": TRAIN_DAYS, "n_folds": N_FOLDS, "n_meters": N_METERS},
        "results": {
            name: {"mae_mean": round(float(np.mean(v)), 4), "n_obs": len(v)}
            for name, _ in configs
            if (v := all_maes[name])
        },
    }
    out_path = ROOT / "assets" / "lgbm_v2_validation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSauvegarde : {out_path}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTemps total : {time.time()-t0:.1f}s", flush=True)
