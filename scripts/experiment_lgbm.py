#!/usr/bin/env python3
"""
scripts/experiment_lgbm.py — Backtest LightGBM DMSF vs Ridge et baselines

Mesure la MAE sur 50 compteurs x 3 folds rolling (meme protocole que
forecast_baseline_metrics.json Phase 0 v2).

Usage : python scripts/experiment_lgbm.py
Sortie : assets/lgbm_backtest_results.json
"""
import sys, time, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

STEPS = 48
N_METERS = 50
TRAIN_DAYS = 60
N_FOLDS = 3


def mae(a, b):
    """MAE scalaire."""
    return float(np.mean(np.abs(np.asarray(a).ravel() - np.asarray(b).ravel())))


def rolling_backtest(series, fn, train_days=TRAIN_DAYS, n_folds=N_FOLDS):
    """fn(train) -> np.ndarray (STEPS,). Retourne liste de (mae, mae_per_horizon)."""
    n, pts = len(series), train_days * STEPS
    gap = max(1, (n - pts - STEPS * 2) // max(n_folds - 1, 1))
    results = []
    for fold in range(n_folds):
        ts = pts + fold * gap
        if ts + STEPS > n:
            break
        train, test = series[max(0, ts - pts):ts], series[ts:ts + STEPS]
        if len(train) < STEPS * 5:
            continue
        try:
            pred = fn(train)
            mae_val = mae(test, pred)
            mae_per_h = [float(abs(test[h] - pred[h])) for h in range(min(STEPS, len(pred)))]
            results.append((mae_val, mae_per_h))
        except Exception as e:
            print(f"    [fold {fold}] {e}", flush=True)
    return results


def naive_j1_fn(train):
    """Naive J-1 : repete le dernier jour connu."""
    return train[-STEPS:].astype(float)


def lgbm_dmsf_fn(train):
    """LightGBM DMSF via LGBMForecaster."""
    from models.forecaster import LGBMForecaster
    m = LGBMForecaster()
    m.fit(train.astype(float))
    return m.predict(STEPS)


def ridge_fn(train):
    """Ridge autoregressif via RidgeForecaster."""
    from models.forecaster import RidgeForecaster
    m = RidgeForecaster()
    m.fit(train.astype(float))
    return m.predict(STEPS)


def run_all():
    """Execute le backtest complet et sauvegarde les resultats."""
    from utils.parser import parse_timeseries, parse_labels

    print("Chargement donnees...", flush=True)
    df = parse_timeseries(str(ROOT / "RES2-6-9.csv"), max_meters=N_METERS)

    lbl_path = ROOT / "RES2-6-9-labels.csv"
    labels = {}
    if lbl_path.exists():
        try:
            labels = parse_labels(str(lbl_path))  # {meter_id_str: 0|1}
        except Exception as e:
            print(f"  Labels non charges: {e}", flush=True)

    configs = {
        "lgbm": lgbm_dmsf_fn,
        "ridge": ridge_fn,
        "naive_last_day": naive_j1_fn,
    }

    all_maes = {k: [] for k in configs}
    all_per_h = {k: [] for k in configs}
    per_class = {"rs": {k: [] for k in configs}, "rp": {k: [] for k in configs}}

    meters = [m for m in df["meter_id"].unique()
              if (df["meter_id"] == m).sum() > TRAIN_DAYS * STEPS + STEPS * 6]
    meters = meters[:N_METERS]

    for mi, mid in enumerate(meters):
        s = df[df["meter_id"] == mid].sort_values("ts")["kw"].values.astype(float)

        # Determiner la classe RS ou RP (0=RP, 1=RS)
        lbl = labels.get(str(mid))
        cls = None if lbl is None else ("rp" if lbl == 0 else "rs")

        print(f"  [{mi+1:2d}/{len(meters)}] {mid}  n={len(s)}  cls={cls or '?'}",
              flush=True)

        for name, fn in configs.items():
            t0 = time.time()
            fold_results = rolling_backtest(s, fn)
            dt = time.time() - t0
            if fold_results:
                fold_maes = [r[0] for r in fold_results]
                fold_per_h = [r[1] for r in fold_results]
                all_maes[name].extend(fold_maes)
                all_per_h[name].extend(fold_per_h)
                if cls:
                    per_class[cls][name].extend(fold_maes)
                print(f"    {name}: MAE={np.mean(fold_maes):.4f}  t={dt:.1f}s",
                      flush=True)

    # Synthese
    print("\n" + "=" * 70, flush=True)
    print(f"{'Modele':<18} {'MAE moy':>9} {'vs naive':>10} {'% gain':>9}", flush=True)
    print("-" * 70, flush=True)
    ref = float(np.mean(all_maes["naive_last_day"])) if all_maes["naive_last_day"] else 1.0
    for name, vals in all_maes.items():
        if not vals:
            continue
        m = float(np.mean(vals))
        gain = (ref - m) / ref * 100
        print(f"  {name:<16}  {m:>8.4f}   {ref:>8.4f}  {gain:>+8.1f}%", flush=True)
    print("=" * 70, flush=True)

    # Construction du JSON de sortie
    output = {
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "n_folds_per_meter": N_FOLDS,
            "train_window_days": TRAIN_DAYS,
            "horizon_steps": STEPS,
        },
        "models": {},
        "per_class": {"rs": {}, "rp": {}},
        "win_rates": {},
    }

    naive_maes = all_maes.get("naive_last_day", [])
    for name, vals in all_maes.items():
        if not vals:
            continue
        per_h_mat = np.array(all_per_h[name])
        output["models"][name] = {
            "mae_mean": float(np.mean(vals)),
            "mae_std": float(np.std(vals)),
            "n_obs": len(vals),
            "mae_per_horizon": [round(float(per_h_mat[:, h].mean()), 4)
                                 for h in range(per_h_mat.shape[1])],
        }
        for cls in ("rs", "rp"):
            cls_vals = per_class[cls][name]
            if cls_vals:
                output["per_class"][cls][name] = {
                    "mae_mean": float(np.mean(cls_vals)),
                    "n_obs": len(cls_vals),
                }
        # Win rate vs naive
        if naive_maes and name != "naive_last_day":
            wins = sum(v < n for v, n in zip(vals, naive_maes))
            output["win_rates"][name] = {
                "vs_naive_last_day": wins / len(vals),
                "n_compared": len(vals),
            }

    out_path = ROOT / "assets" / "lgbm_backtest_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nResultats sauvegardes dans : {out_path}", flush=True)
    return output


if __name__ == "__main__":
    t0 = time.time()
    run_all()
    print(f"\nTemps total : {time.time()-t0:.1f}s", flush=True)
