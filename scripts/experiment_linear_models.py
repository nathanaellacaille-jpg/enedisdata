#!/usr/bin/env python3
"""
scripts/experiment_linear_models.py — Backtest WSN, NLinear, DLinear vs Ridge vs naive

5 compteurs x 3 folds, meme protocole que experiment_lgbm_v2.py.
Usage : python scripts/experiment_linear_models.py
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
# IDs fixes pour reproductibilite — meme echantillon que lgbm_v2_validation.json
METER_IDS = [
    "47000903308", "476866365062", "763769451041", "917755392634", "980841892564",
]
L = 192            # fenetre d'entree NLinear / DLinear
LOOKBACK_WSN = 7 * STEPS  # 336 — 7 slots journaliers pour WSN
RIDGE_LAMBDA = 1e-3
MA_KERNEL = 25


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


def ridge(train):
    """RidgeForecaster (reference)."""
    from models.forecaster import RidgeForecaster
    m = RidgeForecaster()
    m.fit(train.astype(float))
    return m.predict(STEPS)


def _moving_average_rows(X, k=MA_KERNEL):
    """Moving average par ligne, padding bord replique (mode edge)."""
    pad = k // 2
    X_pad = np.pad(X, ((0, 0), (pad, pad)), mode='edge')
    cs = np.cumsum(X_pad, axis=1)
    cs = np.hstack([np.zeros((X.shape[0], 1)), cs])
    return (cs[:, k:] - cs[:, :-k]) / k  # (n_samp, L)


def weighted_seasonal_naive(train):
    """WSN: 48 RidgeCV independants, 7 features = meme slot sur les 7 jours passes."""
    from sklearn.linear_model import RidgeCV
    alphas = np.logspace(-4, 3, 20)
    n = len(train)
    if n < LOOKBACK_WSN + STEPS:
        raise ValueError(f"Serie trop courte: {n}")

    # origins: indices de debut de fenetre de test
    origins = np.arange(LOOKBACK_WSN, n - STEPS + 1)
    day_offsets = np.arange(1, 8) * STEPS  # [48, 96, ..., 336]
    preds = np.zeros(STEPS)

    for h in range(STEPS):
        X = np.stack([train[origins + h - d] for d in day_offsets], axis=1)  # (n_samp, 7)
        y_h = train[origins + h]
        mdl = RidgeCV(alphas=alphas)
        mdl.fit(X, y_h)
        x_pred = np.array([train[n + h - d] for d in day_offsets]).reshape(1, -1)
        preds[h] = float(mdl.predict(x_pred)[0])

    return preds


def nlinear(train):
    """NLinear MIMO: soustrait window[-1], lstsq + ridge manuel λ=1e-3, L=192 lags."""
    n = len(train)
    if n < L + STEPS:
        raise ValueError(f"Serie trop courte: {n}")

    n_samp = n - L - STEPS + 1
    i_win = np.arange(n_samp)[:, None] + np.arange(L)[None, :]
    X_raw = train[i_win].astype(float)        # (n_samp, L)
    last = X_raw[:, -1:]                       # (n_samp, 1) — valeur de reference

    X = X_raw - last                           # centrage NLinear
    i_tgt = np.arange(n_samp)[:, None] + np.arange(L, L + STEPS)[None, :]
    Y = train[i_tgt].astype(float) - last     # cibles centrees (n_samp, STEPS)

    # W = (X.T X + λI)^{-1} X.T Y
    A = X.T @ X + RIDGE_LAMBDA * np.eye(L)
    W = np.linalg.solve(A, X.T @ Y)           # (L, STEPS)

    window = train[-L:].astype(float)
    return W.T @ (window - window[-1]) + window[-1]


def dlinear(train):
    """DLinear MIMO: trend+seasonal decompos, deux projections ridge λ=1e-3 summees."""
    n = len(train)
    if n < L + STEPS:
        raise ValueError(f"Serie trop courte: {n}")

    n_samp = n - L - STEPS + 1
    i_win = np.arange(n_samp)[:, None] + np.arange(L)[None, :]
    X_raw = train[i_win].astype(float)         # (n_samp, L)

    trend = _moving_average_rows(X_raw)        # (n_samp, L)
    seasonal = X_raw - trend                   # (n_samp, L)

    # Centrage coherent : trend[-1]+seasonal[-1] = window[-1]
    trend_last = trend[:, -1:]
    seas_last = seasonal[:, -1:]
    center_ref = X_raw[:, -1:]                 # = trend_last + seas_last

    X_full = np.hstack([trend - trend_last, seasonal - seas_last])  # (n_samp, 2L)
    i_tgt = np.arange(n_samp)[:, None] + np.arange(L, L + STEPS)[None, :]
    Y = train[i_tgt].astype(float) - center_ref  # (n_samp, STEPS)

    two_L = 2 * L
    A = X_full.T @ X_full + RIDGE_LAMBDA * np.eye(two_L)
    W = np.linalg.solve(A, X_full.T @ Y)       # (2L, STEPS)

    window = train[-L:].astype(float)
    trend_w = np.convolve(window, np.ones(MA_KERNEL) / MA_KERNEL, mode='same')
    seas_w = window - trend_w
    x_pred = np.concatenate([trend_w - trend_w[-1], seas_w - seas_w[-1]])
    return W.T @ x_pred + window[-1]


def main():
    """Execute le backtest des modeles lineaires."""
    from utils.parser import parse_timeseries, parse_labels

    print(f"Chargement ({N_METERS} compteurs, {N_FOLDS} folds, train={TRAIN_DAYS}j)...", flush=True)
    df = parse_timeseries(str(ROOT / "RES2-6-9.csv"), max_meters=None)

    lbl_path = ROOT / "RES2-6-9-labels.csv"
    labels = {}
    if lbl_path.exists():
        try:
            labels = parse_labels(str(lbl_path))
        except Exception:
            pass

    configs = [
        ("wsn", weighted_seasonal_naive),
        ("nlinear", nlinear),
        ("dlinear", dlinear),
        ("ridge", ridge),
        ("naive_last_day", naive_j1),
    ]

    all_maes = {k: [] for k, _ in configs}

    meters = [m for m in METER_IDS if m in df["meter_id"].values][:N_METERS]

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
                print(f"    {name:<22}: MAE={np.mean(vals):.4f}  t={dt:.1f}s", flush=True)

    print("\n=== Resultats ===", flush=True)
    ref = float(np.mean(all_maes["naive_last_day"])) if all_maes["naive_last_day"] else 1.0
    for name, _ in configs:
        vals = all_maes[name]
        if not vals:
            continue
        m = float(np.mean(vals))
        gain = (ref - m) / ref * 100
        print(f"  {name:<22} MAE={m:.4f}  vs_naive={gain:+.1f}%", flush=True)

    out = {
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "train_days": TRAIN_DAYS, "n_folds": N_FOLDS, "n_meters": N_METERS,
            "L": L, "ridge_lambda": RIDGE_LAMBDA, "ma_kernel": MA_KERNEL,
        },
        "results": {
            name: {"mae_mean": round(float(np.mean(v)), 4), "n_obs": len(v)}
            for name, _ in configs
            if (v := all_maes[name])
        },
    }
    out_path = ROOT / "assets" / "linear_models_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSauvegarde : {out_path}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTemps total : {time.time()-t0:.1f}s", flush=True)
