#!/usr/bin/env python3
"""
scripts/experiment_nlinear_global.py — NLinear global (500 compteurs) vs local vs zeroshot

Protocol par fold sur les 5 compteurs de reference :
  nlinear_local    : W entraine sur le seul compteur cible (meme que experiment_linear_models)
  nlinear_global   : W entraine sur les 500 compteurs (dont la cible) — "pooled"
  nlinear_zeroshot : W entraine sur les 499 autres, zero-shot sur la cible (leave-one-out)
  ridge            : RidgeForecaster local (reference)
  naive_last_day   : baseline

Le global XtX/XtY est accumule de facon incrementale (O(L^2) memoire fixe),
un solve (192x192) suffit pour les 48 horizons.
Usage : python scripts/experiment_nlinear_global.py
"""
import sys, time, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

STEPS = 48
TRAIN_DAYS = 90
N_FOLDS = 3
L = 192
RIDGE_LAMBDA = 1e-3
# IDs de reference fixes — meme echantillon que lgbm_v2_validation.json
METER_IDS = [
    "47000903308", "476866365062", "763769451041", "917755392634", "980841892564",
]
# Gap de rolling backtest calcule pour n=17472 (longueur des compteurs de reference)
_PTS = TRAIN_DAYS * STEPS          # 4320
_N_REF = 17472
_GAP = max(1, (_N_REF - _PTS - STEPS * 2) // max(N_FOLDS - 1, 1))  # 6528


def mae(a, b):
    """MAE scalaire."""
    return float(np.mean(np.abs(np.asarray(a).ravel() - np.asarray(b).ravel())))


def _local_matrices(train: np.ndarray):
    """Calcule XtX et XtY pour une serie d'entrainement (NLinear centrage window[-1])."""
    n = len(train)
    n_samp = n - L - STEPS + 1
    if n_samp <= 0:
        return None, None
    i_win = np.arange(n_samp)[:, None] + np.arange(L)[None, :]
    X_raw = train[i_win].astype(np.float64)
    last = X_raw[:, -1:]
    X = X_raw - last
    i_tgt = np.arange(n_samp)[:, None] + np.arange(L, L + STEPS)[None, :]
    Y = train[i_tgt].astype(np.float64) - last
    return X.T @ X, X.T @ Y   # (L,L), (L,STEPS)


def _solve(XtX, XtY):
    """Ridge solve : W = (XtX + λI)^{-1} XtY."""
    return np.linalg.solve(XtX + RIDGE_LAMBDA * np.eye(L), XtY)  # (L, STEPS)


def _predict(W, window):
    """Prediction NLinear : W.T @ (window - window[-1]) + window[-1]."""
    x = window.astype(np.float64) - window[-1]
    return W.T @ x + window[-1]


def accumulate_global(df, all_meters, fold):
    """Accumule XtX/XtY sur tous les compteurs pour un fold donne."""
    XtX = np.zeros((L, L))
    XtY = np.zeros((L, STEPS))
    n_ok = 0
    for mid in all_meters:
        s = df[df["meter_id"] == mid].sort_values("ts")["kw"].values.astype(np.float32)
        n = len(s)
        ts = _PTS + fold * _GAP
        if ts > n - STEPS:
            ts = n - STEPS
        if ts < _PTS:
            continue
        train = s[max(0, ts - _PTS):ts].astype(np.float64)
        xtx, xty = _local_matrices(train)
        if xtx is None:
            continue
        XtX += xtx
        XtY += xty
        n_ok += 1
    return XtX, XtY, n_ok


def main():
    """Execute le backtest NLinear global vs local vs zeroshot."""
    from utils.parser import parse_timeseries, parse_labels

    print(f"Chargement de tous les compteurs (max_meters=None)...", flush=True)
    t_load = time.time()
    df = parse_timeseries(str(ROOT / "RES2-6-9.csv"), max_meters=None)
    all_meters = sorted(df["meter_id"].astype(str).unique())
    print(f"  {len(all_meters)} compteurs charges en {time.time()-t_load:.1f}s", flush=True)

    lbl_path = ROOT / "RES2-6-9-labels.csv"
    labels = {}
    if lbl_path.exists():
        try:
            from utils.parser import parse_labels
            labels = parse_labels(str(lbl_path))
        except Exception:
            pass

    # Pre-calcul des matrices globales pour chaque fold (3 x accumulation sur ~500 compteurs)
    print(f"\nAccumulation globale ({N_FOLDS} folds x {len(all_meters)} compteurs)...", flush=True)
    global_matrices = {}
    for fold in range(N_FOLDS):
        t0 = time.time()
        XtX, XtY, n_ok = accumulate_global(df, all_meters, fold)
        global_matrices[fold] = (XtX, XtY)
        print(f"  fold {fold} : {n_ok} compteurs inclus  t={time.time()-t0:.1f}s", flush=True)

    configs = [
        "nlinear_global",
        "nlinear_zeroshot",
        "nlinear_local",
        "ridge",
        "naive_last_day",
    ]
    all_maes = {k: [] for k in configs}

    print(f"\nEvaluation sur {len(METER_IDS)} compteurs de reference...", flush=True)
    ref_meters = [m for m in METER_IDS if m in df["meter_id"].astype(str).values]

    for mi, mid in enumerate(ref_meters):
        s = df[df["meter_id"] == mid].sort_values("ts")["kw"].values.astype(np.float32)
        lbl = labels.get(str(mid))
        cls = None if lbl is None else ("rp" if lbl == 0 else "rs")
        print(f"  [{mi+1}/{len(ref_meters)}] {mid}  n={len(s)}  cls={cls or '?'}", flush=True)

        fold_maes = {k: [] for k in configs}

        for fold in range(N_FOLDS):
            ts = _PTS + fold * _GAP
            if ts + STEPS > len(s):
                break
            train = s[max(0, ts - _PTS):ts].astype(np.float64)
            test = s[ts:ts + STEPS].astype(np.float64)
            if len(train) < L + STEPS:
                continue

            XtX_all, XtY_all = global_matrices[fold]

            # Matrices locales du compteur cible
            XtX_loc, XtY_loc = _local_matrices(train)
            if XtX_loc is None:
                continue

            # W_global : tous les compteurs dont la cible
            W_global = _solve(XtX_all, XtY_all)

            # W_zeroshot : tous sauf la cible (leave-one-out exact)
            W_zs = _solve(XtX_all - XtX_loc, XtY_all - XtY_loc)

            # W_local : cible uniquement
            W_local = _solve(XtX_loc, XtY_loc)

            window = train[-L:]
            fold_maes["nlinear_global"].append(mae(test, _predict(W_global, window)))
            fold_maes["nlinear_zeroshot"].append(mae(test, _predict(W_zs, window)))
            fold_maes["nlinear_local"].append(mae(test, _predict(W_local, window)))

            # Ridge local
            from models.forecaster import RidgeForecaster
            m_ridge = RidgeForecaster()
            m_ridge.fit(train)
            fold_maes["ridge"].append(mae(test, m_ridge.predict(STEPS)))

            # Naive J-1
            fold_maes["naive_last_day"].append(mae(test, train[-STEPS:]))

        for k in configs:
            if fold_maes[k]:
                all_maes[k].extend(fold_maes[k])
                print(f"    {k:<22}: MAE={np.mean(fold_maes[k]):.4f}", flush=True)

    print("\n=== Resultats ===", flush=True)
    ref = float(np.mean(all_maes["naive_last_day"])) if all_maes["naive_last_day"] else 1.0
    for k in configs:
        vals = all_maes[k]
        if not vals:
            continue
        m = float(np.mean(vals))
        gain = (ref - m) / ref * 100
        print(f"  {k:<22} MAE={m:.4f}  vs_naive={gain:+.1f}%", flush=True)

    out = {
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "train_days": TRAIN_DAYS, "n_folds": N_FOLDS,
            "n_ref_meters": len(ref_meters), "n_global_meters": len(all_meters),
            "L": L, "ridge_lambda": RIDGE_LAMBDA,
        },
        "results": {
            k: {"mae_mean": round(float(np.mean(v)), 4), "n_obs": len(v)}
            for k in configs
            if (v := all_maes[k])
        },
    }
    out_path = ROOT / "assets" / "nlinear_global_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSauvegarde : {out_path}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTemps total : {time.time()-t0:.1f}s", flush=True)
