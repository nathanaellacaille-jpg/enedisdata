#!/usr/bin/env python3
"""
scripts/compute_nlinear_global_weights.py — Pre-calcule W_global pour NLinear (500 compteurs)

Accumule XtX et XtY sur les 90 derniers jours de chaque compteur,
resout W = (XtX + λI)^{-1} XtY et sauvegarde assets/nlinear_global_weights.npy.
Usage : python scripts/compute_nlinear_global_weights.py
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

L = 192
STEPS = 48
TRAIN_DAYS = 90
RIDGE_LAMBDA = 1e-3
PTS = TRAIN_DAYS * STEPS  # 4320


def main():
    """Calcule et sauvegarde W_global."""
    from utils.parser import parse_timeseries

    print("Chargement de tous les compteurs...", flush=True)
    t0 = time.time()
    df = parse_timeseries(str(ROOT / "RES2-6-9.csv"), max_meters=None)
    all_meters = sorted(df["meter_id"].astype(str).unique())
    print(f"  {len(all_meters)} compteurs  t={time.time()-t0:.1f}s", flush=True)

    XtX = np.zeros((L, L))
    XtY = np.zeros((L, STEPS))
    n_ok = 0

    print("Accumulation XtX / XtY...", flush=True)
    t0 = time.time()
    for mid in all_meters:
        s = df[df["meter_id"] == mid].sort_values("ts")["kw"].values.astype(np.float64)
        train = s[-PTS:] if len(s) >= PTS + STEPS else s[:-STEPS]
        if len(train) < L + STEPS:
            continue
        n_samp = len(train) - L - STEPS + 1
        i_win = np.arange(n_samp)[:, None] + np.arange(L)[None, :]
        X_raw = train[i_win]
        last = X_raw[:, -1:]
        X = X_raw - last
        i_tgt = np.arange(n_samp)[:, None] + np.arange(L, L + STEPS)[None, :]
        Y = train[i_tgt] - last
        XtX += X.T @ X
        XtY += X.T @ Y
        n_ok += 1

    print(f"  {n_ok} compteurs inclus  t={time.time()-t0:.1f}s", flush=True)

    print("Solve ridge...", flush=True)
    W = np.linalg.solve(XtX + RIDGE_LAMBDA * np.eye(L), XtY)  # (L, STEPS)

    out = ROOT / "assets" / "nlinear_global_weights.npy"
    np.save(str(out), W)
    print(f"Sauvegarde : {out}  shape={W.shape}  ({out.stat().st_size // 1024} KB)", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTemps total : {time.time()-t0:.1f}s", flush=True)
