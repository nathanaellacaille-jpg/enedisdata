"""Phase 0 — Diagnostic chiffre de la classification RS/RP.

Produit une baseline propre avant toute modification :
- distribution des labels
- metriques CV5 (accuracy, F1, rappel/precision RS, AUC)
- matrice de confusion agregee
- top erreurs avec leurs features
- courbe d'apprentissage (sature-t-on ou plus de data aiderait ?)

A relancer apres chaque iteration de Phase 1/2/3 pour mesurer le gain.

Usage : python scripts/phase0_diagnostic.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_LBL_PATH, DEFAULT_TS_PATH
from models.classifier import EnergyClassifier
from utils.features import extract_features
from utils.parser import parse_labels, parse_timeseries


def main() -> None:
    """Execute le diagnostic complet."""
    print("=" * 72)
    print("PHASE 0 — DIAGNOSTIC CLASSIFICATION RS/RP")
    print("=" * 72)

    # 1. Chargement
    print("\n[1/6] Chargement des donnees...")
    if not DEFAULT_TS_PATH.exists():
        print(f"  ERREUR : {DEFAULT_TS_PATH} introuvable.")
        sys.exit(1)
    df = parse_timeseries(str(DEFAULT_TS_PATH), max_meters=None)
    labels = parse_labels(str(DEFAULT_LBL_PATH))
    print(f"  Timeseries : {df['meter_id'].nunique()} compteurs, {len(df):,} points")
    print(f"  Labels     : {len(labels)} ids")

    # 2. Features
    print("\n[2/6] Extraction des features...")
    features = extract_features(df)
    print(f"  Features : {features.shape[0]} compteurs x {features.shape[1]} colonnes")
    print(f"  Colonnes : {list(features.columns)}")

    # 3. Intersection
    common = [mid for mid in features.index if str(mid) in labels]
    X = features.loc[common]
    y = np.array([labels[str(mid)] for mid in common])
    n_rp = int((y == 0).sum())
    n_rs = int((y == 1).sum())
    print(f"\n[3/6] Distribution des labels (intersection features x labels = {len(y)})")
    print(f"  RP : {n_rp} ({n_rp / len(y) * 100:.1f} %)")
    print(f"  RS : {n_rs} ({n_rs / len(y) * 100:.1f} %)")
    print(f"  Desequilibre : {max(n_rp, n_rs) / min(n_rp, n_rs):.2f}x")

    # 4. CV5 baseline
    print("\n[4/6] CV5 baseline (EnergyClassifier actuel, seuil tune par PR curve)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics: dict[str, list[float]] = {
        "accuracy": [], "f1_weighted": [],
        "recall_rs": [], "precision_rs": [],
        "auc": [], "threshold": [],
    }
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    y_proba_all: list[float] = []

    for tr_idx, te_idx in cv.split(X, y):
        clf = EnergyClassifier()
        clf.fit(X.iloc[tr_idx], y[tr_idx])
        proba = clf.predict_proba(X.iloc[te_idx])
        pred = (proba >= clf.threshold_).astype(int)
        metrics["threshold"].append(clf.threshold_)

        metrics["accuracy"].append(accuracy_score(y[te_idx], pred))
        metrics["f1_weighted"].append(f1_score(y[te_idx], pred, average="weighted"))
        metrics["recall_rs"].append(recall_score(y[te_idx], pred, pos_label=1, zero_division=0))
        metrics["precision_rs"].append(precision_score(y[te_idx], pred, pos_label=1, zero_division=0))
        metrics["auc"].append(roc_auc_score(y[te_idx], proba))

        y_true_all.extend(y[te_idx].tolist())
        y_pred_all.extend(pred.tolist())
        y_proba_all.extend(proba.tolist())

    print(f"  {'Metrique':<18s} {'Moyenne':>10s} {'+/- std':>10s}")
    print(f"  {'-' * 18} {'-' * 10} {'-' * 10}")
    for name, vals in metrics.items():
        print(f"  {name:<18s} {np.mean(vals):>10.4f} {np.std(vals):>10.4f}")

    # 5. Matrice de confusion + erreurs
    print("\n[5/6] Matrice de confusion (CV5 agregee)")
    cm = confusion_matrix(y_true_all, y_pred_all)
    print(f"               Pred RP    Pred RS")
    print(f"  Vrai RP    {cm[0, 0]:8d}   {cm[0, 1]:8d}    (rappel RP : {cm[0, 0] / cm[0].sum():.3f})")
    print(f"  Vrai RS    {cm[1, 0]:8d}   {cm[1, 1]:8d}    (rappel RS : {cm[1, 1] / cm[1].sum():.3f})")

    preds_df = pd.DataFrame({
        "meter_id": common,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "y_proba": y_proba_all,
    })
    errors = preds_df[preds_df["y_true"] != preds_df["y_pred"]]
    fn = errors[(errors["y_true"] == 1) & (errors["y_pred"] == 0)]  # missed RS
    fp = errors[(errors["y_true"] == 0) & (errors["y_pred"] == 1)]  # false RS
    print(f"\n  Total erreurs : {len(errors)} / {len(preds_df)} ({len(errors) / len(preds_df) * 100:.1f} %)")
    print(f"  Faux negatifs RS (manques)   : {len(fn)}")
    print(f"  Faux positifs RS (sur-detect): {len(fp)}")

    # Worst confidence errors
    worst_fn = fn.nsmallest(5, "y_proba")
    worst_fp = fp.nlargest(5, "y_proba")
    print(f"\n  Top 5 RS manquees (proba la plus basse) — confiance la plus erronee :")
    for _, row in worst_fn.iterrows():
        feat = X.loc[row["meter_id"]]
        print(f"    {row['meter_id']} : proba={row['y_proba']:.3f} "
              f"| we/wd={feat['ratio_we_wd']:.2f} zero={feat['zero_ratio']:.2f} "
              f"gap={feat['max_gap_days']:.0f}j active={feat['active_days_ratio']:.2f}")

    print(f"\n  Top 5 RP classees RS (proba la plus haute) — sur-confiance erronee :")
    for _, row in worst_fp.iterrows():
        feat = X.loc[row["meter_id"]]
        print(f"    {row['meter_id']} : proba={row['y_proba']:.3f} "
              f"| we/wd={feat['ratio_we_wd']:.2f} zero={feat['zero_ratio']:.2f} "
              f"gap={feat['max_gap_days']:.0f}j active={feat['active_days_ratio']:.2f}")

    # 6. Learning curve
    print("\n[6/6] Courbe d'apprentissage (peut prendre 1-2 min)...")
    from sklearn.ensemble import HistGradientBoostingClassifier
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=6,
            min_samples_leaf=10, l2_regularization=0.5,
            class_weight="balanced", random_state=42,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20,
        )),
    ])
    train_sizes_abs = [50, 100, 150, 200, 250, 300, 350]
    train_fracs = [n / len(X) for n in train_sizes_abs if n < len(X) * 0.8]
    train_n, train_scores, test_scores = learning_curve(
        pipe, X, y,
        train_sizes=train_fracs,
        cv=cv, scoring="f1_weighted", n_jobs=-1, random_state=42,
    )
    print(f"  {'N_train':>8s}  {'F1 train':>10s}  {'F1 test':>10s}  {'Ecart':>8s}")
    print(f"  {'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 8}")
    for n, tr, te in zip(train_n, train_scores.mean(axis=1), test_scores.mean(axis=1)):
        gap = tr - te
        print(f"  {int(n):>8d}  {tr:>10.4f}  {te:>10.4f}  {gap:>8.4f}")

    if train_scores.mean(axis=1)[-1] - test_scores.mean(axis=1)[-1] > 0.10:
        verdict = "OVERFIT : plus de regularisation ou moins de features aiderait"
    elif test_scores.mean(axis=1)[-1] - test_scores.mean(axis=1)[-3] > 0.02:
        verdict = "PLUS DE DATA AIDERAIT : la courbe test n'a pas plafonne"
    else:
        verdict = "PLATEAU : modele/features sont la limite, pas la quantite de data"
    print(f"\n  Verdict : {verdict}")

    print("\n" + "=" * 72)
    print(f"BASELINE A BATTRE : F1 weighted = {np.mean(metrics['f1_weighted']):.4f}, "
          f"Recall RS = {np.mean(metrics['recall_rs']):.4f}, "
          f"AUC = {np.mean(metrics['auc']):.4f}")
    print("=" * 72)

    # Dump baseline metrics en JSON pour affichage dans la page Streamlit
    # (evite de recomputer le CV5 a chaque cold start Cloud).
    import json
    from datetime import datetime
    out = ROOT / "assets" / "baseline_metrics.json"
    out.parent.mkdir(exist_ok=True)
    payload = {
        "computed_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
            "n_rp": n_rp,
            "n_rs": n_rs,
            "imbalance_ratio": round(max(n_rp, n_rs) / min(n_rp, n_rs), 2),
        },
        "cv5": {
            "accuracy": float(np.mean(metrics["accuracy"])),
            "accuracy_std": float(np.std(metrics["accuracy"])),
            "f1_weighted": float(np.mean(metrics["f1_weighted"])),
            "f1_weighted_std": float(np.std(metrics["f1_weighted"])),
            "recall_rs": float(np.mean(metrics["recall_rs"])),
            "recall_rs_std": float(np.std(metrics["recall_rs"])),
            "precision_rs": float(np.mean(metrics["precision_rs"])),
            "precision_rs_std": float(np.std(metrics["precision_rs"])),
            "auc": float(np.mean(metrics["auc"])),
            "auc_std": float(np.std(metrics["auc"])),
            "threshold_mean": float(np.mean(metrics["threshold"])),
        },
        "confusion": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
            "recall_rp": float(cm[0, 0] / cm[0].sum()),
            "recall_rs": float(cm[1, 1] / cm[1].sum()),
            "n_errors": int(len(errors)),
            "n_false_negatives_rs": int(len(fn)),
            "n_false_positives_rs": int(len(fp)),
        },
    }
    # Seuil PR-optimal calcule sur les probas OOF (vrai optimum vs moyenne des per-fold)
    from sklearn.metrics import precision_recall_curve as _prc
    _y_true_arr = np.array(y_true_all)
    _y_proba_arr = np.array(y_proba_all)
    _p, _r, _t = _prc(_y_true_arr, _y_proba_arr)
    _f1s = 2 * _p * _r / (_p + _r + 1e-12)
    _best = int(np.argmax(_f1s[:-1])) if len(_f1s) > 1 else 0
    threshold_pr_optimal = float(_t[_best]) if _best < len(_t) else 0.5
    payload["cv5"]["threshold_pr_optimal"] = threshold_pr_optimal
    print(f"\n  Seuil PR-optimal sur OOF : {threshold_pr_optimal:.4f}")

    # OOF predictions : permet a la page Streamlit de recomputer les metriques
    # pour n'importe quel seuil sans refitter le modele (slider interactif).
    payload["oof"] = {
        "meter_ids": [str(m) for m in common],
        "y_true": [int(v) for v in y_true_all],
        "y_proba": [float(v) for v in y_proba_all],
    }

    # Permutation importance sur le pipeline entraine sur l'ensemble du dataset
    print("\n[7/7] Calcul des permutation importances (n_jobs=1 pour eviter blowup RAM)...")
    full_clf = EnergyClassifier()
    full_clf.fit(X, y)
    imps = full_clf.feature_importances(X, y).head(10)
    payload["feature_importances_top10"] = [
        {"feature": str(k), "importance": float(v)} for k, v in imps.items()
    ]
    print("  Top 10 features par importance :")
    for k, v in imps.items():
        print(f"    {k:30s} {v:.4f}")

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\nMetriques baseline ecrites dans {out}")


if __name__ == "__main__":
    main()
