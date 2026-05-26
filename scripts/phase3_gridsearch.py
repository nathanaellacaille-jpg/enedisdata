"""Phase 3a — GridSearchCV des hyperparams HistGBT pour les figer.

Lance une recherche reduite (8-12 combos) en CV5 sur le dataset complet.
Affiche le meilleur set et son score, a hardcoder ensuite dans
models/classifier.py.

Usage : python scripts/phase3_gridsearch.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_LBL_PATH, DEFAULT_TS_PATH
from utils.features import extract_features
from utils.parser import parse_labels, parse_timeseries


def main() -> None:
    """Execute la grid search."""
    print("=" * 72)
    print("PHASE 3a — GRIDSEARCH HistGBT")
    print("=" * 72)

    print("\n[1/3] Chargement...")
    df = parse_timeseries(str(DEFAULT_TS_PATH), max_meters=None)
    labels = parse_labels(str(DEFAULT_LBL_PATH))
    features = extract_features(df)
    common = [mid for mid in features.index if str(mid) in labels]
    X = features.loc[common]
    y = np.array([labels[str(mid)] for mid in common])
    print(f"  {len(X)} echantillons, {X.shape[1]} features")

    print("\n[2/3] GridSearchCV (12 combos * CV5 = 60 fits)...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            class_weight="balanced", random_state=42,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
        )),
    ])
    grid = {
        "clf__max_iter": [300, 600],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__max_depth": [4, 6],
        "clf__l2_regularization": [0.5, 1.5],
        "clf__min_samples_leaf": [5, 10],
    }
    # Cap a 24 combos max (limite a 5 dims = 96 combos, on reduit)
    # Strategy : combiner que les valeurs principales
    grid_reduced = {
        "clf__max_iter": [400],                  # fixe (early stopping gere)
        "clf__learning_rate": [0.03, 0.05, 0.08],
        "clf__max_depth": [4, 6, 8],
        "clf__l2_regularization": [0.3, 1.0],
        "clf__min_samples_leaf": [5, 15],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        pipe, grid_reduced,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    gs.fit(X, y)

    print("\n[3/3] Resultats")
    print(f"  Meilleur F1 weighted CV5 : {gs.best_score_:.4f}")
    print(f"  Meilleurs params :")
    for k, v in gs.best_params_.items():
        print(f"    {k:30s} = {v}")

    # Top 5 combos
    res = pd.DataFrame(gs.cv_results_).sort_values("rank_test_score").head(5)
    print(f"\n  Top 5 combos :")
    cols = ["mean_test_score", "std_test_score"] + [c for c in res.columns if c.startswith("param_")]
    print(res[cols].to_string(index=False))

    print("\n" + "=" * 72)
    print("A HARDCODER dans models/classifier.py :")
    params = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    print(f"  HistGradientBoostingClassifier({', '.join(f'{k}={v!r}' for k, v in params.items())},")
    print(f"      class_weight='balanced', random_state=42,")
    print(f"      early_stopping=True, validation_fraction=0.15, n_iter_no_change=20)")
    print("=" * 72)


if __name__ == "__main__":
    main()
