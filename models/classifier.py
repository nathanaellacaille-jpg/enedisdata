import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Hyperparams trouves par GridSearchCV (scripts/phase3_gridsearch.py)
# sur 500 echantillons / 26 features, CV5, F1 weighted = 0.9454.
_HGBT_PARAMS = dict(
    max_iter=400,
    learning_rate=0.08,
    max_depth=4,
    min_samples_leaf=5,
    l2_regularization=1.0,
    class_weight="balanced",
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
)


def _make_stacking() -> StackingClassifier:
    """Construit le stacking HistGBT + RF + LogReg avec meta LogReg.

    Le meta-learner combine les probas des 3 modeles via une regression
    logistique entrainee sur les predictions out-of-fold (CV5 interne).
    """
    return StackingClassifier(
        estimators=[
            ("hgbt", HistGradientBoostingClassifier(**_HGBT_PARAMS)),
            ("rf", RandomForestClassifier(
                n_estimators=400, max_depth=10, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )),
            ("lr", LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced", random_state=42,
            )),
        ],
        final_estimator=LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=42,
        ),
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )


class EnergyClassifier:
    """Pipeline StandardScaler -> Stacking(HistGBT + RF + LogReg) -> meta LogReg.

    Le seuil decisionnel optimal est appris sur le train via PR curve (max F1)
    et expose via self.threshold_ apres fit.
    """

    def __init__(self):
        """Initialise le pipeline scaler + stacking."""
        self._pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", _make_stacking()),
        ])
        self._feature_names: list = []
        self.threshold_: float = 0.5

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "EnergyClassifier":
        """Entraine le pipeline puis calcule le seuil optimal F1 par CV interne."""
        self._feature_names = list(X.columns)
        self._pipe.fit(X, y)
        self.threshold_ = self._tune_threshold(X, y)
        return self

    def _tune_threshold(self, X: pd.DataFrame, y: np.ndarray) -> float:
        """Seuil F1-optimal sur les probas du train.

        Le meta-learner du Stacking utilise deja des probas out-of-fold (cv=5 interne)
        pour son entrainement, donc les probas predites sur le train sont peu biaisees.
        On evite ainsi une CV5 supplementaire qui multiplie le compute par 5.
        """
        proba = self._pipe.predict_proba(X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, proba)
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
        best_idx = int(np.argmax(f1s[:-1])) if len(f1s) > 1 else 0
        return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Labels predits selon self.threshold_ optimal."""
        return (self.predict_proba(X) >= self.threshold_).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilite d'etre RS (classe 1)."""
        return self._pipe.predict_proba(X)[:, 1]

    def feature_importances(self, X: pd.DataFrame, y: np.ndarray) -> pd.Series:
        """Permutation importance sur l'espace original.

        n_jobs=1 : eviter le multiprocessing qui copie le pipeline complet
        dans chaque worker (~1 GB par copie avec le Stacking) → OOM sur Cloud.
        Calcul one-shot via scripts/phase0_diagnostic.py.
        """
        from sklearn.inspection import permutation_importance
        result = permutation_importance(
            self._pipe, X, y,
            n_repeats=3,
            random_state=42,
            scoring="f1_weighted",
            n_jobs=1,
        )
        return pd.Series(
            result.importances_mean,
            index=X.columns,
        ).sort_values(ascending=False)

    def save(self, path: str) -> None:
        """Sauvegarde le modele via pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path: str) -> "EnergyClassifier":
        """Charge un modele depuis un fichier pickle."""
        with open(path, "rb") as f:
            return pickle.load(f)
