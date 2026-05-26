import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class EnergyClassifier:
    """Pipeline StandardScaler -> HistGradientBoostingClassifier (0=RP, 1=RS).

    Calcule un seuil decisionnel optimal sur le train via PR curve (max F1)
    plutot que d'utiliser un seuil hardcode. Le seuil optimal est expose via
    `self.threshold_` apres fit.
    """

    def __init__(self):
        """Initialise le pipeline HistGBT + scaler."""
        self._pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(
                max_iter=400,
                learning_rate=0.05,
                max_depth=6,
                min_samples_leaf=10,
                l2_regularization=0.5,
                class_weight="balanced",
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
            )),
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
        """Trouve le seuil qui maximise F1 sur les proba out-of-fold (CV5 interne)."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_proba = np.zeros(len(y))
        for tr, te in cv.split(X, y):
            inner = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", HistGradientBoostingClassifier(
                    max_iter=400, learning_rate=0.05, max_depth=6,
                    min_samples_leaf=10, l2_regularization=0.5,
                    class_weight="balanced", random_state=42,
                    early_stopping=True, validation_fraction=0.15,
                    n_iter_no_change=20,
                )),
            ])
            inner.fit(X.iloc[tr], y[tr])
            oof_proba[te] = inner.predict_proba(X.iloc[te])[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, oof_proba)
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
        """Permutation importance sur l'espace original."""
        from sklearn.inspection import permutation_importance
        result = permutation_importance(
            self._pipe, X, y,
            n_repeats=10,
            random_state=42,
            scoring="f1_weighted",
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
