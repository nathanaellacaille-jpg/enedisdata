import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from config import CLF_N_COMP_PCA, CLF_N_TREES


class EnergyClassifier:
    """Pipeline StandardScaler -> PCA -> RandomForestClassifier (0=RP, 1=RS)."""

    def __init__(self):
        """Initialise le pipeline de classification."""
        self._pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=CLF_N_COMP_PCA, random_state=42)),
            ("clf", RandomForestClassifier(n_estimators=CLF_N_TREES, random_state=42, n_jobs=-1)),
        ])
        self._feature_names: list = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "EnergyClassifier":
        """Entraine le pipeline sur X et y."""
        self._feature_names = list(X.columns)
        self._pipe.fit(X.values, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Labels predits (0=RP, 1=RS)."""
        return self._pipe.predict(X.values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilite d'etre RS (classe 1)."""
        return self._pipe.predict_proba(X.values)[:, 1]

    def feature_importances(self, X: pd.DataFrame) -> pd.Series:
        """Importances depliees depuis PCA vers features originales."""
        clf = self._pipe.named_steps["clf"]
        pca = self._pipe.named_steps["pca"]
        # Importance de chaque composante PCA
        comp_imp = clf.feature_importances_           # shape (n_components,)
        # Decommposition : contribution de chaque feature originale via les vecteurs propres
        components = pca.components_                  # shape (n_components, n_features)
        feature_imp = np.abs(components).T @ comp_imp # shape (n_features,)
        feature_imp /= feature_imp.sum() + 1e-12
        names = self._feature_names or list(X.columns)
        return pd.Series(feature_imp, index=names).sort_values(ascending=False)

    def save(self, path: str) -> None:
        """Sauvegarde le modele via pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path: str) -> "EnergyClassifier":
        """Charge un modele depuis un fichier pickle."""
        with open(path, "rb") as f:
            return pickle.load(f)
