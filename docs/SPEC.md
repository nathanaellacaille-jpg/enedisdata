# SPEC.md

Spécification complète du projet Enedis Analytics.
Lire en entier avant d'écrire la moindre ligne de code.

## Contexte

Dataset open data Enedis RES2-6-9kVA.
Format : CSV avec colonnes `id`, `horodate`, `valeur`.
Valeurs en kW, pas 30 minutes (48 points/jour).
Énergie journalière = somme des valeurs × 0.5.

## Structure des fichiers

### config.py

```python
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

@dataclass(frozen=True)
class Palette:
    RS: str = "#0F172A"
    RP: str = "#0F172A"
    RS_soft: str = "#F8FAFC"
    RP_soft: str = "#F1F5F9"
    LR: str = "#0F172A"
    ARIMA: str = "#475569"
    LSTM: str = "#94A3B8"
    NAIVE: str = "#CBD5E1"
    REAL: str = "#0F172A"
    GRID: str = "#F8FAFC"
    BORDER: str = "#E2E8F0"
    TEXT: str = "#0F172A"
    TEXT_MUTED: str = "#64748B"
    MULTI: List[str] = field(default_factory=lambda: [
        "#0F172A","#1E293B","#334155","#475569",
        "#64748B","#94A3B8","#CBD5E1","#E2E8F0",
    ])

PAL = Palette()

STEPS_PER_DAY = 48
STEPS_PER_HOUR = 2

CLF_TEST_SIZE = 0.30
CLF_N_COMP_PCA = 5
CLF_N_TREES = 300

FCST_N_LAGS = 48
FCST_N_FOURIER = 3
FCST_HORIZON_H = 24
FCST_ARIMA_ORDER = (2, 1, 2)

LSTM_SEQ_LEN = 96
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_EPOCHS = 60
LSTM_LR = 1e-3

GEN_DEFAULT_N = 10
GEN_NOISE_STD = 0.15

DAY_FR = {"Monday":"Lundi","Tuesday":"Mardi","Wednesday":"Mercredi",
          "Thursday":"Jeudi","Friday":"Vendredi","Saturday":"Samedi","Sunday":"Dimanche"}
DAY_FR_SHORT = {"Monday":"Lun","Tuesday":"Mar","Wednesday":"Mer",
                "Thursday":"Jeu","Friday":"Ven","Saturday":"Sam","Sunday":"Dim"}
```

Ajouter les fonctions `_make_rp_profile()` et `_make_rs_profile()` retournant
des np.ndarray de taille 48, normalisés entre 0 et 1 :
- RP : pic matin 7h-9h (amplitude 0.45) + pic soir 18h-22h (amplitude 0.60) + base 0.18
- RS : base 0.06 + pic soir 20h (amplitude 0.35)

### utils/parser.py

```python
def parse_timeseries(file) -> pd.DataFrame:
    """Lit un CSV Enedis, retourne df [meter_id, ts, kw]."""

def parse_labels(file) -> dict:
    """Lit un CSV id,label, retourne {meter_id: int}."""
```

- Auto-détection séparateur (`;`, `,`, `\t`)
- Colonnes normalisées : `meter_id` (str), `ts` (datetime64 UTC), `kw` (float64)
- `ValueError` avec message lisible si colonnes manquantes ou format invalide

### utils/features.py

```python
def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les features par compteur depuis df [meter_id, ts, kw]."""
```

Features retournées (index = meter_id) :

| Feature | Description |
|---|---|
| `ratio_we_wd` | Énergie WE / énergie semaine |
| `cv_daily_energy` | Coeff. variation énergie journalière |
| `peak_hour_ratio` | Énergie 18h-22h / énergie totale |
| `night_ratio` | Énergie 0h-6h / énergie totale |
| `fourier_amp_1/2/3` | Amplitudes des 3 premières harmoniques sur profil moyen |

### utils/metrics.py

```python
def compute_metrics(y_true, y_pred) -> dict:
    """Retourne MAE, RMSE, MAPE, R2. MAPE protégée contre division par zéro."""
```

### models/classifier.py

```python
class EnergyClassifier:
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "EnergyClassifier":
        """Pipeline StandardScaler → PCA → RandomForestClassifier."""
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Labels prédits (0=RP, 1=RS)."""
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilité d'être RS."""
    def feature_importances(self, X: pd.DataFrame) -> pd.Series:
        """Importances dépliées depuis PCA vers features originales."""
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> "EnergyClassifier": ...
```

### models/forecaster.py

```python
def make_lag_features(series: np.ndarray, n_lags: int) -> np.ndarray:
    """Construit la matrice de lags temporels."""

def make_fourier_features(n: int, n_harmonics: int) -> np.ndarray:
    """Features saisonnières sin/cos."""

class RidgeForecaster:
    def fit(self, X, y): ...
    def predict(self, X) -> np.ndarray: ...
    def coef_series(self) -> pd.Series: ...

class ARIMAForecaster:
    def fit(self, series: np.ndarray, order=(2,1,2)): ...
    def predict(self, h: int) -> np.ndarray: ...
    def summary(self) -> str: ...

class LSTMForecaster:
    """Optionnel — activé si use_lstm=True dans la page."""
    def fit(self, series: np.ndarray, callback=None): ...
    def predict(self, h: int) -> np.ndarray: ...
```

### models/generator.py

```python
class CurveGenerator:
    def fit(self, df: pd.DataFrame, labels: dict | None) -> "CurveGenerator":
        """Calcule profils moyens par classe. Fallback sur profils de référence si labels=None."""
    def generate(self, n: int, curve_type: str, n_days=7, noise_std=0.15) -> pd.DataFrame:
        """curve_type in {'RS','RP','mixed'}. Retourne df [curve_id, day, slot, kw, curve_type]."""
    def profile_stats(self) -> dict:
        """Énergie moyenne et std par type."""
```

## Pages

### pages/1_classification.py

Sidebar :
- Upload CSV timeseries (requis)
- Upload CSV labels (optionnel)
- Sélection compteur à analyser

Tabs :
1. **Résultat** — métriques (conso moy, pic, ratio WE/WD, énergie/j), courbe brute annotée, jauge RS/RP (colorscale gris)
2. **Vue d'ensemble** — scatter features, distribution des prédictions par classe, heatmap corrélation features
3. **Explication** — top 5 features, radar vs profil de référence
4. **Performance** — matrice de confusion + accuracy + F1 (si labels fournis)
5. **Export** — CSV des prédictions

### pages/2_prevision.py

Sidebar :
- Upload CSV timeseries
- Selectbox compteur
- Checkboxes : activer ARIMA / activer LSTM

Tabs :
1. **Graphique** — historique + prévisions superposées, ligne de début prévision, annotation pic d'erreur
2. **Analyse** — tableau métriques par modèle, verdict meilleur modèle
3. **Technique** — coefficients Ridge, diagnostic ARIMA, courbe loss LSTM
4. **Horizon** — MAE par heure d'horizon pour chaque modèle
5. **Guide** — description courte des 3 méthodes, 2-3 phrases chacune

### pages/3_generation.py

Sidebar :
- Upload CSV timeseries (optionnel, pour calibration)
- Upload CSV labels (optionnel)
- Radio : RS / RP / Mixte
- Slider : nombre de courbes (1-100)
- Slider : nombre de jours (1-30)
- Slider : courbes à visualiser (1-20)

Tabs :
1. **Courbes** — superposées ou grille individuelle (radio)
2. **Profils** — profil moyen RS vs RP, ratio WE/semaine par type
3. **Comparaison** — deux courbes côte à côte, tableau de stats
4. **Statistiques** — distribution énergie journalière, heatmap puissance (colorscale gris)
5. **Export** — CSV toutes courbes + JSON résumé stats

## app.py

```python
st.set_page_config(page_title="Enedis Analytics", page_icon=None, layout="wide")
# charge assets/style.css
# header HTML :
# <div class="app-header">
#   <div class="app-title">Enedis Analytics</div>
#   <div class="app-subtitle">Classification · Prévision · Génération</div>
# </div>
# st.navigation([pg1, pg2, pg3], position="sidebar")
# sidebar footer : badge "RES2-6-9 kVA" + 2 lignes info en 11px
```

## Contraintes de déploiement

- `streamlit run app.py` suffit
- Zéro variable d'environnement requise
- Données chargées uniquement via upload dans l'UI — pas de chemin hardcodé
- L'app doit fonctionner sans données uploadées (état vide propre, message neutre)
- LSTM désactivé par défaut (optionnel, checkbox dans sidebar page 2)
