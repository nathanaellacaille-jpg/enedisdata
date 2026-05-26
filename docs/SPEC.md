# SPEC.md

Reference technique de l'etat actuel du projet Enedis Analytics.
A jour au 2026-05-26.

## Contexte

Dataset open data Enedis RES2-6-9kVA. Format CSV avec colonnes `id` (case-insensitive), `horodate` (datetime UTC ISO), `valeur` (Wh par demi-heure). Valeurs converties en kW par le parser (`valeur / 500`). Pas de 30 minutes, 48 points/jour. Energie journaliere = somme des kW * 0.5.

Le dataset contient 500 compteurs uniques. Les labels (`RES2-6-9-labels.csv`, 500 ids) sont versionnes dans le repo.

## Arborescence

```
app.py
config.py
requirements.txt
RES2-6-9-labels.csv      tracke (9 KB)
RES2-6-9.csv             gitignore, telecharge depuis GitHub Release au cold start
.streamlit/config.toml   theme light
assets/style.css         CSS global
utils/
  parser.py              CSV -> DataFrame [meter_id (category), ts (UTC), kw (float32)]
  features.py            17 features par compteur
  metrics.py             MAE, RMSE, MAPE, R2
  data_loader.py         resolution path local / cache /tmp / download URL
  corpus.py              corpus synthetique de fallback (300 courbes x 14 jours par classe)
models/
  classifier.py          EnergyClassifier (StandardScaler + RandomForest)
  forecaster.py          RidgeForecaster, ARIMAForecaster, LSTMForecaster
  generator.py           CurveGenerator (mode parametrique + bootstrap)
pages/
  1_classification.py
  2_prevision.py
  3_generation.py
docs/
  SPEC.md
  AUDIT.md
  UI.md
```

## config.py

Constantes principales (extrait, voir le fichier pour la liste complete) :

```python
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_TS_PATH = ROOT_DIR / "RES2-6-9.csv"
DEFAULT_LBL_PATH = ROOT_DIR / "RES2-6-9-labels.csv"
DATA_URL_TS = os.environ.get(
    "ENEDIS_TS_URL",
    "https://github.com/nathanaellacaille-jpg/enedisdata/releases/download/data-v1/RES2-6-9.csv",
)

STEPS_PER_DAY = 48

CLF_TEST_SIZE = 0.30
CLF_N_TREES = 300            # legacy, n'est plus utilise depuis le passage HistGBT

FCST_N_LAGS = 384            # 8 jours de lags (Ridge selectionne j-1..j-8 via L2)
FCST_N_FOURIER = 3
FCST_HORIZON_H = 24
FCST_ARIMA_ORDER = (2, 1, 2)

LSTM_SEQ_LEN = 48
LSTM_HIDDEN = 48
LSTM_LAYERS = 2
LSTM_EPOCHS = 40
LSTM_LR = 1e-3
LSTM_BATCH_SIZE = 64

GEN_NOISE_STD = 0.15
GEN_NOISE_RHO = 0.7          # autocorrelation AR(1) du bruit
GEN_CORPUS_N = 300           # courbes par classe dans le corpus built-in
GEN_CORPUS_DAYS = 14

_env_cap = os.environ.get("ENEDIS_MAX_METERS", "500")
MAX_METERS_UPLOAD: int | None = None if _env_cap.lower() in ("none", "0", "all") else int(_env_cap)
```

Palette : voir [UI.md](UI.md). `PAL.MULTI` pour l'UI en niveaux de gris, `PAL.ACCENT` pour les traces de graphiques uniquement.

## utils/parser.py

```python
def parse_timeseries(file, max_meters: int | None = MAX_METERS_UPLOAD) -> pd.DataFrame:
    """Lit un CSV Enedis par chunks de 50 000 lignes."""
```

- Auto-detection separateur (`;`, `,`, tab) sur les 4 premiers KB.
- Colonnes renommees : `meter_id` (categorical), `ts` (datetime64 UTC), `kw` (float32).
- `meter_id` en dtype `category` : reduit la RAM d'environ 60x vs str.
- Arret precoce des que `max_meters` IDs uniques sont vus (None = pas de cap).
- `ValueError` si colonnes manquantes.

```python
def parse_labels(file) -> dict:
    """{meter_id: int}."""
```

## utils/data_loader.py

```python
def load_default_ts() -> pd.DataFrame | None:
    """Charge le CSV timeseries (local prioritaire, sinon telecharge depuis DATA_URL_TS)."""

def load_default_labels() -> dict | None:
    """Charge le CSV labels (tracke dans git)."""
```

Resolution du chemin timeseries :
1. `DEFAULT_TS_PATH` (racine repo, mode local)
2. `/tmp/enedis-data/RES2-6-9.csv` (cache disque persistant)
3. Telechargement streame depuis `DATA_URL_TS` (barre de progression Streamlit, ecriture atomique via fichier `.part`)

Cache `@st.cache_data` keye sur `(path, mtime, size, max_meters)` : invalide si le fichier change.

## utils/features.py

26 features par compteur (index = `meter_id`). Refonte Phase 1+2 (2026-05-26) :
ajout de 6 features ciblees sur les "RS occupees" (RS qui ressemblent aux RP)
et extension Fourier de 3 a 6 harmoniques.

| Famille | Feature | Description |
|---|---|---|
| Presence | `zero_ratio` | Proportion de slots quasi nuls (kW < 0.05) |
| Presence | `max_gap_days` | Plus longue absence en jours consecutifs |
| Presence | `n_absence_periods` | Nombre de periodes d'absence |
| Presence | `active_days_ratio` | Taux de jours avec consommation |
| Presence | `seasonal_presence_gap` | Ecart de presence ete vs hiver |
| Presence | `vacation_weeks` | Plus longue serie de semaines a faible energie |
| Periodicite | `autocorr_lag48` | Autocorrelation jour-precedent |
| Temporel | `ratio_we_wd` | Energie weekend / energie semaine |
| Temporel | `peak_hour_ratio` | Energie 18h-22h / total |
| Temporel | `night_ratio` | Energie 0h-6h / total |
| Temporel | `morning_ratio` | Energie 6h-9h / total |
| Temporel | `peak_hour_std` | Variabilite jour-a-jour de l'heure du pic du soir |
| Temporel | `night_amplitude` | Amplitude relative kW nuit (veille appareils vs vide reel) |
| Variabilite | `cv_daily_energy` | CV de l'energie journaliere |
| Variabilite | `cv_weekly` | CV de l'energie hebdomadaire |
| Variabilite | `weekly_entropy` | Entropie de Shannon du profil hebdo (normalisee) |
| Variabilite | `dow_consistency` | Variance inter-semaine du profil par jour de la semaine |
| Saisonnalite | `seasonal_ratio` | Ratio kW ete / kW hiver |
| Saisonnalite | `summer_weekend_boost` | Ratio kW WE ete / WE hiver |
| Distribution | `skewness` | Asymetrie de la distribution kW |
| Fourier | `fourier_amp_1/2/3/4/5/6` | Amplitudes des 6 premieres harmoniques journalieres |

## utils/metrics.py

```python
def compute_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, MAPE, R2. MAPE protegee contre division par zero."""
```

## models/classifier.py

```python
class EnergyClassifier:
    """Pipeline StandardScaler -> StackingClassifier(HistGBT + RF + LogReg)
    avec meta-learner LogisticRegression.

    Le seuil decisionnel optimal est appris sur le train via PR curve (max F1),
    expose via self.threshold_ apres fit.
    """

    def fit(self, X, y) -> "EnergyClassifier"
    def predict(self, X) -> np.ndarray             # seuil dynamique self.threshold_
    def predict_proba(self, X) -> np.ndarray       # probabilite de la classe RS
    def feature_importances(self, X, y) -> pd.Series  # permutation importance, f1_weighted
    def save(self, path) / load(self, path)
```

**Architecture Phase 3** (2026-05-26) :
- Base learners (entrees du stacking) :
  - HistGBT : params trouves par GridSearchCV (`max_iter=400, learning_rate=0.08, max_depth=4, min_samples_leaf=5, l2_regularization=1.0`)
  - RandomForest : `n_estimators=400, max_depth=10, min_samples_leaf=5`
  - LogisticRegression : `C=1.0, max_iter=1000`
  - Tous : `class_weight="balanced"`, `random_state=42`
- Meta-learner : LogisticRegression sur les probas out-of-fold (CV5 interne)
- Le `StackingClassifier` de sklearn gere la CV interne (cv=5) pour generer les meta-features sans leakage.

Le seuil decisionnel monte autour de 0.71 (probas du stacking ecrasees vers le haut par le meta LogReg + class_weight). Plus de constante `CLF_RS_THRESHOLD` hardcodee.

**Performance baseline (CV5, 500 compteurs)** — progression dans le temps :
| Metrique | RF baseline (Phase 0) | HistGBT + features (Phase 1+2) | Stacking + grid (Phase 3) |
|---|---|---|---|
| F1 weighted | 0.908 | 0.932 | **0.938** |
| Recall RS | 0.581 | 0.789 | **0.832** |
| Precision RS | 0.775 | 0.767 | 0.771 |
| AUC | 0.912 | 0.959 | **0.969** |
| RS manquees | 30/72 | 15/72 | **12/72** |

Plafond actuel : les 12 RS encore manquees ont `ratio_we_wd ~1.0`, `active=1.0`, `zero_ratio bas` — RS "occupees toute l'annee" indiscernables des RP dans les features actuelles. Lever ce plafond demanderait des donnees externes (meteo, geolocalisation, type d'habitation).

## models/forecaster.py

```python
def make_lag_features(series, n_lags) -> np.ndarray
def make_fourier_features(n, n_harmonics=3, period=48, offset=0) -> np.ndarray
    # periode FIXE a 48 (journaliere), conforme Taylor & McSharry (2008)

class RidgeForecaster:
    # 384 lags (8 jours) + 6 colonnes Fourier (3 harmoniques x sin/cos)
    # Selection naturelle via regularisation L2

class ARIMAForecaster:
    # ordre fixe (2,1,2), pas de saisonnalite
    # cf. AUDIT.md : SARIMA recommande

class LSTMForecaster:
    # 2 couches, 48 unites cachees, 40 epochs, batch 64
    # optionnel, active via checkbox sidebar
```

## models/generator.py

```python
class CurveGenerator:
    def fit(self, df: pd.DataFrame, labels: dict | None) -> "CurveGenerator"

    def generate(self, n, curve_type, n_days=7, noise_std=GEN_NOISE_STD) -> pd.DataFrame
        # Mode parametrique : profil moyen par classe + bruit AR(1) en espace normalise [0,1]
        #                    + facteur journalier log-normal pour l'amplitude

    def generate_bootstrap(self, n, curve_type, n_days=7, noise_std=GEN_NOISE_STD) -> pd.DataFrame
        # Mode reechantillonnage : tire des journees reelles de la classe
        #                         + bruit AR(1) leger pour decorreler

    def similarity_report(self, real_df, labels, gen_df, curve_type) -> dict
        # pearson_profile, wasserstein_energy, mean_energy_gen/real,
        # peak_gen/real, we_ratio_gen/real, discriminative_score (LR test gen-vs-real)
```

Fallback automatique sur `utils/corpus.py:load_builtin_corpus()` si aucune calibration disponible.

## Pages

Les 3 pages suivent le meme pattern : sidebar pour le chargement (auto via `data_loader`) + selection compteur, contenu principal en vue lineaire ou tabbed selon la page. Aucun `file_uploader` — l'app ne demande aucun input fichier a l'utilisateur.

### pages/1_classification.py

Sidebar :
- Chargement automatique du timeseries + labels via `load_default_ts()` / `load_default_labels()`.
- Selectbox compteur a analyser (parmi les ids charges).

Vue lineaire (pas de tabs) :
1. Metriques compteur (conso moyenne, pic, ratio WE/SD, energie/j).
2. Jauge probabilite RS + classe predite.
3. Courbe de charge brute.
4. Profil moyen 24 h vs references RS/RP (`_make_rp_profile`, `_make_rs_profile`).
5. Top 5 facteurs determinants (permutation importance).
6. Positionnement du compteur sur scatter `zero_ratio` vs `ratio_we_wd`.
7. Performance : accuracy / F1 / recall RS via CV5 stratifiee + matrice de confusion.

### pages/2_prevision.py

Sidebar :
- Chargement automatique timeseries.
- Selectbox compteur.

Vue tabbed :
1. **Prevision** : historique + courbes des modeles superposees, ligne de separation train/test, annotation pic d'ecart Ridge.
2. **Resultats** : metriques par modele, verdict meilleur modele.
3. **Precision par heure** : MAE par pas d'horizon.
4. **Comment ca marche** : descriptions courtes Ridge / ARIMA / LSTM.

Train/test split : dernier `FCST_HORIZON_H * 2 = 48` pas mis de cote pour evaluation.

### pages/3_generation.py

Sidebar :
- Chargement automatique. Caption "X compteurs · Y points" ou "Corpus de reference (jeu de donnees absent)" en fallback.

Vue lineaire :
- Radio RS / RP.
- Bouton implicite : 50 courbes generees en mode bootstrap, 7 jours.
- Graphique des 50 courbes superposees + moyenne.
- Qualite de generation : profil moyen reel vs genere + distribution d'energie journaliere, metriques "Ressemblance de profil" (Pearson) et "Indiscernabilite" (score discriminatif).
- KPIs de coherence : energie moyenne, pic, ratio WE — chacun compare au reel.
- Bouton download CSV.

## app.py

```python
st.set_page_config(page_title="Enedis Analytics", layout="wide", initial_sidebar_state="expanded")
# Preload manuel des modules pour contourner le bug Python 3.14 + Streamlit 1.57
# (KeyError sur certains imports via runner _mpa_v1)
_preload("config", ...)              # config en premier (importe par tous les autres)
_preload("utils", ...)               # puis utils.* et models.*
_preload("utils.data_loader", ...)
# ...
# Charge assets/style.css
# Header HTML : "Enedis Analytics" + sous-titre
# Navigation : st.navigation([pg1, pg2, pg3], position="sidebar")
```

## Contraintes de deploiement

- `streamlit run app.py` suffit en local si `RES2-6-9.csv` est present.
- En l'absence du CSV local : telechargement automatique depuis `DATA_URL_TS` (GitHub Release publique, ~400 MB).
- Variables d'environnement optionnelles : `ENEDIS_MAX_METERS`, `ENEDIS_TS_URL`.
- Python >= 3.11. Deploye sur Streamlit Cloud avec Python 3.14.
- Premier cold start sur Cloud : 2-4 minutes (telecharger + parser + extraire features pour 500 compteurs). Cache disque `/tmp/enedis-data/` persiste tant que le container vit.
