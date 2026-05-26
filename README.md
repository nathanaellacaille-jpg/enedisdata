# Enedis Analytics

Analyse, prevision et generation de courbes de charge residentielles sur donnees Enedis open data.

Enedis Analytics est une application Streamlit pour la caracterisation de profils de consommation basse tension (RES2-6-9kVA). Elle couvre trois taches : classification residence secondaire / residence principale, prevision J+1, et generation de courbes synthetiques. Destinee a des usages d'exploration et de prototypage sur donnees ouvertes Enedis.

App deployee : https://enedisdata-pcv6jat7c92bp3fqqp494m.streamlit.app/

## Structure

```
app.py                  point d'entree Streamlit
config.py               palette, constantes, chemins et URL du dataset
utils/                  parsing, features, metriques, data_loader, corpus
models/                 classifier, forecaster, generator
pages/                  3 pages Streamlit (classification, prevision, generation)
assets/style.css        CSS global
docs/SPEC.md            specification technique a jour
docs/AUDIT.md           audit scientifique (snapshot 2026-04-04, recos partiellement implementees)
docs/UI.md              regles d'interface
```

## Installation

```bash
git clone https://github.com/nathanaellacaille-jpg/enedisdata
cd enedisdata
pip install -r requirements.txt
streamlit run app.py
```

## Donnees

- Dataset : Enedis open data RES2-6-9kVA (courbes de charge fictives)
- Format attendu : CSV avec colonnes `id`, `horodate`, `valeur` (separateur `;`, `,` ou tab auto-detecte)
- Telechargement open data : https://opendata.enedis.fr/datasets/courbes-de-charges-fictives-res2-6-9

**Chargement automatique** — plus de drag-and-drop. L'app cherche le CSV dans cet ordre :

1. `RES2-6-9.csv` a la racine du repo (mode local)
2. `/tmp/enedis-data/RES2-6-9.csv` (cache disque persistant entre runs)
3. Telechargement streame depuis `DATA_URL_TS` (defaut : GitHub Release `data-v1`)

Le fichier labels `RES2-6-9-labels.csv` (~9 KB, 500 ids) est tracke directement dans git.

### Configuration via variables d'environnement

| Variable | Defaut | Effet |
|---|---|---|
| `ENEDIS_MAX_METERS` | `500` | Plafond de compteurs charges. `none` / `0` / `all` = pas de cap. Sur Streamlit Cloud le dataset n'a que 500 compteurs uniques de toute facon. |
| `ENEDIS_TS_URL` | URL Release `data-v1` | Surcharge l'URL de fallback (heberge ailleurs). |

## Pages

**Classification RS/RP** — `pages/1_classification.py`

Pipeline : extraction de **26 features** (ratio WE/semaine, presence/absence, entropie hebdo, variabilite de l'heure du pic, signatures saisonnieres, 6 harmoniques Fourier, etc.) -> StandardScaler -> **StackingClassifier** (HistGBT + RandomForest + LogReg) avec meta LogReg. HistGBT params trouves par GridSearchCV. Seuil decisionnel appris dynamiquement par PR curve sur CV5 interne (max F1). Importances par permutation.

Sur 500 compteurs labellises (85.6 % RP / 14.4 % RS) : **F1 weighted 0.938, Recall RS 0.832, Precision RS 0.771, AUC 0.969** (CV5). 12 RS manquees sur 72 (vs 30 dans la baseline RandomForest initiale).

Vue lineaire (non-onglets) : metriques compteur, courbe de charge, profil moyen vs references RS/RP, facteurs determinants, positionnement parmi tous les compteurs, performance globale + matrice de confusion.

**Prevision J+1** — `pages/2_prevision.py`

Trois modeles sur les 48 pas (24 h) suivants :
- Ridge regression sur 384 lags (8 jours) + features Fourier (periode 48, 3 harmoniques)
- ARIMA(2,1,2)
- LSTM (2 couches, 48 unites cachees, 40 epochs)

Train/test : la derniere journee de la serie est mise de cote a l'entrainement et sert d'evaluation. Metriques : MAE, RMSE, MAPE, R2.

**Generation de courbes** — `pages/3_generation.py`

Generateur conditionnel RS/RP calibre sur les donnees reelles. Deux modes :
- **Parametrique** : profil moyen par type + bruit AR(1) (rho=0.7) en espace normalise + facteur d'amplitude journalier log-normal
- **Bootstrap** : reechantillonnage de slots reels par type, perturbation legere par bruit AR(1)

Fallback corpus built-in (`utils/corpus.py`) si le jeu de donnees n'est pas disponible.

Validation de qualite : similarite de profil (Pearson), distribution d'energie journaliere (Wasserstein), indiscernabilite (test de classification gen-vs-reel).

## Choix methodologiques

Les justifications scientifiques sont dans `docs/AUDIT.md`. Plusieurs recommandations de cet audit ont ete implementees depuis (Fourier a periode fixe, permutation importance, bruit AR(1) dans le generateur, validation par similarity_report). Restent ouvertes : SARIMA, features meteo, split train/test temporel pour la classification.

References principales : McLoughlin et al. (2012), Taylor & McSharry (2008), Kong et al. (2019), Fekri et al. (2020), Yildiz et al. (2021).

## Licence

Donnees Enedis sous Licence Ouverte v2.0 (Etalab).
