# Enedis Analytics

Analyse, prevision et generation de courbes de charge residentielles sur donnees Enedis open data.

Enedis Analytics est une application Streamlit pour la caracterisation de profils de consommation basse tension (RES2-6-9kVA). Elle couvre trois taches : classification residence secondaire / residence principale, prevision J+1, et generation de courbes synthetiques. Destinee a des usages d'exploration et de prototypage sur donnees ouvertes Enedis.

App deployee : https://enedisdata-pcv6jat7c92bp3fqqp494m.streamlit.app/

## Structure

```
app.py                  point d'entree Streamlit
config.py               palette, constantes, chemins et URL du dataset
utils/                  parsing, features, metriques, data_loader
models/                 classifier, forecaster, generator
pages/                  3 pages Streamlit (classification, prevision, generation)
assets/style.css        CSS global
docs/SPEC.md            specification technique a jour
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

## Pages

**Classification RS/RP** : `pages/1_classification.py`

Pipeline : extraction de **26 features** (ratio WE/semaine, presence/absence, entropie hebdo, variabilite de l'heure du pic, signatures saisonnieres, 6 harmoniques Fourier, etc.) -> StandardScaler -> **StackingClassifier** (HistGBT + RandomForest + LogReg) avec meta LogReg. HistGBT params trouves par GridSearchCV. Seuil decisionnel appris dynamiquement par PR curve sur CV5 interne (max F1). Importances par permutation.

Sur 500 compteurs labellises (85.6 % RP / 14.4 % RS) : **F1 weighted 0.943, Recall RS 0.765, Precision RS 0.835, AUC 0.969, accuracy 0.944** (CV5). Environ 17 RS manquees sur 72.

Vue lineaire (non-onglets) : metriques compteur, courbe de charge, profil moyen vs references RS/RP, facteurs determinants, positionnement parmi tous les compteurs, performance globale + matrice de confusion.

**Prevision J+1** : `pages/2_prevision.py`

Trois modeles sur les 48 pas (24 h) suivants, compares a une reference naive (dernier jour repete slot a slot) :
- **LightGBM v2** : Direct Multi-Step Forecasting, 48 modeles independants (un par pas horizon), 29 features domaine-metier (meme slot J-1/J-2/J-7, moyenne et ecart-type 7j, lags, delta journalier, Fourier 6 harmoniques, calendrier one-hot), entraine sur le residu vs J-1
- **Ridge** : regression lineaire regularisee (RidgeCV) sur 192 lags + Fourier 6 harmoniques + calendrier explicite, StandardScaler, apprise sur le residu vs J-1
- **NLinear global** : projection lineaire pre-entrainee une fois sur les 500 compteurs (poids 192x48), aucun entrainement a la volee

Evaluation : les 24 dernieres heures de chaque serie sont mises de cote pour la vue par compteur ; la performance globale provient d'un backtest rolling (50 compteurs x 3 folds). Metriques : MAE, RMSE.

**Generation de courbes** : `pages/3_generation.py`

Generateur conditionnel RS/RP calibre sur les donnees reelles. Deux modes :
- **Parametrique** : profil moyen par type + bruit AR(1) (rho=0.7) en espace normalise + facteur d'amplitude journalier log-normal
- **Bootstrap** : reechantillonnage de journees reelles par type, conditionne semaine/weekend, perturbation legere par facteur journalier et bruit AR(1)

Validation de qualite : similarite de profil (Pearson), distribution d'energie journaliere (Wasserstein), indiscernabilite (test de classification gen-vs-reel).

## Choix methodologiques

Choix implementes : stacking pour la classification avec seuil F1-optimal, prevision residuelle vs J-1 (Ridge, LightGBM DMSF, NLinear global), Fourier a periode fixe, permutation importance, bruit AR(1) dans le generateur, validation par similarity_report. Pistes ouvertes : features meteo, split train/test temporel pour la classification.
