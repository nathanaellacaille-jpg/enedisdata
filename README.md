# Enedis Analytics

Analyse, prevision et generation de courbes de charge residentielles sur donnees Enedis open data.

Enedis Analytics est une application Streamlit pour la caracterisation de profils de consommation basse tension (RES2-6-9kVA). Elle couvre trois taches : classification residence secondaire / residence principale, prevision J+1, et generation de courbes synthetiques. Destinee a des usages d'exploration et de prototypage sur donnees ouvertes Enedis.

## Structure

```
app.py                  point d'entree Streamlit
config.py               palette et constantes
utils/                  parsing, features, metriques
models/                 classifier, forecaster, generator
pages/                  3 pages Streamlit
assets/style.css        CSS global
docs/SPEC.md            specification technique
docs/AUDIT.md           audit scientifique
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
- Format attendu : CSV avec colonnes `id`, `horodate`, `valeur`
- Telechargement : https://opendata.enedis.fr/datasets/courbes-de-charges-fictives-res2-6-9
- Limite upload : 200 Mo (Streamlit Cloud) — l'app extrait automatiquement 50 compteurs

## Pages

**Classification RS/RP**
Pipeline : extraction de features (ratio WE/semaine, variabilite journaliere,
amplitudes Fourier) -> PCA(5) -> Random Forest(300 arbres).
Sortie : probabilite RS par compteur, importance des features par permutation, matrice de confusion.

**Prevision J+1**
Trois modeles compares sur les 48 pas suivants :
Ridge Regression (lags + Fourier, periode 48), ARIMA(2,1,2), LSTM optionnel.
Metriques : MAE, RMSE, MAPE, R2.

**Generation de courbes**
Generateur conditionnel RS/RP par profil moyen calibre sur donnees reelles
+ bruit gaussien par slot. Sortie : courbes synthetiques exportables en CSV.

## Choix methodologiques

Les justifications scientifiques detaillees sont dans `docs/AUDIT.md`.
References principales : McLoughlin et al. (2012), Taylor & McSharry (2008),
Kong et al. (2019), Fekri et al. (2020).

## Licence

Donnees Enedis sous Licence Ouverte v2.0 (Etalab).
