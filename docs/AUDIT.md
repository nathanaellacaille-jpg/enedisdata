# Audit scientifique — Enedis Analytics

Date du snapshot : 2026-04-04. Perimetre : code source complet (config, utils, models, pages).
Aucune modification de code n'est recommandee sans validation sur donnees labellisees.

## Mise a jour 2026-05-26 — etat des recommandations

Plusieurs reco de l'audit ont ete implementees depuis. Le contenu ci-dessous est conserve comme reference historique mais ne reflete plus l'etat actuel pour les points suivants :

| Reco audit | Etat 2026-05-26 |
|---|---|
| #1 — Fourier `period` fixe a 48 dans `RidgeForecaster` | **Fait.** `models/forecaster.py:make_fourier_features(n, n_harmonics=3, period=48, offset=0)`. |
| #3 — Validation de la generation (DTW/MMD) | **Fait (proxy).** `CurveGenerator.similarity_report` retourne Pearson sur profil, Wasserstein sur energie, score discriminatif (LR test gen-vs-real). DTW formel non implemente. |
| #4 — Calibrer `GEN_NOISE_STD` sur les donnees | **Fait partiellement.** Le generateur calcule la variabilite empirique par slot et la combine avec le bruit AR(1) lors du `fit`. |
| #5 — Bruit AR(1) au lieu de i.i.d. dans le generateur | **Fait.** `GEN_NOISE_RHO=0.7` dans config, applique dans `generate` et `generate_bootstrap`. Mode bootstrap ajoute (reechantillonnage de slots reels). |
| #7 — Permutation importance au lieu de decomposition PCA | **Fait.** `EnergyClassifier.feature_importances` utilise `sklearn.inspection.permutation_importance` (n_repeats=10, scoring f1_weighted). La PCA a ete entierement retiree du pipeline. |
| Pipeline classification | **Change.** Plus de PCA : `StandardScaler -> RandomForest(300, class_weight=balanced)`. Seuil decisionnel abaisse a 0.35 pour gain de rappel RS. CV5 stratifiee en plus du split simple. |
| #6 — Split train/test temporel (classif) | Non fait. Toujours `train_test_split` aleatoire. |
| #2 — SARIMA au lieu de ARIMA(2,1,2) | Non fait. ARIMA fixe conserve. |
| #8 — Plus de 3 harmoniques Fourier dans `extract_features` | Non fait. Toujours 3. |
| Features meteo | Non fait. |

Les features ont par ailleurs ete enrichies depuis l'audit : 17 features au total (vs 7 initiales) — ajout de `zero_ratio`, `max_gap_days`, `n_absence_periods`, `active_days_ratio`, `seasonal_presence_gap`, `autocorr_lag48`, `morning_ratio`, `cv_weekly`, `skewness`.

---

## 1. Classification RS/RP

### Features

Les sept features implementees dans `utils/features.py` sont globalement coherentes avec
la litterature sur la caracterisation des profils residentiels basse tension.

- `ratio_we_wd` : le contraste weekend/semaine est la feature la plus discriminante pour
  detecter les residences secondaires (RS). Les RS presentent une forte activite le weekend
  et une consommation de base faible en semaine. Cette feature est explicitement mentionnee
  dans les travaux de McLoughlin et al. (2012) et Granell et al. (2015) sur la segmentation
  des profils residentiels.

- `cv_daily_energy` : la variabilite journaliere est plus elevee pour les RS (longues periodes
  d'inoccupation). Feature pertinente, utilisee dans Haben et al. (2014).

- `peak_hour_ratio` (18h-22h) et `night_ratio` (0h-6h) : ratios d'energie temporelle.
  Pertinents mais moins discriminants que `ratio_we_wd` pour la distinction RS/RP.
  McLoughlin et al. (2012) signalent que le pic vesperal est present dans les deux classes.

- `fourier_amp_1/2/3` : trois harmoniques sur le profil moyen journalier.
  Standard dans la litterature (Haben et al. 2014 utilisent jusqu'a dix harmoniques).
  Trois harmoniques capturent la structure journaliere basique mais peuvent manquer
  des patterns infra-journaliers fins. Extension a 6-8 harmoniques recommandee.

**Probleme identifie** : les amplitudes de Fourier sont calculees via `np.fft.rfft` sur le
profil moyen de 48 slots et normalisees par `len(profile)` (48). Ce calcul est correct
pour des amplitudes absolues mais n'est pas normalise par la variance du profil.
Deux compteurs de puissance tres differente peuvent avoir des amplitudes Fourier tres
differentes meme avec un profil temporel identique. Une normalisation par la puissance
maximale ou moyenne serait plus robuste.

### Pipeline PCA + Random Forest

La reduction PCA (5 composantes sur 7 features) apporte peu de compression
(71 % des dimensions conservees) et introduit un cout en interpretabilite.
Le Random Forest est intrinsiquement robuste a la multicolinearite et n'exige pas
de pre-reduction dimensionnelle pour 7 features. La PCA est justifiable a partir
de plusieurs dizaines de features (Granell et al. 2015 l'appliquent sur 53 features).

L'estimation des importances par decomposition via les vecteurs propres PCA
(`np.abs(components).T @ comp_imp`) est une approximation. Elle ne correspond pas
aux importances reelles sur l'espace original. Les valeurs SHAP ou la permutation
importance (Breiman 2001) sont plus rigoureuses et deja disponibles dans scikit-learn.

### Seuil et evaluation

Le seuil de classification a 0.5 est standard pour les probabilites non calibrees.
Si le jeu de donnees est desequilibre (proportion RS/RP inconnue dans l'open data),
un seuil optimise sur la courbe ROC serait preferable.

La separation train/test (`CLF_TEST_SIZE=0.30`) est effectuee de facon aleatoire.
Pour des series temporelles, cela introduit un risque de leakage temporel : des points
de la meme periode peuvent se retrouver dans train et test pour des compteurs differents.
Une separation strictement temporelle (ex. les 30 derniers jours en test) est recommandee.

---

## 2. Prevision J+1

### Ridge avec lags et Fourier

L'approche ARX (autoregressif avec features exogenes) est bien documentee pour la
prevision de charge a court terme (Taylor & McSharry 2008 ; Hong et al. 2016,
Global Energy Forecasting Competition). `FCST_N_LAGS=48` (24h) capture la periodicite
journaliere, ce qui est standard.

**Probleme identifie** : dans `make_fourier_features(n, n_harmonics)`, la periode est
relative a `n` (nombre de points d'entrainement moins les lags), non fixee a 48 (periode
journaliere) ou 336 (periode hebdomadaire). Les features Fourier generees ne correspondent
donc pas a des harmoniques de frequence physique fixe, ce qui reduit leur interpretabilite
et potentiellement leur pouvoir predictif. La formulation correcte utilise une periode
fixe : `sin(2 * pi * k * t / 48)` pour les harmoniques journalieres.

La prediction recursive sur 48 pas accumule les erreurs de facon quadratique.
La litterature privilegie les approches MIMO (multi-input multi-output) ou direct
multi-step pour les horizons superieurs a 6 pas (Ben Taieb et al. 2012).

### ARIMA(2,1,2)

L'ordre fixe (2,1,2) est sous-optimal pour des series echantillonnees a 30 minutes.
La saisonnalite dominante est journaliere (periode 48) et hebdomadaire (periode 336).
Un ARIMA non saisonnier ne capture pas ces composantes ; il produit systematiquement
une regression vers la moyenne sur les horizons superieurs a quelques heures.
SARIMA(p,d,q)(P,D,Q,48) ou SARIMA(p,d,q)(P,D,Q,336) est la reference standard
(Box & Jenkins 1970 ; Taylor 2003 pour les donnees electricite intra-journalieres).
Sans selection automatique de l'ordre (ex. AIC/BIC via `auto_arima`), le choix
(2,1,2) n'est pas justifiable a priori sur l'ensemble des compteurs.

### LSTM (SEQ_LEN=96, HIDDEN=64, LAYERS=2, EPOCHS=60)

L'architecture est coherente avec Kong et al. (2019) et Shi et al. (2018) sur les
smart meters. SEQ_LEN=96 (48h) est une fenetrage adequate.

**Points faibles** :
- L'entrainement est realise sur le dataset entier en un seul batch
  (`X = torch.tensor(...)` sans DataLoader). Sur de longues series, cela consomme
  beaucoup de memoire et empeche la regularisation par mini-batches.
- Il n'y a pas de jeu de validation separe pour l'arret precoce (early stopping).
  60 epochs sans regularisation risque le surapprentissage sur des series courtes.
- La prediction recursive step-by-step accumule les erreurs comme pour Ridge.

### Metriques

MAE, RMSE, MAPE et R² sont le quartet standard de la litterature de prevision energetique
(Hong et al. 2016). La protection contre la division par zero dans `compute_metrics`
exclut les valeurs exactement nulles, mais pas les valeurs proches de zero. Pour les
compteurs RS la nuit (consommation quasi nulle), le MAPE reste numeriquement instable.
Le sMAPE (symmetric MAPE) ou le WAPE (weighted APE) sont plus robustes dans ce cas.

---

## 3. Generation de courbes

L'approche profil moyen + bruit gaussien additif et independant est la methode la plus
simple envisageable. Elle presente deux limites structurelles :

1. **Incoherence temporelle** : le bruit est tire independamment pour chaque slot.
   Les courbes generees peuvent presenter des transitions brusques non physiques entre
   slots consecutifs. La consommation reelle est fortement autocorrelee a l'echelle
   30 minutes.

2. **Bruit non calibre** : `GEN_NOISE_STD=0.15` est une valeur arbitraire.
   La variance reelle du bruit varie selon l'heure, le type de compteur et la saison.
   Une calibration sur donnees reelles (ecart-type empirique par slot) produirait
   des courbes plus plausibles.

La litterature recente propose des alternatives nettement superieures :
- CGAN conditionnel (Fekri et al. 2020) : capture la structure de dependance temporelle
  et la distribution marginale simultanement.
- VAE (variational autoencoder) : espace latent continu, interpolation entre profils.
- Bootstrap sur donnees reelles (Yildiz et al. 2021) : reutilise directement la
  distribution empirique, sans hypothese paramétrique.

Il n'existe aucune metrique de validation de la qualite des courbes generees.
Le Dynamic Time Warping (DTW) ou le Maximum Mean Discrepancy (MMD) sont les
references pour evaluer la similitude entre distributions de courbes.

---

## 4. Features engineering — points transversaux

La normalisation `StandardScaler` est appliquee correctement avant PCA dans le
classifieur. En revanche, les features de lags dans `RidgeForecaster` ne sont pas
normalisees. Bien que les lags soient tous en kW (meme unite), l'absence de
standardisation peut biaiser les coefficients Ridge si les amplitudes varient
fortement entre compteurs.

L'absence de features meteorologiques est la limite la plus documentee. La temperature
exterieure explique 30 a 60 % de la variance de la consommation residentielles selon
Fan et al. (2012). Sur le dataset Enedis RES2-6-9kVA, l'impact est potentiellement
moindre (usage eclairage et electromenager dominant), mais l'integration de la
temperature comme covariate ameliore systematiquement la prevision.

---

## 5. Synthese

| Composant | Approche implementee | Standard litterature | Points forts | Points faibles | Priorite |
|---|---|---|---|---|---|
| Classification | PCA(5) + RF(300) | Oui | Interpretable, robuste | PCA injustifiee sur 7 features, leakage temporel possible | Moyenne |
| Prevision Ridge | Lags(48) + Fourier | Partiel | Rapide, interpretable | Periode Fourier non fixe, prediction recursive | Haute |
| Prevision ARIMA | (2,1,2) fixe | Non | Simple a ajuster | Pas de saisonnalite, ordre non selectionne | Haute |
| Prevision LSTM | Many-to-one, 2 couches | Partiel | Architecture correcte | Pas de mini-batches, pas d'early stopping | Moyenne |
| Generation | Profil moyen + bruit | Non standard | Simple, rapide | Bruit independant, non calibre, pas de validation | Haute |
| Features | 7 features + Fourier | Partiel | Features pertinentes | Fourier non normalise par variance, pas de meteo | Basse |

---

## 6. Recommandations

1. **Corriger la periode des features Fourier dans `RidgeForecaster`** (impact direct sur
   la qualite de prevision) : fixer la periode a 48 au lieu de `n`.
   Ref. : Taylor & McSharry (2008), *IEEE Trans. Power Systems*.

2. **Remplacer ARIMA(2,1,2) par SARIMA avec selection automatique de l'ordre**
   (AIC/BIC, librairie `pmdarima`). Ajouter une saisonnalite periode 48.
   Ref. : Taylor (2003), *Int. J. Forecasting*.

3. **Ajouter une validation de la generation** : calculer le DTW moyen entre courbes
   generees et courbes reelles pour chaque type (RS/RP).
   Ref. : Yildiz et al. (2021), *Applied Energy*.

4. **Calibrer `GEN_NOISE_STD` sur les donnees** : remplacer la valeur fixe par l'ecart-type
   empirique par slot, calcule lors du `CurveGenerator.fit()`.

5. **Introduire un autocorrelation du bruit dans le generateur** : remplacer le bruit i.i.d.
   par un processus AR(1) pour chaque courbe generee.
   Ref. : Fekri et al. (2020), *IEEE Trans. Smart Grid*.

6. **Separer train/test temporellement dans le classifieur** pour eviter le leakage.
   Utiliser les N derniers jours comme test plutot qu'un split aleatoire.

7. **Remplacer l'importance PCA decomposee par la permutation importance** de scikit-learn
   (`sklearn.inspection.permutation_importance`) pour une attribution plus fiable.

8. **Augmenter les harmoniques Fourier dans `extract_features` de 3 a 6-8** pour capturer
   des patterns infra-journaliers plus fins.
   Ref. : Haben et al. (2014), *IEEE Trans. Smart Grid*.

---

## References

- Haben S. et al. (2014). "A new error measure for forecasts of household-level,
  high resolution electrical energy consumption." *IEEE Trans. Smart Grid* 5(4).
- McLoughlin F. et al. (2012). "Characterising domestic electricity consumption patterns
  by dwelling and occupant socio-economic variables." *Energy and Buildings* 48.
- Granell R. et al. (2015). "Impacts of raw data temporal resolution." *Energy Conversion
  and Management* 89.
- Taylor J.W. (2003). "Short-term electricity demand forecasting using double seasonal
  exponential smoothing." *J. Operational Research Society* 54(8).
- Taylor J.W., McSharry P.E. (2008). "Short-term load forecasting methods." *IEEE Trans.
  Power Systems* 23(3).
- Hong T. et al. (2016). "Probabilistic energy forecasting: Global Energy Forecasting
  Competition 2014." *Int. J. Forecasting* 32(3).
- Kong W. et al. (2019). "Short-term residential load forecasting based on LSTM recurrent
  neural network." *IEEE Trans. Smart Grid* 10(1).
- Shi H. et al. (2018). "Deep learning for household load forecasting." *IEEE Trans. Smart
  Grid* 9(6).
- Fekri M.N. et al. (2020). "Generating energy data for machine learning with recurrent
  generative adversarial networks." *Energies* 13(1).
- Yildiz B. et al. (2021). "Recent advances in the analysis of residential electricity
  consumption and applications of smart meter data." *Applied Energy* 282.
- Ben Taieb S. et al. (2012). "A review and comparison of strategies for multi-step ahead
  time series forecasting based on the NN5 forecasting competition." *Expert Systems with
  Applications* 39(8).
- Fan S. et al. (2012). "Short-term load forecasting based on an adaptive hybrid method."
  *IEEE Trans. Power Systems* 27(1).
