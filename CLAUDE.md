# CLAUDE.md

Lire ce fichier en entier avant toute action. Ces règles sont non-négociables.

## Architecture

```
enedis/
├── app.py                  point d'entrée Streamlit
├── config.py               palette + constantes (source unique)
├── requirements.txt
├── .streamlit/config.toml  thème light
├── assets/style.css        CSS global (source unique)
├── utils/                  parser, features, metrics
├── models/                 classifier, forecaster, generator
├── pages/                  1_classification, 2_prevision, 3_generation
└── docs/                   SPEC.md
```

## Règles code

- Python 3.11+
- Jamais de couleur hardcodée dans pages/ — toujours `PAL.xxx` de config.py
- Couleurs de traces de graphiques : `PAL.ACCENT[i]` (palette vive autorisée pour distinguer les courbes). L'UI/CSS reste en niveaux de gris.
- Jamais d'import circulaire : pages/ importe uniquement utils/ et models/
- `st.cache_data` sur toute lecture de fichier et tout calcul reproductible
- `st.cache_resource` sur tout entraînement de modèle
- Jamais `st.write()` — utiliser les composants natifs Streamlit
- `delta_color="off"` sur chaque `st.metric` sans exception
- Docstring une ligne sur chaque fonction

## Règles UI

- UI/CSS en niveaux de gris uniquement ; couleurs vives réservées aux traces (`PAL.ACCENT`).
- Zéro emoji dans tout le code : labels, titres, markdown, f-strings, commentaires, captions.
- Labels de métriques : 2-3 mots max.
- Appliquer systématiquement, y compris dans les fonctions helper et les f-strings.

## Vérifications avant de déclarer une tâche terminée

1. `grep -rn $'[\U0001F300-\U0001FFFF\U00002600-\U000027BF]' .` → vide
2. `grep -n "#EF4444\|#3B82F6\|#10B981\|#F59E0B\|#8B5CF6\|#60A5FA\|#1D4ED8\|#DBEAFE\|#FEE2E2\|#FEF9C3" pages/` → vide
3. `grep -n "st\.metric" pages/*.py | grep -v "delta_color"` → vide
4. `streamlit run app.py` sans erreur sur les 3 pages
