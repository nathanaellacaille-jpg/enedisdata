# UI.md

Règles d'interface. Référencées par CLAUDE.md — s'appliquent à chaque fichier du projet.

## Principe

Interface sobre, light mode, inspiration Apple/Linear.
Aucune couleur décorative dans l'UI (CSS, métriques, bordures, fonds, texte) :
tout passe par le contraste et l'espacement.
Exception : les **traces de graphiques** peuvent utiliser la palette colorée
`PAL.ACCENT` pour distinguer les courbes (voir section Graphiques Plotly).

## Thème Streamlit — .streamlit/config.toml

```toml
[theme]
base = "light"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8FAFC"
textColor = "#0F172A"
font = "sans serif"

[server]
headless = true
```

## Palette — config.py

Niveaux de gris pour l'UI. Jamais de couleur sémantique (rouge/vert/bleu)
dans le CSS, les métriques, bordures, fonds ou texte.
Les couleurs vives sont réservées aux **traces de graphiques** via `PAL.ACCENT`.

```python
LR       = "#0F172A"
ARIMA    = "#475569"
LSTM     = "#94A3B8"
REAL     = "#0F172A"
BORDER   = "#E2E8F0"
TEXT     = "#0F172A"
TEXT_MUTED = "#64748B"
MULTI    = ["#0F172A","#1E293B","#334155","#475569","#64748B","#94A3B8","#CBD5E1","#E2E8F0"]
ACCENT   = ["#2563EB","#F59E0B","#10B981","#DB2777","#8B5CF6","#0891B2"]  # traces seulement
```

## CSS — assets/style.css

Règles clés :

- Font : Inter, Segoe UI — poids 400 et 500 uniquement, jamais 600/700
- Header : `border-bottom: 0.5px solid #E2E8F0` — pas de fond coloré
- Sidebar : `background: #FAFAFA`, `border-right: 0.5px solid #E2E8F0`
- Borders : toujours `0.5px solid` — jamais plus épais sauf élément actif (1.5px max)
- Border-radius : 6px pour composants, 8px pour cards
- Zéro gradient, zéro box-shadow décoratif
- Supprimer `#MainMenu`, `footer`, `[data-testid="stDecoration"]`

Métriques :
```css
background: #F8FAFC
label: 11px, color #64748B, weight 400
valeur: 22px, color #0F172A, weight 500
delta: 11px, color #64748B — toujours delta_color="off" côté Python
```

Boutons :
```css
background: transparent
border: 0.5px solid #CBD5E1
hover: background #F1F5F9
```

Tabs actif :
```css
border-bottom: 1.5px solid #0F172A
font-weight: 500
```

## Graphiques Plotly

Appliquer sur chaque `fig.update_layout()` :

```python
fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Inter, sans-serif", size=12, color="#0F172A"),
    margin=dict(l=16, r=16, t=32, b=16),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
        font=dict(size=11), bgcolor="rgba(0,0,0,0)", borderwidth=0,
    ),
    xaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0",
               tickfont=dict(size=11, color="#64748B")),
    yaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0",
               tickfont=dict(size=11, color="#64748B")),
)
```

Traces :
- Distinguer les courbes par couleur via `PAL.ACCENT[i]`, combinée si besoin
  à `line_dash` ("dash"/"dot") et à l'épaisseur (courbe principale jusqu'à ~2.8)
- Heatmap et gauge restent en niveaux de gris (pas de couleur sémantique)
- Colorscale heatmap : `[[0,"#FFFFFF"],[0.5,"#94A3B8"],[1,"#0F172A"]]`
- Gauge steps : `["#F8FAFC","#E2E8F0","#334155"]`

## Texte dans l'UI

- Zéro emoji dans tout le code : labels, titres, markdown, f-strings, commentaires, st.tab, st.radio, st.selectbox, tooltips, captions
- Texte court et factuel — pas de phrase explicative superflue
- Titres de page : nom seul, sans ponctuation décorative
- Labels de métriques : 2-3 mots max
- Captions : une phrase, ton neutre
