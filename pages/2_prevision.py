import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import PAL, FCST_HORIZON_H, FCST_N_LAGS, STEPS_PER_DAY, ROOT_DIR
from models.forecaster import LGBMForecaster, RidgeForecaster
from utils.data_loader import load_default_ts
from utils.metrics import compute_metrics


# Mapping nom JSON -> libelle affiche (cf. assets/forecast_baseline_metrics.json)
_MODEL_LABELS = {
    "lgbm": "LightGBM",
    "ridge": "Ridge",
    "naive_last_day": "Reference",
    "naive_weekly": "Hebdo",
    "seasonal_mean": "Moy. saisonniere",
    "naive_persistence": "Persistance",
}


@st.cache_data
def _load_forecast_baseline() -> dict | None:
    """Charge assets/forecast_baseline_metrics.json (Phase 0 v2) ou None."""
    p = Path(ROOT_DIR) / "assets" / "forecast_baseline_metrics.json"
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _render_perf_banner(metrics: dict) -> None:
    """Affiche le top 4 modeles MAE + table detaillee depuis Phase 0 v2."""
    models = metrics["models"]
    win_rates = metrics.get("win_rates_vs_naive_weekly", {})
    sorted_models = sorted(models.items(), key=lambda kv: kv[1]["mae_mean"])

    sample = metrics.get("sample", {})
    n_meters = sample.get("n_meters", "?")
    n_folds = sample.get("n_folds_per_meter", "?")

    st.markdown("**Performance globale des modeles**")
    st.caption(f"Backtest {n_meters} compteurs x {n_folds} folds rolling. "
               f"MAE moyenne en kW (plus bas = meilleur).")

    cols = st.columns(4)
    for i, (name, m) in enumerate(sorted_models[:4]):
        label = _MODEL_LABELS.get(name, name)
        cols[i].metric(f"#{i + 1} {label}", f"{m['mae_mean']:.3f} kW", delta_color="off")

    with st.expander("Detail complet et taux de victoire vs Hebdo (semaine -7j)"):
        rows = []
        for name, m in sorted_models:
            wr = win_rates.get(name, {})
            rows.append({
                "Modele": _MODEL_LABELS.get(name, name),
                "MAE (kW)": round(m["mae_mean"], 3),
                "RMSE (kW)": round(m.get("rmse_mean", float("nan")), 3),
                "Bat l'Hebdo": f"{wr.get('vs_naive_weekly', 0) * 100:.0f}%" if wr else "—",
                "Gain median": f"{wr.get('median_gain_pct', 0):+.1f}%" if wr else "—",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            f"Phase 0 v2 — calcule le {metrics.get('computed_at', '?')}. "
            "Les baselines naives (Reference = dernier jour repete, Hebdo = -7j, "
            "Moy. saisonniere = moyenne (jour, slot), Persistance = derniere valeur) "
            "servent de niveaux de comparaison."
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _plotly_base() -> dict:
    """Retourne le layout de base pour les graphiques Plotly."""
    return dict(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12, color=PAL.TEXT),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=11), bgcolor="rgba(0,0,0,0)", borderwidth=0,
        ),
        xaxis=dict(gridcolor="#F1F5F9", linecolor=PAL.BORDER,
                   tickfont=dict(size=11, color=PAL.TEXT_MUTED)),
        yaxis=dict(gridcolor="#F1F5F9", linecolor=PAL.BORDER,
                   tickfont=dict(size=11, color=PAL.TEXT_MUTED)),
    )


@st.cache_resource
def _train_ridge(series_key: str, series: list) -> RidgeForecaster:
    """Entraine le forecaster Ridge."""
    arr = np.array(series, dtype=float)
    mdl = RidgeForecaster()
    mdl.fit(arr)
    return mdl


@st.cache_resource
def _train_lgbm(series_key: str, series: list) -> LGBMForecaster:
    """Entraine le forecaster LightGBM DMSF (48 modeles)."""
    arr = np.array(series, dtype=float)
    mdl = LGBMForecaster()
    mdl.fit(arr)
    return mdl


def _naive_forecast(series: np.ndarray, h: int) -> np.ndarray:
    """Prevision naive : repete le dernier jour connu de la serie d'entrainement."""
    day = series[-STEPS_PER_DAY:]
    reps = (h // STEPS_PER_DAY) + 1
    return np.tile(day, reps)[:h]


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("**Donnees**")
    df = load_default_ts()
    st.session_state["_ts_df"] = df
    selected = None

    if df is not None:
        n_meters = df["meter_id"].nunique()
        st.caption(f"{n_meters} compteurs charges · {len(df):,} points")
        meter_ids = sorted(df["meter_id"].unique().tolist())
        selected = st.selectbox("Compteur", meter_ids, key="fcst_meter")

    st.markdown(
        '<div class="sidebar-footer">'
        '<div class="sidebar-badge">RES2-6-9 kVA</div><br>'
        'Enedis open data<br>Prevision 24h'
        "</div>",
        unsafe_allow_html=True,
    )

# ── main ─────────────────────────────────────────────────────────────────────

st.markdown("## Prevision")

# Bandeau performance globale : Phase 0 v2 (50 compteurs x 3 folds)
_baseline = _load_forecast_baseline()
if _baseline is not None:
    _render_perf_banner(_baseline)
    st.markdown("---")

if df is None or selected is None:
    st.caption("Jeu de donnees indisponible.")
    st.stop()

meter_df = df[df["meter_id"] == selected].sort_values("ts").reset_index(drop=True)
series = meter_df["kw"].values.astype(float)
ts_index = meter_df["ts"].values

if len(series) < FCST_N_LAGS + FCST_HORIZON_H * 2 + STEPS_PER_DAY:
    min_days = (FCST_N_LAGS + FCST_HORIZON_H * 2 + STEPS_PER_DAY) // STEPS_PER_DAY
    st.warning(f"Serie trop courte pour la prevision (minimum {min_days} jours).")
    st.stop()

horizon = FCST_HORIZON_H * 2

# Capping fenetre d'entrainement a 90 jours pour tenir dans 1 GB Streamlit Cloud.
# Phase 0 v2 a montre que MAE est equivalente entre 60-90 jours et serie complete
# (la saisonnalite hebdo + 4 jours de lags suffisent), donc pas de perte de qualite.
TRAIN_WINDOW_PTS = 90 * STEPS_PER_DAY
if len(series) > TRAIN_WINDOW_PTS + horizon:
    train_series = series[-TRAIN_WINDOW_PTS - horizon:-horizon]
    train_ts = ts_index[-TRAIN_WINDOW_PTS - horizon:-horizon]
else:
    train_series = series[:-horizon]
    train_ts = ts_index[:-horizon]
test_series = series[-horizon:]
test_ts = ts_index[-horizon:]

# Cache key inclut la taille de la fenetre pour invalider proprement
series_key = f"{selected}_{len(train_series)}"

# ── Chargement sequentiel des modeles (un par un pour eviter les pics RAM) ────

# 1. Ridge — rapide, pas de spinner
ridge = _train_ridge(series_key, train_series.tolist())
ridge_pred = ridge.predict(horizon)
naive_pred = _naive_forecast(train_series, horizon)

# 2. LightGBM DMSF — 48 modeles, spinner car premier chargement peut prendre ~15s
with st.spinner("Entrainement LightGBM..."):
    lgbm = _train_lgbm(series_key, train_series.tolist())
lgbm_pred = lgbm.predict(horizon)

last_train_ts = pd.Timestamp(train_ts[-1])
future_ts = test_ts
y_eval = test_series

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["Prevision", "Resultats", "Precision par heure", "Comment ca marche"])

# ── Tab 1 : Prevision ─────────────────────────────────────────────────────────
with tab1:
    n_hist = min(len(series), STEPS_PER_DAY * 7)
    hist_ts = ts_index[-n_hist:]
    hist_kw = series[-n_hist:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_ts, y=hist_kw,
        mode="lines", name="Historique",
        line=dict(color=PAL.REAL, width=2),
    ))
    fig.add_vline(x=str(last_train_ts), line_width=1, line_dash="solid", line_color=PAL.BORDER)
    fig.add_annotation(
        x=str(last_train_ts), y=1, yref="paper",
        text="Donnees connues", showarrow=False,
        font=dict(size=10, color=PAL.TEXT_MUTED),
        xanchor="right", xshift=-6,
    )
    # Predictions : du plus fonce (meilleur esperé) au plus clair (reference)
    fig.add_trace(go.Scatter(
        x=future_ts, y=lgbm_pred,
        mode="lines", name="LightGBM",
        line=dict(color=PAL.ACCENT[1], width=1.5, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=future_ts, y=ridge_pred,
        mode="lines", name="Ridge",
        line=dict(color=PAL.ACCENT[0], width=1.5, dash="longdash"),
    ))
    fig.add_trace(go.Scatter(
        x=future_ts, y=naive_pred,
        mode="lines", name="Reference",
        line=dict(color=PAL.TEXT_MUTED, width=1.5, dash="dot"),
    ))

    fig.update_layout(
        **_plotly_base(),
        margin=dict(l=16, r=16, t=32, b=16),
        title=f"Prevision 24h — {selected}",
        yaxis_title="kW",
    )
    st.plotly_chart(fig, width="stretch")

# ── Tab 2 : Resultats ─────────────────────────────────────────────────────────
with tab2:
    rows = [("LightGBM", lgbm_pred), ("Ridge", ridge_pred), ("Reference", naive_pred)]

    mae_vals = {name: compute_metrics(y_eval, pred)["MAE"] for name, pred in rows}
    best = min(mae_vals, key=mae_vals.get)
    mae_ref = mae_vals["Reference"]

    c1, c2 = st.columns(2)
    c1.metric("Meilleur modele", best, delta_color="off")
    c2.metric("Erreur reference (kW)", f"{mae_ref:.2f}", delta_color="off")

    # Classement MAE sous forme de barres
    sorted_names = sorted(mae_vals, key=mae_vals.get)
    sorted_mae = [mae_vals[n] for n in sorted_names]
    bar_colors = [PAL.ACCENT[0] if n == best else PAL.MULTI[4] for n in sorted_names]

    fig_bar = go.Figure(go.Bar(
        x=sorted_mae,
        y=sorted_names,
        orientation="h",
        marker_color=bar_colors,
        width=0.4,
    ))
    fig_bar.update_layout(
        **_plotly_base(),
        margin=dict(l=16, r=16, t=32, b=16),
        title="Erreur moyenne par modele (MAE en kW) — moins = mieux",
        xaxis_title="kW",
    )
    st.plotly_chart(fig_bar, width="stretch")

# ── Tab 3 : Precision par heure ───────────────────────────────────────────────
with tab3:
    hours = np.arange(1, horizon + 1) / 2.0

    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(
        x=hours, y=np.abs(y_eval - lgbm_pred),
        mode="lines", name="LightGBM",
        line=dict(color=PAL.ACCENT[1], width=1.5, dash="dash"),
    ))
    fig_h.add_trace(go.Scatter(
        x=hours, y=np.abs(y_eval - ridge_pred),
        mode="lines", name="Ridge",
        line=dict(color=PAL.ACCENT[0], width=1.5),
    ))
    fig_h.add_trace(go.Scatter(
        x=hours, y=np.abs(y_eval - naive_pred),
        mode="lines", name="Reference",
        line=dict(color=PAL.TEXT_MUTED, width=1.5, dash="dot"),
    ))
    fig_h.update_layout(
        **_plotly_base(),
        margin=dict(l=16, r=16, t=32, b=16),
        title="Ecart de prevision selon l'heure",
        xaxis_title="Heure de prevision",
        yaxis_title="Ecart (kW)",
    )
    st.plotly_chart(fig_h, width="stretch")

# ── Tab 4 : Comment ca marche ─────────────────────────────────────────────────
with tab4:
    st.markdown("**LightGBM** (DMSF, 48 modeles, n_lags=192, Fourier 6 harmoniques, calendrier)")
    st.caption(
        "Direct Multi-Step Forecasting : un LGBMRegressor distinct par pas horizon (h=0..47), "
        "entraine sur le residu vs J-1. Pas d'accumulation d'erreur contrairement a l'autoregressif. "
        "Parametres conservateurs pour Streamlit Cloud : n_estimators=200, num_leaves=31."
    )
    st.markdown("**Ridge** (n_lags=192, Fourier 6 harmoniques, StandardScaler, calendrier explicite)")
    st.caption(
        "Regression lineaire regularisee sur les 4 derniers jours de lags + harmoniques journalieres "
        "+ jour-de-la-semaine en one-hot + indicateur weekend. Apres tuning Phase 1, bat la Reference "
        "de 7% en moyenne sur 50 compteurs."
    )
    st.markdown("**Reference** (repeter le dernier jour)")
    st.caption(
        "Repete simplement la consommation du jour precedent slot par slot. Niveau de comparaison "
        "incontournable : sur smart meter 30 min sans meteo, 'demain = hier' est tres dur a battre "
        "et plusieurs modeles fancy echouent a faire mieux."
    )
