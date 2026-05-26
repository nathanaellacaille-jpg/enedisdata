import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import PAL, FCST_HORIZON_H, FCST_N_LAGS, STEPS_PER_DAY, MAX_METERS_UPLOAD
from models.forecaster import RidgeForecaster
from utils.metrics import compute_metrics
from utils.parser import parse_timeseries


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


@st.cache_data
def _load_ts(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Charge et parse le CSV timeseries."""
    import io
    return parse_timeseries(io.BytesIO(file_bytes), max_meters=MAX_METERS_UPLOAD)


@st.cache_resource
def _train_ridge(series_key: str, series: list) -> RidgeForecaster:
    """Entraine le forecaster Ridge."""
    arr = np.array(series, dtype=float)
    mdl = RidgeForecaster()
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
    ts_file = st.file_uploader("Timeseries CSV", type=["csv"], key="fcst_ts")

    if ts_file is not None:
        if st.session_state.get("_ts_file_name") != ts_file.name:
            try:
                st.session_state["_ts_df"] = _load_ts(ts_file.getvalue(), ts_file.name)
                st.session_state["_ts_file_name"] = ts_file.name
            except ValueError as e:
                st.error(str(e))

    df = st.session_state.get("_ts_df")
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

if df is None or selected is None:
    st.caption("Chargez un fichier timeseries CSV pour commencer.")
    st.stop()

meter_df = df[df["meter_id"] == selected].sort_values("ts").reset_index(drop=True)
series = meter_df["kw"].values.astype(float)
ts_index = meter_df["ts"].values

if len(series) < FCST_N_LAGS + FCST_HORIZON_H * 2 + STEPS_PER_DAY:
    min_days = (FCST_N_LAGS + FCST_HORIZON_H * 2 + STEPS_PER_DAY) // STEPS_PER_DAY
    st.warning(f"Serie trop courte pour la prevision (minimum {min_days} jours).")
    st.stop()

horizon = FCST_HORIZON_H * 2
series_key = f"{selected}_{len(series)}"

train_series = series[:-horizon]
test_series = series[-horizon:]
train_ts = ts_index[:-horizon]
test_ts = ts_index[-horizon:]

ridge = _train_ridge(series_key, train_series.tolist())
ridge_pred = ridge.predict(horizon)
naive_pred = _naive_forecast(train_series, horizon)

last_train_ts = pd.Timestamp(train_ts[-1])
future_ts = test_ts
y_eval = test_series

# Metriques
m_ridge = compute_metrics(y_eval, ridge_pred)
m_naive = compute_metrics(y_eval, naive_pred)

# Tabs
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
    # Ligne de separation donnees connues / prevision
    fig.add_vline(x=str(last_train_ts), line_width=1, line_dash="solid", line_color=PAL.BORDER)
    fig.add_annotation(
        x=str(last_train_ts), y=1, yref="paper",
        text="Donnees connues", showarrow=False,
        font=dict(size=10, color=PAL.TEXT_MUTED),
        xanchor="right", xshift=-6,
    )
    fig.add_trace(go.Scatter(
        x=future_ts, y=ridge_pred,
        mode="lines", name="Prevision",
        line=dict(color=PAL.LR, width=1.5, dash="longdash"),
    ))
    fig.add_trace(go.Scatter(
        x=future_ts, y=naive_pred,
        mode="lines", name="Reference (jour precedent)",
        line=dict(color=PAL.TEXT_MUTED, width=1.5, dash="dot"),
    ))

    # Annotation pic d'ecart
    diff = np.abs(y_eval - ridge_pred)
    peak_idx = int(np.argmax(diff))
    peak_ts = future_ts[peak_idx] if peak_idx < len(future_ts) else future_ts[-1]
    fig.add_annotation(
        x=str(peak_ts), y=float(ridge_pred[peak_idx]),
        text="Ecart max",
        showarrow=True, arrowhead=2, arrowcolor=PAL.TEXT_MUTED,
        font=dict(size=10, color=PAL.TEXT_MUTED),
    )

    fig.update_layout(
        **_plotly_base(),
        margin=dict(l=16, r=16, t=32, b=16),
        title=f"Prevision 24h — {selected}",
        yaxis_title="kW",
    )
    st.plotly_chart(fig, width="stretch")

# ── Tab 2 : Resultats ─────────────────────────────────────────────────────────
with tab2:
    mae_ridge = m_ridge["MAE"]
    mae_naive = m_naive["MAE"]
    gain = (mae_naive - mae_ridge) / mae_naive * 100 if mae_naive > 0 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Erreur prevision (kW)", f"{mae_ridge:.2f}", delta_color="off")
    c2.metric("Erreur reference (kW)", f"{mae_naive:.2f}", delta_color="off")
    c3.metric("Gain vs reference", f"{gain:.0f}%", delta_color="off")

    st.caption(
        "L'erreur est la difference moyenne entre la consommation prevue et la consommation reelle, "
        "exprimee en kilowatts. La reference repete simplement le jour precedent."
    )

# ── Tab 3 : Precision par heure ───────────────────────────────────────────────
with tab3:
    hours = np.arange(1, horizon + 1) / 2.0

    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(
        x=hours, y=np.abs(y_eval - ridge_pred),
        mode="lines", name="Prevision",
        line=dict(color=PAL.LR, width=1.5),
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
    st.markdown("**Prevision intelligente**")
    st.caption(
        "Le modele analyse les 8 derniers jours de consommation pour predire les 24 prochaines heures. "
        "Il apprend automatiquement les habitudes journalieres et hebdomadaires du compteur."
    )
    st.markdown("**Reference simple**")
    st.caption(
        "Pour comparaison, la reference repete simplement la consommation du jour precedent. "
        "Un bon modele doit systematiquement faire mieux que cette reference."
    )
