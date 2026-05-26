import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import PAL, FCST_HORIZON_H, FCST_N_LAGS, STEPS_PER_DAY, FCST_ARIMA_ORDER, MAX_METERS_UPLOAD
from models.forecaster import RidgeForecaster, ARIMAForecaster, LSTMForecaster
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


@st.cache_resource
def _train_arima(series_key: str, series: list) -> ARIMAForecaster:
    """Entraine le forecaster ARIMA."""
    arr = np.array(series, dtype=float)
    mdl = ARIMAForecaster()
    mdl.fit(arr, order=FCST_ARIMA_ORDER)
    return mdl


@st.cache_resource
def _train_lstm(series_key: str, series: list) -> LSTMForecaster:
    """Entraine le forecaster LSTM."""
    arr = np.array(series, dtype=float)
    mdl = LSTMForecaster()
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

# ── Chargement sequentiel des modeles (un par un pour eviter les pics RAM) ────

# 1. Ridge — rapide, pas de spinner
ridge = _train_ridge(series_key, train_series.tolist())
ridge_pred = ridge.predict(horizon)
naive_pred = _naive_forecast(train_series, horizon)

# 2. ARIMA — moyenne complexite, chargement apres Ridge libere
arima_pred = None
with st.spinner("Chargement ARIMA..."):
    try:
        arima = _train_arima(series_key, train_series.tolist())
        arima_pred = arima.predict(horizon)
    except Exception as e:
        st.warning(f"ARIMA indisponible : {e}")

# 3. LSTM — le plus lourd, chargement en dernier
lstm_pred = None
with st.spinner("Chargement LSTM..."):
    try:
        lstm_model = _train_lstm(series_key, train_series.tolist())
        lstm_pred = lstm_model.predict(horizon)
    except Exception as e:
        st.warning(f"LSTM indisponible : {e}")

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
        x=future_ts, y=ridge_pred,
        mode="lines", name="Ridge",
        line=dict(color=PAL.ACCENT[0], width=1.5, dash="longdash"),
    ))
    if arima_pred is not None:
        fig.add_trace(go.Scatter(
            x=future_ts, y=arima_pred,
            mode="lines", name="ARIMA",
            line=dict(color=PAL.ACCENT[1], width=1.5, dash="dash"),
        ))
    if lstm_pred is not None:
        fig.add_trace(go.Scatter(
            x=future_ts, y=lstm_pred,
            mode="lines", name="LSTM",
            line=dict(color=PAL.ACCENT[2], width=1.5, dash="dashdot"),
        ))
    fig.add_trace(go.Scatter(
        x=future_ts, y=naive_pred,
        mode="lines", name="Reference",
        line=dict(color=PAL.TEXT_MUTED, width=1.5, dash="dot"),
    ))

    # Annotation pic d'ecart Ridge
    diff = np.abs(y_eval - ridge_pred)
    peak_idx = int(np.argmax(diff))
    peak_ts = future_ts[peak_idx] if peak_idx < len(future_ts) else future_ts[-1]
    fig.add_annotation(
        x=str(peak_ts), y=float(ridge_pred[peak_idx]),
        text="Ecart max Ridge",
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
    rows = [("Ridge", ridge_pred), ("Reference", naive_pred)]
    if arima_pred is not None:
        rows.append(("ARIMA", arima_pred))
    if lstm_pred is not None:
        rows.append(("LSTM", lstm_pred))

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
        x=hours, y=np.abs(y_eval - ridge_pred),
        mode="lines", name="Ridge",
        line=dict(color=PAL.ACCENT[0], width=1.5),
    ))
    if arima_pred is not None:
        fig_h.add_trace(go.Scatter(
            x=hours, y=np.abs(y_eval - arima_pred),
            mode="lines", name="ARIMA",
            line=dict(color=PAL.ACCENT[1], width=1.5, dash="dash"),
        ))
    if lstm_pred is not None:
        fig_h.add_trace(go.Scatter(
            x=hours, y=np.abs(y_eval - lstm_pred),
            mode="lines", name="LSTM",
            line=dict(color=PAL.ACCENT[2], width=1.5, dash="dashdot"),
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
    st.markdown("**Ridge**")
    st.caption(
        "Analyse les 8 derniers jours de consommation pour predire les 24 prochaines heures. "
        "Rapide et fiable, il apprend les habitudes journalieres et hebdomadaires du compteur."
    )
    st.markdown("**ARIMA**")
    st.caption(
        "Detecte les tendances et les repetitions a court terme dans la courbe de consommation. "
        "Performant quand la serie suit des patterns reguliers."
    )
    st.markdown("**LSTM**")
    st.caption(
        "Reseau de neurones capable de memoriser des comportements complexes sur plusieurs jours. "
        "Le plus puissant sur des donnees abondantes."
    )
    st.markdown("**Reference**")
    st.caption(
        "Repete simplement la consommation du jour precedent. "
        "Sert de base de comparaison : un bon modele doit faire mieux."
    )
