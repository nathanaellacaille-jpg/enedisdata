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
def _train_ridge(series_key: str, series: list, start_ts: pd.Timestamp | None = None) -> RidgeForecaster:
    """Entraine le forecaster Ridge."""
    arr = np.array(series, dtype=float)
    mdl = RidgeForecaster()
    mdl.fit(arr, start_ts=start_ts)
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
    use_arima = False
    use_lstm = False

    if df is not None:
        n_meters = df["meter_id"].nunique()
        st.caption(f"{n_meters} compteurs charges · {len(df):,} points")
        meter_ids = sorted(df["meter_id"].unique().tolist())
        selected = st.selectbox("Compteur", meter_ids, key="fcst_meter")

    st.markdown("**Modeles**")
    use_arima = st.checkbox("Activer ARIMA", value=False, key="use_arima")
    use_lstm = st.checkbox("Activer LSTM", value=False, key="use_lstm")

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

# Horizon en pas (24h = 48 pas)
horizon = FCST_HORIZON_H * 2
series_key = f"{selected}_{len(series)}"

# Split train/test : dernier horizon = test set, jamais vu a l'entrainement
train_series = series[:-horizon]
test_series = series[-horizon:]
train_ts = ts_index[:-horizon]
test_ts = ts_index[-horizon:]

# Entrainement sur train_series uniquement
start_ts = pd.Timestamp(train_ts[0]) if len(train_ts) > 0 else None
ridge = _train_ridge(series_key, train_series.tolist(), start_ts=start_ts)
ridge_pred = ridge.predict(horizon)

naive_pred = _naive_forecast(train_series, horizon)

arima = None
arima_pred = None
if use_arima:
    with st.spinner("Ajustement ARIMA..."):
        try:
            arima = _train_arima(series_key, train_series.tolist())
            arima_pred = arima.predict(horizon)
        except Exception as e:
            st.warning(f"ARIMA : {e}")

lstm_model = None
lstm_pred = None
if use_lstm:
    with st.spinner("Entrainement LSTM..."):
        try:
            lstm_model = _train_lstm(series_key, train_series.tolist())
            lstm_pred = lstm_model.predict(horizon)
        except Exception as e:
            st.warning(f"LSTM : {e}")

# Les predictions s'alignent sur la periode de test (timestamps reels)
last_train_ts = pd.Timestamp(train_ts[-1])
future_ts = test_ts

# Evaluation honnete : test_series n'a pas ete vu a l'entrainement
y_eval = test_series
eval_ts = test_ts

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Graphique", "Analyse", "Technique", "Horizon", "Guide"])

# ── Tab 1 : Graphique ─────────────────────────────────────────────────────────
with tab1:
    n_hist = min(len(series), STEPS_PER_DAY * 7)
    hist_ts = ts_index[-n_hist:]
    hist_kw = series[-n_hist:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_ts, y=hist_kw,
        mode="lines", name="Historique",
        line=dict(color=PAL.REAL, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=future_ts, y=ridge_pred,
        mode="lines", name="Ridge",
        line=dict(color=PAL.LR, width=1.5, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=future_ts, y=naive_pred,
        mode="lines", name="Naif",
        line=dict(color=PAL.NAIVE, width=1.5, dash="dot"),
    ))
    if arima_pred is not None:
        fig.add_trace(go.Scatter(
            x=future_ts, y=arima_pred,
            mode="lines", name="ARIMA",
            line=dict(color=PAL.ARIMA, width=1.5, dash="dashdot"),
        ))
    if lstm_pred is not None:
        fig.add_trace(go.Scatter(
            x=future_ts, y=lstm_pred,
            mode="lines", name="LSTM",
            line=dict(color=PAL.LSTM, width=1.5),
        ))

    # Ligne de separation train / test
    fig.add_vline(x=str(last_train_ts), line_width=1, line_dash="solid", line_color=PAL.BORDER)

    # Annotation pic d'erreur Ridge
    if y_eval is not None:
        diff = np.abs(y_eval - ridge_pred)
        peak_idx = int(np.argmax(diff))
        peak_ts = future_ts[peak_idx] if peak_idx < len(future_ts) else future_ts[-1]
        fig.add_annotation(
            x=str(peak_ts), y=float(ridge_pred[peak_idx]),
            text="Pic d'ecart",
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

# ── Tab 2 : Analyse ───────────────────────────────────────────────────────────
with tab2:
    if y_eval is None:
        st.caption("Serie trop courte pour calculer les metriques.")
    else:
        rows = []
        for name, pred in [("Ridge", ridge_pred), ("Naif", naive_pred)]:
            m = compute_metrics(y_eval, pred)
            rows.append({"Modele": name, **m})
        if arima_pred is not None:
            m = compute_metrics(y_eval, arima_pred)
            rows.append({"Modele": "ARIMA", **m})
        if lstm_pred is not None:
            m = compute_metrics(y_eval, lstm_pred)
            rows.append({"Modele": "LSTM", **m})

        metrics_df = pd.DataFrame(rows).set_index("Modele")
        metrics_df = metrics_df.round(4)
        st.dataframe(metrics_df, width="stretch")

        best = metrics_df["MAE"].idxmin()
        st.caption(f"Meilleur modele selon MAE : {best}")

# ── Tab 3 : Technique ─────────────────────────────────────────────────────────
with tab3:
    st.markdown("**Coefficients Ridge**")
    coef = ridge.coef_series()
    top_coef = coef.reindex(coef.abs().nlargest(20).index)
    fig_coef = go.Figure(go.Bar(
        x=top_coef.values[::-1],
        y=top_coef.index[::-1],
        orientation="h",
        marker_color=PAL.MULTI[0],
    ))
    fig_coef.update_layout(**_plotly_base(), margin=dict(l=16, r=16, t=32, b=16), title="Top 20 coefficients Ridge", xaxis_title="Valeur")
    st.plotly_chart(fig_coef, width="stretch")

    if arima is not None:
        st.markdown("**Resume ARIMA**")
        st.code(arima.summary(), language="text")

    if lstm_model is not None and lstm_model.losses:
        st.markdown("**Courbe de perte LSTM**")
        fig_loss = go.Figure(go.Scatter(
            x=list(range(1, len(lstm_model.losses) + 1)),
            y=lstm_model.losses,
            mode="lines",
            line=dict(color=PAL.LSTM, width=1.5),
        ))
        fig_loss.update_layout(**_plotly_base(), margin=dict(l=16, r=16, t=32, b=16), title="Loss LSTM", xaxis_title="Epoch", yaxis_title="MSE")
        st.plotly_chart(fig_loss, width="stretch")

# ── Tab 4 : Horizon ───────────────────────────────────────────────────────────
with tab4:
    if y_eval is None:
        st.caption("Serie trop courte.")
    else:
        fig_h = go.Figure()
        hours = np.arange(1, horizon + 1) / 2.0  # en heures
        for name, pred, color in [
            ("Ridge", ridge_pred, PAL.LR),
            ("Naif", naive_pred, PAL.NAIVE),
        ]:
            mae_per_step = np.abs(y_eval - pred)
            fig_h.add_trace(go.Scatter(
                x=hours, y=mae_per_step,
                mode="lines", name=name,
                line=dict(color=color, width=1.5),
            ))
        if arima_pred is not None:
            mae_per_step = np.abs(y_eval - arima_pred)
            fig_h.add_trace(go.Scatter(
                x=hours, y=mae_per_step,
                mode="lines", name="ARIMA",
                line=dict(color=PAL.ARIMA, width=1.5, dash="dash"),
            ))
        if lstm_pred is not None:
            mae_per_step = np.abs(y_eval - lstm_pred)
            fig_h.add_trace(go.Scatter(
                x=hours, y=mae_per_step,
                mode="lines", name="LSTM",
                line=dict(color=PAL.LSTM, width=1.5),
            ))
        fig_h.update_layout(
            **_plotly_base(),
            margin=dict(l=16, r=16, t=32, b=16),
            title="MAE par pas d'horizon",
            xaxis_title="Horizon (heures)",
            yaxis_title="MAE (kW)",
        )
        st.plotly_chart(fig_h, width="stretch")

# ── Tab 5 : Guide ─────────────────────────────────────────────────────────────
with tab5:
    st.markdown("**Ridge**")
    st.caption(
        "Regression lineaire regularisee sur les lags temporels et des features de Fourier. "
        "Rapide, interpretable, adapte aux series avec forte periodicite journaliere."
    )
    st.markdown("**ARIMA**")
    st.caption(
        "Modele autoressif integre a moyenne mobile. "
        "Capture les tendances et l'autocorrelation a court terme. "
        "Ordre (2,1,2) par defaut."
    )
    st.markdown("**LSTM**")
    st.caption(
        "Reseau de neurones recurrent a memoire longue. "
        "Peut capturer des dependances non lineaires complexes. "
        "Necessite plus de donnees et de temps d'entrainement."
    )
