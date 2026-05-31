import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import PAL, FCST_HORIZON_H, STEPS_PER_DAY, ROOT_DIR
from models.forecaster import LGBMForecasterV2, NLinearGlobalForecaster, RidgeForecaster, LGBM_V2_LOOKBACK
from utils.data_loader import load_default_ts
from utils.metrics import compute_metrics


_MODEL_LABELS = {
    "lgbm": "LightGBM",
    "ridge": "Ridge",
    "nlinear": "NLinear",
    "naive_last_day": "Reference",
}


@st.cache_data
def _load_forecast_baseline() -> dict | None:
    p = Path(ROOT_DIR) / "assets" / "forecast_baseline_metrics.json"
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _render_perf_banner(metrics: dict) -> None:
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
        cols[i].metric(f"#{i + 1} {_MODEL_LABELS.get(name, name)}", f"{m['mae_mean']:.3f} kW", delta_color="off")

    with st.expander("Detail complet et taux de victoire vs Reference (J-7)"):
        rows = []
        for name, m in sorted_models:
            wr = win_rates.get(name, {})
            v = wr.get("vs_naive_weekly") if wr else None
            g = wr.get("median_gain_pct") if wr else None
            rows.append({
                "Modele": _MODEL_LABELS.get(name, name),
                "MAE (kW)": round(m["mae_mean"], 3),
                "RMSE (kW)": round(m.get("rmse_mean") or float("nan"), 3),
                "Bat Reference": f"{v * 100:.0f}%" if v is not None else "-",
                "Gain median": f"{g:+.1f}%" if g is not None else "-",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            f"Backtest calcule le {metrics.get('computed_at', '?')}. "
            "Reference = dernier jour repete slot a slot. "
            "NLinear = projection lineaire pre-entrainee sur les 500 compteurs. "
            "LightGBM evalue sur un sous-ensemble reduit (cout eleve)."
        )


def _plotly_base() -> dict:
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
    mdl = RidgeForecaster()
    mdl.fit(np.array(series, dtype=float))
    return mdl


@st.cache_resource
def _train_lgbm(series_key: str, series: list) -> LGBMForecasterV2:
    mdl = LGBMForecasterV2()
    mdl.fit(np.array(series, dtype=float))
    return mdl


@st.cache_resource
def _load_nlinear(series_key: str, series: list) -> NLinearGlobalForecaster | None:
    try:
        mdl = NLinearGlobalForecaster()
        mdl.fit(np.array(series, dtype=float))
        return mdl
    except FileNotFoundError:
        return None


def _naive_forecast(series: np.ndarray, h: int) -> np.ndarray:
    return np.tile(series[-STEPS_PER_DAY:], (h // STEPS_PER_DAY) + 1)[:h]


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

st.markdown("## Prevision")

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

_MIN_PTS = LGBM_V2_LOOKBACK + STEPS_PER_DAY * 2 + FCST_HORIZON_H * 2
if len(series) < _MIN_PTS:
    st.warning(f"Serie trop courte pour la prevision (minimum {_MIN_PTS // STEPS_PER_DAY + 1} jours).")
    st.stop()

horizon = FCST_HORIZON_H * 2
TRAIN_WINDOW_PTS = 90 * STEPS_PER_DAY
if len(series) > TRAIN_WINDOW_PTS + horizon:
    train_series = series[-TRAIN_WINDOW_PTS - horizon:-horizon]
    train_ts = ts_index[-TRAIN_WINDOW_PTS - horizon:-horizon]
else:
    train_series = series[:-horizon]
    train_ts = ts_index[:-horizon]
test_series = series[-horizon:]
test_ts = ts_index[-horizon:]

series_key = f"{selected}_{len(train_series)}"

ridge = _train_ridge(series_key, train_series.tolist())
ridge_pred = ridge.predict(horizon)
naive_pred = _naive_forecast(train_series, horizon)

nlinear = _load_nlinear(series_key, train_series.tolist())
nlinear_pred = nlinear.predict(horizon) if nlinear is not None else None

with st.spinner("Entrainement LightGBM..."):
    lgbm = _train_lgbm(series_key, train_series.tolist())
lgbm_pred = lgbm.predict(horizon)

last_train_ts = pd.Timestamp(train_ts[-1])

tab1, tab2, tab3, tab4 = st.tabs(["Prevision", "Resultats", "Precision par heure", "Comment ca marche"])

with tab1:
    hist_ts = ts_index[-min(len(series), STEPS_PER_DAY * 7):]
    hist_kw = series[-min(len(series), STEPS_PER_DAY * 7):]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_ts, y=hist_kw, mode="lines", name="Historique",
                             line=dict(color=PAL.REAL, width=2)))
    fig.add_vline(x=str(last_train_ts), line_width=1, line_dash="solid", line_color=PAL.BORDER)
    fig.add_annotation(x=str(last_train_ts), y=1, yref="paper", text="Donnees connues",
                       showarrow=False, font=dict(size=10, color=PAL.TEXT_MUTED),
                       xanchor="right", xshift=-6)
    fig.add_trace(go.Scatter(x=test_ts, y=lgbm_pred, mode="lines", name="LightGBM",
                             line=dict(color=PAL.ACCENT[1], width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=test_ts, y=ridge_pred, mode="lines", name="Ridge",
                             line=dict(color=PAL.ACCENT[0], width=1.5, dash="longdash")))
    if nlinear_pred is not None:
        fig.add_trace(go.Scatter(x=test_ts, y=nlinear_pred, mode="lines", name="NLinear",
                                 line=dict(color=PAL.ACCENT[2], width=1.5, dash="dashdot")))
    fig.add_trace(go.Scatter(x=test_ts, y=naive_pred, mode="lines", name="Reference",
                             line=dict(color=PAL.TEXT_MUTED, width=1.5, dash="dot")))
    fig.update_layout(**_plotly_base(), margin=dict(l=16, r=16, t=72, b=16),
                      title=dict(text=f"Prevision 24h : {selected}", y=0.98, yanchor="top", yref="container"),
                      yaxis_title="kW")
    st.plotly_chart(fig, width="stretch")

with tab2:
    rows = [("LightGBM", lgbm_pred), ("Ridge", ridge_pred), ("Reference", naive_pred)]
    if nlinear_pred is not None:
        rows.insert(2, ("NLinear", nlinear_pred))

    mae_vals = {name: compute_metrics(test_series, pred)["MAE"] for name, pred in rows}
    best = min(mae_vals, key=mae_vals.get)

    c1, c2 = st.columns(2)
    c1.metric("Meilleur modele", best, delta_color="off")
    c2.metric("Erreur reference (kW)", f"{mae_vals['Reference']:.2f}", delta_color="off")

    sorted_names = sorted(mae_vals, key=mae_vals.get)
    fig_bar = go.Figure(go.Bar(
        x=[mae_vals[n] for n in sorted_names],
        y=sorted_names,
        orientation="h",
        marker_color=[PAL.ACCENT[0] if n == best else PAL.MULTI[4] for n in sorted_names],
        width=0.4,
    ))
    fig_bar.update_layout(**_plotly_base(), margin=dict(l=16, r=16, t=72, b=16),
                          title=dict(text="Erreur moyenne par modele (MAE en kW), moins = mieux",
                                     y=0.98, yanchor="top", yref="container"),
                          xaxis_title="kW")
    st.plotly_chart(fig_bar, width="stretch")

with tab3:
    hours = np.arange(1, horizon + 1) / 2.0
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(x=hours, y=np.abs(test_series - lgbm_pred), mode="lines",
                               name="LightGBM", line=dict(color=PAL.ACCENT[1], width=1.5, dash="dash")))
    fig_h.add_trace(go.Scatter(x=hours, y=np.abs(test_series - ridge_pred), mode="lines",
                               name="Ridge", line=dict(color=PAL.ACCENT[0], width=1.5)))
    if nlinear_pred is not None:
        fig_h.add_trace(go.Scatter(x=hours, y=np.abs(test_series - nlinear_pred), mode="lines",
                                   name="NLinear", line=dict(color=PAL.ACCENT[2], width=1.5, dash="dashdot")))
    fig_h.add_trace(go.Scatter(x=hours, y=np.abs(test_series - naive_pred), mode="lines",
                               name="Reference", line=dict(color=PAL.TEXT_MUTED, width=1.5, dash="dot")))
    fig_h.update_layout(**_plotly_base(), margin=dict(l=16, r=16, t=72, b=16),
                        title=dict(text="Ecart de prevision selon l'heure",
                                   y=0.98, yanchor="top", yref="container"),
                        xaxis_title="Heure de prevision", yaxis_title="Ecart (kW)")
    st.plotly_chart(fig_h, width="stretch")

with tab4:
    st.markdown("**LightGBM v2** (DMSF, 48 modeles, 29 features domaine-metier, Fourier 6 harmoniques, calendrier)")
    st.caption(
        "Direct Multi-Step Forecasting : un LGBMRegressor distinct par pas horizon (h=0..47), "
        "entraine sur le residu vs J-1. Features : meme slot J-1/J-2/J-7, moyenne 7j, ecart-type 7j, "
        "lag-1/lag-2, delta journalier, Fourier x6, one-hot jour-de-semaine. "
        "29 features vs 192 lags bruts (V1) : moins de colinearity, meilleure generalisation."
    )
    st.markdown("**Ridge** (n_lags=192, Fourier 6 harmoniques, StandardScaler, calendrier explicite)")
    st.caption(
        "Regression lineaire regularisee sur les 4 derniers jours de lags + harmoniques journalieres "
        "+ jour-de-la-semaine en one-hot + indicateur weekend. Apres tuning Phase 1, bat la Reference "
        "de 7% en moyenne sur 50 compteurs."
    )
    if nlinear_pred is not None:
        st.markdown("**NLinear** (projection lineaire globale, pre-entraine sur 500 compteurs)")
        st.caption(
            "Soustrait la derniere valeur de la fenetre avant projection : pred = W.T @ (window - window[-1]) + window[-1]. "
            "W (192x48) pre-calcule une fois sur l'ensemble des 500 compteurs par accumulation des equations normales "
            "(O(L^2) memoire fixe). Prediction = un produit matrice-vecteur, aucun entrainement a la volee. "
            "Sur le backtest 5 compteurs x 3 folds : MAE 0.646, +11% vs Reference."
        )
    st.markdown("**Reference** (repeter le dernier jour)")
    st.caption(
        "Repete simplement la consommation du jour precedent slot par slot. Niveau de comparaison "
        "incontournable : sur smart meter 30 min sans meteo, 'demain = hier' est tres dur a battre "
        "et plusieurs modeles fancy echouent a faire mieux."
    )
