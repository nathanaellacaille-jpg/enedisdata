import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from config import PAL, CLF_TEST_SIZE, _make_rp_profile, _make_rs_profile
from models.classifier import EnergyClassifier
from utils.features import extract_features
from utils.parser import parse_timeseries, parse_labels


st.set_page_config(page_title="Classification", layout="wide")


# ── helpers ──────────────────────────────────────────────────────────────────

def _plotly_base() -> dict:
    """Retourne le layout de base pour les graphiques Plotly."""
    return dict(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12, color=PAL.TEXT),
        margin=dict(l=16, r=16, t=32, b=16),
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
    """Charge et parse le CSV timeseries depuis les bytes."""
    import io
    return parse_timeseries(io.BytesIO(file_bytes))


@st.cache_data
def _load_labels(file_bytes: bytes, file_name: str) -> dict:
    """Charge et parse le CSV labels depuis les bytes."""
    import io
    return parse_labels(io.BytesIO(file_bytes))


@st.cache_data
def _compute_features(df_json: str) -> pd.DataFrame:
    """Calcule les features depuis le dataframe serialise."""
    df = pd.read_json(df_json, orient="split")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return extract_features(df)


@st.cache_resource
def _train_model(features_json: str, labels_json: str):
    """Entraine le classifieur et retourne (model, X_test, y_test, y_pred)."""
    feat = pd.read_json(features_json, orient="split")
    labels_dict = pd.read_json(labels_json, typ="series").to_dict()
    common = [mid for mid in feat.index if str(mid) in labels_dict]
    if len(common) < 4:
        return None, None, None, None
    X = feat.loc[common]
    y = np.array([labels_dict[str(mid)] for mid in common])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CLF_TEST_SIZE, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    clf = EnergyClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, X_test, y_test, y_pred


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("**Donnees**")
    ts_file = st.file_uploader("Timeseries CSV", type=["csv"], key="clf_ts")
    lbl_file = st.file_uploader("Labels CSV (optionnel)", type=["csv"], key="clf_lbl")

    if ts_file:
        try:
            df = _load_ts(ts_file.getvalue(), ts_file.name)
            meter_ids = sorted(df["meter_id"].unique().tolist())
            selected = st.selectbox("Compteur", meter_ids, key="clf_meter")
        except ValueError as e:
            st.error(str(e))
            df = None
            selected = None
    else:
        df = None
        selected = None

    st.markdown(
        '<div class="sidebar-footer">'
        '<div class="sidebar-badge">RES2-6-9 kVA</div><br>'
        'Enedis open data<br>Classification RS / RP'
        "</div>",
        unsafe_allow_html=True,
    )

# ── main ─────────────────────────────────────────────────────────────────────

st.markdown("## Classification")

if df is None:
    st.caption("Chargez un fichier timeseries CSV pour commencer.")
    st.stop()

# Features globales
features_json = _compute_features(df.to_json(orient="split", date_format="iso"))
features = pd.read_json(features_json, orient="split")

# Labels + modele
labels = None
clf = None
if lbl_file:
    try:
        labels = _load_labels(lbl_file.getvalue(), lbl_file.name)
        labels_series = pd.Series(labels)
        clf, X_test, y_test, y_pred = _train_model(
            features_json,
            labels_series.to_json(),
        )
    except ValueError as e:
        st.error(str(e))

# Predictions sur tout le dataset
if clf is not None:
    all_pred = clf.predict(features)
    all_proba = clf.predict_proba(features)
    pred_series = pd.Series(all_pred, index=features.index)
    proba_series = pd.Series(all_proba, index=features.index)
else:
    pred_series = pd.Series(index=features.index, dtype=int)
    proba_series = pd.Series(index=features.index, dtype=float)

# Donnees du compteur selectionne
meter_df = df[df["meter_id"] == selected].sort_values("ts")
meter_feat = features.loc[[selected]] if selected in features.index else pd.DataFrame()

# Metriques du compteur
daily = meter_df.groupby(meter_df["ts"].dt.date)["kw"].sum() * 0.5
mean_conso = daily.mean() if not daily.empty else 0.0
peak_kw = meter_df["kw"].max()
we_wd = meter_feat["ratio_we_wd"].iloc[0] if not meter_feat.empty else 0.0
energy_j = mean_conso

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Resultat", "Vue d'ensemble", "Explication", "Performance", "Export"])

# ── Tab 1 : Resultat ──────────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Conso moy. (kWh/j)", f"{mean_conso:.1f}", delta_color="off")
    c2.metric("Pic (kW)", f"{peak_kw:.2f}", delta_color="off")
    c3.metric("Ratio WE/SD", f"{we_wd:.2f}", delta_color="off")
    c4.metric("Energie/j (kWh)", f"{energy_j:.1f}", delta_color="off")

    st.markdown("")

    # Courbe brute
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=meter_df["ts"], y=meter_df["kw"],
        mode="lines", name="Puissance (kW)",
        line=dict(color=PAL.REAL, width=1.5),
    ))
    fig.update_layout(
        **_plotly_base(),
        title="Courbe de charge",
        yaxis_title="kW",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Jauge RS/RP
    if not proba_series.empty and selected in proba_series.index:
        proba_rs = float(proba_series[selected])
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba_rs * 100,
            number={"suffix": "%", "font": {"size": 28, "color": PAL.TEXT}},
            title={"text": "Score RS", "font": {"size": 13, "color": PAL.TEXT_MUTED}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"size": 10, "color": PAL.TEXT_MUTED}},
                "bar": {"color": PAL.TEXT, "thickness": 0.25},
                "steps": [
                    {"range": [0, 33], "color": "#F8FAFC"},
                    {"range": [33, 66], "color": "#E2E8F0"},
                    {"range": [66, 100], "color": PAL.TEXT},
                ],
                "borderwidth": 0,
                "bgcolor": "white",
            },
        ))
        fig_gauge.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=16, r=16, t=32, b=16),
            font=dict(family="Inter, sans-serif", size=12, color=PAL.TEXT),
            height=220,
        )
        col_g, col_info = st.columns([1, 2])
        with col_g:
            st.plotly_chart(fig_gauge, use_container_width=True)
        with col_info:
            label_txt = "RS" if proba_rs >= 0.5 else "RP"
            st.metric("Classe predite", label_txt, delta_color="off")
    else:
        st.caption("Chargez des labels pour obtenir la classification.")

# ── Tab 2 : Vue d'ensemble ────────────────────────────────────────────────────
with tab2:
    if len(features) < 2:
        st.caption("Donnees insuffisantes pour la vue d'ensemble.")
    else:
        # Scatter features
        col_x = st.selectbox("Axe X", options=list(features.columns), index=0, key="scatter_x")
        col_y = st.selectbox("Axe Y", options=list(features.columns), index=1, key="scatter_y")

        if clf is not None and not pred_series.empty:
            colors = [PAL.MULTI[0] if p == 1 else PAL.MULTI[4] for p in pred_series.reindex(features.index, fill_value=0)]
            labels_txt = ["RS" if p == 1 else "RP" for p in pred_series.reindex(features.index, fill_value=0)]
        else:
            colors = [PAL.MULTI[2]] * len(features)
            labels_txt = ["?" for _ in range(len(features))]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=features[col_x], y=features[col_y],
            mode="markers",
            marker=dict(color=colors, size=7, opacity=0.8),
            text=[f"{mid}<br>{lbl}" for mid, lbl in zip(features.index, labels_txt)],
            hovertemplate="%{text}<br>X=%{x:.3f}<br>Y=%{y:.3f}<extra></extra>",
        ))
        fig2.update_layout(**_plotly_base(), xaxis_title=col_x, yaxis_title=col_y)
        st.plotly_chart(fig2, use_container_width=True)

        # Distribution predictions
        if clf is not None and not pred_series.empty:
            counts = pred_series.value_counts().sort_index()
            fig_dist = go.Figure(go.Bar(
                x=["RP", "RS"],
                y=[counts.get(0, 0), counts.get(1, 0)],
                marker_color=[PAL.MULTI[4], PAL.MULTI[0]],
                width=0.4,
            ))
            fig_dist.update_layout(**_plotly_base(), title="Distribution des classes predites", yaxis_title="Compteurs")
            st.plotly_chart(fig_dist, use_container_width=True)

        # Heatmap correlation
        corr = features.corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale=[[0, "#FFFFFF"], [0.5, "#94A3B8"], [1, "#0F172A"]],
            zmin=-1, zmax=1,
        ))
        fig_corr.update_layout(**_plotly_base(), title="Correlation features")
        st.plotly_chart(fig_corr, use_container_width=True)

# ── Tab 3 : Explication ───────────────────────────────────────────────────────
with tab3:
    if clf is None or meter_feat.empty:
        st.caption("Modele non disponible. Chargez des labels.")
    else:
        importances = clf.feature_importances(features)
        top5 = importances.head(5)

        fig_imp = go.Figure(go.Bar(
            x=top5.values[::-1],
            y=top5.index[::-1],
            orientation="h",
            marker_color=PAL.MULTI[0],
            width=0.5,
        ))
        fig_imp.update_layout(**_plotly_base(), title="Top 5 features", xaxis_title="Importance")
        st.plotly_chart(fig_imp, use_container_width=True)

        # Radar vs profil de reference
        rp_ref = _make_rp_profile()
        rs_ref = _make_rs_profile()
        meter_profile = meter_df.copy()
        meter_profile["slot"] = meter_profile["ts"].dt.hour * 2 + meter_profile["ts"].dt.minute // 30
        mp = meter_profile.groupby("slot")["kw"].mean().reindex(range(48), fill_value=0.0).values
        if mp.max() > 0:
            mp = mp / mp.max()

        slots = list(range(48))
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatter(
            x=slots, y=mp, mode="lines", name="Compteur",
            line=dict(color=PAL.REAL, width=1.5),
        ))
        fig_radar.add_trace(go.Scatter(
            x=slots, y=rp_ref, mode="lines", name="Ref RP",
            line=dict(color=PAL.ARIMA, width=1.5, dash="dash"),
        ))
        fig_radar.add_trace(go.Scatter(
            x=slots, y=rs_ref, mode="lines", name="Ref RS",
            line=dict(color=PAL.LSTM, width=1.5, dash="dot"),
        ))
        fig_radar.update_layout(
            **_plotly_base(),
            title="Profil moyen vs references",
            xaxis_title="Slot (30 min)",
            yaxis_title="Puissance normalisee",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ── Tab 4 : Performance ───────────────────────────────────────────────────────
with tab4:
    if clf is None or y_test is None:
        st.caption("Chargez des labels pour evaluer les performances.")
    else:
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        m1, m2 = st.columns(2)
        m1.metric("Accuracy", f"{acc:.2%}", delta_color="off")
        m2.metric("F1 (pondere)", f"{f1:.3f}", delta_color="off")

        # Matrice de confusion
        labels_names = ["RP", "RS"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=labels_names,
            y=labels_names,
            colorscale=[[0, "#FFFFFF"], [0.5, "#94A3B8"], [1, "#0F172A"]],
            showscale=True,
            text=cm.astype(str),
            texttemplate="%{text}",
        ))
        fig_cm.update_layout(
            **_plotly_base(),
            title="Matrice de confusion",
            xaxis_title="Predit",
            yaxis_title="Reel",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

# ── Tab 5 : Export ────────────────────────────────────────────────────────────
with tab5:
    if pred_series.empty:
        st.caption("Aucune prediction disponible.")
    else:
        export_df = pd.DataFrame({
            "meter_id": pred_series.index,
            "classe_predite": pred_series.values,
            "label": ["RS" if v == 1 else "RP" for v in pred_series.values],
        })
        if not proba_series.empty:
            export_df["proba_rs"] = proba_series.reindex(pred_series.index).values

        st.dataframe(export_df, use_container_width=True)
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("Telecharger CSV", csv_bytes, file_name="predictions.csv", mime="text/csv")
