import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold

from config import PAL, CLF_TEST_SIZE, CLF_N_TREES, CLF_RS_THRESHOLD, MAX_METERS_UPLOAD, _make_rp_profile, _make_rs_profile
from models.classifier import EnergyClassifier
from utils.features import extract_features
from utils.parser import parse_timeseries, parse_labels

_FEAT_LABELS = {
    "zero_ratio": "Taux d'absence",
    "ratio_we_wd": "Ratio WE / semaine",
    "max_gap_days": "Max jours absents",
    "n_absence_periods": "Nb periodes d'absence",
    "active_days_ratio": "Taux de jours occupes",
    "seasonal_presence_gap": "Ecart presence ete-hiver",
    "autocorr_lag48": "Autocorr. jour precedent",
    "peak_hour_ratio": "Pic soir (18h-22h)",
    "night_ratio": "Conso nuit",
    "morning_ratio": "Conso matin",
    "cv_daily_energy": "Variabilite quotidienne",
    "cv_weekly": "Variabilite hebdo",
    "seasonal_ratio": "Ratio ete / hiver",
    "skewness": "Asymetrie",
    "fourier_amp_1": "Periodicite J",
    "fourier_amp_2": "Periodicite J/2",
    "fourier_amp_3": "Periodicite J/3",
}


# ── helpers ──────────────────────────────────────────────────────────────────

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
    """Charge et parse le CSV timeseries depuis les bytes."""
    import io
    return parse_timeseries(io.BytesIO(file_bytes), max_meters=MAX_METERS_UPLOAD)


@st.cache_data
def _load_labels(file_bytes: bytes, file_name: str) -> dict:
    """Charge et parse le CSV labels depuis les bytes."""
    import io
    return parse_labels(io.BytesIO(file_bytes))


@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_json(date_format='iso')})
def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les features par compteur."""
    return extract_features(df)


@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_json(date_format='iso')})
def _predict_all(features: pd.DataFrame, labels: dict) -> tuple:
    """Predit classes et probabilites pour tous les compteurs (cache par dataset)."""
    clf, *_ = _train_model(features, labels)
    if clf is None:
        return pd.Series(index=features.index, dtype=int), pd.Series(index=features.index, dtype=float)
    proba = clf.predict_proba(features)
    return (
        pd.Series((proba >= CLF_RS_THRESHOLD).astype(int), index=features.index),
        pd.Series(proba, index=features.index),
    )


@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_json(date_format='iso')})
def _compute_corr(features: pd.DataFrame) -> pd.DataFrame:
    """Calcule la matrice de correlation des features (cache par dataset)."""
    return features.corr()


@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_json(date_format='iso')})
def _compute_importances(features: pd.DataFrame, labels: dict) -> pd.Series:
    """Calcule la permutation importance (cache par dataset)."""
    clf, *_ = _train_model(features, labels)
    if clf is None:
        return pd.Series(dtype=float)
    common = [mid for mid in features.index if str(mid) in labels]
    if len(common) >= 2:
        X_imp = features.loc[common]
        y_imp = np.array([labels[str(mid)] for mid in common])
    else:
        X_imp = features
        y_imp = pd.Series(index=features.index, dtype=int).fillna(0).values.astype(int)
    return clf.feature_importances(X_imp, y_imp)


@st.cache_resource(hash_funcs={pd.DataFrame: lambda df: df.to_json(date_format='iso')})
def _train_model(features: pd.DataFrame, labels: dict):
    """Entraine le classifieur et retourne (model, X_test, y_test, y_proba_test, cv_scores)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    common = [mid for mid in features.index if str(mid) in labels]
    if len(common) < 4:
        return None, None, None, None, None
    X = features.loc[common]
    y = np.array([labels[str(mid)] for mid in common])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CLF_TEST_SIZE, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    clf = EnergyClassifier()
    clf.fit(X_train, y_train)
    y_proba_test = clf.predict_proba(X_test)

    # Validation croisée stratifiée 5 folds, métriques calculées au seuil CLF_RS_THRESHOLD
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, recs = [], [], []
    for tr_idx, te_idx in cv.split(X, y):
        cv_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=CLF_N_TREES, random_state=42, n_jobs=-1, class_weight="balanced")),
        ])
        cv_pipe.fit(X.iloc[tr_idx], y[tr_idx])
        proba_fold = cv_pipe.predict_proba(X.iloc[te_idx])[:, 1]
        pred_fold = (proba_fold >= CLF_RS_THRESHOLD).astype(int)
        accs.append(accuracy_score(y[te_idx], pred_fold))
        f1s.append(f1_score(y[te_idx], pred_fold, average="weighted"))
        recs.append(recall_score(y[te_idx], pred_fold, pos_label=1, zero_division=0))
    cv_scores = {
        "accuracy": float(np.mean(accs)),
        "f1": float(np.mean(f1s)),
        "recall_rs": float(np.mean(recs)),
        "accuracy_std": float(np.std(accs)),
        "f1_std": float(np.std(f1s)),
        "recall_rs_std": float(np.std(recs)),
    }
    return clf, X_test, y_test, y_proba_test, cv_scores


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("**Donnees**")
    ts_file = st.file_uploader("Timeseries CSV", type=["csv"], key="clf_ts")
    lbl_file = st.file_uploader("Labels CSV (optionnel)", type=["csv"], key="clf_lbl")

    if ts_file is not None:
        if st.session_state.get("_ts_file_name") != ts_file.name:
            try:
                st.session_state["_ts_df"] = _load_ts(ts_file.getvalue(), ts_file.name)
                st.session_state["_ts_file_name"] = ts_file.name
            except ValueError as e:
                st.error(str(e))

    if lbl_file is not None:
        if st.session_state.get("_labels_file_name") != lbl_file.name:
            try:
                st.session_state["_labels"] = _load_labels(lbl_file.getvalue(), lbl_file.name)
                st.session_state["_labels_file_name"] = lbl_file.name
            except ValueError as e:
                st.error(str(e))

    df = st.session_state.get("_ts_df")
    if df is not None:
        n_meters = df["meter_id"].nunique()
        st.caption(f"{n_meters} compteurs charges · {len(df):,} points")
        meter_ids = sorted(df["meter_id"].unique().tolist())
        selected = st.selectbox("Compteur", meter_ids, key="clf_meter")
    else:
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
features = _compute_features(df)

# Labels + modele
labels = st.session_state.get("_labels")
clf = None
y_proba_test = None
cv_scores = None
if labels is not None:
    clf, X_test, y_test, y_proba_test, cv_scores = _train_model(features, labels)
y_pred = (y_proba_test >= CLF_RS_THRESHOLD).astype(int) if y_proba_test is not None else None

# Predictions sur tout le dataset
if clf is not None:
    pred_series, proba_series = _predict_all(features, labels)
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

# ── Vue unifiee ────────────────────────────────────────────────────────────────

st.markdown(f"### Compteur {selected}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Conso moy. (kWh/j)", f"{mean_conso:.1f}", delta_color="off")
c2.metric("Pic (kW)", f"{peak_kw:.2f}", delta_color="off")
c3.metric("Ratio WE/SD", f"{we_wd:.2f}", delta_color="off")
c4.metric("Energie/j (kWh)", f"{energy_j:.1f}", delta_color="off")

# Verdict de classification
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
                {"range": [66, 100], "color": "#334155"},
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
        st.plotly_chart(fig_gauge, width="stretch")
    with col_info:
        label_txt = "RS" if proba_rs >= CLF_RS_THRESHOLD else "RP"
        st.metric("Classe predite", label_txt, delta_color="off")
else:
    st.caption("Chargez des labels pour obtenir la classification.")

st.markdown("---")

# Courbe de charge
st.markdown("### Courbe de charge")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=meter_df["ts"], y=meter_df["kw"],
    mode="lines", name="Puissance (kW)",
    line=dict(color=PAL.ACCENT[0], width=1.5),
))
fig.update_layout(
    **_plotly_base(),
    margin=dict(l=16, r=16, t=32, b=16),
    yaxis_title="kW",
)
st.plotly_chart(fig, width="stretch")

# Explication : profil + facteurs (necessite le modele)
if clf is not None and not meter_feat.empty:
    st.markdown("---")
    st.markdown("### Profil de consommation")

    rp_ref = _make_rp_profile()
    rs_ref = _make_rs_profile()
    meter_profile = meter_df.copy()
    meter_profile["slot"] = meter_profile["ts"].dt.hour * 2 + meter_profile["ts"].dt.minute // 30
    mp = meter_profile.groupby("slot")["kw"].mean().reindex(range(48), fill_value=0.0).values
    if mp.max() > 0:
        mp = mp / mp.max()

    slots = list(range(48))
    fig_radar = go.Figure()
    tick_vals = list(range(0, 48, 4))
    tick_text = [f"{h}h" for h in range(0, 24, 2)]
    fig_radar.add_trace(go.Scatter(
        x=slots, y=mp, mode="lines", name="Compteur",
        line=dict(color=PAL.ACCENT[0], width=2.8),
    ))
    fig_radar.add_trace(go.Scatter(
        x=slots, y=rp_ref, mode="lines", name="Ref residence principale (RP)",
        line=dict(color=PAL.ACCENT[1], width=1.8, dash="dash"),
    ))
    fig_radar.add_trace(go.Scatter(
        x=slots, y=rs_ref, mode="lines", name="Ref residence secondaire (RS)",
        line=dict(color=PAL.ACCENT[2], width=1.8, dash="dot"),
    ))
    _radar_layout = {
        **_plotly_base(),
        "margin": dict(l=16, r=16, t=32, b=16),
        "title": "Profil de consommation vs references",
        "yaxis_title": "Puissance (normalisee)",
        "xaxis": dict(
            gridcolor="#F1F5F9", linecolor=PAL.BORDER,
            tickfont=dict(size=11, color=PAL.TEXT_MUTED),
            tickvals=tick_vals, ticktext=tick_text,
        ),
    }
    fig_radar.update_layout(**_radar_layout)
    st.plotly_chart(fig_radar, width="stretch")

    st.markdown("---")
    st.markdown("### Facteurs determinants")
    importances = _compute_importances(features, labels)
    top5 = importances.head(5)
    fig_imp = go.Figure(go.Bar(
        x=top5.values[::-1],
        y=[_FEAT_LABELS.get(f, f) for f in top5.index[::-1]],
        orientation="h",
        marker_color=PAL.ACCENT[0],
        width=0.5,
    ))
    fig_imp.update_layout(**_plotly_base(), margin=dict(l=16, r=16, t=32, b=16), title="Facteurs les plus determinants", xaxis_title="Importance")
    st.plotly_chart(fig_imp, width="stretch")

# Positionnement du compteur parmi les autres
if clf is not None and not pred_series.empty and len(features) >= 2:
    st.markdown("---")
    st.markdown("### Positionnement parmi les compteurs")

    col_x, col_y = "zero_ratio", "ratio_we_wd"
    preds = pred_series.reindex(features.index, fill_value=0)
    colors = [PAL.ACCENT[2] if p == 1 else PAL.ACCENT[1] for p in preds]
    labels_txt = ["RS" if p == 1 else "RP" for p in preds]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=features[col_x], y=features[col_y],
        mode="markers",
        marker=dict(color=colors, size=7, opacity=0.7),
        text=[f"{mid}<br>{lbl}" for mid, lbl in zip(features.index, labels_txt)],
        hovertemplate="%{text}<br>X=%{x:.3f}<br>Y=%{y:.3f}<extra></extra>",
        name="Compteurs",
    ))
    if selected in features.index:
        fig2.add_trace(go.Scatter(
            x=[features.loc[selected, col_x]], y=[features.loc[selected, col_y]],
            mode="markers",
            marker=dict(color="rgba(0,0,0,0)", size=16, line=dict(color=PAL.TEXT, width=2)),
            text=[f"{selected} (selectionne)"],
            hovertemplate="%{text}<extra></extra>",
            name="Compteur selectionne",
        ))
    fig2.update_layout(
        **_plotly_base(),
        margin=dict(l=16, r=16, t=32, b=16),
        xaxis_title=_FEAT_LABELS.get(col_x, col_x),
        yaxis_title=_FEAT_LABELS.get(col_y, col_y),
    )
    st.plotly_chart(fig2, width="stretch")

# Performance du modele
if clf is not None and y_test is not None and y_proba_test is not None:
    st.markdown("---")
    st.markdown("### Performance du modele")

    if cv_scores is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Precision globale",
            f"{cv_scores['accuracy']:.2%}",
            f"± {cv_scores['accuracy_std']:.2%}",
            delta_color="off",
        )
        c2.metric(
            "Equilibre RS / RP",
            f"{cv_scores['f1']:.2%}",
            f"± {cv_scores['f1_std']:.2%}",
            delta_color="off",
        )
        c3.metric(
            "Detection residences secondaires",
            f"{cv_scores['recall_rs']:.2%}",
            f"± {cv_scores['recall_rs_std']:.2%}",
            delta_color="off",
        )

    cm = confusion_matrix(y_test, y_pred)
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
        margin=dict(l=16, r=16, t=32, b=16),
        title="Matrice de confusion",
        xaxis_title="Predit",
        yaxis_title="Reel",
    )
    st.plotly_chart(fig_cm, width="stretch")
