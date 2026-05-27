import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from config import PAL, CLF_TEST_SIZE, ROOT_DIR, _make_rp_profile, _make_rs_profile
from models.classifier import EnergyClassifier
from utils.data_loader import load_default_ts, load_default_labels
from utils.features import extract_features


@st.cache_data
def _load_baseline_metrics() -> dict | None:
    """Charge les metriques baseline (CV5 pre-calcule par scripts/phase0_diagnostic.py)."""
    import json
    p = ROOT_DIR / "assets" / "baseline_metrics.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)

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
    "fourier_amp_4": "Periodicite J/4",
    "fourier_amp_5": "Periodicite J/5",
    "fourier_amp_6": "Periodicite J/6",
    "weekly_entropy": "Entropie hebdo",
    "peak_hour_std": "Variabilite heure pic",
    "dow_consistency": "Coherence jour-semaine",
    "summer_weekend_boost": "Boost WE ete",
    "night_amplitude": "Amplitude nuit",
    "vacation_weeks": "Vacances longues (sem.)",
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


@st.cache_data(hash_funcs={pd.DataFrame: lambda df: (len(df), df.shape[1], str(df.index[0]) if len(df) else "", str(df.index[-1]) if len(df) else "")})
def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les features par compteur."""
    return extract_features(df)


@st.cache_data(hash_funcs={pd.DataFrame: lambda df: (len(df), df.shape[1], str(df.index[0]) if len(df) else "", str(df.index[-1]) if len(df) else "")})
def _predict_all(features: pd.DataFrame, labels: dict) -> tuple:
    """Predit classes et probabilites pour tous les compteurs (cache par dataset)."""
    clf, *_ = _train_model(features, labels)
    if clf is None:
        return pd.Series(index=features.index, dtype=int), pd.Series(index=features.index, dtype=float)
    proba = clf.predict_proba(features)
    return (
        pd.Series((proba >= clf.threshold_).astype(int), index=features.index),
        pd.Series(proba, index=features.index),
    )


@st.cache_resource(hash_funcs={pd.DataFrame: lambda df: (len(df), df.shape[1], str(df.index[0]) if len(df) else "", str(df.index[-1]) if len(df) else "")})
def _train_model(features: pd.DataFrame, labels: dict):
    """Entraine EnergyClassifier (Stacking, threshold tune) et retourne (model, X_test, y_test, y_proba_test, cv_scores).

    Metriques calculees uniquement sur le holdout du train_test_split (zero refit supplementaire).
    Le CV5 imbriquant 5x EnergyClassifier etait trop lourd pour Streamlit Cloud (1 GB RAM).
    Les vraies metriques CV5 + perm restent dispo en local via scripts/phase0_diagnostic.py.
    """
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
    y_pred_test = (y_proba_test >= clf.threshold_).astype(int)
    cv_scores = {
        "accuracy": float(accuracy_score(y_test, y_pred_test)),
        "f1": float(f1_score(y_test, y_pred_test, average="weighted")),
        "recall_rs": float(recall_score(y_test, y_pred_test, pos_label=1, zero_division=0)),
        "accuracy_std": 0.0,
        "f1_std": 0.0,
        "recall_rs_std": 0.0,
        "threshold_mean": float(clf.threshold_),
    }
    return clf, X_test, y_test, y_proba_test, cv_scores


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("**Donnees**")
    df = load_default_ts()
    st.session_state["_ts_df"] = df
    labels_loaded = load_default_labels()
    if labels_loaded is not None:
        st.session_state["_labels"] = labels_loaded

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
    st.caption("Jeu de donnees indisponible.")
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
y_pred = (y_proba_test >= clf.threshold_).astype(int) if (clf is not None and y_proba_test is not None) else None

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
        threshold = clf.threshold_ if clf is not None else 0.5
        label_txt = "RS" if proba_rs >= threshold else "RP"
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

    baseline_imp = _load_baseline_metrics()
    if baseline_imp and "feature_importances_top10" in baseline_imp:
        st.markdown("---")
        st.markdown("### Facteurs determinants")
        top5 = baseline_imp["feature_importances_top10"][:5]
        names = [_FEAT_LABELS.get(d["feature"], d["feature"]) for d in top5]
        vals = [d["importance"] for d in top5]
        fig_imp = go.Figure(go.Bar(
            x=vals[::-1],
            y=names[::-1],
            orientation="h",
            marker_color=PAL.ACCENT[0],
            width=0.5,
        ))
        fig_imp.update_layout(**_plotly_base(), margin=dict(l=16, r=16, t=32, b=16), title="Facteurs les plus determinants", xaxis_title="Importance (permutation)")
        st.plotly_chart(fig_imp, width="stretch")

# Performance du modele
if clf is not None and y_test is not None and y_proba_test is not None:
    st.markdown("---")
    st.markdown("### Performance du modele")

    baseline = _load_baseline_metrics()
    if baseline is not None:
        cv5 = baseline["cv5"]
        st.caption(
            f"Validation croisee 5 folds sur {baseline['dataset']['n_samples']} compteurs "
            f"({baseline['dataset']['n_rp']} RP / {baseline['dataset']['n_rs']} RS, "
            f"calculee le {baseline['computed_at'][:10]})."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Precision globale",
            f"{cv5['accuracy']:.2%}",
            f"± {cv5['accuracy_std']:.2%}",
            delta_color="off",
        )
        c2.metric(
            "Equilibre RS / RP",
            f"{cv5['f1_weighted']:.2%}",
            f"± {cv5['f1_weighted_std']:.2%}",
            delta_color="off",
        )
        c3.metric(
            "Detection residences secondaires",
            f"{cv5['recall_rs']:.2%}",
            f"± {cv5['recall_rs_std']:.2%}",
            delta_color="off",
        )

        conf = baseline["confusion"]
        cm_arr = np.array([[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]])
    else:
        cm_arr = confusion_matrix(y_test, y_pred)

    labels_names = ["RP", "RS"]
    fig_cm = go.Figure(go.Heatmap(
        z=cm_arr,
        x=labels_names,
        y=labels_names,
        colorscale=[[0, "#FFFFFF"], [0.5, "#94A3B8"], [1, "#0F172A"]],
        showscale=True,
        text=cm_arr.astype(str),
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
