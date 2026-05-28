import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from config import PAL, CLF_TEST_SIZE, ROOT_DIR
from models.classifier import EnergyClassifier
from utils.data_loader import load_default_ts, load_default_labels
from utils.features import extract_features


@st.cache_data
def _load_baseline_inner(path_str: str, mtime: float, size: int) -> dict | None:
    """Charge le JSON depuis disque (cache invalide quand mtime/size change)."""
    import json
    with open(path_str, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_baseline_metrics() -> dict | None:
    """Charge les metriques baseline (CV5 pre-calcule par scripts/phase0_diagnostic.py).

    Pass mtime+size en argument pour invalider le cache automatiquement
    quand le fichier change sur Cloud (sans ca, le cache @st.cache_data garde
    l'ancien JSON en memoire apres un deploy).
    """
    p = ROOT_DIR / "assets" / "baseline_metrics.json"
    if not p.exists():
        return None
    s = p.stat()
    return _load_baseline_inner(str(p), s.st_mtime, s.st_size)

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
def _predict_all(features: pd.DataFrame, labels: dict) -> pd.Series:
    """Probabilite RS pour tous les compteurs (cache par dataset)."""
    clf = _train_model(features, labels)
    if clf is None:
        return pd.Series(index=features.index, dtype=float)
    return pd.Series(clf.predict_proba(features), index=features.index)


@st.cache_resource(hash_funcs={pd.DataFrame: lambda df: (len(df), df.shape[1], str(df.index[0]) if len(df) else "", str(df.index[-1]) if len(df) else "")})
def _train_model(features: pd.DataFrame, labels: dict):
    """Entraine EnergyClassifier (Stacking, seuil appris) sur un holdout stratifie."""
    common = [mid for mid in features.index if str(mid) in labels]
    if len(common) < 4:
        return None
    X = features.loc[common]
    y = np.array([labels[str(mid)] for mid in common])
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=CLF_TEST_SIZE, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    clf = EnergyClassifier()
    clf.fit(X_train, y_train)
    return clf


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

    # Slider seuil decision : defaut = seuil PR-optimal CV5 du JSON baseline.
    # Plus bas → plus de RS detectees (rappel ↑, precision ↓).
    _baseline_for_threshold = _load_baseline_metrics()
    if _baseline_for_threshold is not None:
        # Defaut = seuil "balanced" (max moyenne recall_RP + recall_RS),
        # typiquement proche de 0.5. Plus equitable que le PR-F1-optimal qui
        # favorise la classe majoritaire.
        _default_thr = float(
            _baseline_for_threshold["cv5"].get("threshold_balanced")
            or _baseline_for_threshold["cv5"].get("threshold_pr_optimal", 0.5)
        )
        st.markdown("**Seuil de decision**")
        threshold_override = st.slider(
            "Probabilite >= seuil -> RS",
            min_value=0.20, max_value=0.95,
            value=_default_thr, step=0.01,
            help="Plus bas: detecte plus de RS (rappel ↑). Plus haut: classifications plus surs (precision ↑).",
            key="clf_threshold_slider",
        )
    else:
        threshold_override = None

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
if labels is not None:
    clf = _train_model(features, labels)
    # Surcharge le seuil par celui du slider sidebar (defaut = PR-optimal CV5).
    if clf is not None and threshold_override is not None:
        clf.threshold_ = threshold_override

# Predictions sur tout le dataset
if clf is not None:
    proba_series = _predict_all(features, labels)
else:
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

# ── Performance globale (bloc top, KPI globaux) ───────────────────────────────

if clf is not None:
    st.markdown("### Performance globale du modele")

    baseline = _load_baseline_metrics()
    if baseline is not None:
        cv5 = baseline["cv5"]
        has_oof = "oof" in baseline and baseline["oof"].get("y_proba")

        if has_oof:
            default_threshold = float(cv5.get("threshold_balanced") or cv5.get("threshold_pr_optimal", 0.5))
            threshold = threshold_override if threshold_override is not None else default_threshold
            is_default = abs(threshold - default_threshold) < 1e-3

            st.caption(
                f"Validation croisee 5 folds sur {baseline['dataset']['n_samples']} compteurs "
                f"({baseline['dataset']['n_rp']} RP / {baseline['dataset']['n_rs']} RS). "
                f"Seuil de decision : **{threshold:.2f}**."
            )

            oof_y = np.array(baseline["oof"]["y_true"])
            oof_proba = np.array(baseline["oof"]["y_proba"])
            oof_pred = (oof_proba >= threshold).astype(int)
            acc = float(accuracy_score(oof_y, oof_pred))
            f1w = float(f1_score(oof_y, oof_pred, average="weighted"))
            rec_rs = float(recall_score(oof_y, oof_pred, pos_label=1, zero_division=0))
            cm_arr = confusion_matrix(oof_y, oof_pred)
            acc_delta = f"± {cv5['accuracy_std']:.2%}" if is_default else None
            f1_delta = f"± {cv5['f1_weighted_std']:.2%}" if is_default else None
            rec_delta = f"± {cv5['recall_rs_std']:.2%}" if is_default else None
        else:
            st.caption(
                f"Validation croisee 5 folds sur {baseline['dataset']['n_samples']} compteurs "
                f"({baseline['dataset']['n_rp']} RP / {baseline['dataset']['n_rs']} RS)."
            )
            acc = cv5["accuracy"]
            f1w = cv5["f1_weighted"]
            rec_rs = cv5["recall_rs"]
            acc_delta = f"± {cv5['accuracy_std']:.2%}"
            f1_delta = f"± {cv5['f1_weighted_std']:.2%}"
            rec_delta = f"± {cv5['recall_rs_std']:.2%}"
            conf = baseline.get("confusion", {})
            cm_arr = np.array([[conf.get("tn", 0), conf.get("fp", 0)],
                               [conf.get("fn", 0), conf.get("tp", 0)]])

        col_perf_a, col_perf_b = st.columns([2, 1])
        with col_perf_a:
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision globale", f"{acc:.2%}", acc_delta, delta_color="off")
            c2.metric("Equilibre RS / RP", f"{f1w:.2%}", f1_delta, delta_color="off")
            c3.metric("Detection residences secondaires", f"{rec_rs:.2%}", rec_delta, delta_color="off")
        with col_perf_b:
            fig_cm = go.Figure(go.Heatmap(
                z=cm_arr,
                x=["RP", "RS"],
                y=["RP", "RS"],
                colorscale=[[0, "#FFFFFF"], [0.5, "#94A3B8"], [1, "#0F172A"]],
                showscale=False,
                text=cm_arr.astype(str),
                texttemplate="%{text}",
                textfont=dict(size=14),
            ))
            fig_cm.update_layout(
                **_plotly_base(),
                margin=dict(l=8, r=8, t=24, b=8),
                title="Matrice de confusion",
                xaxis_title="Predit",
                yaxis_title="Reel",
                height=240,
            )
            st.plotly_chart(fig_cm, width="stretch")

    st.markdown("---")

# ── Detail compteur selectionne ───────────────────────────────────────────────

st.markdown(f"### Compteur {selected}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Conso moy. (kWh/j)", f"{mean_conso:.1f}", delta_color="off")
c2.metric("Pic (kW)", f"{peak_kw:.2f}", delta_color="off")
c3.metric("Ratio WE/SD", f"{we_wd:.2f}", delta_color="off")
c4.metric("Energie/j (kWh)", f"{energy_j:.1f}", delta_color="off")

# Verdict de classification : metric simple + progress bar (plus leger que gauge Plotly)
if not proba_series.empty and selected in proba_series.index:
    proba_rs = float(proba_series[selected])
    threshold = clf.threshold_ if clf is not None else 0.5
    label_txt = "RS" if proba_rs >= threshold else "RP"
    confidence = max(proba_rs, 1 - proba_rs)

    col_v_a, col_v_b = st.columns([1, 2])
    with col_v_a:
        st.metric("Classe predite", label_txt, delta_color="off")
    with col_v_b:
        st.metric(
            "Probabilite RS",
            f"{proba_rs:.0%}",
            delta=f"confiance {confidence:.0%}",
            delta_color="off",
        )
        st.progress(proba_rs, text=f"Seuil decision : {threshold:.2f}")
else:
    st.caption("Chargez des labels pour obtenir la classification.")

# Timeline d'occupation : visualise les VRAIS signaux qui discriminent RS/RP
# (absences, jours actifs, ratio saisonnier).
if not meter_df.empty:
    st.markdown("---")
    st.markdown("### Timeline d'occupation")

    _daily = meter_df.groupby(meter_df["ts"].dt.date)["kw"].sum() * 0.5
    _daily.index = pd.to_datetime(_daily.index)
    _daily = _daily.sort_index()

    # Seuil d'absence : 0.5 kWh/jour (meme valeur que dans features.py)
    _absent = _daily < 0.5
    _active_pct = float((~_absent).mean() * 100) if len(_daily) > 0 else 0.0
    _n_absent = int(_absent.sum())

    # Detection longue absence consecutive (max gap days)
    _max_gap = 0
    _cur = 0
    _gap_end_idx = -1
    for i, v in enumerate(_absent.values):
        if v:
            _cur += 1
            if _cur > _max_gap:
                _max_gap = _cur
                _gap_end_idx = i
        else:
            _cur = 0
    _max_gap_start_date = _daily.index[_gap_end_idx - _max_gap + 1] if _gap_end_idx >= 0 and _max_gap > 0 else None
    _max_gap_end_date = _daily.index[_gap_end_idx] if _gap_end_idx >= 0 else None

    # KPI compacts au-dessus du graphique
    k1, k2, k3 = st.columns(3)
    k1.metric("Jours actifs", f"{_active_pct:.0f} %", delta_color="off")
    k2.metric("Jours absents", str(_n_absent), delta_color="off")
    k3.metric("Plus longue absence", f"{_max_gap} j", delta_color="off")

    # Bar chart par jour : couleur differente actif vs absent
    colors = [PAL.ACCENT[0] if not a else "#E2E8F0" for a in _absent.values]
    fig_tl = go.Figure(go.Bar(
        x=_daily.index, y=_daily.values,
        marker_color=colors,
        hovertemplate="%{x|%d %b %Y} : %{y:.1f} kWh<extra></extra>",
    ))
    if _max_gap_start_date is not None and _max_gap >= 7:
        # Highlight bandeau pour les longues absences (>= 1 semaine)
        fig_tl.add_vrect(
            x0=_max_gap_start_date, x1=_max_gap_end_date,
            fillcolor="#FCA5A5", opacity=0.15, layer="below", line_width=0,
            annotation_text=f"Absence {_max_gap}j",
            annotation_position="top left",
            annotation=dict(font=dict(size=10, color=PAL.TEXT_MUTED)),
        )
    fig_tl.update_layout(
        **_plotly_base(),
        margin=dict(l=16, r=16, t=8, b=16),
        yaxis_title="Energie quotidienne (kWh)",
        showlegend=False,
        height=240,
    )
    st.plotly_chart(fig_tl, width="stretch")
    st.caption(
        "Bleu : jours actifs (≥ 0.5 kWh). Gris : jours absents. "
        "Surlignage : plus longue periode d'absence consecutive. "
        "Les RS se distinguent typiquement par des absences saisonnieres prolongees."
    )

if clf is not None and not meter_feat.empty:
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

# (La section Performance globale a ete deplacee en haut de page.)
