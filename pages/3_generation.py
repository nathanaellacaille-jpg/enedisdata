import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import PAL, GEN_NOISE_STD, STEPS_PER_DAY
from models.generator import CurveGenerator
from utils.corpus import load_builtin_corpus
from utils.data_loader import load_default_ts, load_default_labels


# ── helpers ───────────────────────────────────────────────────────────────────

def _plotly_base() -> dict:
    """Layout Plotly minimal partage par tous les graphiques."""
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
def _fit_generator(ts_key: str, lbl_key: str, _df: pd.DataFrame | None, _labels: dict | None) -> CurveGenerator:
    """Calibre le generateur — corpus built-in si le jeu de donnees est absent."""
    gen = CurveGenerator()
    if _df is not None and _labels is not None:
        gen.fit(_df, _labels)
    else:
        df_b, labels_b = load_builtin_corpus()
        gen.fit(df_b, labels_b)
    return gen


@st.cache_data
def _generate(ts_key: str, lbl_key: str, n: int, curve_type: str, n_days: int,
              _df: pd.DataFrame | None, _labels: dict | None, mode: str = "parametric") -> pd.DataFrame:
    """Genere n courbes en mode parametrique ou reechantillonnage."""
    gen = _fit_generator(ts_key, lbl_key, _df, _labels)
    if mode == "bootstrap":
        return gen.generate_bootstrap(n, curve_type, n_days, GEN_NOISE_STD)
    return gen.generate(n, curve_type, n_days, GEN_NOISE_STD)


@st.cache_data
def _sample_real_curves(
    ts_key: str,
    lbl_key: str,
    _df: pd.DataFrame,
    _labels: dict,
    curve_type: str,
    n: int,
    n_days: int,
) -> pd.DataFrame:
    """Tire n meters reels aleatoirement et extrait leurs n_days derniers jours connus."""
    label_val = 1 if curve_type == "RS" else 0
    ids = [k for k, v in _labels.items() if v == label_val]
    meter_set = set(_df["meter_id"].astype(str).unique())
    available = [m for m in ids if m in meter_set]
    if not available:
        return pd.DataFrame(columns=["curve_id", "day", "slot", "kw", "curve_type"])
    rng = np.random.default_rng(42)
    sampled = list(rng.choice(available, size=min(n, len(available)), replace=False))
    sub = _df[_df["meter_id"].isin(sampled)].copy()
    sub["meter_id"] = sub["meter_id"].astype(str)
    sub["date"] = sub["ts"].dt.date
    sub["slot"] = sub["ts"].dt.hour * 2 + sub["ts"].dt.minute // 30
    parts = []
    for curve_id, mid in enumerate(sampled):
        msub = sub[sub["meter_id"] == mid]
        dates = sorted(msub["date"].unique())[-n_days:]
        for day_idx, date in enumerate(dates):
            dsub = msub[msub["date"] == date][["slot", "kw"]].copy()
            dsub["curve_id"] = curve_id
            dsub["day"] = day_idx
            dsub["curve_type"] = curve_type
            parts.append(dsub)
    if not parts:
        return pd.DataFrame(columns=["curve_id", "day", "slot", "kw", "curve_type"])
    result = pd.concat(parts, ignore_index=True)
    return result[["curve_id", "day", "slot", "kw", "curve_type"]]


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("**Calibration**")
    real_df = load_default_ts()
    labels = load_default_labels()
    if real_df is not None and labels is not None:
        st.caption(f"{real_df['meter_id'].nunique()} compteurs · {len(real_df):,} points")
    else:
        st.caption("Corpus de reference (jeu de donnees absent)")

    st.markdown(
        '<div class="sidebar-footer">'
        '<div class="sidebar-badge">RES2-6-9 kVA</div><br>'
        'Enedis open data<br>Generation synthetique'
        "</div>",
        unsafe_allow_html=True,
    )


# ── page ──────────────────────────────────────────────────────────────────────

st.markdown("## Generation")

has_real = real_df is not None and labels is not None
ts_key = "default_ts" if has_real else "corpus"
lbl_key = "default_lbl" if has_real else "corpus"

if has_real:
    st.caption(f"Calibre sur {real_df['meter_id'].nunique()} compteurs reels — {len(real_df):,} points")
else:
    corpus_df, corpus_labels = load_builtin_corpus()
    real_df = corpus_df
    labels = corpus_labels
    st.caption(f"Corpus de reference — {corpus_df['meter_id'].nunique()} courbes synthetiques")


# ── controles ─────────────────────────────────────────────────────────────────

curve_type = st.radio("Type", ["RS", "RP"], horizontal=True, key="gen_type")

N_DAYS = 7
gen_df = _generate(ts_key, lbl_key, 50, curve_type, N_DAYS, real_df, labels, "bootstrap")
main_df = (
    _sample_real_curves(ts_key, lbl_key, real_df, labels, curve_type, 50, N_DAYS)
    if has_real
    else gen_df
)


# ── graphique principal : N courbes sur 7 jours ──────────────────────────────

main_df_sorted = main_df.sort_values(["curve_id", "day", "slot"]).copy()
main_df_sorted["t"] = main_df_sorted["day"] * STEPS_PER_DAY + main_df_sorted["slot"]
total_slots = N_DAYS * STEPS_PER_DAY

fig_main = go.Figure()
for cid in sorted(main_df_sorted["curve_id"].unique()):
    c = main_df_sorted[main_df_sorted["curve_id"] == cid]
    fig_main.add_trace(go.Scatter(
        x=c["t"], y=c["kw"],
        mode="lines", showlegend=False,
        line=dict(color="#CBD5E1", width=0.8),
        hoverinfo="skip",
    ))

mean_curve = main_df_sorted.groupby("t")["kw"].mean()
_mean_label = "Moyenne des courbes reelles" if has_real else "Moyenne des courbes generees"
fig_main.add_trace(go.Scatter(
    x=mean_curve.index, y=mean_curve.values,
    mode="lines", name=_mean_label,
    line=dict(color=PAL.ACCENT[0], width=2),
))

day_ticks = list(range(0, total_slots, STEPS_PER_DAY))
day_labels = [f"J{d + 1}" for d in range(N_DAYS)]
_main_title = (
    f"50 courbes {curve_type} reelles — 7 derniers jours"
    if has_real
    else f"50 courbes {curve_type} generees — 7 jours"
)
fig_main.update_layout(
    **_plotly_base(),
    margin=dict(l=16, r=16, t=32, b=16),
    title=_main_title,
    yaxis_title="Puissance (kW)",
    height=380,
)
fig_main.update_xaxes(tickmode="array", tickvals=day_ticks, ticktext=day_labels)
st.plotly_chart(fig_main, width="stretch")


# ── validation : similarite ──────────────────────────────────────────────────

st.markdown("### Qualite de generation")

gen = _fit_generator(ts_key, lbl_key, real_df, labels)
report = gen.similarity_report(real_df, labels, gen_df, curve_type)

if not report["has_real"]:
    st.caption("Validation indisponible.")
else:
    val_a, val_b = st.columns(2)

    with val_a:
        hours = [s / 2 for s in range(STEPS_PER_DAY)]
        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(
            x=hours, y=report["profile_real"],
            mode="lines", name="Reel",
            line=dict(color=PAL.ACCENT[0], width=1.5),
        ))
        fig_a.add_trace(go.Scatter(
            x=hours, y=report["profile_gen"],
            mode="lines", name="Genere",
            line=dict(color=PAL.ACCENT[1], width=1.5, dash="dash"),
        ))
        fig_a.update_layout(
            **_plotly_base(),
            margin=dict(l=16, r=16, t=32, b=16),
            title="Profil moyen sur 24 h",
            xaxis_title="Heure",
            yaxis_title="kW",
            height=280,
        )
        st.plotly_chart(fig_a, width="stretch")
        st.metric("Ressemblance de profil", f"{report['pearson_profile'] * 100:.0f} %", delta_color="off")
        if has_real and report["discriminative_score"] is not None:
            indiscernabilite = max(0.0, 1.0 - abs(report["discriminative_score"] - 0.5) * 2) * 100
            st.metric(
                "Indiscernabilite",
                f"{indiscernabilite:.0f} %",
                delta="ideal : 100 %",
                delta_color="off",
            )

    with val_b:
        e_real = report["energy_real"]
        e_gen = report["energy_gen"]
        lo = float(min(e_real.min(), e_gen.min()))
        hi = float(max(e_real.max(), e_gen.max()))
        size = (hi - lo) / 24 if hi > lo else 1.0
        fig_b = go.Figure()
        fig_b.add_trace(go.Histogram(
            x=e_real, xbins=dict(start=lo, end=hi, size=size),
            name="Reel", marker_color=PAL.ACCENT[0], opacity=0.55, histnorm="probability",
        ))
        fig_b.add_trace(go.Histogram(
            x=e_gen, xbins=dict(start=lo, end=hi, size=size),
            name="Genere", marker_color=PAL.ACCENT[1], opacity=0.55, histnorm="probability",
        ))
        fig_b.update_layout(
            **_plotly_base(),
            margin=dict(l=16, r=16, t=32, b=16),
            title="Distribution energie journaliere",
            xaxis_title="kWh / jour",
            yaxis_title="Frequence",
            barmode="overlay",
            height=280,
        )
        st.plotly_chart(fig_b, width="stretch")
        st.metric("Ecart energetique moyen", f"{report['wasserstein_energy']:.2f} kWh", delta_color="off")


# ── KPIs coherence globale ────────────────────────────────────────────────────

st.markdown("### Coherence globale")

k1, k2, k3 = st.columns(3)
with k1:
    delta_e = f"reel : {report['mean_energy_real']:.2f}" if report["mean_energy_real"] is not None else None
    st.metric("Energie moyenne / jour", f"{report['mean_energy_gen']:.2f} kWh",
              delta=delta_e, delta_color="off")
with k2:
    delta_p = f"reel : {report['peak_real']:.2f}" if report["peak_real"] is not None else None
    st.metric("Puissance de pointe", f"{report['peak_gen']:.2f} kW",
              delta=delta_p, delta_color="off")
with k3:
    delta_r = f"reel : {report['we_ratio_real']:.2f}" if report["we_ratio_real"] is not None else None
    st.metric("Rapport weekend", f"{report['we_ratio_gen']:.2f}",
              delta=delta_r, delta_color="off")
