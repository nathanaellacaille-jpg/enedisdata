import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import PAL, GEN_NOISE_STD, MAX_METERS_UPLOAD, STEPS_PER_DAY
from models.generator import CurveGenerator
from utils.parser import parse_timeseries, parse_labels
from utils.corpus import load_builtin_corpus


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
def _fit_generator(ts_key: str, lbl_key: str, ts_bytes: bytes | None, lbl_bytes: bytes | None) -> CurveGenerator:
    """Calibre le generateur — corpus built-in si aucun upload."""
    gen = CurveGenerator()
    if ts_bytes and lbl_bytes:
        df = parse_timeseries(io.BytesIO(ts_bytes), max_meters=MAX_METERS_UPLOAD)
        labels = parse_labels(io.BytesIO(lbl_bytes))
    else:
        df, labels = load_builtin_corpus()
    gen.fit(df, labels)
    return gen


@st.cache_data
def _load_real(ts_bytes: bytes, ts_key: str) -> pd.DataFrame:
    """Charge la timeseries reelle pour la validation."""
    return parse_timeseries(io.BytesIO(ts_bytes), max_meters=MAX_METERS_UPLOAD)


@st.cache_data
def _load_labels_dict(lbl_bytes: bytes, lbl_key: str) -> dict:
    """Charge le dict de labels meter_id -> int."""
    return parse_labels(io.BytesIO(lbl_bytes))


@st.cache_data
def _generate(ts_key: str, lbl_key: str, n: int, curve_type: str, n_days: int,
              ts_bytes: bytes | None, lbl_bytes: bytes | None, mode: str = "parametric") -> pd.DataFrame:
    """Genere n courbes en mode parametrique ou reechantillonnage."""
    gen = _fit_generator(ts_key, lbl_key, ts_bytes, lbl_bytes)
    if mode == "bootstrap":
        return gen.generate_bootstrap(n, curve_type, n_days, GEN_NOISE_STD)
    return gen.generate(n, curve_type, n_days, GEN_NOISE_STD)


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("**Calibration (optionnel)**")
    ts_file = st.file_uploader("Timeseries CSV", type=["csv"], key="gen_ts")
    lbl_file = st.file_uploader("Labels CSV", type=["csv"], key="gen_lbl")

    if ts_file is not None and st.session_state.get("_gen_ts_file_name") != ts_file.name:
        st.session_state["_gen_ts_bytes"] = ts_file.getvalue()
        st.session_state["_gen_ts_file_name"] = ts_file.name

    if lbl_file is not None and st.session_state.get("_gen_lbl_file_name") != lbl_file.name:
        st.session_state["_gen_lbl_bytes"] = lbl_file.getvalue()
        st.session_state["_gen_lbl_file_name"] = lbl_file.name

    st.markdown(
        '<div class="sidebar-footer">'
        '<div class="sidebar-badge">RES2-6-9 kVA</div><br>'
        'Enedis open data<br>Generation synthetique'
        "</div>",
        unsafe_allow_html=True,
    )


# ── page ──────────────────────────────────────────────────────────────────────

st.markdown("## Generation")

ts_bytes = st.session_state.get("_gen_ts_bytes")
lbl_bytes = st.session_state.get("_gen_lbl_bytes")
ts_key = st.session_state.get("_gen_ts_file_name", "none")
lbl_key = st.session_state.get("_gen_lbl_file_name", "none")

has_upload = ts_bytes is not None and lbl_bytes is not None
corpus_df, corpus_labels = load_builtin_corpus()

if has_upload:
    real_df = _load_real(ts_bytes, ts_key)
    labels = _load_labels_dict(lbl_bytes, lbl_key)
    st.caption(f"Calibre sur {real_df['meter_id'].nunique()} compteurs reels — {len(real_df):,} points")
else:
    real_df = corpus_df
    labels = corpus_labels
    st.caption(f"Corpus de reference — {corpus_df['meter_id'].nunique()} courbes synthetiques")


# ── controles ─────────────────────────────────────────────────────────────────

curve_type = st.radio("Type", ["RS", "RP"], horizontal=True, key="gen_type")

N_DAYS = 7
gen_df = _generate(ts_key, lbl_key, 50, curve_type, N_DAYS, ts_bytes, lbl_bytes, "bootstrap")


# ── graphique principal : N courbes sur 7 jours ──────────────────────────────

gen_df_sorted = gen_df.sort_values(["curve_id", "day", "slot"]).copy()
gen_df_sorted["t"] = gen_df_sorted["day"] * STEPS_PER_DAY + gen_df_sorted["slot"]
total_slots = N_DAYS * STEPS_PER_DAY

fig_main = go.Figure()
for cid in sorted(gen_df_sorted["curve_id"].unique()):
    c = gen_df_sorted[gen_df_sorted["curve_id"] == cid]
    fig_main.add_trace(go.Scatter(
        x=c["t"], y=c["kw"],
        mode="lines", showlegend=False,
        line=dict(color="#CBD5E1", width=0.8),
        hoverinfo="skip",
    ))

mean_curve = gen_df_sorted.groupby("t")["kw"].mean()
fig_main.add_trace(go.Scatter(
    x=mean_curve.index, y=mean_curve.values,
    mode="lines", name="Moyenne des courbes generees",
    line=dict(color=PAL.REAL, width=2),
))

day_ticks = list(range(0, total_slots, STEPS_PER_DAY))
day_labels = [f"J{d + 1}" for d in range(N_DAYS)]
fig_main.update_layout(
    **_plotly_base(),
    margin=dict(l=16, r=16, t=32, b=16),
    title=f"50 courbes {curve_type} sur 7 jours (pas 30 min)",
    yaxis_title="Puissance (kW)",
    height=380,
)
fig_main.update_xaxes(tickmode="array", tickvals=day_ticks, ticktext=day_labels)
st.plotly_chart(fig_main, width="stretch")


# ── validation : similarite ──────────────────────────────────────────────────

st.markdown("### Similarite avec les donnees reelles")

gen = _fit_generator(ts_key, lbl_key, ts_bytes, lbl_bytes)
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
            line=dict(color=PAL.REAL, width=1.5),
        ))
        fig_a.add_trace(go.Scatter(
            x=hours, y=report["profile_gen"],
            mode="lines", name="Genere",
            line=dict(color=PAL.TEXT_MUTED, width=1.5, dash="dash"),
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
        st.metric("Correlation de forme (Pearson)", f"{report['pearson_profile']:.3f}", delta_color="off")
        if has_upload and report["discriminative_score"] is not None:
            st.metric(
                "Score discriminant (1-NN)",
                f"{report['discriminative_score']:.3f}",
                delta="ideal : 0.500",
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
            name="Reel", marker_color=PAL.REAL, opacity=0.55, histnorm="probability",
        ))
        fig_b.add_trace(go.Histogram(
            x=e_gen, xbins=dict(start=lo, end=hi, size=size),
            name="Genere", marker_color=PAL.TEXT_MUTED, opacity=0.55, histnorm="probability",
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
        st.metric("Distance de Wasserstein", f"{report['wasserstein_energy']:.2f} kWh", delta_color="off")


# ── KPIs coherence globale ────────────────────────────────────────────────────

st.markdown("### Coherence globale")

k1, k2, k3 = st.columns(3)
with k1:
    delta_e = f"reel : {report['mean_energy_real']:.2f}" if report["mean_energy_real"] is not None else None
    st.metric("Energie moyenne / jour", f"{report['mean_energy_gen']:.2f} kWh",
              delta=delta_e, delta_color="off")
with k2:
    delta_p = f"reel : {report['peak_real']:.2f}" if report["peak_real"] is not None else None
    st.metric("Pic", f"{report['peak_gen']:.2f} kW",
              delta=delta_p, delta_color="off")
with k3:
    delta_r = f"reel : {report['we_ratio_real']:.2f}" if report["we_ratio_real"] is not None else None
    st.metric("Ratio WE / semaine", f"{report['we_ratio_gen']:.2f}",
              delta=delta_r, delta_color="off")


# ── export ────────────────────────────────────────────────────────────────────

st.markdown("### Export")
csv_bytes = gen_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Telecharger le CSV",
    csv_bytes,
    file_name=f"courbes_{curve_type}_{n_curves}.csv",
    mime="text/csv",
)
