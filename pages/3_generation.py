import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import PAL, GEN_DEFAULT_N, GEN_NOISE_STD, MAX_METERS_UPLOAD, _make_rp_profile, _make_rs_profile
from models.generator import CurveGenerator
from utils.parser import parse_timeseries, parse_labels


st.set_page_config(page_title="Generation", layout="wide")


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
    """Charge le CSV timeseries."""
    import io
    return parse_timeseries(io.BytesIO(file_bytes), max_meters=MAX_METERS_UPLOAD)


@st.cache_data
def _load_labels(file_bytes: bytes, file_name: str) -> dict:
    """Charge le CSV labels."""
    import io
    return parse_labels(io.BytesIO(file_bytes))


@st.cache_resource
def _fit_generator(ts_key: str, lbl_key: str, ts_bytes: bytes | None, lbl_bytes: bytes | None) -> CurveGenerator:
    """Cree et entraine le generateur."""
    import io
    gen = CurveGenerator()
    df = parse_timeseries(io.BytesIO(ts_bytes), max_meters=MAX_METERS_UPLOAD) if ts_bytes else None
    labels = parse_labels(io.BytesIO(lbl_bytes)) if lbl_bytes else None
    gen.fit(df, labels)
    return gen


@st.cache_data
def _generate(ts_key: str, lbl_key: str, n: int, curve_type: str, n_days: int, noise: float,
              ts_bytes: bytes | None, lbl_bytes: bytes | None) -> pd.DataFrame:
    """Genere les courbes et retourne le dataframe."""
    gen = _fit_generator(ts_key, lbl_key, ts_bytes, lbl_bytes)
    return gen.generate(n, curve_type, n_days, noise)


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("**Calibration (optionnel)**")
    ts_file = st.file_uploader("Timeseries CSV", type=["csv"], key="gen_ts")
    lbl_file = st.file_uploader("Labels CSV", type=["csv"], key="gen_lbl")

    if ts_file is not None:
        if st.session_state.get("_ts_file_name") != ts_file.name:
            try:
                ts_bytes_new = ts_file.getvalue()
                st.session_state["_ts_df"] = _load_ts(ts_bytes_new, ts_file.name)
                st.session_state["_ts_bytes"] = ts_bytes_new
                st.session_state["_ts_file_name"] = ts_file.name
            except ValueError as e:
                st.error(str(e))

    if lbl_file is not None:
        if st.session_state.get("_labels_file_name") != lbl_file.name:
            try:
                lbl_bytes_new = lbl_file.getvalue()
                st.session_state["_labels"] = _load_labels(lbl_bytes_new, lbl_file.name)
                st.session_state["_lbl_bytes"] = lbl_bytes_new
                st.session_state["_labels_file_name"] = lbl_file.name
            except ValueError as e:
                st.error(str(e))

    st.markdown("**Parametres**")
    curve_type_label = st.radio("Type", ["RS", "RP", "Mixte"], key="gen_type")
    curve_type = {"RS": "RS", "RP": "RP", "Mixte": "mixed"}[curve_type_label]
    n_curves = st.slider("Nombre de courbes", 1, 100, GEN_DEFAULT_N, key="gen_n")
    n_days = st.slider("Nombre de jours", 1, 30, 7, key="gen_days")
    n_viz = st.slider("Courbes a visualiser", 1, 20, min(5, n_curves), key="gen_viz")

    st.markdown(
        '<div class="sidebar-footer">'
        '<div class="sidebar-badge">RES2-6-9 kVA</div><br>'
        'Enedis open data<br>Generation synthetique'
        "</div>",
        unsafe_allow_html=True,
    )

# ── generation ────────────────────────────────────────────────────────────────

st.markdown("## Generation")

ts_bytes = st.session_state.get("_ts_bytes")
lbl_bytes = st.session_state.get("_lbl_bytes")
ts_key = st.session_state.get("_ts_file_name", "none")
lbl_key = st.session_state.get("_labels_file_name", "none")

gen_df = _generate(ts_key, lbl_key, n_curves, curve_type, n_days, GEN_NOISE_STD, ts_bytes, lbl_bytes)

if ts_bytes:
    _ts_info = st.session_state.get("_ts_df")
    if _ts_info is not None:
        n_meters = _ts_info["meter_id"].nunique()
        st.caption(f"{n_meters} compteurs charges · {len(_ts_info):,} points")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Courbes", "Profils", "Comparaison", "Statistiques", "Export"])

# ── Tab 1 : Courbes ──────────────────────────────────────────────────────────
with tab1:
    display_mode = st.radio("Affichage", ["Superposees", "Grille"], horizontal=True, key="gen_display")

    # Un jour particulier
    day_sel = st.selectbox("Jour", list(range(n_days)), key="gen_day")
    day_df = gen_df[gen_df["day"] == day_sel]
    curves_to_show = sorted(day_df["curve_id"].unique())[:n_viz]

    if display_mode == "Superposees":
        fig = go.Figure()
        for i, cid in enumerate(curves_to_show):
            cdata = day_df[day_df["curve_id"] == cid]
            ct = cdata["curve_type"].iloc[0]
            dash = "solid" if ct == "RS" else "dash"
            color = PAL.MULTI[i % len(PAL.MULTI)]
            fig.add_trace(go.Scatter(
                x=cdata["slot"], y=cdata["kw"],
                mode="lines", name=f"{ct}-{cid}",
                line=dict(color=color, width=1.5, dash=dash),
            ))
        fig.update_layout(**_plotly_base(), margin=dict(l=16, r=16, t=32, b=16), title=f"Courbes — jour {day_sel}", xaxis_title="Slot (30 min)", yaxis_title="kW")
        st.plotly_chart(fig, width="stretch")
    else:
        cols = 2
        rows_needed = (len(curves_to_show) + cols - 1) // cols
        for row in range(rows_needed):
            row_cols = st.columns(cols)
            for col_idx in range(cols):
                cidx = row * cols + col_idx
                if cidx >= len(curves_to_show):
                    break
                cid = curves_to_show[cidx]
                cdata = day_df[day_df["curve_id"] == cid]
                ct = cdata["curve_type"].iloc[0]
                fig_g = go.Figure(go.Scatter(
                    x=cdata["slot"], y=cdata["kw"],
                    mode="lines",
                    line=dict(color=PAL.REAL, width=1.5),
                ))
                fig_g.update_layout(**{
                    **_plotly_base(),
                    "title": f"{ct}-{cid}",
                    "height": 200,
                    "margin": dict(l=8, r=8, t=28, b=8),
                })
                row_cols[col_idx].plotly_chart(fig_g, width="stretch")

# ── Tab 2 : Profils ──────────────────────────────────────────────────────────
with tab2:
    rp_ref = _make_rp_profile()
    rs_ref = _make_rs_profile()

    fig_prof = go.Figure()
    fig_prof.add_trace(go.Scatter(
        x=list(range(48)), y=rs_ref,
        mode="lines", name="Profil RS",
        line=dict(color=PAL.RS, width=1.5),
    ))
    fig_prof.add_trace(go.Scatter(
        x=list(range(48)), y=rp_ref,
        mode="lines", name="Profil RP",
        line=dict(color=PAL.RP, width=1.5, dash="dash"),
    ))
    fig_prof.update_layout(**_plotly_base(), margin=dict(l=16, r=16, t=32, b=16), title="Profils moyens RS vs RP", xaxis_title="Slot (30 min)", yaxis_title="Puissance normalisee")
    st.plotly_chart(fig_prof, width="stretch")

    # Ratio WE/semaine par type
    gen_df_day = gen_df.copy()
    gen_df_day["is_we"] = gen_df_day["day"] % 7 >= 5
    ratio_data = gen_df_day.groupby(["curve_type", "is_we"])["kw"].sum().unstack(fill_value=0)
    ratio_data.columns = [str(c) for c in ratio_data.columns]

    if "True" in ratio_data.columns and "False" in ratio_data.columns:
        ratio_we = ratio_data["True"] / (ratio_data["False"] + 1e-8)
        fig_ratio = go.Figure(go.Bar(
            x=ratio_we.index.tolist(),
            y=ratio_we.values.tolist(),
            marker_color=[PAL.MULTI[0], PAL.MULTI[3]],
            width=0.4,
        ))
        fig_ratio.update_layout(**_plotly_base(), margin=dict(l=16, r=16, t=32, b=16), title="Ratio WE/semaine par type", yaxis_title="Ratio")
        st.plotly_chart(fig_ratio, width="stretch")

# ── Tab 3 : Comparaison ──────────────────────────────────────────────────────
with tab3:
    curve_ids = sorted(gen_df["curve_id"].unique().tolist())
    if len(curve_ids) >= 2:
        c1_id = st.selectbox("Courbe A", curve_ids, index=0, key="comp_a")
        c2_id = st.selectbox("Courbe B", curve_ids, index=1, key="comp_b")
        day_c = st.selectbox("Jour", list(range(n_days)), key="comp_day")

        c1_data = gen_df[(gen_df["curve_id"] == c1_id) & (gen_df["day"] == day_c)]
        c2_data = gen_df[(gen_df["curve_id"] == c2_id) & (gen_df["day"] == day_c)]

        col_a, col_b = st.columns(2)
        for col, cdata, label in [(col_a, c1_data, f"Courbe {c1_id}"), (col_b, c2_data, f"Courbe {c2_id}")]:
            fig_c = go.Figure(go.Scatter(
                x=cdata["slot"], y=cdata["kw"],
                mode="lines",
                line=dict(color=PAL.REAL, width=1.5),
            ))
            fig_c.update_layout(
                **_plotly_base(),
                margin=dict(l=16, r=16, t=32, b=16),
                title=label,
                xaxis_title="Slot", yaxis_title="kW", height=250,
            )
            col.plotly_chart(fig_c, width="stretch")

        # Tableau stats
        stats = []
        for cdata, label in [(c1_data, f"Courbe {c1_id}"), (c2_data, f"Courbe {c2_id}")]:
            energy = cdata["kw"].sum() * 0.5
            peak = cdata["kw"].max()
            mean_kw = cdata["kw"].mean()
            ct = cdata["curve_type"].iloc[0] if not cdata.empty else "?"
            stats.append({"Courbe": label, "Type": ct, "Energie (kWh)": round(energy, 2),
                          "Pic (kW)": round(peak, 3), "Moy (kW)": round(mean_kw, 3)})
        st.dataframe(pd.DataFrame(stats), width="stretch")
    else:
        st.caption("Au moins 2 courbes requises.")

# ── Tab 4 : Statistiques ──────────────────────────────────────────────────────
with tab4:
    # Distribution energie journaliere
    daily_energy = gen_df.groupby(["curve_id", "day"])["kw"].sum() * 0.5
    fig_dist = go.Figure(go.Histogram(
        x=daily_energy.values,
        nbinsx=20,
        marker_color=PAL.MULTI[0],
        opacity=0.85,
    ))
    fig_dist.update_layout(**_plotly_base(), margin=dict(l=16, r=16, t=32, b=16), title="Distribution energie journaliere (kWh)", xaxis_title="kWh/j", yaxis_title="Frequence")
    st.plotly_chart(fig_dist, width="stretch")

    # Heatmap puissance : slot x curve_id (premier jour)
    pivot_ids = sorted(gen_df["curve_id"].unique())[:n_viz]
    heat_df = gen_df[(gen_df["day"] == 0) & (gen_df["curve_id"].isin(pivot_ids))]
    pivot = heat_df.pivot_table(index="slot", columns="curve_id", values="kw", aggfunc="mean")

    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[[0, "#FFFFFF"], [0.5, "#94A3B8"], [1, "#0F172A"]],
    ))
    fig_heat.update_layout(
        **_plotly_base(),
        margin=dict(l=16, r=16, t=32, b=16),
        title="Puissance par slot et courbe (jour 0)",
        xaxis_title="Courbe",
        yaxis_title="Slot (30 min)",
    )
    st.plotly_chart(fig_heat, width="stretch")

# ── Tab 5 : Export ────────────────────────────────────────────────────────────
with tab5:
    st.dataframe(gen_df.head(200), width="stretch")

    csv_bytes = gen_df.to_csv(index=False).encode("utf-8")
    st.download_button("Telecharger CSV", csv_bytes, file_name="courbes_synthetiques.csv", mime="text/csv")

    # JSON resume stats
    gen_inst = CurveGenerator()
    gen_inst.fit(
        parse_timeseries(__import__("io").BytesIO(ts_bytes)) if ts_bytes else None,
        parse_labels(__import__("io").BytesIO(lbl_bytes)) if lbl_bytes else None,
    )
    stats_dict = gen_inst.profile_stats()
    stats_dict["n_curves"] = int(n_curves)
    stats_dict["n_days"] = int(n_days)
    stats_dict["curve_type"] = curve_type
    json_bytes = json.dumps(stats_dict, indent=2).encode("utf-8")
    st.download_button("Telecharger JSON stats", json_bytes, file_name="stats.json", mime="application/json")
