import sys
import types
import importlib
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Fix Python 3.14 + Streamlit : sys.modules ne pre-enregistre pas les packages
# locaux dans le contexte exec() de Streamlit, ce qui provoque KeyError: 'utils'
for _pkg in ("utils", "models"):
    if _pkg not in sys.modules:
        sys.modules[_pkg] = types.ModuleType(_pkg)

import utils.parser, utils.features, utils.metrics  # noqa: E401, F401
import models.classifier, models.forecaster, models.generator  # noqa: E401, F401

st.set_page_config(page_title="Enedis Analytics", page_icon=None, layout="wide", initial_sidebar_state="expanded")

# Charge le CSS global
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="app-header">'
    '<div class="app-title">Enedis Analytics</div>'
    '<div class="app-subtitle">Classification · Prevision · Generation</div>'
    "</div>",
    unsafe_allow_html=True,
)

# Navigation
pg1 = st.Page("pages/1_classification.py", title="Classification", default=True)
pg2 = st.Page("pages/2_prevision.py", title="Prevision")
pg3 = st.Page("pages/3_generation.py", title="Generation")

pg = st.navigation([pg1, pg2, pg3], position="sidebar")

with st.sidebar:
    st.markdown(
        '<div class="sidebar-footer">'
        '<div class="sidebar-badge">RES2-6-9 kVA</div><br>'
        'Enedis open data<br>v1.0'
        "</div>",
        unsafe_allow_html=True,
    )

pg.run()
