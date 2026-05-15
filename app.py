import sys
import importlib.util
import streamlit as st
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _ensure_package(name: str) -> None:
    """Enregistre un package local dans sys.modules avec __path__ correct.

    Contourne KeyError: 'utils' sur Python 3.14 + Streamlit 1.57 où le runner
    interne (_mpa_v1) execute app.py sans que les packages locaux soient dans
    sys.modules, ce qui fait echouer l'import de sous-modules (utils.parser…).
    """
    if name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        name,
        _root / name / "__init__.py",
        submodule_search_locations=[str(_root / name)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)


_ensure_package("utils")
_ensure_package("models")

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
