import sys
import importlib.util
import streamlit as st
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _preload(name: str, path: Path, is_pkg: bool = False) -> None:
    """Charge un module dans sys.modules via importlib, sans instruction import.

    Contourne le bug Python 3.14 + Streamlit 1.57 : le runner interne
    (_mpa_v1 / page.py) execute app.py dans un contexte ou l'instruction
    'import utils.parser' echoue systematiquement (KeyError ou 'not a package')
    car Python ne peut pas resoudre les packages locaux depuis ce contexte.
    En chargeant les fichiers directement avec spec_from_file_location on
    court-circuite entierement le mecanisme d'import defaillant.

    IMPORTANT : on doit aussi setter l'attribut sur le package parent, sinon
    `from utils.X import Y` echoue avec ImportError sur Python 3.14 strict
    meme si le module est correctement dans sys.modules.
    """
    if name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        name, path,
        submodule_search_locations=[str(path.parent)] if is_pkg else None,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    # Attache au package parent pour faire fonctionner `from parent.child import X`.
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent_mod = sys.modules.get(parent_name)
        if parent_mod is not None:
            setattr(parent_mod, child_name, mod)


_preload("config",            _root / "config.py")
_preload("utils",             _root / "utils/__init__.py",       is_pkg=True)
_preload("utils.parser",      _root / "utils/parser.py")
_preload("utils.features",    _root / "utils/features.py")
_preload("utils.metrics",     _root / "utils/metrics.py")
_preload("utils.data_loader", _root / "utils/data_loader.py")
_preload("utils.corpus",      _root / "utils/corpus.py")
_preload("models",            _root / "models/__init__.py",      is_pkg=True)
_preload("models.classifier", _root / "models/classifier.py")
_preload("models.forecaster", _root / "models/forecaster.py")
_preload("models.generator",  _root / "models/generator.py")

st.set_page_config(page_title="Enedis Analytics", page_icon=None, layout="wide", initial_sidebar_state="expanded")

# Charge le CSS global
css_path = _root / "assets" / "style.css"
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
