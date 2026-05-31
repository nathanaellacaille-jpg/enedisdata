import sys
import importlib.util
import streamlit as st
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


for _stale in [m for m in list(sys.modules) if m == "config"
               or m == "utils" or m.startswith("utils.")
               or m == "models" or m.startswith("models.")]:
    del sys.modules[_stale]


def _preload(name: str, path: Path, is_pkg: bool = False) -> None:
    spec = importlib.util.spec_from_file_location(
        name, path,
        submodule_search_locations=[str(path.parent)] if is_pkg else None,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
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
_preload("models",            _root / "models/__init__.py",      is_pkg=True)
_preload("models.classifier", _root / "models/classifier.py")
_preload("models.forecaster", _root / "models/forecaster.py")
_preload("models.generator",  _root / "models/generator.py")

st.set_page_config(page_title="Enedis Analytics", page_icon=None, layout="wide", initial_sidebar_state="expanded")

css_path = _root / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.markdown(
    '<div class="app-header">'
    '<div class="app-title">Enedis Analytics</div>'
    '<div class="app-subtitle">Classification · Prevision · Generation</div>'
    "</div>",
    unsafe_allow_html=True,
)

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
