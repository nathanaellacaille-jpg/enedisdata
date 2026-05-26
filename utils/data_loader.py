from pathlib import Path
import pandas as pd
import streamlit as st

from config import DEFAULT_TS_PATH, DEFAULT_LBL_PATH
from utils.parser import parse_timeseries, parse_labels


@st.cache_data(show_spinner="Chargement du jeu de donnees complet...")
def _load_ts_cached(path_str: str, mtime: float, size: int) -> pd.DataFrame:
    """Parse le CSV timeseries depuis le disque (cache invalide si fichier modifie)."""
    return parse_timeseries(path_str, max_meters=None)


@st.cache_data(show_spinner="Chargement des labels...")
def _load_labels_cached(path_str: str, mtime: float, size: int) -> dict:
    """Parse le CSV labels depuis le disque (cache invalide si fichier modifie)."""
    return parse_labels(path_str)


def _stat(path: Path) -> tuple[float, int]:
    """Retourne (mtime, size) pour invalider le cache si le fichier change."""
    s = path.stat()
    return s.st_mtime, s.st_size


def load_default_ts() -> pd.DataFrame | None:
    """Charge le CSV timeseries par defaut (None si introuvable)."""
    p = Path(DEFAULT_TS_PATH)
    if not p.exists():
        st.error(f"Fichier introuvable : {p}")
        return None
    mtime, size = _stat(p)
    return _load_ts_cached(str(p), mtime, size)


def load_default_labels() -> dict | None:
    """Charge le CSV labels par defaut (None si introuvable)."""
    p = Path(DEFAULT_LBL_PATH)
    if not p.exists():
        return None
    mtime, size = _stat(p)
    return _load_labels_cached(str(p), mtime, size)
