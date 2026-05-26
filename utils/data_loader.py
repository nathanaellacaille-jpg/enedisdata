import os
import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st

from config import DEFAULT_TS_PATH, DEFAULT_LBL_PATH, DATA_URL_TS, ROOT_DIR, MAX_METERS_UPLOAD
from utils.parser import parse_timeseries, parse_labels


CACHE_DIR = Path(tempfile.gettempdir()) / "enedis-data"


def _download(url: str, dest: Path) -> bool:
    """Stream un fichier depuis url vers dest, avec barre de progression Streamlit."""
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    placeholder = st.empty()
    bar = st.progress(0.0)
    try:
        placeholder.caption(f"Telechargement du jeu de donnees depuis {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "enedis-app/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            downloaded = 0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        bar.progress(min(downloaded / total, 1.0))
        tmp.replace(dest)
        placeholder.caption(f"Telecharge : {downloaded / 1e6:.1f} MB")
        bar.empty()
        return True
    except Exception as e:
        bar.empty()
        placeholder.error(f"Echec du telechargement ({url}) : {e}")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False


def _resolve_ts_path() -> Path | None:
    """Renvoie le chemin du CSV timeseries (local ou cache telecharge)."""
    p = Path(DEFAULT_TS_PATH)
    if p.exists():
        return p
    cached = CACHE_DIR / "RES2-6-9.csv"
    if cached.exists():
        return cached
    if DATA_URL_TS and _download(DATA_URL_TS, cached):
        return cached
    return None


@st.cache_data(show_spinner="Chargement du jeu de donnees...")
def _load_ts_cached(path_str: str, mtime: float, size: int, max_meters: int | None) -> pd.DataFrame:
    """Parse le CSV timeseries depuis le disque (cache invalide si fichier modifie)."""
    return parse_timeseries(path_str, max_meters=max_meters)


@st.cache_data(show_spinner="Chargement des labels...")
def _load_labels_cached(path_str: str, mtime: float, size: int) -> dict:
    """Parse le CSV labels depuis le disque (cache invalide si fichier modifie)."""
    return parse_labels(path_str)


def _stat(path: Path) -> tuple[float, int]:
    """Retourne (mtime, size) pour invalider le cache si le fichier change."""
    s = path.stat()
    return s.st_mtime, s.st_size


def load_default_ts() -> pd.DataFrame | None:
    """Charge le CSV timeseries (local prioritaire, sinon telecharge depuis DATA_URL_TS)."""
    p = _resolve_ts_path()
    if p is None:
        st.error(
            "Jeu de donnees indisponible.\n\n"
            f"- Chemin local : `{DEFAULT_TS_PATH}` (absent)\n"
            f"- Cache : `{CACHE_DIR / 'RES2-6-9.csv'}` (absent)\n"
            f"- URL : `{DATA_URL_TS}`\n"
            f"- cwd : `{os.getcwd()}`\n\n"
            "Verifier que la release GitHub existe et que le tag/fichier correspond."
        )
        return None
    mtime, size = _stat(p)
    return _load_ts_cached(str(p), mtime, size, MAX_METERS_UPLOAD)


def load_default_labels() -> dict | None:
    """Charge le CSV labels (tracke dans git, doit etre present)."""
    p = Path(DEFAULT_LBL_PATH)
    if not p.exists():
        st.warning(f"Labels introuvables : `{p}`")
        return None
    mtime, size = _stat(p)
    return _load_labels_cached(str(p), mtime, size)
