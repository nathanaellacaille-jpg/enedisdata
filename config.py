from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import numpy as np

import os

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_TS_PATH = ROOT_DIR / "RES2-6-9.csv"
DEFAULT_LBL_PATH = ROOT_DIR / "RES2-6-9-labels.csv"

# URL de fallback : telechargee si DEFAULT_TS_PATH absent (Streamlit Cloud).
# Surchargeable via variable d'environnement / secret Streamlit ENEDIS_TS_URL.
DATA_URL_TS = os.environ.get(
    "ENEDIS_TS_URL",
    "https://github.com/nathanaellacaille-jpg/enedisdata/releases/download/data-v1/RES2-6-9.csv",
)


@dataclass(frozen=True)
class Palette:
    """Palette de couleurs (niveaux de gris uniquement)."""

    LR: str = "#0F172A"
    ARIMA: str = "#475569"
    LSTM: str = "#94A3B8"
    REAL: str = "#0F172A"
    BORDER: str = "#E2E8F0"
    TEXT: str = "#0F172A"
    TEXT_MUTED: str = "#64748B"
    MULTI: List[str] = field(default_factory=lambda: [
        "#0F172A", "#1E293B", "#334155", "#475569",
        "#64748B", "#94A3B8", "#CBD5E1", "#E2E8F0",
    ])
    # Couleurs de traces (graphiques uniquement, jamais dans l'UI/CSS)
    ACCENT: List[str] = field(default_factory=lambda: [
        "#2563EB", "#F59E0B", "#10B981", "#DB2777", "#8B5CF6", "#0891B2",
    ])


PAL = Palette()

STEPS_PER_DAY = 48

CLF_TEST_SIZE = 0.30
CLF_N_TREES = 300  # legacy : utilise par les anciens snapshots, plus reference dans le code actuel

FCST_N_LAGS = 192          # 4 jours de lags (Phase 1 tuning : > 192 est marginal, < est sous-optimal)
FCST_N_FOURIER = 6         # 6 harmoniques journalieres (audit reco #8, marginal sur prevision)
FCST_HORIZON_H = 24
FCST_ARIMA_ORDER = (1, 1, 1)              # Phase 1 : (2,1,2) trop complexe, (1,1,1) meilleur ET plus rapide
FCST_SARIMA_SEASONAL = (1, 0, 1, 48)      # saisonnalite journaliere 48 (audit reco #2)

LSTM_SEQ_LEN = 192      # 4 jours (Phase 1 tuning : meilleur que 336 et 672 sur le set tuning)
LSTM_HIDDEN = 64        # Phase 1 : 64 > 32 (capacite legerement superieure debloque 2/5 wins)
LSTM_LAYERS = 1
LSTM_EPOCHS = 30        # max ; early stopping coupe avant
LSTM_LR = 2e-3
LSTM_BATCH_SIZE = 64
LSTM_PATIENCE = 4
LSTM_MIN_DELTA = 1e-3
LSTM_VAL_FRAC = 0.10
LSTM_DROPOUT = 0.0
LSTM_USE_ROLLING = True # Phase 1 : rolling mean/std 24h en input -> meilleur jeu de features

GEN_NOISE_STD = 0.15
GEN_NOISE_RHO = 0.7  # autocorrélation AR(1) entre slots consécutifs
GEN_CORPUS_N = 300    # courbes par classe dans le corpus de référence built-in
GEN_CORPUS_DAYS = 14  # jours par courbe dans le corpus

_env_cap = os.environ.get("ENEDIS_MAX_METERS", "500")
MAX_METERS_UPLOAD: int | None = None if _env_cap.lower() in ("none", "0", "all") else int(_env_cap)


def _make_rp_profile() -> np.ndarray:
    """Profil de reference RP normalise entre 0 et 1 (48 slots)."""
    t = np.arange(48)
    base = 0.18
    matin = 0.45 * np.exp(-0.5 * ((t - 16) / 2.5) ** 2)   # pic 7h-9h  (slot 14-18)
    soir = 0.60 * np.exp(-0.5 * ((t - 40) / 4.0) ** 2)    # pic 18h-22h (slot 36-44)
    profile = base + matin + soir
    return profile / profile.max()


def _make_rs_profile() -> np.ndarray:
    """Profil de reference RS normalise entre 0 et 1 (48 slots)."""
    t = np.arange(48)
    base = 0.06
    soir = 0.35 * np.exp(-0.5 * ((t - 40) / 3.0) ** 2)    # pic 20h (slot 40)
    profile = base + soir
    return profile / profile.max()
