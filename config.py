from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass(frozen=True)
class Palette:
    """Palette de couleurs (niveaux de gris uniquement)."""

    RS: str = "#0F172A"
    RP: str = "#0F172A"
    RS_soft: str = "#F8FAFC"
    RP_soft: str = "#F1F5F9"
    LR: str = "#0F172A"
    ARIMA: str = "#475569"
    LSTM: str = "#94A3B8"
    NAIVE: str = "#CBD5E1"
    REAL: str = "#0F172A"
    GRID: str = "#F8FAFC"
    BORDER: str = "#E2E8F0"
    TEXT: str = "#0F172A"
    TEXT_MUTED: str = "#64748B"
    MULTI: List[str] = field(default_factory=lambda: [
        "#0F172A", "#1E293B", "#334155", "#475569",
        "#64748B", "#94A3B8", "#CBD5E1", "#E2E8F0",
    ])


PAL = Palette()

STEPS_PER_DAY = 48
STEPS_PER_HOUR = 2

CLF_TEST_SIZE = 0.30
CLF_N_COMP_PCA = 5
CLF_N_TREES = 300

FCST_N_LAGS = 48
FCST_N_FOURIER = 3
FCST_HORIZON_H = 24
FCST_ARIMA_ORDER = (2, 1, 2)

LSTM_SEQ_LEN = 96
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_EPOCHS = 60
LSTM_LR = 1e-3

GEN_DEFAULT_N = 10
GEN_NOISE_STD = 0.15

MAX_METERS_UPLOAD = 200

DAY_FR = {
    "Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi",
    "Thursday": "Jeudi", "Friday": "Vendredi", "Saturday": "Samedi", "Sunday": "Dimanche",
}
DAY_FR_SHORT = {
    "Monday": "Lun", "Tuesday": "Mar", "Wednesday": "Mer",
    "Thursday": "Jeu", "Friday": "Ven", "Saturday": "Sam", "Sunday": "Dim",
}


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
