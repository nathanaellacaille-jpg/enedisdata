import numpy as np
import pandas as pd
from config import STEPS_PER_DAY


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les features par compteur depuis df [meter_id, ts, kw]."""
    records = []
    for meter_id, grp in df.groupby("meter_id"):
        rec = _features_for_meter(meter_id, grp)
        records.append(rec)
    result = pd.DataFrame(records).set_index("meter_id")
    return result


def _features_for_meter(meter_id: str, grp: pd.DataFrame) -> dict:
    """Calcule les features pour un seul compteur."""
    grp = grp.sort_values("ts").copy()
    grp["date"] = grp["ts"].dt.date
    grp["hour"] = grp["ts"].dt.hour
    grp["month"] = grp["ts"].dt.month
    grp["dayofweek"] = grp["ts"].dt.dayofweek  # 0=Mon, 6=Sun

    daily = grp.groupby("date")["kw"].sum() * 0.5  # kWh par jour
    dow_map = grp.groupby("date")["dayofweek"].first()

    wd_mask = dow_map[dow_map < 5].index
    we_mask = dow_map[dow_map >= 5].index

    wd_energy = daily[daily.index.isin(wd_mask)].mean() if len(wd_mask) > 0 else 0.0
    we_energy = daily[daily.index.isin(we_mask)].mean() if len(we_mask) > 0 else 0.0
    ratio_we_wd = (we_energy / wd_energy) if wd_energy > 0 else 0.0

    cv_daily_energy = (daily.std() / daily.mean()) if daily.mean() > 0 else 0.0

    total = grp["kw"].sum()
    peak_mask = (grp["hour"] >= 18) & (grp["hour"] < 22)
    night_mask = grp["hour"] < 6
    morning_mask = (grp["hour"] >= 6) & (grp["hour"] < 9)
    peak_hour_ratio = grp.loc[peak_mask, "kw"].sum() / total if total > 0 else 0.0
    night_ratio = grp.loc[night_mask, "kw"].sum() / total if total > 0 else 0.0
    morning_ratio = grp.loc[morning_mask, "kw"].sum() / total if total > 0 else 0.0

    summer_kw = grp.loc[grp["month"].isin([6, 7, 8]), "kw"].sum()
    winter_kw = grp.loc[grp["month"].isin([12, 1, 2]), "kw"].sum()
    seasonal_ratio = (summer_kw / winter_kw) if winter_kw > 0 else 1.0

    grp["week"] = grp["ts"].dt.to_period("W")
    weekly_energy = grp.groupby("week")["kw"].sum() * 0.5
    cv_weekly = (weekly_energy.std() / weekly_energy.mean()) if weekly_energy.mean() > 0 else 0.0

    # Profil moyen sur 48 slots
    grp["slot"] = (grp["ts"].dt.hour * 2 + grp["ts"].dt.minute // 30)
    mean_profile = grp.groupby("slot")["kw"].mean().reindex(range(STEPS_PER_DAY), fill_value=0.0).values
    f1, f2, f3 = _fourier_amplitudes(mean_profile, 3)

    return {
        "meter_id": meter_id,
        "ratio_we_wd": ratio_we_wd,
        "cv_daily_energy": cv_daily_energy,
        "peak_hour_ratio": peak_hour_ratio,
        "night_ratio": night_ratio,
        "morning_ratio": morning_ratio,
        "seasonal_ratio": seasonal_ratio,
        "cv_weekly": cv_weekly,
        "fourier_amp_1": f1,
        "fourier_amp_2": f2,
        "fourier_amp_3": f3,
    }


def _fourier_amplitudes(profile: np.ndarray, n: int) -> list:
    """Retourne les n premieres amplitudes de Fourier du profil."""
    fft = np.fft.rfft(profile)
    amps = np.abs(fft[1: n + 1]) / len(profile)
    # Pad si necessaire
    while len(amps) < n:
        amps = np.append(amps, 0.0)
    return list(amps[:n])
