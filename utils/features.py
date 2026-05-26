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

    # zero_ratio : proportion de slots quasi nuls (RS = longues absences)
    series_kw = grp["kw"].values
    zero_ratio = float((series_kw < 0.05).mean())

    # autocorr_lag48 : autocorrélation à j-1 (RS a des transitions brusques zéro/non-zéro)
    if len(series_kw) > 48:
        x1, x2 = series_kw[48:], series_kw[:-48]
        denom = x1.std() * x2.std()
        autocorr_lag48 = float(np.corrcoef(x1, x2)[0, 1]) if denom > 1e-8 else 0.0
    else:
        autocorr_lag48 = 0.0

    # max_gap_days : plus longue séquence de jours consécutifs à conso < 0.5 kWh (absence RS)
    absent = (daily < 0.5).values
    max_gap = 0
    cur = 0
    n_absence_periods = 0
    for v in absent:
        if v:
            cur += 1
            if cur > max_gap:
                max_gap = cur
        else:
            if cur >= 3:
                n_absence_periods += 1
            cur = 0
    if cur >= 3:
        n_absence_periods += 1
    max_gap_days = max_gap

    # active_days_ratio : proportion de jours occupés (RS = beaucoup de jours vides)
    active_days_ratio = float((daily >= 0.5).mean()) if len(daily) > 0 else 0.0

    # seasonal_presence_gap : écart de présence été - hiver dans [-1, 1] (RS = occupation saisonnière marquée)
    month_map = grp.groupby("date")["month"].first()
    summer_dates = month_map[month_map.isin([6, 7, 8])].index
    winter_dates = month_map[month_map.isin([12, 1, 2])].index
    summer_active = (daily[daily.index.isin(summer_dates)] >= 0.5).mean() if len(summer_dates) > 0 else 0.0
    winter_active = (daily[daily.index.isin(winter_dates)] >= 0.5).mean() if len(winter_dates) > 0 else 0.0
    seasonal_presence_gap = float(summer_active - winter_active)

    # skewness : asymétrie de la distribution (RS : beaucoup de zéros + pics rares → fortement asymétrique)
    _std = series_kw.std()
    skewness = float(((series_kw - series_kw.mean()) ** 3).mean() / (_std ** 3)) if _std > 1e-8 else 0.0

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
        "zero_ratio": zero_ratio,
        "autocorr_lag48": autocorr_lag48,
        "max_gap_days": max_gap_days,
        "n_absence_periods": n_absence_periods,
        "active_days_ratio": active_days_ratio,
        "seasonal_presence_gap": seasonal_presence_gap,
        "skewness": skewness,
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
