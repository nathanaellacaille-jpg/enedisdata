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
    grp["dayofweek"] = grp["ts"].dt.dayofweek

    daily = grp.groupby("date")["kw"].sum() * 0.5
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

    grp["week"] = grp["ts"].dt.tz_convert(None).dt.to_period("W")
    weekly_energy = grp.groupby("week")["kw"].sum() * 0.5
    cv_weekly = (weekly_energy.std() / weekly_energy.mean()) if weekly_energy.mean() > 0 else 0.0

    series_kw = grp["kw"].values
    zero_ratio = float((series_kw < 0.05).mean())

    if len(series_kw) > 48:
        x1, x2 = series_kw[48:], series_kw[:-48]
        denom = x1.std() * x2.std()
        autocorr_lag48 = float(np.corrcoef(x1, x2)[0, 1]) if denom > 1e-8 else 0.0
    else:
        autocorr_lag48 = 0.0

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

    active_days_ratio = float((daily >= 0.5).mean()) if len(daily) > 0 else 0.0

    month_map = grp.groupby("date")["month"].first()
    summer_dates = month_map[month_map.isin([6, 7, 8])].index
    winter_dates = month_map[month_map.isin([12, 1, 2])].index
    summer_active = (daily[daily.index.isin(summer_dates)] >= 0.5).mean() if len(summer_dates) > 0 else 0.0
    winter_active = (daily[daily.index.isin(winter_dates)] >= 0.5).mean() if len(winter_dates) > 0 else 0.0
    seasonal_presence_gap = float(summer_active - winter_active)

    _std = series_kw.std()
    skewness = float(((series_kw - series_kw.mean()) ** 3).mean() / (_std ** 3)) if _std > 1e-8 else 0.0

    grp["slot"] = (grp["ts"].dt.hour * 2 + grp["ts"].dt.minute // 30)
    mean_profile = grp.groupby("slot")["kw"].mean().reindex(range(STEPS_PER_DAY), fill_value=0.0).values
    fourier_amps = _fourier_amplitudes(mean_profile, 6)


    dow_energy = grp.groupby("dayofweek")["kw"].sum()
    if dow_energy.sum() > 0:
        p = (dow_energy / dow_energy.sum()).values
        p = p[p > 0]
        weekly_entropy = float(-(p * np.log2(p)).sum() / np.log2(7))
    else:
        weekly_entropy = 0.0

    evening = grp[(grp["hour"] >= 17) & (grp["hour"] < 23)].copy()
    if len(evening) > 0:
        evening_peak_per_day = evening.loc[evening.groupby("date")["kw"].idxmax()][["date", "hour"]]
        peak_hour_std = float(evening_peak_per_day["hour"].std()) if len(evening_peak_per_day) > 1 else 0.0
    else:
        peak_hour_std = 0.0

    grp["slot_dow"] = grp["dayofweek"] * 48 + grp["slot"]
    pivot = grp.pivot_table(index="week", columns="slot_dow", values="kw", aggfunc="mean")
    dow_consistency = float(pivot.std(axis=0).mean()) if not pivot.empty else 0.0

    summer_we = grp.loc[grp["month"].isin([6, 7, 8]) & (grp["dayofweek"] >= 5), "kw"].mean()
    winter_we = grp.loc[grp["month"].isin([12, 1, 2]) & (grp["dayofweek"] >= 5), "kw"].mean()
    if pd.notna(summer_we) and pd.notna(winter_we) and winter_we > 0.01:
        summer_weekend_boost = float(summer_we / winter_we)
    else:
        summer_weekend_boost = 1.0

    night_kw = grp.loc[night_mask, "kw"]
    if len(night_kw) > 0 and night_kw.max() > 0:
        night_amplitude = float((night_kw.max() - night_kw.min()) / (night_kw.mean() + 1e-6))
    else:
        night_amplitude = 0.0

    weekly_low = (weekly_energy < weekly_energy.median() * 0.3).values
    cur_run = 0
    max_run = 0
    for v in weekly_low:
        if v:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    vacation_weeks = float(max_run)

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
        "fourier_amp_1": fourier_amps[0],
        "fourier_amp_2": fourier_amps[1],
        "fourier_amp_3": fourier_amps[2],
        "fourier_amp_4": fourier_amps[3],
        "fourier_amp_5": fourier_amps[4],
        "fourier_amp_6": fourier_amps[5],
        "weekly_entropy": weekly_entropy,
        "peak_hour_std": peak_hour_std,
        "dow_consistency": dow_consistency,
        "summer_weekend_boost": summer_weekend_boost,
        "night_amplitude": night_amplitude,
        "vacation_weeks": vacation_weeks,
    }


def _fourier_amplitudes(profile: np.ndarray, n: int) -> list:
    """Retourne les n premieres amplitudes de Fourier du profil."""
    fft = np.fft.rfft(profile)
    amps = np.abs(fft[1: n + 1]) / len(profile)
    while len(amps) < n:
        amps = np.append(amps, 0.0)
    return list(amps[:n])
