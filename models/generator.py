import numpy as np
import pandas as pd
from config import STEPS_PER_DAY, GEN_NOISE_STD, _make_rp_profile, _make_rs_profile


class CurveGenerator:
    """Generateur de courbes de charge synthetiques."""

    def __init__(self):
        """Initialise le generateur avec les profils de reference."""
        self._profiles = {
            "RP": _make_rp_profile(),
            "RS": _make_rs_profile(),
        }
        self._scales = {"RP": 3.0, "RS": 1.2}  # kW amplitude reference

    def fit(self, df: pd.DataFrame, labels: dict | None) -> "CurveGenerator":
        """Calcule profils moyens par classe. Fallback sur profils de reference si labels=None."""
        if df is None or labels is None:
            return self
        for label_val, label_name in [(0, "RP"), (1, "RS")]:
            ids = [k for k, v in labels.items() if v == label_val]
            sub = df[df["meter_id"].isin(ids)]
            if sub.empty:
                continue
            sub = sub.copy()
            sub["slot"] = sub["ts"].dt.hour * 2 + sub["ts"].dt.minute // 30
            profile = sub.groupby("slot")["kw"].mean().reindex(range(STEPS_PER_DAY), fill_value=0.0).values
            if profile.max() > 0:
                self._profiles[label_name] = profile / profile.max()
                self._scales[label_name] = sub.groupby("slot")["kw"].mean().max()
        return self

    def generate(self, n: int, curve_type: str, n_days: int = 7, noise_std: float = GEN_NOISE_STD) -> pd.DataFrame:
        """Genere n courbes sur n_days. curve_type in {'RS','RP','mixed'}."""
        records = []
        for i in range(n):
            if curve_type == "mixed":
                ct = "RS" if i % 2 == 0 else "RP"
            else:
                ct = curve_type
            profile = self._profiles[ct]
            scale = self._scales[ct]
            for day in range(n_days):
                # Variation journaliere
                day_factor = 1.0 + np.random.normal(0, noise_std * 0.5)
                for slot in range(STEPS_PER_DAY):
                    noise = np.random.normal(0, noise_std)
                    kw = max(0.0, (profile[slot] + noise) * scale * day_factor)
                    records.append({
                        "curve_id": i,
                        "day": day,
                        "slot": slot,
                        "kw": kw,
                        "curve_type": ct,
                    })
        return pd.DataFrame(records)

    def profile_stats(self) -> dict:
        """Energie moyenne et std par type."""
        stats = {}
        for name, profile in self._profiles.items():
            scale = self._scales[name]
            daily_energy = profile.sum() * scale * 0.5  # kWh/jour
            stats[name] = {"mean_kwh_day": round(daily_energy, 2), "std": 0.0}
        return stats
