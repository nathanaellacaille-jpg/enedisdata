import numpy as np
import pandas as pd
from config import STEPS_PER_DAY, GEN_NOISE_STD, GEN_NOISE_RHO, _make_rp_profile, _make_rs_profile


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance DTW entre deux series de meme longueur (numpy pur, O(n²))."""
    n = len(a)
    D = np.full((n + 1, n + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, n])


class CurveGenerator:
    """Generateur de courbes de charge synthetiques."""

    def __init__(self):
        """Initialise le generateur avec les profils de reference."""
        self._profiles = {
            "RP": _make_rp_profile(),
            "RS": _make_rs_profile(),
        }
        self._scales = {"RP": 3.0, "RS": 1.2}  # kW amplitude reference
        self.noise_std_by_slot = {
            "RS": np.full(STEPS_PER_DAY, GEN_NOISE_STD),
            "RP": np.full(STEPS_PER_DAY, GEN_NOISE_STD),
        }

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
                self._scales[label_name] = float(sub.groupby("slot")["kw"].mean().max())
            # Calibration ecart-type par slot sur donnees reelles
            pivot = sub.pivot_table(index="ts", columns="meter_id", values="kw")
            pivot = pivot.copy()
            pivot["slot"] = np.arange(len(pivot)) % STEPS_PER_DAY
            slot_std = pivot.groupby("slot").std().mean(axis=1)
            self.noise_std_by_slot[label_name] = (
                slot_std.reindex(range(STEPS_PER_DAY), fill_value=GEN_NOISE_STD).values
            )
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
            slot_stds = self.noise_std_by_slot[ct]
            for day in range(n_days):
                day_factor = 1.0 + np.random.normal(0, noise_std * 0.5)
                # Bruit AR(1) : corrélation temporelle entre slots consécutifs
                ar_noise = np.zeros(STEPS_PER_DAY)
                ar_noise[0] = np.random.normal(0, float(slot_stds[0]))
                innov_scale = np.sqrt(1.0 - GEN_NOISE_RHO ** 2)
                for t in range(1, STEPS_PER_DAY):
                    innov = np.random.normal(0, float(slot_stds[t]) * innov_scale)
                    ar_noise[t] = GEN_NOISE_RHO * ar_noise[t - 1] + innov
                for slot in range(STEPS_PER_DAY):
                    kw = max(0.0, (profile[slot] + ar_noise[slot]) * scale * day_factor)
                    records.append({
                        "curve_id": i,
                        "day": day,
                        "slot": slot,
                        "kw": kw,
                        "curve_type": ct,
                    })
        return pd.DataFrame(records)

    def quality_scores(self, gen_df: pd.DataFrame) -> dict:
        """DTW moyen entre chaque courbe generee et le profil reel calibre."""
        scores = {}
        for ct in gen_df["curve_type"].unique():
            real = self._profiles[ct] * self._scales[ct]
            dtw_vals = [
                _dtw_distance(real, grp.sort_values("slot")["kw"].values)
                for (_, _), grp in gen_df[gen_df["curve_type"] == ct].groupby(["curve_id", "day"])
            ]
            scores[ct] = round(float(np.mean(dtw_vals)), 3) if dtw_vals else 0.0
        return scores

    def profile_stats(self) -> dict:
        """Energie moyenne et std estimee par type (std via variance du day_factor)."""
        stats = {}
        for name, profile in self._profiles.items():
            scale = self._scales[name]
            mean_kwh = float(profile.sum() * float(scale) * 0.5)
            # day_factor ~ N(1, GEN_NOISE_STD*0.5) → std energie = mean * GEN_NOISE_STD*0.5
            std_kwh = mean_kwh * GEN_NOISE_STD * 0.5
            stats[name] = {"mean_kwh_day": round(mean_kwh, 2), "std": round(std_kwh, 2)}
        return stats
