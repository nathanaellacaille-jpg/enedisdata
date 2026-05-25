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
        self._scales = {"RP": 3.0, "RS": 1.2}
        # Spread log-normal de l'amplitude entre compteurs (calibré par fit)
        self._scale_log_std = {"RP": 0.3, "RS": 0.3}
        self.noise_std_by_slot = {
            "RS": np.full(STEPS_PER_DAY, GEN_NOISE_STD),
            "RP": np.full(STEPS_PER_DAY, GEN_NOISE_STD),
        }

    def fit(self, df: pd.DataFrame, labels: dict | None) -> "CurveGenerator":
        """Calibre profils et bruit par classe sur donnees reelles.

        Strategie :
        - Chaque compteur est normalise sur son propre pic → forme pure [0,1].
        - Le profil de classe = moyenne de ces formes (non biaisee par l'amplitude).
        - L'echelle = mediane des pics individuels (robuste aux outliers).
        - La diversite d'amplitude est capturee par un spread log-normal calibre.
        - Le bruit AR(1) = variabilite de kw_normalise par slot → espace [0,1],
          sans pollution par l'heterogeneite inter-compteurs.
        """
        if df is None or labels is None:
            return self
        for label_val, label_name in [(0, "RP"), (1, "RS")]:
            ids = [k for k, v in labels.items() if v == label_val]
            sub = df[df["meter_id"].isin(ids)]
            if sub.empty:
                continue
            sub = sub.copy()
            sub["slot"] = sub["ts"].dt.hour * 2 + sub["ts"].dt.minute // 30

            # Pic moyen par compteur (sur le profil slot-moyen)
            slot_means = sub.groupby(["meter_id", "slot"])["kw"].mean()
            meter_peaks = slot_means.groupby("meter_id").max()
            valid = meter_peaks[meter_peaks > 0].index
            if valid.empty:
                continue

            # Tableau (n_compteurs × 48) des profils normalisés au pic de chaque compteur
            norm_matrix = (
                slot_means.loc[valid]
                .unstack("slot")
                .reindex(columns=range(STEPS_PER_DAY), fill_value=0.0)
                .div(meter_peaks[valid], axis=0)
                .values
            )  # shape (n_valid, 48), valeurs dans [0, 1]

            # Profil de classe = moyenne des formes normalisées, re-normalisée au max
            mean_shape = norm_matrix.mean(axis=0)
            if mean_shape.max() > 0:
                self._profiles[label_name] = mean_shape / mean_shape.max()

            # Échelle = médiane des pics individuels ; spread log-normal pour generate()
            peaks_arr = meter_peaks[valid].values
            self._scales[label_name] = float(np.median(peaks_arr))
            self._scale_log_std[label_name] = float(
                np.clip(np.std(np.log(peaks_arr + 1e-9)), 0.05, 1.5)
            )

            # Bruit : pattern relatif intra-compteur par slot
            # 1) std de kw_norm par (meter_id, slot) → variabilité jour/jour propre à chaque compteur
            # 2) moyenne sur les compteurs → pattern typique en espace [0,1]
            # 3) normalisation au mean=1 → patron relatif (quels slots sont plus bruités)
            # 4) mise à l'échelle par GEN_NOISE_STD → niveau conservateur pour courbes lisibles
            # La variabilité absolue du dataset (0.5-0.8 normalisé) est intentionnellement
            # ignorée : on veut des courbes représentatives, pas des tirages aléatoires bruts.
            sub_v = sub[sub["meter_id"].isin(valid)].copy()
            sub_v["meter_peak"] = sub_v["meter_id"].map(meter_peaks)
            sub_v["kw_norm"] = sub_v["kw"] / sub_v["meter_peak"]
            intra_std = (
                sub_v.groupby(["meter_id", "slot"])["kw_norm"]
                .std()
                .groupby("slot")
                .mean()
                .reindex(range(STEPS_PER_DAY), fill_value=GEN_NOISE_STD)
            )
            mean_intra = intra_std.mean()
            rel_pattern = intra_std / max(mean_intra, 1e-9)   # normalisé, mean≈1.0
            # Cap adaptatif : proportionnel à la dynamique du profil calibré.
            # Garantit SNR ≥ ~1.3 → corr individuelle ≥ 0.79 quelle que soit la
            # "platitude" du profil réel (RP très plat = profil_std faible → bruit réduit).
            profile_std = float(self._profiles[label_name].std())
            noise_cap = min(0.20, profile_std * 0.75)
            self.noise_std_by_slot[label_name] = np.clip(
                GEN_NOISE_STD * rel_pattern.values, 0.0, noise_cap
            )

        return self

    def generate(self, n: int, curve_type: str, n_days: int = 7, noise_std: float = GEN_NOISE_STD) -> pd.DataFrame:
        """Genere n courbes sur n_days. curve_type in {'RS','RP','mixed'}.

        Chaque courbe tire sa propre amplitude dans la distribution log-normale
        calibree par fit() → diversite realiste des niveaux de consommation.
        Le parametre noise_std scale le bruit AR(1) et le facteur journalier.
        """
        # Ratio par rapport au niveau de bruit par defaut (controle UI)
        noise_ratio = noise_std / max(GEN_NOISE_STD, 1e-9)
        records = []
        for i in range(n):
            if curve_type == "mixed":
                ct = "RS" if i % 2 == 0 else "RP"
            else:
                ct = curve_type
            profile = self._profiles[ct]
            slot_stds = self.noise_std_by_slot[ct] * noise_ratio

            # Amplitude log-normale propre à cette courbe (diversité inter-compteurs)
            log_mean = np.log(max(self._scales[ct], 1e-9))
            scale = float(np.clip(
                np.exp(np.random.normal(log_mean, self._scale_log_std[ct])),
                self._scales[ct] * 0.05,
                self._scales[ct] * 8.0,
            ))

            for day in range(n_days):
                # Facteur journalier borné : ±~15 % autour de 1 au niveau de bruit par défaut
                day_factor = float(np.clip(
                    1.0 + np.random.normal(0, noise_std * 0.3),
                    0.3, 2.0,
                ))
                # Bruit AR(1) en espace normalisé [0,1]
                ar_noise = np.zeros(STEPS_PER_DAY)
                ar_noise[0] = np.random.normal(0, float(slot_stds[0]))
                innov_scale = np.sqrt(1.0 - GEN_NOISE_RHO ** 2)
                for t in range(1, STEPS_PER_DAY):
                    innov = np.random.normal(0, float(slot_stds[t]) * innov_scale)
                    ar_noise[t] = GEN_NOISE_RHO * ar_noise[t - 1] + innov
                for slot in range(STEPS_PER_DAY):
                    raw = (profile[slot] + ar_noise[slot]) * scale * day_factor
                    # Clip [0, 4×scale] : protège contre spikes extrêmes
                    kw = float(np.clip(raw, 0.0, scale * 4.0))
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
        """Energie moyenne et std par type."""
        stats = {}
        for name, profile in self._profiles.items():
            scale = self._scales[name]
            daily_energy = float(profile.sum() * float(scale) * 0.5)
            stats[name] = {"mean_kwh_day": round(daily_energy, 2), "std": 0.0}
        return stats
