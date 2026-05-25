import numpy as np
import pandas as pd
from config import STEPS_PER_DAY, GEN_NOISE_STD, GEN_NOISE_RHO, _make_rp_profile, _make_rs_profile


def _wasserstein_1d(x: np.ndarray, y: np.ndarray, n_q: int = 200) -> float:
    """Distance Wasserstein-1 entre deux echantillons 1D via quantiles."""
    if len(x) == 0 or len(y) == 0:
        return 0.0
    q = np.linspace(0.0, 1.0, n_q)
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


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
        # Jitter de timing du pic entre compteurs (en slots), calibre par fit
        self._peak_jitter_slots = {"RP": 0.0, "RS": 0.0}
        self.noise_std_by_slot = {
            "RS": np.full(STEPS_PER_DAY, GEN_NOISE_STD),
            "RP": np.full(STEPS_PER_DAY, GEN_NOISE_STD),
        }

    def fit(self, df: pd.DataFrame, labels: dict | None) -> "CurveGenerator":
        """Calibre profils, scale et bruit par classe sur donnees reelles.

        Strategie :
        - Profil de classe = moyenne des formes normalisees au pic, puis
          'sharpening' (profile ** alpha) pour matcher le ratio peak/energy
          reel (le profil moyen est mecaniquement plus plat que les profils
          individuels → energie surestimee a pic fixe sans sharpening).
        - Scale calibre sur l'energie journaliere mediane (pas le pic), pour
          que les courbes generees aient la bonne consommation totale.
        - Diversite d'amplitude = log-normal fit sur les energies par compteur.
        - Bruit AR(1) = variabilite intra-compteur en espace normalise [0,1].
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
            sub["date"] = sub["ts"].dt.date

            # Pic et energie journaliers par (compteur, date), agreges en mediane par compteur
            slot_means = sub.groupby(["meter_id", "slot"])["kw"].mean()
            daily_stats = sub.groupby(["meter_id", "date"]).agg(
                peak=("kw", "max"), energy_half=("kw", "sum")
            )
            daily_stats["energy"] = daily_stats["energy_half"] * 0.5
            meter_peaks_daily = daily_stats.groupby("meter_id")["peak"].median()
            meter_energy = daily_stats.groupby("meter_id")["energy"].median()
            # On normalise toujours par le pic-of-slot-means pour la forme (lisse, robuste)
            meter_peaks_smooth = slot_means.groupby("meter_id").max()
            valid = meter_peaks_smooth[
                (meter_peaks_smooth > 0)
                & (meter_peaks_daily.reindex(meter_peaks_smooth.index) > 0)
                & (meter_energy.reindex(meter_peaks_smooth.index) > 0)
            ].index
            if valid.empty:
                continue
            meter_peaks = meter_peaks_smooth  # alias conserve pour la suite

            # Tableau (n_compteurs × 48) des profils normalisés au pic de chaque compteur
            norm_matrix = (
                slot_means.loc[valid]
                .unstack("slot")
                .reindex(columns=range(STEPS_PER_DAY), fill_value=0.0)
                .div(meter_peaks[valid], axis=0)
                .values
            )  # shape (n_valid, 48), valeurs dans [0, 1]

            mean_shape = norm_matrix.mean(axis=0)
            if mean_shape.max() <= 0:
                continue
            mean_shape = mean_shape / mean_shape.max()

            # Sharpening : profile ** alpha + plancher additif calibre pour que
            # sum(profile)*0.5 == median(energy) / median(peak_journalier).
            # Le pic-cible est le pic median d'une journee individuelle (pas le pic
            # moyen-temporel). Le plancher (~1 % du pic) modelise la consommation
            # toujours allumee → evite la sur-occurrence de slots a zero.
            FLOOR = 0.01
            median_peak = float(np.median(meter_peaks_daily[valid].values))
            median_energy = float(np.median(meter_energy[valid].values))
            target_sum_half = median_energy / max(median_peak, 1e-9)
            alpha_grid = np.linspace(0.5, 25.0, 246)

            def _sharpen(p, a):
                q = np.maximum(p ** a, FLOOR)
                return q / q.max()

            sums_half = np.array([_sharpen(mean_shape, a).sum() * 0.5 for a in alpha_grid])
            alpha = float(alpha_grid[np.argmin(np.abs(sums_half - target_sum_half))])
            self._profiles[label_name] = _sharpen(mean_shape, alpha)

            # Jitter de timing : std des positions de pic des compteurs reels,
            # restreint a la plage typique du pic du soir (15h-23h = slots 30-46)
            # pour ignorer les compteurs sans pic structure (argmax aberrant).
            # Permet a generate() de decaler le profil de chaque courbe pour que
            # la moyenne de N courbes generees reproduise le lissage temporel reel.
            peak_slots = norm_matrix.argmax(axis=1)
            in_evening = (peak_slots >= 30) & (peak_slots <= 46)
            jitter_std = float(np.std(peak_slots[in_evening])) if in_evening.any() else 2.0
            self._peak_jitter_slots[label_name] = float(np.clip(jitter_std, 1.0, 4.0))

            # Scale calibre sur l'energie (apres sharpening), log-normal sur energies
            profile_sum_half = float(self._profiles[label_name].sum() * 0.5)
            required_scales = meter_energy[valid].values / max(profile_sum_half, 1e-9)
            self._scales[label_name] = float(np.median(required_scales))
            self._scale_log_std[label_name] = float(
                np.clip(np.std(np.log(required_scales + 1e-9)), 0.05, 1.5)
            )

            # Bruit : pattern relatif intra-compteur par slot
            # 1) std de kw_norm par (meter_id, slot) → variabilité jour/jour propre à chaque compteur
            # 2) moyenne sur les compteurs → pattern typique en espace [0,1]
            # 3) normalisation au mean=1 → patron relatif (quels slots sont plus bruités)
            # 4) mise à l'échelle par GEN_NOISE_STD → niveau conservateur pour courbes lisibles
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
            slot_stds = self.noise_std_by_slot[ct] * noise_ratio

            # Decalage de timing par courbe : reproduit l'etalement temporel des
            # pics entre compteurs sans aplatir le profil individuel.
            jitter = self._peak_jitter_slots[ct]
            offset = int(round(np.random.normal(0, jitter))) if jitter > 0 else 0
            profile = np.roll(self._profiles[ct], offset)

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
                    # Bruit multiplicatif : variabilite proportionnelle au profil.
                    # Un foyer en veille (baseline) varie peu en absolu ; un foyer
                    # en pic varie davantage. Evite aussi le clipping a 0 quand
                    # le profil est tres bas.
                    raw = profile[slot] * (1.0 + ar_noise[slot]) * scale * day_factor
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

    def similarity_report(
        self,
        real_df: pd.DataFrame | None,
        labels: dict | None,
        gen_df: pd.DataFrame,
        curve_type: str,
    ) -> dict:
        """Compare courbes generees vs donnees reelles pour un type donne.

        Retourne profils journaliers, distributions d'energie et metriques de
        similarite (Pearson sur la forme, Wasserstein sur l'energie). Si pas
        de donnees reelles fournies, has_real=False et les comparaisons sont None.
        """
        report = {
            "has_real": False,
            "pearson_profile": None,
            "wasserstein_energy": None,
            "mean_energy_real": None,
            "mean_energy_gen": None,
            "peak_real": None,
            "peak_gen": None,
            "we_ratio_real": None,
            "we_ratio_gen": None,
            "profile_real": None,
            "profile_gen": None,
            "energy_real": None,
            "energy_gen": None,
        }

        gen_sub = gen_df[gen_df["curve_type"] == curve_type]
        if gen_sub.empty:
            return report

        profile_gen = (
            gen_sub.groupby("slot")["kw"].mean()
            .reindex(range(STEPS_PER_DAY), fill_value=0.0)
            .values
        )
        daily_gen = gen_sub.groupby(["curve_id", "day"])["kw"].sum().reset_index()
        daily_gen["energy"] = daily_gen["kw"] * 0.5
        daily_gen["is_we"] = daily_gen["day"] % 7 >= 5
        energy_gen = daily_gen["energy"].values
        we_avg_gen = daily_gen.loc[daily_gen["is_we"], "energy"].mean() if daily_gen["is_we"].any() else 0.0
        wd_avg_gen = daily_gen.loc[~daily_gen["is_we"], "energy"].mean() if (~daily_gen["is_we"]).any() else 0.0

        peaks_gen = gen_sub.groupby(["curve_id", "day"])["kw"].max()
        report["profile_gen"] = profile_gen
        report["energy_gen"] = energy_gen
        report["mean_energy_gen"] = float(energy_gen.mean()) if len(energy_gen) else 0.0
        report["peak_gen"] = float(peaks_gen.median()) if len(peaks_gen) else 0.0
        report["we_ratio_gen"] = float(we_avg_gen / max(wd_avg_gen, 1e-9)) if wd_avg_gen else 0.0

        if real_df is None or labels is None:
            return report

        label_val = 1 if curve_type == "RS" else 0
        real_ids = [k for k, v in labels.items() if v == label_val]
        sub = real_df[real_df["meter_id"].isin(real_ids)]
        if sub.empty:
            return report

        sub = sub.copy()
        sub["slot"] = sub["ts"].dt.hour * 2 + sub["ts"].dt.minute // 30
        sub["date"] = sub["ts"].dt.date
        sub["dow"] = sub["ts"].dt.dayofweek

        profile_real = (
            sub.groupby("slot")["kw"].mean()
            .reindex(range(STEPS_PER_DAY), fill_value=0.0)
            .values
        )
        daily = sub.groupby(["meter_id", "date"]).agg(
            kw_sum=("kw", "sum"), dow=("dow", "first")
        )
        daily["energy"] = daily["kw_sum"] * 0.5
        daily["is_we"] = daily["dow"] >= 5
        energy_real = daily["energy"].values
        we_avg_real = daily.loc[daily["is_we"], "energy"].mean() if daily["is_we"].any() else 0.0
        wd_avg_real = daily.loc[~daily["is_we"], "energy"].mean() if (~daily["is_we"]).any() else 0.0

        pearson = 0.0
        if profile_real.std() > 0 and profile_gen.std() > 0:
            pearson = float(np.corrcoef(profile_real, profile_gen)[0, 1])

        report["has_real"] = True
        report["pearson_profile"] = pearson
        report["wasserstein_energy"] = _wasserstein_1d(energy_real, energy_gen)
        report["profile_real"] = profile_real
        report["energy_real"] = energy_real
        peaks_real = sub.groupby(["meter_id", "date"])["kw"].max()
        report["mean_energy_real"] = float(np.mean(energy_real)) if len(energy_real) else 0.0
        report["peak_real"] = float(peaks_real.median()) if len(peaks_real) else 0.0
        report["we_ratio_real"] = (
            float(we_avg_real / max(wd_avg_real, 1e-9)) if wd_avg_real else 0.0
        )
        return report

    def profile_stats(self) -> dict:
        """Energie moyenne et std par type."""
        stats = {}
        for name, profile in self._profiles.items():
            scale = self._scales[name]
            daily_energy = float(profile.sum() * float(scale) * 0.5)
            stats[name] = {"mean_kwh_day": round(daily_energy, 2), "std": 0.0}
        return stats
