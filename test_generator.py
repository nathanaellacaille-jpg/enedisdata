"""Test interne du CurveGenerator sur les vrais fichiers CSV Enedis."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from utils.parser import parse_timeseries, parse_labels
from models.generator import CurveGenerator
from config import STEPS_PER_DAY, GEN_NOISE_STD

SEP = "-" * 65
OK = "[OK]"
KO = "[ECHEC]"


def check(cond, label_ok, label_ko=None):
    lbl = label_ko or label_ok
    if cond:
        print(f"  {OK} {label_ok}")
    else:
        print(f"  {KO} {lbl}")
        raise AssertionError(lbl)


def section(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


# ── 1. Chargement ────────────────────────────────────────────────────
section("1. Chargement CSV")
df     = parse_timeseries("RES2-6-9-200.csv")
labels = parse_labels(open("RES2-6-9-labels.csv", "rb"))

n_rp = sum(v == 0 for v in labels.values())
n_rs = sum(v == 1 for v in labels.values())
print(f"  Lignes   : {len(df):,}")
print(f"  Compteurs: {df['meter_id'].nunique()}")
print(f"  Labels   : {len(labels)}  RP={n_rp}  RS={n_rs}")
print(f"  kw range : {df['kw'].min():.3f} .. {df['kw'].max():.3f} kW")


# ── 2. fit() ─────────────────────────────────────────────────────────
section("2. fit() — calibration des profils et du bruit")
gen = CurveGenerator()
gen.fit(df, labels)

for ct in ("RP", "RS"):
    p    = gen._profiles[ct]
    s    = gen._scales[ct]
    ls   = gen._scale_log_std[ct]
    stds = gen.noise_std_by_slot[ct]

    print(f"\n  {ct}:")
    print(f"    profil   max={p.max():.4f}  min={p.min():.3f}")
    print(f"    scale    median={s:.3f} kW  log_std={ls:.3f}")
    print(f"    slot_std min={stds.min():.4f}  max={stds.max():.4f}  mean={stds.mean():.4f}")

    check(abs(p.max() - 1.0) < 1e-6,      f"{ct}: profil normalise (max=1.0)")
    check(p.min() >= 0.0,                  f"{ct}: profil >= 0")
    check(stds.max() <= 0.20 + 1e-9,      f"{ct}: slot_std <= 0.20")
    check(stds.min() >= 0.0,              f"{ct}: slot_std >= 0")
    # Niveau conservateur : toujours <= profile_std * 0.75 (cap adaptatif)
    profile_std = float(p.std())
    noise_cap_expected = min(0.20, profile_std * 0.75)
    check(stds.max() <= noise_cap_expected + 1e-9,
          f"{ct}: slot_std respecte cap adaptatif ({stds.max():.4f} <= {noise_cap_expected:.4f})",
          f"{ct}: slot_std depasse le cap adaptatif")
    check(ls > 0.05,  f"{ct}: log_std > 0.05 — diversite amplitude calibree")
    check(s  > 0.0,   f"{ct}: scale > 0")


# ── 2b. Correlation individuelle (lisibilite des courbes) ────────────
section("2b. Correlation individuelle — courbes visuellement lisibles")

def _aligned_corr(c_norm, ref, max_shift=6):
    """Cross-correlation max sur +- max_shift slots (tolere le jitter de timing)."""
    return max(float(np.corrcoef(c_norm, np.roll(ref, k))[0, 1]) for k in range(-max_shift, max_shift + 1))

for ct in ("RP", "RS"):
    np.random.seed(42)
    gdf_ind = gen.generate(n=30, curve_type=ct, n_days=1)
    ref = gen._profiles[ct]
    corrs = []
    for cid in gdf_ind["curve_id"].unique():
        c = gdf_ind[gdf_ind["curve_id"] == cid]["kw"].values
        c_norm = c / (c.max() + 1e-9)
        corrs.append(_aligned_corr(c_norm, ref))
    corr_mean = np.mean(corrs)
    corr_min  = np.min(corrs)
    print(f"  {ct}: corr individuelle moy={corr_mean:.3f}  min={corr_min:.3f}")
    check(corr_mean > 0.75,
          f"{ct}: courbes lisibles (corr_moy={corr_mean:.3f} > 0.75)",
          f"{ct}: courbes bruitees (corr_moy={corr_mean:.3f} <= 0.75)")
    check(corr_min > 0.50,
          f"{ct}: pas de courbe chaotique (corr_min={corr_min:.3f} > 0.50)",
          f"{ct}: au moins une courbe chaotique (corr_min={corr_min:.3f})")


# ── 3. Diversite d'amplitude ─────────────────────────────────────────
section("3. Diversite d'amplitude entre courbes generees")
np.random.seed(42)
gen_df = gen.generate(n=100, curve_type="mixed", n_days=7)

for ct in ("RP", "RS"):
    sub = gen_df[gen_df["curve_type"] == ct]
    amp  = sub.groupby("curve_id")["kw"].mean()
    cv   = amp.std() / amp.mean()
    print(f"  {ct}: amplitude mean={amp.mean():.3f}  std={amp.std():.3f}  CV={cv:.3f}")
    check(cv > 0.10,
          f"{ct}: diversite reelle (CV={cv:.3f} > 0.10)",
          f"{ct}: amplitude trop uniforme (CV={cv:.3f} <= 0.10)")


# ── 4. Sante des valeurs ─────────────────────────────────────────────
section("4. Sante des valeurs generees (100 courbes x 7 jours)")
for ct in ("RP", "RS"):
    sub   = gen_df[gen_df["curve_type"] == ct]
    kw    = sub["kw"].values
    s_med = gen._scales[ct]

    neg    = int((kw < 0).sum())
    spikes = int((kw > s_med * 8.0 + 1e-6).sum())
    pct0   = (kw == 0.0).sum() / len(kw) * 100

    print(f"\n  {ct} — {len(kw):,} pts (scale mediane={s_med:.3f} kW) :")
    print(f"    kw : min={kw.min():.3f}  max={kw.max():.3f}  "
          f"mean={kw.mean():.3f}  std={kw.std():.3f}")
    print(f"    negatifs : {neg}  |  spikes>8x : {spikes}  |  zeros : {pct0:.1f}%")

    check(neg == 0,    f"{ct}: 0 valeur negative")
    check(spikes == 0, f"{ct}: 0 spike > 8x scale")
    check(pct0 < 5.0,
          f"{ct}: zeros < 5 % ({pct0:.1f} %)",
          f"{ct}: trop de zeros ({pct0:.1f} % >= 5 %) — bruit trop eleve")


# ── 5. Coherence de forme ─────────────────────────────────────────────
section("5. Coherence de forme (correlation de Pearson)")
for ct in ("RP", "RS"):
    sub   = gen_df[gen_df["curve_type"] == ct]
    mean_curve = sub.groupby("slot")["kw"].mean().values
    ref   = gen._profiles[ct] * gen._scales[ct]
    corr  = np.corrcoef(mean_curve, ref)[0, 1]
    print(f"  {ct}: corr(profil moyen genere, reference) = {corr:.4f}")
    # Seuil 0.85 : le profil moyen genere est lisse par le jitter de timing,
    # il ne matche plus 1:1 le profil de reference (qui est non-jittered).
    check(corr > 0.85,
          f"{ct}: forme coherente (corr={corr:.4f} > 0.85)",
          f"{ct}: forme degradee (corr={corr:.4f} <= 0.85)")


# ── 6. Separabilite RP / RS ───────────────────────────────────────────
section("6. Separabilite RP / RS")
rp_mean = gen_df[gen_df["curve_type"] == "RP"].groupby("slot")["kw"].mean().values
rs_mean = gen_df[gen_df["curve_type"] == "RS"].groupby("slot")["kw"].mean().values
corr_rp_rs = np.corrcoef(rp_mean / rp_mean.max(), rs_mean / rs_mean.max())[0, 1]
print(f"  Correlation RP_shape vs RS_shape : {corr_rp_rs:.4f}")
check(corr_rp_rs < 0.99,
      f"Profils RP/RS distincts (corr={corr_rp_rs:.4f} < 0.99)",
      f"RP et RS quasi-identiques (corr={corr_rp_rs:.4f})")


# ── 7. Effet du parametre noise_std ──────────────────────────────────
section("7. Effet du parametre noise_std (bruit faible vs fort)")
np.random.seed(0)
df_low  = gen.generate(n=50, curve_type="RP", n_days=5, noise_std=0.05)
df_high = gen.generate(n=50, curve_type="RP", n_days=5, noise_std=0.40)
std_low, std_high = df_low["kw"].std(), df_high["kw"].std()
print(f"  noise_std=0.05 : kw.std()={std_low:.4f}")
print(f"  noise_std=0.40 : kw.std()={std_high:.4f}")
check(std_high > std_low,
      f"noise_std eleve => plus de variance ({std_high:.4f} > {std_low:.4f})",
      f"noise_std sans effet ({std_high:.4f} <= {std_low:.4f})")


# ── 8. similarity_report ─────────────────────────────────────────────
section("8. similarity_report (validation vs donnees reelles)")
np.random.seed(1)
for ct in ("RP", "RS"):
    small = gen.generate(n=20, curve_type=ct, n_days=7)
    rep = gen.similarity_report(df, labels, small, ct)
    print(f"\n  {ct}:")
    print(f"    has_real         = {rep['has_real']}")
    print(f"    pearson_profile  = {rep['pearson_profile']:.3f}")
    print(f"    wasserstein_kWh  = {rep['wasserstein_energy']:.3f}")
    print(f"    energie reel/gen = {rep['mean_energy_real']:.2f} / {rep['mean_energy_gen']:.2f} kWh")
    print(f"    pic     reel/gen = {rep['peak_real']:.2f} / {rep['peak_gen']:.2f} kW")
    print(f"    we_ratio reel/gen= {rep['we_ratio_real']:.2f} / {rep['we_ratio_gen']:.2f}")
    check(rep["has_real"], f"{ct}: has_real True quand donnees fournies")
    check(rep["pearson_profile"] > 0.75,
          f"{ct}: forme correlee (pearson={rep['pearson_profile']:.3f} > 0.75)",
          f"{ct}: forme decorrelee (pearson={rep['pearson_profile']:.3f})")
    check(rep["wasserstein_energy"] >= 0, f"{ct}: wasserstein >= 0")
    check(rep["profile_real"].shape == (STEPS_PER_DAY,), f"{ct}: profile_real shape OK")
    check(rep["profile_gen"].shape == (STEPS_PER_DAY,), f"{ct}: profile_gen shape OK")

# Sans donnees reelles : has_real=False mais stats gen presentes
rep_no = gen.similarity_report(None, None, small, "RS")
check(not rep_no["has_real"], "has_real=False sans donnees reelles")
check(rep_no["pearson_profile"] is None, "pearson_profile None sans reel")
check(rep_no["mean_energy_gen"] > 0, "mean_energy_gen calcule meme sans reel")


# ── 9. profile_stats ─────────────────────────────────────────────────
section("9. profile_stats()")
stats = gen.profile_stats()
for ct, s in stats.items():
    print(f"  {ct}: {s}")
    check(s["mean_kwh_day"] > 0, f"{ct}: energie journaliere positive")


# ── 10. Cas edge ─────────────────────────────────────────────────────
section("10. Cas edge")
CurveGenerator().fit(None, None)
print("  fit(None, None)  : OK")
CurveGenerator().fit(df, {})
print("  fit(df, {})      : OK")
g_rp = CurveGenerator(); g_rp.fit(df, {k: v for k, v in labels.items() if v == 0})
print("  fit(RP only)     : OK")
g_rs = CurveGenerator(); g_rs.fit(df, {k: v for k, v in labels.items() if v == 1})
print("  fit(RS only)     : OK")
d_def = CurveGenerator().generate(n=5, curve_type="mixed", n_days=2)
check((d_def["kw"] >= 0).all(), "Profils par defaut : valeurs >= 0")


# ── 11. Calibre > defaut en fidelite de forme ─────────────────────────
section("11. Calibration ameliore la fidelite vs profils par defaut")
sub_rp = df[df["meter_id"].isin([k for k, v in labels.items() if v == 0])].copy()
sub_rp["slot"] = sub_rp["ts"].dt.hour * 2 + sub_rp["ts"].dt.minute // 30
real_rp = sub_rp.groupby("slot")["kw"].mean().reindex(range(STEPS_PER_DAY), fill_value=0).values
real_rp_n = real_rp / real_rp.max()

def profile_corr(g, ct="RP", seed=99):
    np.random.seed(seed)
    d = g.generate(n=40, curve_type=ct, n_days=7)
    mv = d.groupby("slot")["kw"].mean().values
    return float(np.corrcoef(mv / mv.max(), real_rp_n)[0, 1])

corr_cal = profile_corr(gen)
corr_def = profile_corr(CurveGenerator())
print(f"  Calibre : corr={corr_cal:.4f}")
print(f"  Defaut  : corr={corr_def:.4f}")
check(corr_cal >= corr_def - 0.01,
      f"Calibre >= defaut ({corr_cal:.4f} >= {corr_def:.4f})",
      f"Calibration degrade la forme ({corr_cal:.4f} < {corr_def:.4f})")


section("TOUS LES TESTS PASSES")
