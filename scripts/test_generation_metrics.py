"""Mesure les metriques de generation (bootstrap + parametrique) sur donnees reelles."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.parser import parse_timeseries, parse_labels
from models.generator import CurveGenerator

ROOT = Path(__file__).resolve().parent.parent
TS = ROOT / "RES2-6-9.csv"
LBL = ROOT / "RES2-6-9-labels.csv"

N_METERS = 200


def main():
    print("Chargement...")
    df = parse_timeseries(str(TS), max_meters=N_METERS)
    labels = parse_labels(str(LBL))
    loaded = set(df["meter_id"].astype(str).unique())
    labels = {k: v for k, v in labels.items() if k in loaded}
    print(f"{df['meter_id'].nunique()} compteurs, {len(df):,} points")
    print(f"labels: RP(0)={sum(1 for v in labels.values() if v==0)}, RS(1)={sum(1 for v in labels.values() if v==1)}")

    gen = CurveGenerator()
    gen.fit(df, labels)

    for ct in ["RS", "RP"]:
        print(f"\n===== {ct} =====")
        print(f"  corpus_daily={len(gen._corpus_daily[ct])}, wd={len(gen._corpus_wd[ct])}, we={len(gen._corpus_we[ct])}")
        print(f"  scale={gen._scales[ct]:.2f}")
        for mode in ["bootstrap", "parametric"]:
            if mode == "bootstrap":
                gdf = gen.generate_bootstrap(50, ct, 7, 0.15)
            else:
                gdf = gen.generate(50, ct, 7, 0.15)
            rep = gen.similarity_report(df, labels, gdf, ct)
            if not rep["has_real"]:
                print(f"  [{mode}] has_real=False")
                continue
            indisc = max(0.0, 1.0 - abs((rep["discriminative_score"] or 0.5) - 0.5) * 2) * 100
            print(f"  [{mode:10}] pearson={rep['pearson_profile']*100:4.0f}%  indisc={indisc:4.0f}%  "
                  f"E_gen={rep['mean_energy_gen']:5.2f}/E_real={rep['mean_energy_real']:5.2f}  "
                  f"pk_gen={rep['peak_gen']:4.2f}/pk_real={rep['peak_real']:4.2f}  "
                  f"we_gen={rep['we_ratio_gen']:.2f}/we_real={rep['we_ratio_real']:.2f}  "
                  f"W={rep['wasserstein_energy']:.2f}")


if __name__ == "__main__":
    main()
