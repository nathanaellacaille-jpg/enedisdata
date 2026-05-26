import io
import pandas as pd
from config import MAX_METERS_UPLOAD


_REQUIRED_TS = {"id", "horodate", "valeur"}
_REQUIRED_LBL = {"id", "label"}


def parse_timeseries(file, max_meters: int | None = MAX_METERS_UPLOAD) -> pd.DataFrame:
    """Lit un CSV Enedis par chunks, retourne df [meter_id, ts, kw] (max_meters=None pour tous les compteurs)."""
    # Detecte le separateur sur un echantillon sans consommer le flux
    if hasattr(file, "read"):
        sample_bytes = file.read(4096)
        file.seek(0)
    else:
        with open(file, "rb") as fh:
            sample_bytes = fh.read(4096)
    sample = sample_bytes.decode("utf-8", errors="replace")
    sep = _detect_sep(sample)

    # Lecture par chunks
    chunks = []
    seen_ids: set = set()
    validated = False

    reader = pd.read_csv(file, sep=sep, encoding="utf-8", low_memory=False, chunksize=50_000)
    for chunk in reader:
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        if not validated:
            missing = _REQUIRED_TS - set(chunk.columns)
            if missing:
                raise ValueError(f"Colonnes manquantes : {missing}. Colonnes trouvees : {list(chunk.columns)}")
            validated = True
        chunk = chunk.rename(columns={"id": "meter_id", "horodate": "ts", "valeur": "kw"})
        chunk["meter_id"] = chunk["meter_id"].astype(str)

        # Ajoute les nouveaux ids (cap eventuel via max_meters)
        new_ids = set(chunk["meter_id"].unique()) - seen_ids
        if max_meters is None:
            seen_ids |= new_ids
            filtered = chunk
        else:
            remaining = max_meters - len(seen_ids)
            if new_ids and remaining > 0:
                seen_ids |= set(list(new_ids)[:remaining])
            filtered = chunk[chunk["meter_id"].isin(seen_ids)]

        if not filtered.empty:
            chunks.append(filtered)

        if max_meters is not None and len(seen_ids) >= max_meters:
            break

    if not chunks:
        return pd.DataFrame(columns=["meter_id", "ts", "kw"])

    df = pd.concat(chunks, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    # Valeurs source en Wh/demi-heure → conversion en kW : Wh / 0.5 h / 1000 = Wh / 500
    df["kw"] = (pd.to_numeric(df["kw"], errors="coerce").fillna(0.0) / 500.0).astype("float32")
    # meter_id en categorical : ~1 B/ligne au lieu de ~60 B (str object). Gain RAM ~60x.
    df["meter_id"] = df["meter_id"].astype("category")
    df = df[["meter_id", "ts", "kw"]].sort_values(["meter_id", "ts"]).reset_index(drop=True)
    return df


def parse_labels(file) -> dict:
    """Lit un CSV id,label, retourne {meter_id: int}."""
    raw = file.read() if hasattr(file, "read") else open(file, "rb").read()
    sample = raw[:2048].decode("utf-8", errors="replace")
    sep = _detect_sep(sample)
    df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]
    missing = _REQUIRED_LBL - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}. Colonnes trouvees : {list(df.columns)}")
    df["id"] = df["id"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    return dict(zip(df["id"], df["label"]))


def load_timeseries_to_session(upload_key: str = "ts_file") -> "pd.DataFrame | None":
    """Charge le CSV uploade dans session_state et le retourne."""
    import streamlit as st
    return st.session_state.get("_ts_df")


def _detect_sep(sample: str) -> str:
    """Detecte le separateur CSV parmi ; , tabulation."""
    counts = {";": sample.count(";"), ",": sample.count(","), "\t": sample.count("\t")}
    return max(counts, key=counts.get)
