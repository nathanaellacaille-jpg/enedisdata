import io
import pandas as pd


_REQUIRED_TS = {"id", "horodate", "valeur"}
_REQUIRED_LBL = {"id", "label"}


def parse_timeseries(file) -> pd.DataFrame:
    """Lit un CSV Enedis, retourne df [meter_id, ts, kw]."""
    raw = file.read() if hasattr(file, "read") else open(file, "rb").read()
    sample = raw[:4096].decode("utf-8", errors="replace")
    sep = _detect_sep(sample)
    df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding="utf-8", low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = _REQUIRED_TS - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}. Colonnes trouvees : {list(df.columns)}")
    df = df.rename(columns={"id": "meter_id", "horodate": "ts", "valeur": "kw"})
    df["meter_id"] = df["meter_id"].astype(str)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    df["kw"] = pd.to_numeric(df["kw"], errors="coerce").fillna(0.0).astype("float64")
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


def _detect_sep(sample: str) -> str:
    """Detecte le separateur CSV parmi ; , tabulation."""
    counts = {";": sample.count(";"), ",": sample.count(","), "\t": sample.count("\t")}
    return max(counts, key=counts.get)
