"""Load ecommerce event data from CSV."""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

FUNNEL_EVENTS = ["view", "cart", "purchase"]
FUNNEL_ORDER = {e: i for i, e in enumerate(FUNNEL_EVENTS)}


def load_from_csv(filename: str = "ga4_events.csv") -> pd.DataFrame:
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"{filepath} not found. See data/README.md for download instructions."
        )
    df = pd.read_csv(filepath)
    df = _normalize(df)
    return df


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    if "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
        df["event_date"] = df["event_time"].dt.date

    if "event_type" in df.columns:
        df = df[df["event_type"].isin(FUNNEL_EVENTS)].copy()
        df["funnel_stage"] = df["event_type"].map(FUNNEL_ORDER)

    if "category_code" in df.columns:
        df["category_top"] = df["category_code"].fillna("unknown").str.split(".").str[0]

    return df
