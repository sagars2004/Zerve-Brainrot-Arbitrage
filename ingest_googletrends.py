import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
import requests

# Config (or wire from Input block)
LOOKBACK_DAYS = 90
GEO = "US"  # "" for worldwide
# If you wire an Input block with a .txt upload, set this variable name
# (e.g. serpapi_key_txt). Otherwise leave None and fallback search/env is used.
KEY_INPUT = "serpapikey.txt"
SEARCH_TERMS = [
    "invincible season 3",
    "omni-man",
    "the boys season 5",
    "homelander",
    "vought international",
    "amazon prime video",
]
SERPAPI_URL = "https://serpapi.com/search.json"
_UPLOAD_KEY_PATH = (
    ".zerve_app_uploads/63d32afb-7f0c-463e-8ce3-d1653a6b4703"
    "/af956a41-87e4-4e4b-a07c-b3276b670108/serpapi_key.txt"
)


def _sanitize_col(term: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", term.lower())).strip("_")


def _resolve_serpapi_key(key_input: object) -> str:
    # 1) Wired Input block variable (raw string, bytes, or filepath)
    if key_input is not None:
        if isinstance(key_input, (bytes, bytearray)):
            v = key_input.decode("utf-8").strip()
            if v:
                return v
        p = Path(str(key_input))
        if p.is_file():
            v = p.read_text(encoding="utf-8").strip()
            if v:
                return v
        v = str(key_input).strip()
        if v:
            return v

    # 2) Environment variable
    key = (os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY") or "").strip()
    if key:
        return key

    # 3) Common local/uploaded files
    for candidate in [
        Path(_UPLOAD_KEY_PATH),
        Path.cwd() / _UPLOAD_KEY_PATH,
        Path.cwd() / "serpapi_key.txt",
        Path.cwd() / "serpapi.txt",
        Path.cwd() / "serpapi_api_key.txt",
    ]:
        if candidate.is_file():
            v = candidate.read_text(encoding="utf-8").strip()
            if v:
                return v

    raise RuntimeError(
        "Missing SerpAPI key. Provide via:\n"
        "  1. KEY_INPUT variable (Zerve Input block file upload)\n"
        "  2. SERPAPI_API_KEY or SERPAPI_KEY environment variable\n"
        "  3. serpapi_key.txt in the canvas working directory"
    )


def _to_float(value: object) -> float | None:
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _extract_points(payload: dict[str, Any], term: str) -> list[dict[str, Any]]:
    # SerpAPI Google Trends response shape can vary. We try common containers.
    timeline = payload.get("interest_over_time", {}).get("timeline_data", [])
    if not timeline:
        timeline = payload.get("timeline_data", [])
    if not timeline and "interest_over_time" in payload:
        iot = payload["interest_over_time"]
        if isinstance(iot, list):
            timeline = iot

    rows: list[dict[str, Any]] = []
    for point in timeline:
        ts = point.get("timestamp")
        date_str = point.get("date")
        values = point.get("values") or point.get("value")
        val_num: float | None = None

        if isinstance(values, list) and values:
            first = values[0]
            if isinstance(first, dict):
                # observed forms: {"value": "67"}, {"extracted_value": 67}
                raw_v = first.get("extracted_value", first.get("value"))
            else:
                raw_v = first
            val_num = _to_float(raw_v)
        elif values is not None:
            val_num = _to_float(values)

        if ts is not None:
            d = pd.to_datetime(int(ts), unit="s", utc=True).tz_convert(None).normalize()
        elif date_str:
            d = pd.to_datetime(date_str, errors="coerce")
            if pd.isna(d):
                continue
            d = pd.Timestamp(d).normalize()
        else:
            continue

        rows.append({"date": d, term: val_num})
    return rows


def fetch_term_series(term: str, api_key: str, start: date, end: date, geo: str) -> pd.DataFrame:
    resp = requests.get(
        SERPAPI_URL,
        params={
            "engine": "google_trends",
            "data_type": "TIMESERIES",
            "q": term,
            "geo": geo,
            "api_key": api_key,
            "date": f"{start} {end}",
        },
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("error"):
        raise RuntimeError(f"SerpAPI error for '{term}': {payload['error']}")

    points = _extract_points(payload, term)
    if not points:
        raise RuntimeError(
            f"No timeline points for '{term}'. Response keys: {list(payload.keys())}"
        )

    out = pd.DataFrame(points).sort_values("date").reset_index(drop=True)
    return out


end = datetime.utcnow().date()
start = end - timedelta(days=LOOKBACK_DAYS)
api_key = _resolve_serpapi_key(KEY_INPUT)

parts: list[pd.DataFrame] = []
for i, term in enumerate(SEARCH_TERMS):
    if i > 0:
        time.sleep(0.5)
    part = fetch_term_series(term, api_key=api_key, start=start, end=end, geo=GEO)
    parts.append(part)

trends_daily = parts[0]
for part in parts[1:]:
    trends_daily = trends_daily.merge(part, on="date", how="outer")

rename_map = {t: _sanitize_col(t) for t in SEARCH_TERMS}
trends_daily = trends_daily.rename(columns=rename_map)
trends_daily["date"] = pd.to_datetime(trends_daily["date"], errors="coerce").dt.normalize()
trends_daily = trends_daily.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

signal_cols = [c for c in trends_daily.columns if c != "date"]
trends_daily["trends_composite"] = trends_daily[signal_cols].mean(axis=1)

print(f"Trends rows fetched via SerpAPI: {len(trends_daily)}")
print(f"Date range: {trends_daily['date'].min():%Y-%m-%d} -> {trends_daily['date'].max():%Y-%m-%d}")
print(trends_daily.tail(10).to_string(index=False))