"""
Alpha Vantage AMZN daily bars + volume z-score.
Reads the API key from the uploaded file or the Zerve input block variable.
Uses the free-tier TIME_SERIES_DAILY endpoint (compact outputsize = 100 trading days).
"""

import os
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# If you wired an Input block with a file upload, set this to the variable
# name it produces (e.g. alphavantagekey_txt). Otherwise leave as None and
# the block will search for the key file in the canvas file system.
KEY_INPUT = None  # e.g. alphavantagekey_txt

SYMBOL = "AMZN"
ROLLING_DAYS = 20
AV_URL = "https://www.alphavantage.co/query"

# Known uploaded file path (from canvas file listing)
_UPLOAD_KEY_PATH = (
    ".zerve_app_uploads/63d32afb-7f0c-463e-8ce3-d1653a6b4703"
    "/af956a41-87e4-4e4b-a07c-b3276b670108/alphavantagekey.txt"
)


def _read_api_key(key_input: object) -> str:
    """Resolve the Alpha Vantage API key from multiple possible sources."""
    if key_input is not None:
        if isinstance(key_input, (bytes, bytearray)):
            return key_input.decode("utf-8").strip()
        p = Path(str(key_input))
        if p.is_file():
            return p.read_text(encoding="utf-8").strip()
        return str(key_input).strip()

    # Environment variable
    env = os.getenv("ALPHA_VANTAGE_API_KEY")
    if env:
        return env.strip()

    # Known uploaded file path (use cwd-relative lookup)
    for candidate in [
        Path(_UPLOAD_KEY_PATH),
        Path.cwd() / _UPLOAD_KEY_PATH,
        Path.cwd() / "alphavantagekey.txt",
    ]:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()

    raise RuntimeError(
        "No API key found. Provide it via:\n"
        "  1. KEY_INPUT variable (Zerve Input block file upload)\n"
        "  2. ALPHA_VANTAGE_API_KEY environment variable\n"
        "  3. alphavantagekey.txt in the canvas working directory"
    )


def fetch_daily(api_key: str, symbol: str) -> pd.DataFrame:
    """Fetch daily OHLCV data from Alpha Vantage (compact = last ~100 trading days, free tier)."""
    r = requests.get(
        AV_URL,
        params={
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "compact",  # free-tier compatible; returns ~100 most recent days
            "apikey": api_key,
        },
        timeout=60,
    )
    r.raise_for_status()
    payload = r.json()

    # Handle rate-limit / info messages
    if "Note" in payload:
        raise RuntimeError(f"Alpha Vantage rate-limit notice: {payload['Note']}")
    if "Information" in payload:
        raise RuntimeError(f"Alpha Vantage info message: {payload['Information']}")

    series = payload.get("Time Series (Daily)")
    if not series:
        raise RuntimeError(
            f"Unexpected Alpha Vantage response keys: {list(payload.keys())}\n"
            f"Full response: {payload}"
        )

    rows = []
    for date_str, ohlcv in series.items():
        rows.append(
            {
                "date": pd.to_datetime(date_str),
                "open": float(ohlcv["1. open"]),
                "high": float(ohlcv["2. high"]),
                "low": float(ohlcv["3. low"]),
                "close": float(ohlcv["4. close"]),
                "volume": int(ohlcv["5. volume"]),
            }
        )

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def add_volume_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add rolling volume z-score columns to the dataframe."""
    out = df.copy()
    roll = out["volume"].rolling(window=window, min_periods=window)
    out["volume_roll_mean"] = roll.mean()
    out["volume_roll_std"] = roll.std()
    out["volume_zscore"] = (out["volume"] - out["volume_roll_mean"]) / out["volume_roll_std"]
    return out


# --- Main execution ---
av_api_key = _read_api_key(KEY_INPUT)
print(f"API key loaded: ****************")

amzn_daily = fetch_daily(av_api_key, SYMBOL)
amzn_daily = add_volume_zscore(amzn_daily, ROLLING_DAYS)

print(f"\n{SYMBOL} rows fetched: {len(amzn_daily)} (showing most recent 10 with z-scores)")
print(amzn_daily[["date", "close", "volume", "volume_zscore"]].tail(10).to_string(index=False))

# Summary stats on z-scores
_zscore_clean = amzn_daily["volume_zscore"].dropna()
print(f"\nVolume Z-Score Stats ({ROLLING_DAYS}-day rolling window, n={len(_zscore_clean)}):")
print(f"  Mean : {_zscore_clean.mean():.4f}")
print(f"  Std  : {_zscore_clean.std():.4f}")
print(f"  Min  : {_zscore_clean.min():.4f}")
print(f"  Max  : {_zscore_clean.max():.4f}")
print(f"\nDays with |z-score| > 2 (unusual volume): {(abs(_zscore_clean) > 2).sum()}")
