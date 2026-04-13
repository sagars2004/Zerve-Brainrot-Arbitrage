"""
Polymarket ingestion block for Zerve.

Outputs:
- polymarket_markets: market-level snapshot for relevant keywords
- polymarket_daily: daily aggregated odds + volume + velocity signal
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

LOOKBACK_DAYS = 90
KEYWORDS = [
    "invincible",
    "omni-man",
    "the boys",
    "homelander",
    "vought",
    "amazon",
    "prime video",
]
PROXY_KEYWORDS = [
    "amazon",
    "prime",
    "prime video",
    "streaming",
    "tv",
    "television",
    "earnings",
]

# Gamma API returns rich market objects (question, outcomePrices as JSON strings, volumeNum, etc.).
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
PAGE_SIZE = 500
MAX_MARKETS = 5000
TIMEOUT_S = 45


def _first_volume(*candidates: object) -> float | None:
    for v in candidates:
        x = _to_float(v)
        if x is not None:
            return x
    return None


def _to_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, (bytes, bytearray)):
            return float(value.decode("utf-8"))
        if isinstance(value, str):
            return float(value)
        return None
    except (TypeError, ValueError):
        return None


def _pick_title(row: dict) -> str:
    return str(
        row.get("question")
        or row.get("title")
        or row.get("name")
        or row.get("slug")
        or ""
    )


def _parse_json_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _extract_yes_price(row: dict) -> float | None:
    direct = _to_float(
        row.get("yesPrice")
        or row.get("lastTradePrice")
        or row.get("lastPrice")
        or row.get("outcomePrice")
        or row.get("probability")
    )
    if direct is not None:
        return direct

    bid = _to_float(row.get("bestBid"))
    ask = _to_float(row.get("bestAsk"))
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0

    outcomes = _parse_json_list(row.get("outcomes"))
    outcome_prices = _parse_json_list(row.get("outcomePrices"))
    if outcomes and outcome_prices and len(outcomes) == len(outcome_prices):
        for idx, outcome_name in enumerate(outcomes):
            if str(outcome_name).strip().lower() == "yes":
                return _to_float(outcome_prices[idx])
        # If "Yes" is missing, first price is a useful fallback.
        return _to_float(outcome_prices[0])

    tokens = _parse_json_list(row.get("tokens"))
    for token in tokens:
        if not isinstance(token, dict):
            continue
        outcome_name = str(token.get("outcome") or "").strip().lower()
        if outcome_name == "yes":
            price = _to_float(token.get("price") or token.get("lastPrice"))
            if price is not None:
                return price
    return None


def _parse_timestamp(row: dict, prefer_updated: bool = True) -> pd.Timestamp:
    if prefer_updated:
        raw = (
            row.get("updatedAt")
            or row.get("updated_at")
            or row.get("createdAt")
            or row.get("created_at")
        )
    else:
        raw = (
            row.get("createdAt")
            or row.get("created_at")
            or row.get("updatedAt")
            or row.get("updated_at")
        )
    if raw:
        ts = pd.to_datetime(raw, errors="coerce", utc=True)
        if pd.notna(ts):
            return ts
    return pd.Timestamp.now(tz="UTC")


def _fetch_raw_markets() -> list[dict]:
    """Paginate Gamma /markets until empty or MAX_MARKETS."""
    all_rows: list[dict] = []
    offset = 0
    while len(all_rows) < MAX_MARKETS:
        params = {"limit": PAGE_SIZE, "offset": offset}
        r = requests.get(GAMMA_MARKETS_URL, params=params, timeout=TIMEOUT_S)
        r.raise_for_status()
        payload = r.json()
        batch: list[dict] = []
        if isinstance(payload, list):
            batch = payload
        elif isinstance(payload, dict):
            for key in ("data", "markets", "items"):
                if isinstance(payload.get(key), list):
                    batch = payload[key]
                    break
        if not batch:
            break
        all_rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return all_rows


def _flatten_markets(payload_rows: list[dict]) -> list[dict]:
    flattened: list[dict] = []
    for row in payload_rows:
        if not isinstance(row, dict):
            continue
        nested_markets = row.get("markets")
        if isinstance(nested_markets, list) and nested_markets:
            for market in nested_markets:
                if isinstance(market, dict):
                    merged = {**row, **market}
                    flattened.append(merged)
            continue
        flattened.append(row)
    return flattened


def _build_markets_df(markets: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    primary_hits = 0
    proxy_hits = 0
    for m in markets:
        title = _pick_title(m)
        title_lc = title.lower()
        is_primary = any(k in title_lc for k in KEYWORDS)
        is_proxy = any(k in title_lc for k in PROXY_KEYWORDS)
        if not is_primary and not is_proxy:
            continue
        if is_primary:
            primary_hits += 1
        elif is_proxy:
            proxy_hits += 1

        created_ts = _parse_timestamp(m, prefer_updated=True)
        volume = _first_volume(m.get("volumeNum"), m.get("volumeClob"), m.get("volume"))
        yes_price = _extract_yes_price(m)
        market_id = (
            m.get("id")
            or m.get("conditionId")
            or m.get("marketId")
            or m.get("slug")
            or f"unknown-{created_ts.value}"
        )

        rows.append(
            {
                "market_id": str(market_id),
                "question": title,
                "slug": m.get("slug"),
                "created_ts_utc": created_ts,
                "date": created_ts.tz_convert("UTC").normalize().tz_localize(None),
                "yes_price": yes_price,
                "volume": volume,
                "match_tier": "primary" if is_primary else "proxy",
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["market_id", "question", "slug", "created_ts_utc", "date", "yes_price", "volume"]
        )

    df = pd.DataFrame(rows).sort_values("created_ts_utc").reset_index(drop=True)
    df["yes_price"] = pd.to_numeric(df["yes_price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    # Keep only rows with at least one usable signal.
    df = df[~(df["yes_price"].isna() & df["volume"].isna())].reset_index(drop=True)
    print(f"Primary keyword matches: {primary_hits} | Proxy keyword matches: {proxy_hits}")
    return df


def _build_daily_signal(markets_df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    if markets_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "pm_yes_price_mean",
                "pm_volume_sum",
                "pm_odds_velocity_24h",
                "pm_market_count",
                "pm_primary_market_count",
            ]
        )

    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date()
    scoped = markets_df[markets_df["date"] >= pd.Timestamp(cutoff)].copy()

    daily = (
        scoped.groupby("date", as_index=False)
        .agg(
            pm_yes_price_mean=("yes_price", "mean"),
            pm_volume_sum=("volume", "sum"),
            pm_market_count=("market_id", "nunique"),
            pm_primary_market_count=("match_tier", lambda s: int((s == "primary").sum())),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["pm_odds_velocity_24h"] = daily["pm_yes_price_mean"].diff()
    return daily


raw_markets = _fetch_raw_markets()
flat_markets = _flatten_markets(raw_markets)
polymarket_markets = _build_markets_df(flat_markets)
polymarket_daily = _build_daily_signal(polymarket_markets, LOOKBACK_DAYS)

print(f"Raw markets fetched: {len(raw_markets)}")
print(f"Flattened candidate markets: {len(flat_markets)}")
print(f"Relevant markets (keyword filtered): {len(polymarket_markets)}")
print("polymarket_markets sample:")
print(polymarket_markets.head(10))
print("\npolymarket_daily sample:")
print(polymarket_daily.tail(10))
