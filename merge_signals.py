"""
Merge all daily signals into a single master analysis DataFrame.

Expected upstream variables (from prior Zerve blocks):
- amzn_daily (DataFrame): includes date, close, volume, volume_zscore
- trends_daily (DataFrame): includes date, trends term columns, trends_composite
- polymarket_daily (DataFrame): includes date, pm_yes_price_mean, pm_odds_velocity_24h, etc.
- events_df (DataFrame): includes release_date, show, event metadata

Output:
- master_df: merged daily table with event features for lag analysis.
"""

import pandas as pd


def _normalize_date_col(df: pd.DataFrame, source_col: str = "date", target_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[target_col] = pd.to_datetime(out[source_col], errors="coerce").dt.tz_localize(None)
    out[target_col] = out[target_col].dt.normalize()
    return out


def _attach_event_flags(master: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    out = master.copy()

    event_day_summary = (
        events.groupby("release_date", as_index=False)
        .agg(
            event_count=("event_id", "count"),
            event_shows=("show", lambda s: ", ".join(sorted(set(s)))),
            event_labels=("event_label", lambda s: " | ".join(s)),
            is_dual_drop_day=("is_dual_drop_day", "max"),
        )
        .rename(columns={"release_date": "date"})
    )

    out = out.merge(event_day_summary, on="date", how="left")
    out["is_event_day"] = out["event_count"].fillna(0).astype(int) > 0
    out["event_count"] = out["event_count"].fillna(0).astype(int)
    out["is_dual_drop_day"] = out["is_dual_drop_day"].fillna(False).astype(bool)
    out["event_shows"] = out["event_shows"].fillna("")
    out["event_labels"] = out["event_labels"].fillna("")

    return out


def _nearest_event_distance_days(master: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    out = master.copy()
    # Cast to pd.Timestamp to ensure consistent arithmetic with master date column
    event_dates = sorted(pd.to_datetime(events["release_date"].dropna().unique()))
    if not event_dates:
        out["days_from_nearest_event"] = pd.NA
        out["is_event_window_t5"] = False
        return out

    def nearest_days(d: pd.Timestamp) -> int:
        return int(min(abs((d - ev).days) for ev in event_dates))

    out["days_from_nearest_event"] = out["date"].apply(nearest_days)
    out["is_event_window_t5"] = out["days_from_nearest_event"] <= 5
    return out


# --- Directly reference upstream variables (no globals()) ---
amzn = _normalize_date_col(amzn_daily, source_col="date", target_col="date")
trends = _normalize_date_col(trends_daily, source_col="date", target_col="date")
poly = _normalize_date_col(polymarket_daily, source_col="date", target_col="date")
events = _normalize_date_col(events_df, source_col="release_date", target_col="release_date")

master_df = (
    amzn.merge(trends, on="date", how="outer", suffixes=("", "_trends"))
    .merge(poly, on="date", how="outer", suffixes=("", "_poly"))
    .sort_values("date")
    .reset_index(drop=True)
)

master_df = _attach_event_flags(master_df, events)
master_df = _nearest_event_distance_days(master_df, events)

print(f"master_df shape: {master_df.shape}")
print("Columns:")
print(list(master_df.columns))
print("\nTail sample:")
print(master_df.tail(12).to_string())

null_rate = master_df.isna().mean().sort_values(ascending=False).head(10)
print("\nTop 10 null-rate columns:")
print(null_rate)
