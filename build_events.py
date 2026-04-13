"""
Build canonical episode event table for Brainrot-to-Breadwinner.

Output:
- events_df: one row per episode drop event with ET/UTC timestamps and helpers.
"""

from __future__ import annotations

import pandas as pd

ET_TZ = "America/New_York"

# Calendar year must match your ingest windows (AMZN / Trends). Adjust if your season is a different year.
RAW_EVENTS = [
    {"show": "Invincible", "season": 3, "episode": 7, "release_date": "2026-04-15"},
    {"show": "The Boys", "season": 5, "episode": 3, "release_date": "2026-04-15"},
    {"show": "Invincible", "season": 3, "episode": 8, "release_date": "2026-04-22"},
    {"show": "The Boys", "season": 5, "episode": 4, "release_date": "2026-04-22"},
    {"show": "The Boys", "season": 5, "episode": 5, "release_date": "2026-04-29"},
    {"show": "The Boys", "season": 5, "episode": 6, "release_date": "2026-05-06"},
    {"show": "The Boys", "season": 5, "episode": 7, "release_date": "2026-05-13"},
    {"show": "The Boys", "season": 5, "episode": 8, "release_date": "2026-05-20"},
]


def _build_events_df() -> pd.DataFrame:
    events = pd.DataFrame(RAW_EVENTS)
    events["release_date"] = pd.to_datetime(events["release_date"])

    # All episodes drop at 3:00 AM ET per project brief.
    events["release_ts_et"] = (
        events["release_date"] + pd.Timedelta(hours=3)
    ).dt.tz_localize(ET_TZ)
    events["release_ts_utc"] = events["release_ts_et"].dt.tz_convert("UTC")

    events["event_id"] = (
        events["show"].str.lower().str.replace(" ", "_")
        + "_s"
        + events["season"].astype(str)
        + "_e"
        + events["episode"].astype(str)
    )

    counts_by_day = events.groupby("release_date")["event_id"].transform("count")
    events["is_dual_drop_day"] = counts_by_day > 1
    events["event_label"] = (
        events["show"] + " S" + events["season"].astype(str) + "E" + events["episode"].astype(str)
    )

    return events[
        [
            "event_id",
            "show",
            "season",
            "episode",
            "event_label",
            "release_date",
            "release_ts_et",
            "release_ts_utc",
            "is_dual_drop_day",
        ]
    ].sort_values(["release_date", "show"]).reset_index(drop=True)


events_df = _build_events_df()

print(f"Events rows: {len(events_df)}")
print(events_df)
