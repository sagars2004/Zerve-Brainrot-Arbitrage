"""
Lag analysis block.
Input: master_df
Output: lag_results_df + lag_series_df
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_LAG = 7


def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    if std is None or std == 0 or np.isnan(std):
        return s * np.nan
    return (s - s.mean()) / std


def _lag_corr(x: pd.Series, y: pd.Series, lag: int) -> float:
    # Positive lag means x leads y by `lag` days.
    if lag > 0:
        a = x.shift(lag)
        b = y
    elif lag < 0:
        a = x
        b = y.shift(-lag)
    else:
        a = x
        b = y

    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 8:
        return np.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def analyze_pair(df: pd.DataFrame, x_col: str, y_col: str, pair_name: str, max_lag: int = 7):
    x = _zscore(df[x_col])
    y = _zscore(df[y_col])

    lag_rows = []
    for lag in range(-max_lag, max_lag + 1):
        corr = _lag_corr(x, y, lag)
        lag_rows.append(
            {
                "pair": pair_name,
                "x": x_col,
                "y": y_col,
                "lag_days": lag,
                "corr": corr,
                "abs_corr": np.nan if pd.isna(corr) else abs(corr),
            }
        )

    lag_df = pd.DataFrame(lag_rows)
    best = lag_df.dropna(subset=["abs_corr"]).sort_values("abs_corr", ascending=False).head(1)
    if best.empty:
        summary = {
            "pair": pair_name,
            "x": x_col,
            "y": y_col,
            "best_lag_days": np.nan,
            "best_corr": np.nan,
            "direction_note": "insufficient overlap",
        }
    else:
        r = best.iloc[0]
        best_lag = int(r["lag_days"])
        direction = (
            f"{x_col} leads {y_col} by {best_lag}d" if best_lag > 0
            else (f"{y_col} leads {x_col} by {-best_lag}d" if best_lag < 0 else "same-day co-move")
        )
        summary = {
            "pair": pair_name,
            "x": x_col,
            "y": y_col,
            "best_lag_days": best_lag,
            "best_corr": float(r["corr"]),
            "direction_note": direction,
        }

    return lag_df, summary


def _get_master_df() -> pd.DataFrame:
    """Zerve injects upstream block outputs into exec globals; static analyzers don't see that."""
    g = globals()
    if "master_df" not in g:
        raise RuntimeError("Missing upstream variable: master_df (run merge_signals before this block).")
    out = g["master_df"]
    if not isinstance(out, pd.DataFrame):
        raise RuntimeError(f"master_df must be a pandas DataFrame, got {type(out)!r}")
    return out.copy()


df = _get_master_df()

# Validate that required columns are present
for col in ["trends_composite", "pm_odds_velocity_24h", "volume_zscore"]:
    if col not in df.columns:
        raise RuntimeError(f"Expected column missing in master_df: {col}")

pairs = [
    ("H1 Trends -> Polymarket", "trends_composite", "pm_odds_velocity_24h"),
    ("H2 Polymarket -> AMZN", "pm_odds_velocity_24h", "volume_zscore"),
    ("H3 Trends -> AMZN", "trends_composite", "volume_zscore"),
]

all_lag_rows = []
summaries = []
for pair_name, x_col, y_col in pairs:
    lag_df, summary = analyze_pair(df, x_col, y_col, pair_name, max_lag=MAX_LAG)
    all_lag_rows.append(lag_df)
    summaries.append(summary)

lag_series_df = pd.concat(all_lag_rows, ignore_index=True)
lag_results_df = pd.DataFrame(summaries)

print("Lag summary (best lag per hypothesis):")
print(lag_results_df.to_string(index=False))

print("\nTop lag rows by |corr|:")
print(
    lag_series_df.sort_values("abs_corr", ascending=False)
    .head(12)
    .to_string(index=False)
)
