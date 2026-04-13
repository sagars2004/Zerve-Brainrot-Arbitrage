"""
Brainrot Score block (0–100 daily composite).

Input:
- master_df (from merge_signals): must include trends_composite, volume_zscore,
  pm_yes_price_mean; pm_odds_velocity_24h optional (recomputed from filled prices).

Weights (PRD):
- Google Trends composite: 50%
- AMZN volume z-score: 30% (clip [0, 3], scale to [0, 100])
- Polymarket odds velocity: 20% (|Δ yes_price| from ffill series, min–max to [0, 100])

Output:
- scored_df: master_df plus component columns and brainrot_score
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WT_TRENDS = 0.50
WT_AMZN = 0.30
WT_POLY = 0.20

# Bridge short gaps (e.g. weekends) so daily rows still get an AMZN component when sensible.
AMZN_Z_FFILL_LIMIT = 5


def _require_master() -> pd.DataFrame:
    if "master_df" not in globals():
        raise RuntimeError("Missing upstream variable: master_df (run merge_signals first).")
    return globals()["master_df"].copy()


def _minmax_0_100(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo = np.nanmin(s.to_numpy(dtype=float))
    hi = np.nanmax(s.to_numpy(dtype=float))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo) * 100.0


df = _require_master()

required = ["trends_composite", "volume_zscore", "pm_yes_price_mean"]
for col in required:
    if col not in df.columns:
        raise RuntimeError(f"master_df missing required column: {col}")

# --- Polymarket: ffill level, velocity from filled series (sparse API days) ---
df["pm_yes_price_ffill"] = df["pm_yes_price_mean"].ffill()
df["pm_price_imputed"] = df["pm_yes_price_mean"].isna() & df["pm_yes_price_ffill"].notna()
df["pm_odds_velocity_used"] = df["pm_yes_price_ffill"].diff()
df["pm_velocity_abs"] = df["pm_odds_velocity_used"].abs()

# --- Trends: already 0–100 scale from Google; clip ---
df["br_trends_0_100"] = pd.to_numeric(df["trends_composite"], errors="coerce").clip(0.0, 100.0)

# --- AMZN: volume z clipped [0, 3] -> [0, 100] ---
vz = pd.to_numeric(df["volume_zscore"], errors="coerce")
vz_ffill = vz.ffill(limit=AMZN_Z_FFILL_LIMIT)
df["volume_zscore_for_score"] = vz_ffill
df["br_amzn_0_100"] = (vz_ffill.clip(lower=0.0, upper=3.0) / 3.0) * 100.0

# --- Polymarket velocity contribution ---
df["br_poly_0_100"] = _minmax_0_100(df["pm_velocity_abs"])

# --- Composite ---
df["brainrot_score"] = (
    WT_TRENDS * df["br_trends_0_100"]
    + WT_AMZN * df["br_amzn_0_100"]
    + WT_POLY * df["br_poly_0_100"]
)

# If no cultural signal, do not emit a headline score
df.loc[df["trends_composite"].isna(), "brainrot_score"] = np.nan

scored_df = df

print("Brainrot Score — tail (key columns):")
cols = [
    "date",
    "brainrot_score",
    "br_trends_0_100",
    "br_amzn_0_100",
    "br_poly_0_100",
    "is_event_day",
    "event_labels",
]
cols = [c for c in cols if c in scored_df.columns]
print(scored_df[cols].tail(15).to_string(index=False))

print("\nbrainrot_score describe:")
print(scored_df["brainrot_score"].describe())
