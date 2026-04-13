import pandas as pd
from pytrends.request import TrendReq
from datetime import datetime, timedelta
import time

# Config (or wire from Input block)
LOOKBACK_DAYS = 90
GEO = "US"  # "" for worldwide
SEARCH_TERMS = [
    "invincible season 3",
    "omni-man",
    "the boys season 5",
    "homelander",
    "vought international",
    "amazon prime video",
]

def fetch_batch(pytrends, terms, timeframe, geo):
    pytrends.build_payload(terms, timeframe=timeframe, geo=geo)
    df = pytrends.interest_over_time()
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    return df.reset_index()

# pytrends supports max ~5 terms per request, so split 6 terms into 2 batches
batches = [SEARCH_TERMS[:5], SEARCH_TERMS[5:]]
end = datetime.utcnow().date()
start = end - timedelta(days=LOOKBACK_DAYS)
timeframe = f"{start} {end}"

pytrends = TrendReq(hl="en-US", tz=360)

parts = []
for i, terms in enumerate(batches):
    if not terms:
        continue
    if i > 0:
        time.sleep(2)  # light delay to reduce rate-limit risk
    part = fetch_batch(pytrends, terms, timeframe, GEO)
    parts.append(part)

trends_daily = parts[0]
for part in parts[1:]:
    trends_daily = trends_daily.merge(part, on="date", how="outer")

# Clean column names
trends_daily.columns = [c.lower().replace(" ", "_").replace("-", "_") for c in trends_daily.columns]
trends_daily["date"] = pd.to_datetime(trends_daily["date"]).dt.normalize()
trends_daily = trends_daily.sort_values("date").reset_index(drop=True)

# Optional composite cultural signal (0-100 scale)
signal_cols = [c for c in trends_daily.columns if c != "date"]
trends_daily["trends_composite"] = trends_daily[signal_cols].mean(axis=1)

print(f"Trends rows fetched: {len(trends_daily)}")
print(trends_daily.tail(10))