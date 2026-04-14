"""
Interactive Streamlit dashboard: Brainrot signals + AMZN stock deep-dive.

Run locally after exporting the scored panel from the notebook/Zerve pipeline:

  streamlit run dashboard.py

Data: upload a Parquet or CSV with columns from merge_signals + brainrot_score
(at least date, close, brainrot_score, trends_composite, is_event_day recommended).

Optional: upload lag_series.csv / lag_results.csv (same schema as lag_correlation outputs)
for the pipeline lag panel.
"""

from __future__ import annotations

import os

# Suppress stock_analysis auto-preview on import (that module shows CLI text/Plotly when not quiet).
os.environ.setdefault("STOCK_ANALYSIS_QUIET", "1")

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from stock_analysis import (
    add_stock_derived_columns,
    build_fig_daily_returns,
    build_fig_lag_brainrot_vs_returns,
    build_fig_lag_trends_vs_returns,
    build_fig_price_and_cumret,
    build_fig_rolling_corr_brainrot_returns,
    build_fig_volatility,
    event_return_table,
)
import stock_analysis as stock_analysis_mod
import visualize as visualize_mod

from visualize import (
    _build_fig_brainrot,
    _build_fig_lag_panel,
    _build_fig_signal_stack,
    _ensure_date,
)

_DEFAULT_SA_VIEW = stock_analysis_mod.VIEW_DATE_X_START
_DEFAULT_VZ_VIEW = visualize_mod.VIEW_DATE_X_START


def _read_any_df(upload: BytesIO | None, path: str, name: str) -> pd.DataFrame | None:
    if upload is not None:
        raw = upload.read()
        low = upload.name.lower()
        if low.endswith(".parquet"):
            return pd.read_parquet(BytesIO(raw))
        return pd.read_csv(BytesIO(raw))
    p = (path or "").strip()
    if not p:
        return None
    fp = Path(p).expanduser()
    if not fp.is_file():
        st.error(f"{name}: file not found: {fp}")
        return None
    if fp.suffix.lower() == ".parquet":
        return pd.read_parquet(fp)
    return pd.read_csv(fp)


st.set_page_config(page_title="Brainrot ↔ AMZN", layout="wide")
st.title("Brainrot ↔ AMZN — interactive dashboard")
st.caption(
    "Load your scored daily panel (CSV/Parquet). Stock tab adds returns, vol, rolling correlations, and event-level return table."
)

with st.sidebar:
    st.subheader("Data")
    panel_upload = st.file_uploader("Panel (scored_df)", type=["csv", "parquet"])
    panel_path = st.text_input(
        "Or path to panel file",
        value="",
        placeholder="/path/to/scored_df.parquet",
    )
    st.divider()
    lag_upload = st.file_uploader("Optional: lag_series (CSV/Parquet)", type=["csv", "parquet"])
    lag_path = st.text_input("Or path to lag_series", value="", placeholder="lag_series.csv")
    lr_upload = st.file_uploader("Optional: lag_results", type=["csv", "parquet"])
    lr_path = st.text_input("Or path to lag_results", value="", placeholder="lag_results.csv")
    st.divider()
    use_zoom = st.checkbox(
        "Use Feb 2026 x-axis zoom (matches visualize.py VIEW_DATE_X_START)",
        value=True,
    )

# Streamlit reruns the script; restore defaults when zoom is on so toggling works.
if use_zoom:
    stock_analysis_mod.VIEW_DATE_X_START = _DEFAULT_SA_VIEW
    visualize_mod.VIEW_DATE_X_START = _DEFAULT_VZ_VIEW
else:
    stock_analysis_mod.VIEW_DATE_X_START = None
    visualize_mod.VIEW_DATE_X_START = None

df_panel = _read_any_df(panel_upload, panel_path, "Panel")
if df_panel is None:
    st.info("Upload a panel file or set a valid path to begin.")
    st.stop()

try:
    plot_df = _ensure_date(df_panel)
except Exception as e:
    st.exception(e)
    st.stop()

plot_df = add_stock_derived_columns(plot_df)

dmin = plot_df["date"].min()
dmax = plot_df["date"].max()
c1, c2 = st.columns(2)
with c1:
    start = st.date_input("Window start", value=pd.Timestamp(dmin).date(), min_value=pd.Timestamp(dmin).date())
with c2:
    end = st.date_input("Window end", value=pd.Timestamp(dmax).date(), max_value=pd.Timestamp(dmax).date())

mask = (plot_df["date"] >= pd.Timestamp(start)) & (plot_df["date"] <= pd.Timestamp(end))
win = plot_df.loc[mask].copy()
if win.empty:
    st.warning("No rows in selected date window.")
    st.stop()

st.caption(f"Showing **{len(win)}** rows · close non-null: **{win['close'].notna().sum() if 'close' in win.columns else 0}**")

lag_series_df = _read_any_df(lag_upload, lag_path, "lag_series")
lag_results_df = _read_any_df(lr_upload, lr_path, "lag_results")

tab_ov, tab_st, tab_lag, tab_sum = st.tabs(["Overview", "Stock", "Lag analysis", "Summary"])

with tab_ov:
    st.plotly_chart(_build_fig_brainrot(win), use_container_width=True)
    st.plotly_chart(_build_fig_signal_stack(win), use_container_width=True)

with tab_st:
    st.subheader("Price & cumulative performance")
    st.plotly_chart(build_fig_price_and_cumret(win), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(build_fig_daily_returns(win), use_container_width=True)
    with c2:
        st.plotly_chart(build_fig_volatility(win), use_container_width=True)
    st.subheader("Brainrot vs stock moves")
    st.plotly_chart(build_fig_rolling_corr_brainrot_returns(win), use_container_width=True)
    st.subheader("Episode-drop days — same-day and forward 5-row log-return sum")
    et = event_return_table(win)
    if et.empty:
        st.write("No `is_event_day` rows in this window.")
    else:
        st.dataframe(et, use_container_width=True)
    st.subheader("Stock-specific lead–lag (z-scored)")
    st.plotly_chart(build_fig_lag_trends_vs_returns(win), use_container_width=True)
    st.plotly_chart(build_fig_lag_brainrot_vs_returns(win), use_container_width=True)

with tab_lag:
    if lag_series_df is None or lag_series_df.empty:
        st.info(
            "Upload **lag_series** (and optional **lag_results**) from your lag_correlation block, "
            "or run the pipeline and export CSVs."
        )
    fig_lag = _build_fig_lag_panel(lag_series_df, lag_results_df)
    st.plotly_chart(fig_lag, use_container_width=True)

with tab_sum:
    st.markdown(
        """
**How to export data from Python**

```python
scored_df.to_parquet("scored_df.parquet", index=False)
lag_series_df.to_csv("lag_series.csv", index=False)
lag_results_df.to_csv("lag_results.csv", index=False)
```
"""
    )
    if "brainrot_score" in win.columns:
        s = pd.to_numeric(win["brainrot_score"], errors="coerce").dropna()
        if len(s):
            st.metric("Brainrot score (window)", f"{s.mean():.1f} mean · {s.min():.1f}–{s.max():.1f}")
    if "log_ret_1d" in win.columns:
        r = pd.to_numeric(win["log_ret_1d"], errors="coerce").dropna()
        if len(r):
            st.metric("AMZN daily log return (window)", f"mean {r.mean()*100:.3f}% · vol {r.std()*100:.3f}% (daily)")
