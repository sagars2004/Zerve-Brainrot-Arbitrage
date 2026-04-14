"""
Stock-focused derivatives and charts for AMZN daily panel data.

Expects merged/scored frames with at least: date, close. Optional: brainrot_score,
trends_composite, is_event_day, event_labels.

Used by dashboard.py. Importing this module does **not** print (so Streamlit stays quiet).

**Preview** on load (stdout/stderr only, no image rendering):

- **Streamlit dashboard** sets ``STOCK_ANALYSIS_QUIET=1`` so import stays quiet.
- **Zerve** pasting this file as a module: set **``STOCK_ANALYSIS_FORCE_PREVIEW=1``** on that block so the preview always runs, **or** use **``STOCK_ANALYSIS_CLI=1``** (same as forcing CLI).
- **No ``__file__``** in globals (some ``exec``): preview runs automatically.
- **``python stock_analysis.py``**: preview runs.

Call ``main_cli()`` manually if needed.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _plotly_fig(fig: go.Figure) -> Any:
    return cast(Any, fig)

# Align with visualize / extended_visualizations default zoom.
VIEW_DATE_X_START: str | None = "2026-02-01"

LAG_X_RANGE: tuple[float, float] = (-7.5, 7.5)
ROLLING_CORR_WINDOW = 30
ROLLING_CORR_MIN_PERIODS = 15


def _date_x_range(plot_df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if VIEW_DATE_X_START is None or "date" not in plot_df.columns:
        return None
    dmin = pd.Timestamp(VIEW_DATE_X_START).normalize()
    dmax = pd.to_datetime(plot_df["date"], errors="coerce").max()
    if pd.isna(dmax):
        return None
    dmax = pd.Timestamp(dmax).normalize()
    if dmax <= dmin:
        return None
    return dmin, dmax


def _apply_date_x_range(fig: go.Figure, plot_df: pd.DataFrame, *, subplot_rows: int) -> None:
    xr = _date_x_range(plot_df)
    if not xr:
        return
    for r in range(1, subplot_rows + 1):
        fig.update_xaxes(range=[xr[0], xr[1]], row=r, col=1)


def event_dates(plot_df: pd.DataFrame) -> list[pd.Timestamp]:
    if "is_event_day" not in plot_df.columns or not plot_df["is_event_day"].any():
        return []
    ev = plot_df.loc[plot_df["is_event_day"], "date"].dropna().unique()
    return sorted(pd.to_datetime(ev).tolist())


def _forward_sum_log_ret(log_ret: pd.Series, k: int) -> pd.Series:
    """Sum of the next k rows' log returns (calendar rows; aligns with merged daily panel)."""
    arr = log_ret.to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        chunk = arr[i + 1 : min(i + 1 + k, n)]
        out[i] = float(np.nansum(chunk)) if chunk.size else np.nan
    return pd.Series(out, index=log_ret.index)


def add_stock_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "close" not in out.columns:
        return out
    c = pd.to_numeric(out["close"], errors="coerce")
    out["log_ret_1d"] = np.log(c / c.shift(1))
    out["simple_ret_1d"] = c.pct_change()
    out["fwd_log_ret_5d"] = _forward_sum_log_ret(out["log_ret_1d"], 5)
    out["ret_vol_20d_ann_pct"] = (
        out["log_ret_1d"].rolling(20, min_periods=10).std() * np.sqrt(252) * 100.0
    )
    if "brainrot_score" in out.columns:
        out["roll_corr_brainrot_ret"] = (
            pd.to_numeric(out["brainrot_score"], errors="coerce")
            .rolling(ROLLING_CORR_WINDOW, min_periods=ROLLING_CORR_MIN_PERIODS)
            .corr(pd.to_numeric(out["log_ret_1d"], errors="coerce"))
        )
    return out


def cum_return_index(close: pd.Series) -> pd.Series:
    """Total return index, 100 at the first valid close in the series."""
    c = pd.to_numeric(close, errors="coerce")
    first = c.dropna().iloc[0] if c.notna().any() else np.nan
    if not np.isfinite(first) or first == 0:
        return pd.Series(np.nan, index=close.index)
    return (c / first) * 100.0


def build_fig_price_and_cumret(plot_df: pd.DataFrame) -> go.Figure:
    """AMZN close vs cumulative return index (two stacked panels)."""
    df = add_stock_derived_columns(plot_df)
    if "close" not in df.columns or df["close"].notna().sum() < 2:
        fig = go.Figure()
        fig.update_layout(
            title="AMZN price — need close column with data",
            template="plotly_white",
        )
        return fig

    cum = cum_return_index(df["close"])
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("AMZN close ($)", "Cumulative return index (100 = range start)"),
        row_heights=[0.55, 0.45],
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["close"],
            name="Close",
            line=dict(color="#059669", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=cum,
            name="Cum. ret index",
            line=dict(color="#0f172a", width=1.8),
        ),
        row=2,
        col=1,
    )
    for d in event_dates(df):
        for r in (1, 2):
            _plotly_fig(fig).add_vline(
                x=d,
                line_width=1,
                line_dash="dash",
                line_color="rgba(249,115,22,0.55)",
                row=r,
                col=1,
            )

    fig.update_layout(
        title="Stock: price path and cumulative return (same window as other charts)",
        hovermode="x unified",
        height=520,
        template="plotly_white",
        showlegend=False,
    )
    fig.update_yaxes(title_text="$", row=1, col=1)
    fig.update_yaxes(title_text="Index", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    _apply_date_x_range(fig, df, subplot_rows=2)
    return fig


def build_fig_daily_returns(plot_df: pd.DataFrame) -> go.Figure:
    df = add_stock_derived_columns(plot_df)
    if "log_ret_1d" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="Daily returns — missing close", template="plotly_white")
        return fig

    fig = go.Figure(
        go.Scatter(
            x=df["date"],
            y=df["log_ret_1d"] * 100.0,
            name="Log return",
            line=dict(color="#334155", width=1.2),
            hovertemplate="%{x|%Y-%m-%d}<br>log ret = %{y:.3f} bp<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(100,116,139,0.6)")
    for d in event_dates(df):
        fig.add_vline(x=d, line_width=1, line_dash="dash", line_color="rgba(249,115,22,0.6)")
    fig.update_layout(
        title="AMZN daily log returns (% ; ×100 for readability)",
        yaxis_title="Log return × 100",
        xaxis_title="Date",
        hovermode="x unified",
        height=380,
        template="plotly_white",
    )
    xr = _date_x_range(df)
    if xr:
        fig.update_xaxes(range=[xr[0], xr[1]])
    return fig


def build_fig_rolling_corr_brainrot_returns(plot_df: pd.DataFrame) -> go.Figure:
    df = add_stock_derived_columns(plot_df)
    if "roll_corr_brainrot_ret" not in df.columns or df["roll_corr_brainrot_ret"].isna().all():
        fig = go.Figure()
        fig.update_layout(
            title="Rolling corr(Brainrot, log return) — need brainrot_score + close",
            template="plotly_white",
        )
        return fig

    fig = go.Figure(
        go.Scatter(
            x=df["date"],
            y=df["roll_corr_brainrot_ret"],
            name=f"{ROLLING_CORR_WINDOW}d rolling r",
            line=dict(color="#7c3aed", width=2),
            connectgaps=False,
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
    for d in event_dates(df):
        fig.add_vline(x=d, line_width=1, line_dash="dash", line_color="rgba(249,115,22,0.5)")
    fig.update_layout(
        title=f"Rolling Pearson r: Brainrot score vs AMZN daily log return ({ROLLING_CORR_WINDOW}d window)",
        yaxis_title="Correlation",
        xaxis_title="Date",
        height=380,
        template="plotly_white",
    )
    xr = _date_x_range(df)
    if xr:
        fig.update_xaxes(range=[xr[0], xr[1]])
    return fig


def build_fig_volatility(plot_df: pd.DataFrame) -> go.Figure:
    df = add_stock_derived_columns(plot_df)
    if "ret_vol_20d_ann_pct" not in df.columns or df["ret_vol_20d_ann_pct"].isna().all():
        fig = go.Figure()
        fig.update_layout(title="Realized vol — need close history", template="plotly_white")
        return fig
    fig = go.Figure(
        go.Scatter(
            x=df["date"],
            y=df["ret_vol_20d_ann_pct"],
            name="20d ann. vol",
            line=dict(color="#b45309", width=1.8),
            fill="tozeroy",
            fillcolor="rgba(180,83,9,0.12)",
        )
    )
    for d in event_dates(df):
        fig.add_vline(x=d, line_width=1, line_dash="dash", line_color="rgba(249,115,22,0.45)")
    fig.update_layout(
        title="AMZN realized volatility (20d rolling stdev of log returns, annualized %)",
        yaxis_title="Vol %",
        xaxis_title="Date",
        height=360,
        template="plotly_white",
    )
    xr = _date_x_range(df)
    if xr:
        fig.update_xaxes(range=[xr[0], xr[1]])
    return fig


def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    if std is None or std == 0 or np.isnan(std):
        return s * np.nan
    return (s - s.mean()) / std


def _lag_corr(x: pd.Series, y: pd.Series, lag: int) -> float:
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
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def lag_profile(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    pair_name: str,
    max_lag: int = 7,
) -> pd.DataFrame:
    """Rows per lag; same convention as lag_correlation.py (positive lag ⇒ x leads y)."""
    x = _zscore(df[x_col])
    y = _zscore(df[y_col])
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        c = _lag_corr(x, y, lag)
        rows.append(
            {
                "pair": pair_name,
                "x": x_col,
                "y": y_col,
                "lag_days": lag,
                "corr": c,
                "abs_corr": np.nan if pd.isna(c) else abs(c),
            }
        )
    return pd.DataFrame(rows)


def build_fig_lag_bars(lag_df: pd.DataFrame, *, title: str, subtitle: str = "") -> go.Figure:
    if lag_df.empty or lag_df["corr"].isna().all():
        fig = go.Figure()
        fig.update_layout(title=title + " — insufficient overlap", template="plotly_white")
        return fig
    sub = lag_df.sort_values("lag_days")
    colors = ["#ef4444" if (v is not None and v < 0) else "#22c55e" for v in sub["corr"].fillna(0)]
    fig = go.Figure(
        go.Bar(
            x=sub["lag_days"],
            y=sub["corr"],
            marker_color=colors,
            hovertemplate="Lag %{x} d<br>r = %{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_width=1, line_color="#64748b")
    ymax = float(sub["corr"].max())
    ymin = float(sub["corr"].min())
    pad = 0.08 * (ymax - ymin if ymax > ymin else 1.0)
    fig.update_xaxes(range=list(LAG_X_RANGE))
    fig.update_yaxes(range=[ymin - pad, ymax + pad])
    fig.update_layout(
        title=title + (f"<br><sup>{subtitle}</sup>" if subtitle else ""),
        xaxis_title="Lag (d): + ⇒ x leads y",
        yaxis_title="Pearson r (z-scored series)",
        height=420,
        template="plotly_white",
    )
    return fig


def build_fig_lag_trends_vs_returns(plot_df: pd.DataFrame, max_lag: int = 7) -> go.Figure:
    df = add_stock_derived_columns(plot_df)
    if "trends_composite" not in df.columns or "log_ret_1d" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="Lag Trends → returns — missing columns", template="plotly_white")
        return fig
    lag_df = lag_profile(
        df,
        "trends_composite",
        "log_ret_1d",
        pair_name="H3b Trends -> AMZN log return",
        max_lag=max_lag,
    )
    best = lag_df.dropna(subset=["abs_corr"]).sort_values("abs_corr", ascending=False).head(1)
    sub = f"Peak |r|: lag {int(best.iloc[0]['lag_days'])}, r={float(best.iloc[0]['corr']):.3f}" if not best.empty else ""
    return build_fig_lag_bars(
        lag_df,
        title="Lag correlation: Google Trends composite vs AMZN daily log returns",
        subtitle=sub + " · exploratory; not adjusted for multiple lags",
    )


def build_fig_lag_brainrot_vs_returns(plot_df: pd.DataFrame, max_lag: int = 7) -> go.Figure:
    df = add_stock_derived_columns(plot_df)
    if "brainrot_score" not in df.columns or "log_ret_1d" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="Lag Brainrot → returns — missing columns", template="plotly_white")
        return fig
    lag_df = lag_profile(
        df,
        "brainrot_score",
        "log_ret_1d",
        pair_name="Brainrot -> AMZN log return",
        max_lag=max_lag,
    )
    best = lag_df.dropna(subset=["abs_corr"]).sort_values("abs_corr", ascending=False).head(1)
    sub = f"Peak |r|: lag {int(best.iloc[0]['lag_days'])}, r={float(best.iloc[0]['corr']):.3f}" if not best.empty else ""
    return build_fig_lag_bars(
        lag_df,
        title="Lag correlation: Brainrot score vs AMZN daily log returns",
        subtitle=sub,
    )


def event_return_table(plot_df: pd.DataFrame) -> pd.DataFrame:
    """One row per episode-drop day: same-day and forward 5-row log-return sum."""
    df = add_stock_derived_columns(plot_df)
    if "is_event_day" not in df.columns:
        return pd.DataFrame()
    ev = df[df["is_event_day"]].copy()
    if ev.empty:
        return pd.DataFrame()
    cols = ["date", "log_ret_1d", "fwd_log_ret_5d", "close"]
    if "event_labels" in ev.columns:
        cols.insert(1, "event_labels")
    cols = [c for c in cols if c in ev.columns]
    out = ev[cols].copy()
    if "log_ret_1d" in out.columns:
        out["log_ret_1d_pct"] = out["log_ret_1d"] * 100.0
    if "fwd_log_ret_5d" in out.columns:
        out["fwd_log_ret_5d_pct"] = out["fwd_log_ret_5d"] * 100.0
    return out.sort_values("date").reset_index(drop=True)


def _is_panel_df(df: pd.DataFrame) -> bool:
    return "date" in df.columns and "close" in df.columns


def _load_panel_cli() -> tuple[pd.DataFrame | None, str]:
    import os
    from pathlib import Path

    g = globals()
    for name in ("scored_df", "master_df", "panel_df", "merged_df", "output_df"):
        v = g.get(name)
        if isinstance(v, pd.DataFrame) and _is_panel_df(v):
            return v.copy(), f"global `{name}`"
    for name, v in g.items():
        if name.startswith("__"):
            continue
        if isinstance(v, pd.DataFrame) and _is_panel_df(v):
            return v.copy(), f"global `{name}`"

    def _try_read(fp: Path) -> pd.DataFrame | None:
        if not fp.is_file():
            return None
        try:
            if fp.suffix.lower() == ".parquet":
                return pd.read_parquet(fp)
            return pd.read_csv(fp)
        except Exception:
            return None

    env = os.getenv("SCORED_DF_PATH", "").strip()
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    for p in (
        "scored_df.parquet",
        "scored_df.csv",
        "outputs/scored_df.parquet",
        "outputs/scored_df.csv",
        "data/scored_df.parquet",
        "data/scored_df.csv",
        "master_df.parquet",
        "master_df.csv",
    ):
        candidates.append(Path(p))
    for fp in candidates:
        loaded = _try_read(fp)
        if loaded is not None:
            return loaded, f"file `{fp}`"

    # Zerve uploads are often nested under .zerve_app_uploads/<uuid>/<uuid>/...
    zerve_root = Path(".zerve_app_uploads")
    if zerve_root.exists():
        patterns = (
            "*scored*df*.parquet",
            "*scored*df*.csv",
            "*master*df*.parquet",
            "*master*df*.csv",
            "*panel*.parquet",
            "*panel*.csv",
        )
        for patt in patterns:
            for fp in sorted(zerve_root.rglob(patt)):
                loaded = _try_read(fp)
                if loaded is not None:
                    return loaded, f"Zerve upload `{fp}`"
    return None, "synthetic fallback"


def _synthetic_panel(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(rng.normal(0, 0.4, n))
    trends = np.clip(40 + np.cumsum(rng.normal(0, 2, n)), 0, 100)
    brain = 0.5 * trends + 0.3 * rng.uniform(20, 80, n) + 0.2 * rng.uniform(0, 100, n)
    ev = np.zeros(n, dtype=bool)
    ev[20] = ev[40] = True
    return pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "trends_composite": trends,
            "brainrot_score": brain,
            "is_event_day": ev,
            "event_labels": np.where(ev, "Synthetic ep", ""),
        }
    )


def _cli_env_enabled() -> bool:
    import os

    v = os.environ.get("STOCK_ANALYSIS_CLI", "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _invoked_as_script() -> bool:
    """True when `python path/to/stock_analysis.py` (some hosts never set __name__ == '__main__')."""
    import sys
    from pathlib import Path

    fn = globals().get("__file__")
    if not fn or not sys.argv:
        return False
    try:
        return Path(sys.argv[0]).resolve() == Path(str(fn)).resolve()
    except OSError:
        return False


def _should_run_cli() -> bool:
    return __name__ == "__main__" or _cli_env_enabled() or _invoked_as_script()


def _force_preview_env() -> bool:
    import os

    return os.environ.get("STOCK_ANALYSIS_FORCE_PREVIEW", "").strip().lower() in ("1", "true", "yes", "y", "on")


def _quiet_env() -> bool:
    import os

    return os.environ.get("STOCK_ANALYSIS_QUIET", "").strip().lower() in ("1", "true", "yes", "y", "on")


def _should_suppress_auto_preview() -> bool:
    """
    When True, only run main_cli if _should_run_cli() (env STOCK_ANALYSIS_CLI, etc.).

    Zerve often loads this file as a real module (__name__ == stock_analysis, __file__ set).
    Set env STOCK_ANALYSIS_FORCE_PREVIEW=1 on that block to always run the preview.
    """
    if _force_preview_env():
        return False
    if _quiet_env():
        return True
    name = globals().get("__name__")
    fn = globals().get("__file__")
    has_file = bool(fn and str(fn).strip())
    return name == "stock_analysis" and has_file


def main_cli() -> None:
    """Print smoke / preview to stdout."""
    def _p(*a: object, **k: Any) -> None:
        k.setdefault("flush", True)
        print(*a, **k)

    _p("stock_analysis — AMZN/stock helpers (imported by dashboard.py). CLI smoke / data preview.\n")
    df, source = _load_panel_cli()
    if df is None:
        _p("No panel file found (set SCORED_DF_PATH or place scored_df.parquet / scored_df.csv in cwd).")
        _p("Running synthetic 60-day smoke test instead.\n")
        df = _synthetic_panel()
    else:
        _p(f"Loaded panel from {source}: {len(df)} rows")
        _p(f"Columns: {list(df.columns)[:12]}{'...' if len(df.columns) > 12 else ''}\n")

    df = add_stock_derived_columns(df)
    show_cols = [c for c in ("date", "close", "log_ret_1d", "brainrot_score", "roll_corr_brainrot_ret") if c in df.columns]
    if show_cols:
        _p("Tail (key columns):")
        _p(df[show_cols].tail(8).to_string(index=False))
        _p()

    et = event_return_table(df)
    if not et.empty:
        _p("Event-day return snapshot:")
        _p(et.to_string(index=False))
        _p()

    if (
        "trends_composite" in df.columns
        and "log_ret_1d" in df.columns
        and df["log_ret_1d"].notna().sum() >= 8
    ):
        lag_df = lag_profile(df, "trends_composite", "log_ret_1d", pair_name="CLI Trends -> ret", max_lag=7)
        best = lag_df.dropna(subset=["abs_corr"]).sort_values("abs_corr", ascending=False).head(1)
        if not best.empty:
            r0 = best.iloc[0]
            _p(
                f"Lag preview (Trends vs log_ret): best lag = {int(r0['lag_days'])} d, r = {float(r0['corr']):.4f}"
            )
    else:
        _p("Lag preview skipped (need trends_composite + enough log_ret_1d rows).")

    _p("\nDone. For charts: streamlit run dashboard.py")
    _p(
        "\nTip: if no preview ran, set env "
        "STOCK_ANALYSIS_FORCE_PREVIEW=1 or STOCK_ANALYSIS_CLI=1 for this block."
    )



def _run_cli_bootstrap() -> None:
    if _should_suppress_auto_preview():
        if _should_run_cli():
            main_cli()
    else:
        main_cli()


_run_cli_bootstrap()
