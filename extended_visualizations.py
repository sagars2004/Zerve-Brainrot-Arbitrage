"""
Extended report visualizations — episode windows, lag scatter + regression, rolling correlation.

Upstream globals (Zerve DAG):
- Panel: scored_df or master_df (preferred names). If Zerve uses another name, any global
  DataFrame with columns date, trends_composite, volume_zscore is auto-detected.
- events_df (or any DataFrame with release_date; optional event_label / show / season / episode).
- lag_results_df (optional): best lag for H3; else any DataFrame with pair + best_lag_days columns.

Outputs:
- fig_event_study: 3 rows × N columns — per episode: Trends, PM proxy, AMZN volume z (±5 calendar days)
- fig_scatter_lag: Trends (shifted by best H3 lag) vs AMZN volume z + OLS line + 95% mean-response band
- fig_rolling_corr: 30-day rolling Pearson r(Trends, volume z) + markers on episode-drop days
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

EVENT_WINDOW_DAYS = 5
ROLLING_WINDOW = 30
ROLLING_MIN_PERIODS = 15
DEFAULT_H3_LAG = 1
H3_PAIR_NAME = "H3 Trends -> AMZN"

# Rolling-correlation (and any future calendar plots): x-axis starts here for readability. None = full range.
VIEW_DATE_X_START: str | None = "2026-02-01"


def _plotly_fig(fig: go.Figure) -> Any:
    return cast(Any, fig)


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


def _list_df_globals() -> list[str]:
    return sorted(k for k, v in globals().items() if isinstance(v, pd.DataFrame))


def _panel_df_required_cols() -> frozenset[str]:
    return frozenset({"date", "trends_composite", "volume_zscore"})


def _is_panel_df(df: pd.DataFrame) -> bool:
    return _panel_df_required_cols().issubset(df.columns)


def _get_plot_df() -> tuple[pd.DataFrame, str]:
    """Resolve merged/scored panel; Zerve may name outputs differently than scored_df."""
    g = globals()
    for name in ("scored_df", "master_df", "merged_df", "panel_df", "master", "output_df"):
        v = g.get(name)
        if isinstance(v, pd.DataFrame) and _is_panel_df(v):
            df = v.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
            return df.sort_values("date").reset_index(drop=True), name
    for name, v in g.items():
        if name.startswith("__"):
            continue
        if isinstance(v, pd.DataFrame) and _is_panel_df(v):
            df = v.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
            return df.sort_values("date").reset_index(drop=True), name
    keys = _list_df_globals()
    raise RuntimeError(
        "No DataFrame in scope with columns date, trends_composite, volume_zscore. "
        "Wire the merge_signals / brainrot_score output into this block (or assign it to "
        f"scored_df / master_df). DataFrames currently visible: {keys!r}"
    )


def _get_events_df() -> tuple[pd.DataFrame, str]:
    g = globals()
    for name in ("events_df", "events", "episode_events", "event_table"):
        v = g.get(name)
        if isinstance(v, pd.DataFrame) and "release_date" in v.columns:
            ev = v.copy()
            ev["release_date"] = pd.to_datetime(ev["release_date"], errors="coerce").dt.normalize()
            return ev.sort_values("release_date").reset_index(drop=True), name
    for name, v in g.items():
        if name.startswith("__"):
            continue
        if isinstance(v, pd.DataFrame) and "release_date" in v.columns:
            ev = v.copy()
            ev["release_date"] = pd.to_datetime(ev["release_date"], errors="coerce").dt.normalize()
            return ev.sort_values("release_date").reset_index(drop=True), name
    keys = _list_df_globals()
    raise RuntimeError(
        "No DataFrame with column release_date (build_events output). "
        f"Wire events_df into this block. DataFrames in scope: {keys!r}"
    )


def _h3_best_lag() -> int:
    g = globals()
    for lr_name in ("lag_results_df", "lag_summary_df", "lag_results"):
        lr = g.get(lr_name)
        if isinstance(lr, pd.DataFrame) and not lr.empty and "pair" in lr.columns:
            m = lr[lr["pair"] == H3_PAIR_NAME]
            if not m.empty:
                v = m.iloc[0].get("best_lag_days")
                if pd.notna(v):
                    return int(v)
    for _, v in g.items():
        if isinstance(v, pd.DataFrame) and not v.empty and "pair" in v.columns and "best_lag_days" in v.columns:
            m = v[v["pair"] == H3_PAIR_NAME]
            if not m.empty:
                val = m.iloc[0].get("best_lag_days")
                if pd.notna(val):
                    return int(val)
    return DEFAULT_H3_LAG


def _event_label(row: pd.Series) -> str:
    if "event_label" in row.index and pd.notna(row.get("event_label")):
        return str(row["event_label"])
    return f"{row.get('show', '?')} S{row.get('season', '?')}E{row.get('episode', '?')}"


def _build_event_study_fig(plot_df: pd.DataFrame, events_df: pd.DataFrame) -> go.Figure:
    events_df = events_df.copy()
    n = len(events_df)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(title="Event study — no events in events_df", template="plotly_white")
        return fig

    fig = make_subplots(
        rows=3,
        cols=n,
        shared_yaxes=False,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
        subplot_titles=[_event_label(events_df.iloc[i]) for i in range(n)],
    )

    for col, (_, ev) in enumerate(events_df.iterrows(), start=1):
        center = pd.Timestamp(ev["release_date"]).normalize()
        w0 = center - pd.Timedelta(days=EVENT_WINDOW_DAYS)
        w1 = center + pd.Timedelta(days=EVENT_WINDOW_DAYS)
        win = plot_df[(plot_df["date"] >= w0) & (plot_df["date"] <= w1)].copy()
        if win.empty:
            continue

        x = win["date"]
        trends = pd.to_numeric(win["trends_composite"], errors="coerce")
        if "pm_yes_price_ffill" in win.columns:
            pm = pd.to_numeric(win["pm_yes_price_ffill"], errors="coerce")
        else:
            pm = pd.Series(np.nan, index=win.index, dtype=float)
        if bool(pm.isna().all()) and "pm_yes_price_mean" in win.columns:
            pm = pd.to_numeric(win["pm_yes_price_mean"], errors="coerce").ffill()

        volz = pd.to_numeric(win["volume_zscore"], errors="coerce")

        fig.add_trace(
            go.Scatter(x=x, y=trends, mode="lines+markers", name="Trends", line=dict(color="#0ea5e9"), showlegend=(col == 1)),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(x=x, y=pm, mode="lines+markers", name="PM YES", line=dict(color="#d946ef"), showlegend=(col == 1)),
            row=2,
            col=col,
        )
        fig.add_trace(
            go.Scatter(x=x, y=volz, mode="lines+markers", name="Vol z", line=dict(color="#0f172a"), showlegend=(col == 1)),
            row=3,
            col=col,
        )
        for r in (1, 2, 3):
            _plotly_fig(fig).add_vline(x=center, line_dash="dash", line_color="#f97316", line_width=2, row=r, col=col)

    fig.update_layout(
        title=(
            f"Event study: ±{EVENT_WINDOW_DAYS} calendar days around each drop "
            "(vertical line = release date, 3am ET)"
        ),
        height=520,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(t=100, b=40),
    )
    fig.update_xaxes(tickangle=-35)
    fig.update_yaxes(title_text="Trends", row=1, col=1)
    fig.update_yaxes(title_text="PM YES", row=2, col=1)
    fig.update_yaxes(title_text="Vol z", row=3, col=1)
    return fig


def _ols_line_and_ci(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return x_grid, y_hat, y_hi, y_lo, slope, intercept (mean-response ~95% band)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 5:
        raise ValueError("Too few points for regression")
    slope, intercept = np.polyfit(x, y, 1)
    x_mean = x.mean()
    sxx = np.sum((x - x_mean) ** 2)
    y_hat_all = slope * x + intercept
    resid = y - y_hat_all
    dof = max(n - 2, 1)
    mse = float(np.sum(resid**2) / dof)
    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 80)
    y_line = slope * x_line + intercept
    if sxx <= 0 or mse <= 0:
        return x_line, y_line, y_line, y_line, float(slope), float(intercept)
    se_mean = np.sqrt(mse * (1.0 / n + (x_line - x_mean) ** 2 / sxx))
    t = 1.96
    hi = y_line + t * se_mean
    lo = y_line - t * se_mean
    return x_line, y_line, hi, lo, float(slope), float(intercept)


def _build_scatter_lag_fig(plot_df: pd.DataFrame, lag: int) -> go.Figure:
    df = plot_df.sort_values("date").reset_index(drop=True).copy()
    t = pd.to_numeric(df["trends_composite"], errors="coerce")
    v = pd.to_numeric(df["volume_zscore"], errors="coerce")
    # Same alignment as lag_correlation.analyze_pair: positive lag => x.shift(lag) vs y.
    x_align = t.shift(lag)
    pair = pd.DataFrame({"x": x_align, "y": v}).dropna()

    fig = go.Figure()
    if len(pair) < 8:
        fig.update_layout(
            title=f"Lag scatter (H3 lag={lag}d) — insufficient paired rows",
            template="plotly_white",
        )
        return fig

    xa = pair["x"].to_numpy(dtype=float)
    ya = pair["y"].to_numpy(dtype=float)
    x_line, y_line, y_hi, y_lo, slope, _intercept = _ols_line_and_ci(xa, ya)

    # Band first, then fit line, then points on top for hover visibility.
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_hi,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_lo,
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(124, 58, 237, 0.2)",
            fill="tonexty",
            name="~95% mean-response band",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"OLS (slope={slope:.3f})",
            line=dict(color="#be185d", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pair["x"],
            y=pair["y"],
            mode="markers",
            name="Days",
            marker=dict(size=8, color="#7c3aed", opacity=0.65),
            text=df.loc[pair.index, "date"].dt.strftime("%Y-%m-%d"),
            hovertemplate="%{text}<br>Trends (t−lag)=%{x:.1f}<br>Vol z=%{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=(
            f"Trends vs AMZN volume z at H3 lag = {lag} days "
            f"(x = trends shifted +{lag}, y = volume z; exploratory OLS)"
        ),
        xaxis_title="Google Trends composite (aligned to lead volume by lag)",
        yaxis_title="AMZN volume z-score",
        template="plotly_white",
        height=480,
        margin=dict(t=80, b=60),
    )
    return fig


def _build_rolling_corr_fig(plot_df: pd.DataFrame, events_df: pd.DataFrame) -> go.Figure:
    df = plot_df.sort_values("date").reset_index(drop=True).copy()
    t = pd.to_numeric(df["trends_composite"], errors="coerce")
    v = pd.to_numeric(df["volume_zscore"], errors="coerce")
    roll = t.rolling(window=ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS).corr(v)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=roll,
            mode="lines",
            name=f"{ROLLING_WINDOW}d rolling r(Trends, vol z)",
            line=dict(color="#0369a1", width=2),
            connectgaps=False,
        )
    )

    drop_days = set(pd.to_datetime(events_df["release_date"]).dt.normalize())
    on_drop = df["date"].isin(drop_days)
    if on_drop.any():
        fig.add_trace(
            go.Scatter(
                x=df.loc[on_drop, "date"],
                y=roll[on_drop],
                mode="markers",
                name="Episode drop day",
                marker=dict(size=12, color="#f97316", symbol="diamond", line=dict(color="white", width=1)),
                text=df.loc[on_drop, "date"].dt.strftime("%Y-%m-%d"),
                hovertemplate="%{text}<br>rolling r=%{y:.3f}<extra></extra>",
            )
        )

    for d in sorted(drop_days):
        fig.add_vline(x=d, line_dash="dot", line_color="rgba(249,115,22,0.45)", line_width=1)

    fig.add_hline(y=0.0, line_dash="solid", line_color="#94a3b8", line_width=1)

    fig.update_layout(
        title=(
            f"Rolling correlation (Pearson): Trends composite vs AMZN volume z "
            f"(window={ROLLING_WINDOW} obs, min_periods={ROLLING_MIN_PERIODS})"
        ),
        xaxis_title="Date",
        yaxis_title="Rolling r",
        yaxis=dict(range=[-1.05, 1.05], zeroline=True),
        template="plotly_white",
        height=440,
        margin=dict(t=80, b=60),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    xr = _date_x_range(df)
    if xr:
        fig.update_xaxes(range=[xr[0], xr[1]])
    return fig


def _should_autoshow() -> bool:
    if __name__ == "__main__":
        return True
    return "__file__" not in globals()


# --- Build figures when upstream data exists ---
try:
    _plot_df, _panel_src = _get_plot_df()
    _events, _events_src = _get_events_df()
    _lag = _h3_best_lag()
    fig_event_study = _build_event_study_fig(_plot_df, _events)
    fig_scatter_lag = _build_scatter_lag_fig(_plot_df, _lag)
    fig_rolling_corr = _build_rolling_corr_fig(_plot_df, _events)
    print(
        f"extended_visualizations: panel={_panel_src!r}, events={_events_src!r}, "
        f"H3 scatter lag={_lag}d (lag_results_df or default)."
    )
except RuntimeError as e:
    msg = str(e)
    fig_event_study = go.Figure()
    fig_event_study.update_layout(
        title="Extended viz — data not wired (see console)",
        template="plotly_white",
        annotations=[
            dict(
                text=msg[:500] + ("…" if len(msg) > 500 else ""),
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=11),
                align="center",
            )
        ],
    )
    fig_scatter_lag = fig_event_study
    fig_rolling_corr = fig_event_study
    print(f"extended_visualizations: {e}")

if _should_autoshow():
    fig_event_study.show()
    fig_scatter_lag.show()
    fig_rolling_corr.show()
