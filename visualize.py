"""
Visualization block — Plotly charts for Zerve inline display.

Inputs (from upstream globals):
- scored_df preferred; else master_df if it already contains brainrot_score
- lag_series_df optional (from lag_correlation)
- lag_results_df optional (best lag / r per hypothesis)

Outputs:
- fig_brainrot: Brainrot + component traces, AMZN volume z on secondary axis, episode markers
- fig_signal_stack: stacked panels — Brainrot, AMZN close, volume z-score, optional Polymarket level
- fig_lag_panel: H1 / H2 / H3 lag correlation bars with y=0 baseline and best-lag callouts
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _plotly_fig(fig: go.Figure) -> Any:
    """Plotly stubs mis-type subplot row/col; runtime accepts int."""
    return cast(Any, fig)

LAG_PAIRS: list[tuple[str, str]] = [
    ("H1 Trends -> Polymarket", "H1: Trends → Polymarket"),
    ("H2 Polymarket -> AMZN", "H2: Polymarket → AMZN"),
    ("H3 Trends -> AMZN", "H3: Trends → AMZN"),
]

# Match lag_correlation.py MAX_LAG so empty panels share the same x-axis span as filled ones.
_LAG_X_RANGE: tuple[float, float] = (-7.5, 7.5)

# Time-series x-axis left edge (zoom so January etc. don’t compress the signal). None = auto full range.
VIEW_DATE_X_START: str | None = "2026-02-01"


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


def _apply_date_x_range(fig: go.Figure, plot_df: pd.DataFrame, *, subplot_rows: int | None = None) -> None:
    xr = _date_x_range(plot_df)
    if not xr:
        return
    dmin, dmax = xr
    if subplot_rows is None:
        fig.update_xaxes(range=[dmin, dmax])
    else:
        for r in range(1, subplot_rows + 1):
            _plotly_fig(fig).update_xaxes(range=[dmin, dmax], row=r, col=1)


def _get_plot_frame() -> pd.DataFrame:
    g = globals()
    if "scored_df" in g and isinstance(g["scored_df"], pd.DataFrame):
        return g["scored_df"].copy()
    if "master_df" in g and isinstance(g["master_df"], pd.DataFrame):
        m = g["master_df"]
        if "brainrot_score" in m.columns:
            return m.copy()
    raise RuntimeError(
        "Missing scored_df. Run brainrot_score.py after merge_signals, or ensure master_df has brainrot_score."
    )


def _get_lag_series() -> pd.DataFrame | None:
    g = globals()
    if "lag_series_df" in g and isinstance(g["lag_series_df"], pd.DataFrame):
        return g["lag_series_df"].copy()
    return None


def _get_lag_results() -> pd.DataFrame | None:
    g = globals()
    if "lag_results_df" in g and isinstance(g["lag_results_df"], pd.DataFrame):
        return g["lag_results_df"].copy()
    return None


def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        raise RuntimeError("Plot frame missing required column: date")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.sort_values("date").reset_index(drop=True)


def _upstream_ready() -> bool:
    s = globals().get("scored_df")
    m = globals().get("master_df")
    if isinstance(s, pd.DataFrame) and not s.empty:
        return True
    if isinstance(m, pd.DataFrame) and "brainrot_score" in m.columns:
        return True
    return False


def _empty_figs() -> tuple[go.Figure, go.Figure, go.Figure]:
    fb = go.Figure()
    fb.update_layout(
        title="Brainrot Score — no data yet",
        template="plotly_white",
        annotations=[
            dict(
                text="Run merge_signals → brainrot_score before visualize.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
        ],
    )
    fo = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fo.update_layout(title="Signal stack — no data yet", template="plotly_white")
    fl = make_subplots(rows=1, cols=3, subplot_titles=("H1", "H2", "H3"))
    fl.update_layout(title="Lag correlations — no data yet", template="plotly_white")
    return fb, fo, fl


def _event_dates(plot_df: pd.DataFrame) -> list[pd.Timestamp]:
    if "is_event_day" not in plot_df.columns or not plot_df["is_event_day"].any():
        return []
    ev = plot_df.loc[plot_df["is_event_day"], "date"].dropna().unique()
    return sorted(pd.to_datetime(ev).tolist())


def _add_event_vlines(fig: go.Figure, dates: list[pd.Timestamp], *, rows: int | None = None) -> None:
    for d in dates:
        if rows is None:
            fig.add_vline(x=d, line_width=1, line_dash="dash", line_color="rgba(249,115,22,0.7)")
        else:
            for r in range(1, rows + 1):
                _plotly_fig(fig).add_vline(
                    x=d,
                    line_width=1,
                    line_dash="dash",
                    line_color="rgba(249,115,22,0.55)",
                    row=r,
                    col=1,
                )


def _subtitle_window(plot_df: pd.DataFrame) -> str:
    d0 = plot_df["date"].min()
    d1 = plot_df["date"].max()
    n = len(plot_df)
    n_score = int(plot_df["brainrot_score"].notna().sum()) if "brainrot_score" in plot_df.columns else 0
    n_events = int(plot_df["is_event_day"].sum()) if "is_event_day" in plot_df.columns else 0
    return (
        f"Calendar span: {d0:%Y-%m-%d} → {d1:%Y-%m-%d} · {n} rows · "
        f"{n_score} days with Brainrot score · {n_events} episode-drop flags"
    )


def _build_fig_brainrot(plot_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df["brainrot_score"],
            mode="lines",
            name="Brainrot score",
            line=dict(color="#7c3aed", width=2.5),
            connectgaps=False,
        ),
        secondary_y=False,
    )

    if "br_trends_0_100" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["date"],
                y=plot_df["br_trends_0_100"],
                mode="lines",
                name="Trends (0–100)",
                line=dict(color="#0ea5e9", width=1.5, dash="dot"),
                opacity=0.85,
                connectgaps=False,
            ),
            secondary_y=False,
        )
    if "br_amzn_0_100" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["date"],
                y=plot_df["br_amzn_0_100"],
                mode="lines",
                name="AMZN vol z → 0–100",
                line=dict(color="#059669", width=1.2, dash="dash"),
                opacity=0.8,
                connectgaps=False,
            ),
            secondary_y=False,
        )
    if "br_poly_0_100" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["date"],
                y=plot_df["br_poly_0_100"],
                mode="lines",
                name="Polymarket |Δyes| → 0–100",
                line=dict(color="#d946ef", width=1.2, dash="dot"),
                opacity=0.75,
                connectgaps=False,
            ),
            secondary_y=False,
        )

    if "volume_zscore" in plot_df.columns and plot_df["volume_zscore"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=plot_df["date"],
                y=plot_df["volume_zscore"],
                mode="lines",
                name="AMZN volume z (raw)",
                line=dict(color="#64748b", width=1.5),
                opacity=0.9,
                connectgaps=True,
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Volume z-score", secondary_y=True, zeroline=True)

    if "is_event_day" in plot_df.columns and plot_df["is_event_day"].any():
        ev = plot_df[plot_df["is_event_day"]].dropna(subset=["date", "brainrot_score"])
        if not ev.empty:
            fig.add_trace(
                go.Scatter(
                    x=ev["date"],
                    y=ev["brainrot_score"],
                    mode="markers",
                    name="Episode drop (score)",
                    marker=dict(size=11, color="#f97316", symbol="diamond", line=dict(width=1, color="white")),
                    text=ev.get("event_labels", ""),
                    hovertemplate="%{text}<br>%{x|%Y-%m-%d}<br>score=%{y:.1f}<extra></extra>",
                ),
                secondary_y=False,
            )

    _add_event_vlines(fig, _event_dates(plot_df), rows=None)

    fig.add_annotation(
        x=0.01,
        y=0.02,
        xref="paper",
        yref="paper",
        text="Diamonds: episode-drop days · Dashed lines: same calendar drops · Gray line: raw volume z (right axis).",
        showarrow=False,
        font=dict(size=10, color="#64748b"),
        align="left",
    )

    fig.update_layout(
        title=(
            "Brainrot score + components<br><sup>"
            + _subtitle_window(plot_df)
            + "</sup>"
        ),
        xaxis_title="Date",
        yaxis_title="0–100 scale",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="right", x=1),
        margin=dict(t=100, b=60),
        template="plotly_white",
        yaxis=dict(range=[0, 105]),
    )
    _apply_date_x_range(fig, plot_df, subplot_rows=None)
    return fig


def _build_fig_signal_stack(plot_df: pd.DataFrame) -> go.Figure:
    has_pm = (
        "pm_yes_price_ffill" in plot_df.columns
        and plot_df["pm_yes_price_ffill"].notna().any()
    )
    nrows = 4 if has_pm else 3
    titles = (
        ("Brainrot score", "AMZN close ($)", "Volume z-score (20d)", "Polymarket YES (ffill)")
        if has_pm
        else ("Brainrot score", "AMZN close ($)", "Volume z-score (20d)")
    )
    heights = [0.28, 0.28, 0.24, 0.20] if has_pm else [0.34, 0.33, 0.33]

    fig = make_subplots(
        rows=nrows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=heights,
        subplot_titles=titles,
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df["brainrot_score"],
            name="Brainrot",
            line=dict(color="#7c3aed", width=2),
            connectgaps=False,
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    if "close" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["date"],
                y=plot_df["close"],
                name="Close",
                line=dict(color="#059669", width=2),
                connectgaps=True,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    if "volume_zscore" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["date"],
                y=plot_df["volume_zscore"],
                name="Vol z",
                line=dict(color="#0f172a", width=1.5),
                connectgaps=True,
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        fig.add_hline(y=0.0, line_dash="dot", line_color="rgba(100,116,139,0.6)", row=3, col=1)  # type: ignore[call-arg]
        fig.add_hline(y=2.0, line_dash="dash", line_color="rgba(239,68,68,0.35)", row=3, col=1)  # type: ignore[call-arg]
        fig.add_hline(y=-2.0, line_dash="dash", line_color="rgba(239,68,68,0.35)", row=3, col=1)  # type: ignore[call-arg]

    if has_pm:
        fig.add_trace(
            go.Scatter(
                x=plot_df["date"],
                y=plot_df["pm_yes_price_ffill"],
                name="PM YES ffill",
                line=dict(color="#d946ef", width=1.8),
                connectgaps=True,
                showlegend=False,
            ),
            row=4,
            col=1,
        )
        if "pm_price_imputed" in plot_df.columns:
            imp = plot_df[plot_df["pm_price_imputed"]]
            if not imp.empty:
                fig.add_trace(
                    go.Scatter(
                        x=imp["date"],
                        y=imp["pm_yes_price_ffill"],
                        mode="markers",
                        name="PM imputed day",
                        marker=dict(size=6, color="#f97316", symbol="circle-open"),
                        showlegend=False,
                        hovertemplate="Forward-filled PM price<extra></extra>",
                    ),
                    row=4,
                    col=1,
                )

    _add_event_vlines(fig, _event_dates(plot_df), rows=nrows)

    fig.update_layout(
        title="Stacked signals (aligned calendar dates)",
        hovermode="x unified",
        height=200 + 160 * nrows,
        margin=dict(t=60, b=50),
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Date", row=nrows, col=1)
    _apply_date_x_range(fig, plot_df, subplot_rows=nrows)
    return fig


def _build_fig_lag_panel(lag_df: pd.DataFrame | None) -> go.Figure:
    lr = _get_lag_results()

    subplot_titles: list[str] = []
    for pair_key, short_title in LAG_PAIRS:
        subtitle_extra = ""
        if lr is not None and not lr.empty:
            match = lr[lr["pair"] == pair_key]
            if not match.empty:
                row0 = match.iloc[0]
                lag_v = row0.get("best_lag_days")
                r_v = row0.get("best_corr")
                note = row0.get("direction_note", "")
                if pd.notna(lag_v) and pd.notna(r_v):
                    subtitle_extra = f"<br><sup>Peak |r|: lag {int(lag_v)}, r={float(r_v):.3f}</sup>"
                elif isinstance(note, str) and note:
                    subtitle_extra = f"<br><sup>{note}</sup>"
        subplot_titles.append(f"{short_title}{subtitle_extra}")

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        column_widths=[1, 1, 1],
    )

    if lag_df is None or lag_df.empty:
        fig.update_layout(
            title="Lag correlations — run lag_correlation first",
            template="plotly_white",
            height=420,
        )
        return fig

    ymax = 0.1
    ymin = -0.1

    for col, (pair_key, _short_title) in enumerate(LAG_PAIRS, start=1):
        sub = lag_df[lag_df["pair"] == pair_key].sort_values("lag_days")

        if sub.empty or sub["corr"].isna().all():
            # Invisible trace fixes subplot width/domain; Plotly collapses empty panels otherwise.
            _plotly_fig(fig).add_trace(
                go.Scatter(
                    x=[_LAG_X_RANGE[0], _LAG_X_RANGE[1]],
                    y=[0.0, 0.0],
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )
            _plotly_fig(fig).add_hline(y=0, line_color="#e2e8f0", row=1, col=col)
            # Subplot title already carries lag_results note; do not duplicate "No usable correlations"
            # on the wrong axis — add_annotation(row=col) is unreliable across Plotly versions.
            continue

        colors = ["#ef4444" if (v is not None and v < 0) else "#22c55e" for v in sub["corr"].fillna(0)]
        fig.add_trace(
            go.Bar(
                x=sub["lag_days"],
                y=sub["corr"],
                name=pair_key,
                marker_color=colors,
                showlegend=False,
                hovertemplate="Lag %{x} d<br>r = %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        _plotly_fig(fig).add_hline(y=0, line_width=1, line_color="#64748b", row=1, col=col)
        cmax = float(sub["corr"].max())
        cmin = float(sub["corr"].min())
        ymax = max(ymax, cmax)
        ymin = min(ymin, cmin)

        if lr is not None and not lr.empty:
            m = lr[lr["pair"] == pair_key]
            if not m.empty and pd.notna(m.iloc[0].get("best_lag_days")):
                bl = int(m.iloc[0]["best_lag_days"])
                br = sub[sub["lag_days"] == bl]
                if not br.empty:
                    y_peak = float(br.iloc[0]["corr"])
                    # Anchor by subplot axis id (row/col on annotations is easy to mis-apply).
                    _plotly_fig(fig).add_annotation(
                        x=bl,
                        y=y_peak,
                        xref=f"x{col}",
                        yref=f"y{col}",
                        text=f"best {bl}d",
                        showarrow=True,
                        arrowhead=2,
                        ax=0,
                        ay=-30,
                        font=dict(size=10),
                    )

    pad = 0.08 * (ymax - ymin if ymax > ymin else 1.0)
    y1 = ymin - pad
    y2 = ymax + pad

    for c in (1, 2, 3):
        fig.update_xaxes(range=list(_LAG_X_RANGE), row=1, col=c)
        fig.update_yaxes(range=[y1, y2], row=1, col=c)

    fig.update_xaxes(title_text="Lag (d): + ⇒ x leads y", row=1, col=1)
    fig.update_xaxes(title_text="Lag (d)", row=1, col=2)
    fig.update_xaxes(title_text="Lag (d)", row=1, col=3)
    fig.update_yaxes(title_text="Pearson r", row=1, col=1)
    fig.update_layout(
        title="Lag correlation dashboard (z-scored series; exploratory — not adjusted for multiple lags)",
        height=460,
        margin=dict(t=80, b=50),
        template="plotly_white",
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
    return fig


def _build_figures(plot_df: pd.DataFrame) -> tuple[go.Figure, go.Figure, go.Figure]:
    fig_brainrot = _build_fig_brainrot(plot_df)
    fig_amzn_overlay = _build_fig_signal_stack(plot_df)
    fig_lag_h3 = _build_fig_lag_panel(_get_lag_series())
    return fig_brainrot, fig_amzn_overlay, fig_lag_h3


if not _upstream_ready():
    fig_brainrot, fig_amzn_overlay, fig_lag_h3 = _empty_figs()
    print(
        "visualize: skipped (wire brainrot_score block first: need scored_df or master_df.brainrot_score)."
    )
else:
    _plot_df = _ensure_date(_get_plot_frame())
    fig_brainrot, fig_amzn_overlay, fig_lag_h3 = _build_figures(_plot_df)


def _should_autoshow() -> bool:
    if __name__ == "__main__":
        return True
    return "__file__" not in globals()


if _should_autoshow():
    fig_brainrot.show()
    fig_amzn_overlay.show()
    fig_lag_h3.show()
