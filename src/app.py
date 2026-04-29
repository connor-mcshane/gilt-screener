from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.nonparametric.smoothers_lowess import lowess

from data.pipeline import DEFAULT_PROCESSED_FILE, DEFAULT_TAX_BANDS, load_latest_snapshot

st.set_page_config(
    page_title="UK Gilt Screener",
    page_icon=":pound_banknote:",
    layout="wide",
)

TAX_BAND_LABELS = {
    0.0: "None (0%)",
    0.20: "Basic rate (20%)",
    0.40: "Higher rate (40%)",
    0.45: "Additional rate (45%)",
}


@st.cache_data(show_spinner=False)
def _load_snapshot(path: str) -> pd.DataFrame:
    return load_latest_snapshot(path)


def _fit_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    # get smoothed curve
    if len(x) < 5:
        return None
    smoothed = lowess(y, x, frac=0.3, it=1, return_sorted=True)
    return smoothed[:, 0], smoothed[:, 1]


def _render_yield_curve(
    full: pd.DataFrame,
    highlighted_isins: set[str],
    post_tax_col: str,
    max_ttm: float,
) -> go.Figure:
    # plot all gilts and highlight filtered ones

    base = full.dropna(subset=["time_to_maturity", "yield_to_maturity"]).copy()
    base["is_highlighted"] = base["isin"].isin(highlighted_isins)

    fig = px.scatter(
        base,
        x="time_to_maturity",
        y="yield_to_maturity",
        color="tax_free_return_pct",
        color_continuous_scale="Viridis",
        custom_data=[
            "name", 
            "isin", 
            "coupon", 
            "clean_price", 
            post_tax_col, 
            "redemption_date", 
            "tax_free_return_pct",
            "equivalent_gross_yield"
        ],
    )
    # small dots for all gilts, big dots for highlighted ones
    sizes = base["is_highlighted"].map({True: 20, False: 8})
    opacities = base["is_highlighted"].map({True: 1.0, False: 0.55})
    fig.update_traces(
        marker=dict(size=sizes.tolist(), opacity=opacities.tolist(), line=dict(width=0)),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "ISIN: %{customdata[1]}<br>"
            "Coupon: %{customdata[2]:.3f}%<br>"
            "Maturity: %{customdata[5]}<br>"
            "TTM: %{x:.2f}y<br>"
            "GRY: %{y:.3f}%<br>"
            "Clean price: %{customdata[3]:.2f}<br>"
            "Tax-free portion: %{customdata[6]:.1f}%<br>"
            "Post-tax yield: %{customdata[4]:.3f}%<br>"
            "<b>Equivalent savings rate: %{customdata[7]:.2f}%</b>"
            "<extra></extra>"
        ),
    )

    fit = _fit_curve(
        base["time_to_maturity"].to_numpy(dtype=float),
        base["yield_to_maturity"].to_numpy(dtype=float),
    )
    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=fit[0],
                y=fit[1],
                mode="lines",
                name="Fit (LOWESS)",
                line=dict(color="#6FA8DC", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    snapshot_date = full["snapshot_date"].dropna().iloc[0] if not full.empty else ""
    
    x_max = max_ttm + 1.0
    
    # add top axis for actual years
    if not full.empty and pd.notna(snapshot_date):
        max_years = int(np.ceil(x_max))
        # ticks every 5y
        tick_vals = list(range(0, max_years + 5, 5))
        # convert to dates
        tick_texts = []
        for years in tick_vals:
            # add years to snapshot
            target_year = snapshot_date.year + years
            tick_texts.append(str(target_year))
            
        fig.update_layout(
            xaxis=dict(
                title="Time To Maturity (years)",
                showgrid=True,
                range=[0, x_max],
            ),
            xaxis2=dict(
                title="Maturity Year",
                overlaying="x",
                side="top",
                tickmode="array",
                tickvals=tick_vals,
                ticktext=tick_texts,
                showgrid=False,
                range=[0, x_max],
            ),
        )
        
        # fix traces to use primary axis
        for trace in fig.data:
            trace.xaxis = "x"

    fig.update_layout(
        title=f"UK Gilt Yield Curve {snapshot_date}",
        yaxis_title="Yield To Maturity (%)",
        height=540,
        margin=dict(l=40, r=40, t=80, b=40),
    )
    return fig


def main() -> None:
    st.title("UK Gilt Screener")
    st.caption(
        "Find CGT-efficient UK gilts. Capital gains on conventional gilts are "
        "tax-free for individuals (TCGA 1992 s.115); only coupon income is "
        "taxed. The screener ranks gilts on post-tax yield at your marginal "
        "rate."
    )

    if not DEFAULT_PROCESSED_FILE.exists():
        st.error(
            "No snapshot file found at `data/processed/gilts_latest.parquet`. "
            "This app only reads data baked into the repo (updated by the scheduled "
            "GitHub Action). Commit that file or run the pipeline locally, then redeploy."
        )
        return

    df = _load_snapshot(str(DEFAULT_PROCESSED_FILE))

    if df.empty:
        st.warning("Snapshot file exists but has no rows. Re-run the data pipeline.")
        return

    _sd = df["snapshot_date"].dropna()
    if not _sd.empty:
        st.caption(
            f"Data last updated: **{_sd.iloc[0]}** "
            "(from `gilts_latest.parquet` in the repo)."
        )

    with st.sidebar:
        st.header("Filters")
        tax_rate = st.selectbox(
            "Marginal tax rate",
            options=list(DEFAULT_TAX_BANDS),
            index=2,
            format_func=lambda r: TAX_BAND_LABELS[r],
        )
        post_tax_col = f"post_tax_yield_{int(round(tax_rate * 100))}"
        
        divisor = (1.0 - tax_rate) if tax_rate < 1.0 else 1.0
        df["equivalent_gross_yield"] = df[post_tax_col] / divisor

        coupon_max = float(df["coupon"].max())
        max_coupon = st.slider(
            "Max coupon (%)",
            min_value=0.0,
            max_value=max(coupon_max, 1.0),
            value=min(2.0, coupon_max),
            step=0.125,
        )
        ttm_max = float(df["time_to_maturity"].max() or 50.0)
        max_ttm = st.slider(
            "Max years to maturity",
            min_value=0.0,
            max_value=max(ttm_max, 1.0),
            value=min(30.0, ttm_max),
            step=0.5,
        )
        sub_par_only = st.checkbox(
            "Only gilts trading below par",
            value=False,
            help="Par is 100. Gilts trading below 100 will mature at 100, "
            "giving you a guaranteed capital gain. Because this gain is "
            "tax-free, these are the most tax-efficient gilts.",
        )

    filtered = df.loc[
        df["coupon"].le(max_coupon)
        & df["time_to_maturity"].le(max_ttm)
        & df["time_to_maturity"].gt(0)
    ].copy()
    if sub_par_only:
        filtered = filtered.loc[filtered["clean_price"] < 100.0]

    snapshot_date = df["snapshot_date"].dropna().iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Snapshot date", str(snapshot_date))
    col2.metric("Gilts in universe", f"{len(df):,}")
    col3.metric("Matching filters", f"{len(filtered):,}")
    col4.metric(
        "CGT-attractive",
        f"{int(filtered['cgt_attractive'].sum()):,}",
        help="Coupon <= 2% and clean price < 100",
    )

    if not filtered.empty and tax_rate > 0:
        best_gilt = filtered.loc[filtered[post_tax_col].idxmax()]
        equiv_rate = best_gilt["equivalent_gross_yield"]
        st.success(
            f"💡 **Tax Savings Highlight:** The highest-yielding gilt in your filter is **{best_gilt['name']}**. "
            f"It gives a post-tax yield of **{best_gilt[post_tax_col]:.2f}%**. "
            f"To get that same return after {int(tax_rate*100)}% tax, a regular savings account "
            f"would need to pay an interest rate of **{equiv_rate:.2f}%**!"
        )

    st.plotly_chart(
        _render_yield_curve(df, set(filtered["isin"]), post_tax_col, max_ttm),
        use_container_width=True,
    )
    st.caption(
        "**How to read this chart:** Every dot is a UK gilt. The large, bright dots "
        "match your filters (e.g. low coupon, short maturity). The small, faded dots "
        "are the rest of the market. The colour shows the tax-free portion of the return "
        "(bright yellow = mostly tax-free capital gains, dark purple = mostly taxable "
        "coupon income). For a higher-rate taxpayer, you generally want to look for large, "
        "bright yellow dots near the top of the curve."
    )

    st.subheader("Gilt table")
    table_cols = [
        "name",
        "isin",
        "coupon",
        "redemption_date",
        "time_to_maturity",
        "clean_price",
        "yield_to_maturity",
        "running_yield",
        "tax_free_return_pct",
        post_tax_col,
        "equivalent_gross_yield",
        "cgt_attractive",
    ]
    table = filtered[table_cols].rename(
        columns={
            "name": "Name",
            "isin": "ISIN",
            "coupon": "Coupon (%)",
            "redemption_date": "Maturity",
            "time_to_maturity": "Years to maturity",
            "clean_price": "Clean price",
            "yield_to_maturity": "GRY (%)",
            "running_yield": "Running yield (%)",
            "tax_free_return_pct": "Tax-free portion (%)",
            post_tax_col: f"Post-tax yield @ {int(round(tax_rate * 100))}% (%)",
            "equivalent_gross_yield": "Equivalent Savings Rate (%)",
            "cgt_attractive": "CGT-attractive",
        }
    )
    st.dataframe(
        table.sort_values(f"Post-tax yield @ {int(round(tax_rate * 100))}% (%)", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Export filtered table (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"gilt_screener_{snapshot_date}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
