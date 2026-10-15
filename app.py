"""Forecast to Order - Streamlit dashboard.

Run with:  streamlit run app.py

Forecasts are precomputed by ``run_pipeline.py``; this app only reruns the
inventory simulation (well under a second) when the sidebar settings change.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from run_pipeline import load_sim_inputs
from src.backtest import MODEL_COLUMNS
from src.config import DEFAULT_CONFIG
from src.simulation import (AverageBasedPolicy, ForecastBasedPolicy, SimulationInputs, assign_item_params,
                            make_policies, simulate, summarize)

OUT = Path(__file__).resolve().parent / "outputs"

# Validated palette (see README): policies are categorical slots 1-2, status colours are reserved.
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
POLICY_COLORS = {"Forecast based": "#2a78d6", "Average based": "#eb6834"}
BAND = "rgba(42,120,214,0.16)"
GOOD, CRITICAL, WARNING = "#006300", "#d03b3b", "#fab219"


def rgba(hex_color: str, alpha: float) -> str:
    """CSS rgba() string from a #rrggbb colour."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

st.set_page_config(page_title="Forecast to Order", page_icon="📦", layout="wide")

st.markdown(
    f"""
<style>
  .block-container {{ padding-top: 2rem; max-width: 1400px; }}
  h1, h2, h3 {{ letter-spacing: -0.01em; }}
  .subtitle {{ color: {INK_2}; margin-top: -0.6rem; margin-bottom: 1.2rem; font-size: 1.02rem; }}
  .headline {{ background: #f3f7fd; border: 1px solid rgba(42,120,214,0.25); border-radius: 10px;
               padding: 14px 18px; margin-bottom: 14px; font-size: 1.05rem; color: {INK}; }}
  .headline b.good {{ color: {GOOD}; }}
  .headline b.bad {{ color: {CRITICAL}; }}
  .kpi {{ border: 1px solid rgba(11,11,11,0.10); border-radius: 10px; padding: 14px 16px; background: {SURFACE};
          height: 100%; }}
  .kpi .label {{ color: {INK_2}; font-size: 0.86rem; font-weight: 600; text-transform: uppercase;
                 letter-spacing: 0.03em; }}
  .kpi .help {{ color: {MUTED}; font-size: 0.78rem; margin-bottom: 8px; }}
  .kpi .row {{ display: flex; justify-content: space-between; align-items: baseline; margin: 2px 0; }}
  .kpi .pol {{ color: {INK_2}; font-size: 0.86rem; }}
  .kpi .pol i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }}
  .kpi .val {{ font-size: 1.15rem; font-weight: 600; color: {INK}; font-variant-numeric: tabular-nums; }}
  .kpi .delta {{ margin-top: 8px; padding-top: 8px; border-top: 1px solid {GRID}; font-weight: 700; font-size: 1.02rem; }}
  .kpi .delta.good {{ color: {GOOD}; }}
  .kpi .delta.bad {{ color: {CRITICAL}; }}
  .kpi .delta.flat {{ color: {INK_2}; }}
  .note {{ color: {INK_2}; font-size: 0.88rem; }}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- data
@st.cache_resource
def get_inputs() -> SimulationInputs:
    return load_sim_inputs(OUT / "sim_inputs.npz")


@st.cache_data
def get_tables() -> dict[str, pd.DataFrame]:
    return {
        "fold": pd.read_parquet(OUT / "forecasts_backtest.parquet"),
        "overall": pd.read_csv(OUT / "metrics_overall.csv"),
        "per_fold": pd.read_csv(OUT / "metrics_per_fold.csv"),
        "per_item": pd.read_csv(OUT / "metrics_per_item.csv"),
        "importance": pd.read_csv(OUT / "feature_importance.csv"),
    }


@st.cache_data
def get_run_info() -> dict:
    return json.loads((OUT / "run_info.json").read_text())


@st.cache_data(show_spinner="Re-running the inventory simulation...")
def run_sim(service_level: float, lead_time: int, shelf_mult: float, expiry_aware: bool, safety_days: int) -> pd.DataFrame:
    cfg = replace(DEFAULT_CONFIG.sim, service_level=service_level, lead_time=lead_time,
                  shelf_life_multiplier=shelf_mult, expiry_aware=expiry_aware, safety_days=safety_days)
    return simulate(get_inputs(), cfg, make_policies(cfg))


@st.cache_data(show_spinner="Tracing the cash vs. service trade-off...")
def run_frontier(lead_time: int, shelf_mult: float, expiry_aware: bool) -> pd.DataFrame:
    cfg = replace(DEFAULT_CONFIG.sim, lead_time=lead_time, shelf_life_multiplier=shelf_mult, expiry_aware=expiry_aware)
    policies = {}
    for d in (0, 3, 5, 7, 10, 14, 21):
        policies[f"Average based|{d}"] = AverageBasedPolicy(lead_time + cfg.review_period + d, cfg.review_period)
    for sl in (0.80, 0.90, 0.95, 0.98, 0.99, 0.995):
        policies[f"Forecast based|{sl}"] = ForecastBasedPolicy(sl, cfg.review_period, expiry_aware)
    s = summarize(simulate(get_inputs(), cfg, policies))
    s[["policy", "setting"]] = s["policy"].str.split("|", expand=True)
    s["setting"] = s["setting"].astype(float)
    return s


if not (OUT / "sim_inputs.npz").exists():
    st.error("No pipeline outputs found. Run `python run_pipeline.py` first, then reload this page.")
    st.stop()

inputs = get_inputs()
tables = get_tables()
info = get_run_info()

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Planning settings")
    service_level = st.select_slider(
        "Target service level",
        options=[0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995],
        value=DEFAULT_CONFIG.sim.service_level,
        format_func=lambda v: f"{v:.1%}".replace(".0%", "%"),
        help="How often the forecast-based plan aims to get through an order cycle without running out. "
             "Higher means more safety stock.",
    )
    lead_time = st.slider("Supplier lead time (days)", 1, 14, DEFAULT_CONFIG.sim.lead_time,
                          help="Days between placing an order and receiving it.")
    shelf_mult = st.select_slider(
        "Shelf life", options=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0], value=1.0,
        format_func=lambda v: f"{v:g}x", help="Scales every product's shelf life (1x = 30, 60, 90 or 180 days).",
    )
    with st.expander("Advanced"):
        safety_days = st.slider("Extra days of cover (average-based rule)", 0, 21, DEFAULT_CONFIG.sim.safety_days,
                                help="The average-based rule orders up to average daily sales x "
                                     "(lead time + 7-day order cycle + these extra days).")
        expiry_aware = st.toggle("Forecast plan respects expiry dates", value=True,
                                 help="Never order more than is expected to sell before it expires.")
    st.divider()
    st.caption(
        f"Data: Walmart M5, store {info['store_id']}, top {info['n_items']} products by volume. "
        f"Simulated period {info['simulation_start']} to {info['simulation_end']} ({info['simulation_days']} days). "
        "Orders are placed weekly. Shelf lives and costs are simulated."
    )

log = run_sim(service_level, lead_time, shelf_mult, expiry_aware, safety_days)
summary = summarize(log).set_index("policy")
avg, fc = summary.loc["Average based"], summary.loc["Forecast based"]
days = info["simulation_days"]

# ----------------------------------------------------------------------------- header + KPIs
st.title("Forecast to Order")
st.markdown(
    f"<div class='subtitle'>Expiry-aware inventory planning for {info['n_items']} fast-moving products in one store. "
    f"What happens to lost sales, expired stock and cash when orders follow a demand forecast instead of a "
    f"simple average? Results replay {days} real days of sales ({info['simulation_start']} to {info['simulation_end']}).</div>",
    unsafe_allow_html=True,
)


def money(x: float, signed: bool = False) -> str:
    s = f"${abs(x):,.0f}"
    if signed:
        return ("-" if x < 0 else "+") + s
    return s


def md_money(x: float) -> str:
    """money() for plain Markdown, where a bare $ would start a LaTeX formula."""
    return money(x).replace("$", r"\$")


saving = avg["total_cost"] - fc["total_cost"]
cash_freed = avg["avg_inventory_value"] - fc["avg_inventory_value"]
yearly = saving * 365 / days
if saving >= 0:
    msg = (f"Ordering from the forecast <b class='good'>saves {money(saving)}</b> in lost profit, waste and holding cost "
           f"over {days} days ({saving / avg['total_cost']:.0%} lower, about {money(yearly)} a year for these "
           f"{info['n_items']} products)")
else:
    msg = (f"With these settings forecast-based ordering <b class='bad'>costs {money(-saving)} more</b> over {days} days "
           f"than the average-based rule")
msg += (f", while tying up <b class='{'good' if cash_freed >= 0 else 'bad'}'>{money(abs(cash_freed))} "
        f"{'less' if cash_freed >= 0 else 'more'}</b> cash in stock on an average day.")
st.markdown(f"<div class='headline'>{msg}</div>", unsafe_allow_html=True)


def kpi(label: str, help_: str, a: float, f: float, fmt, lower_is_better: bool = True, delta_fmt=None) -> str:
    diff = f - a
    good = diff < 0 if lower_is_better else diff > 0
    cls = "flat" if abs(diff) < 1e-9 else ("good" if good else "bad")
    dtext = (delta_fmt or fmt)(abs(diff))
    if cls == "flat":
        delta = "No difference"
    else:
        word = ("lower" if diff < 0 else "higher")
        delta = f"{'▼' if diff < 0 else '▲'} {dtext} {word} with forecast"
    sw = lambda p: f"<i style='background:{POLICY_COLORS[p]}'></i>"  # noqa: E731
    return (f"<div class='kpi'><div class='label'>{label}</div><div class='help'>{help_}</div>"
            f"<div class='row'><span class='pol'>{sw('Average based')}Average based</span><span class='val'>{fmt(a)}</span></div>"
            f"<div class='row'><span class='pol'>{sw('Forecast based')}Forecast based</span><span class='val'>{fmt(f)}</span></div>"
            f"<div class='delta {cls}'>{delta}</div></div>")


cols = st.columns(4)
cards = [
    kpi("Lost sales", f"Revenue missed because shelves were empty ({int(avg['stockout_days'])} vs "
        f"{int(fc['stockout_days'])} product-days out of stock)", avg["lost_sales_value"], fc["lost_sales_value"], money),
    kpi("Expired stock", "Purchase value of units thrown away after their expiry date",
        avg["expired_value"], fc["expired_value"], money),
    kpi("Cash tied up in stock", "Average value of inventory on the shelf (at cost)",
        avg["avg_inventory_value"], fc["avg_inventory_value"], money),
    kpi("Fill rate", "Share of customer demand served from stock", avg["fill_rate"], fc["fill_rate"],
        lambda v: f"{v:.1%}", lower_is_better=False, delta_fmt=lambda v: f"{v * 100:.1f} pts"),
]
for c, html in zip(cols, cards):
    c.markdown(html, unsafe_allow_html=True)
st.markdown(
    f"<p class='note' style='margin-top:10px'>Total cost = lost profit on missed sales + expired stock + cost of holding "
    f"stock ({DEFAULT_CONFIG.sim.annual_holding_rate:.0%} a year). Average based: {money(avg['total_cost'])} · "
    f"Forecast based: {money(fc['total_cost'])}.</p>",
    unsafe_allow_html=True,
)


def style_fig(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=90, b=10), plot_bgcolor=SURFACE, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK_2, size=13), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)),
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED))
    return fig


tab_item, tab_trade, tab_acc, tab_how = st.tabs(
    ["Product drilldown", "Cash vs. service trade-off", "Model accuracy", "How it works"])

# ----------------------------------------------------------------------------- drilldown
with tab_item:
    params = {p.item_id: p for p in assign_item_params(
        inputs.items, inputs.prices, replace(DEFAULT_CONFIG.sim, shelf_life_multiplier=shelf_mult))}
    per_item = summarize(log, by=["item_id"])
    ranking = (per_item.pivot(index="item_id", columns="policy", values="total_cost")
               .assign(gain=lambda d: d["Average based"] - d["Forecast based"]).sort_values("gain", ascending=False))
    c1, c2 = st.columns([2, 3])
    # Default to the biggest saver among products with a clean recent history (no long gaps in recorded sales).
    recent = {it: inputs.history[inputs.items.index(it), -56:] for it in ranking.index}
    clean = [it for it in ranking.index if (recent[it] > 0).mean() > 0.9]
    default_idx = ranking.index.get_loc(clean[0]) if clean else 0
    item = c1.selectbox("Product", ranking.index.tolist(), index=default_idx,
                        format_func=lambda i: f"{i}  ·  shelf life {params[i].shelf_life} d  ·  ${params[i].price:.2f}",
                        help="Sorted by how much the forecast-based plan saves on this product.")
    p = params[item]
    c2.markdown(
        f"<p class='note' style='margin-top:2.1rem'>Selling price ${p.price:.2f} · unit cost ${p.unit_cost:.2f} · "
        f"shelf life {p.shelf_life} days · lead time {lead_time} days · orders placed weekly</p>",
        unsafe_allow_html=True,
    )

    # Sales vs forecast
    fold = tables["fold"]
    f_item = fold[fold["item_id"] == item].sort_values("date")
    idx = inputs.items.index(item)
    hist_days = 28
    hist_dates = pd.to_datetime(inputs.dates[0]) - pd.to_timedelta(np.arange(hist_days, 0, -1), unit="D")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f_item["date"], y=f_item["p90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=f_item["date"], y=f_item["p10"], fill="tonexty", fillcolor=BAND, line=dict(width=0),
                             name="Likely range (80%)", hovertemplate="%{y:.0f}"))
    fig.add_trace(go.Scatter(x=hist_dates, y=inputs.history[idx, -hist_days:], name="Sales before the test period",
                             line=dict(color=MUTED, width=1.5), hovertemplate="%{y:.0f}"))
    fig.add_trace(go.Scatter(x=f_item["date"], y=f_item["y"], name="Actual sales", line=dict(color=INK, width=2),
                             hovertemplate="%{y:.0f}"))
    fig.add_trace(go.Scatter(x=f_item["date"], y=f_item["lgbm"], name="Forecast", line=dict(color=POLICY_COLORS["Forecast based"], width=2),
                             hovertemplate="%{y:.1f}"))
    fig.add_trace(go.Scatter(x=f_item["date"], y=f_item["ma28"], name="28-day average", line=dict(color=POLICY_COLORS["Average based"], width=2, dash="dot"),
                             hovertemplate="%{y:.1f}"))
    for d in f_item.groupby("fold")["date"].min():
        fig.add_vline(x=d, line=dict(color=AXIS, width=1, dash="dash"))
    fig.update_layout(title=dict(text="Daily units sold vs. forecast (each dashed line = a new 4-week forecast)",
                                 font=dict(size=15, color=INK)))
    fig.update_yaxes(title_text="Units per day", rangemode="tozero")
    st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    # Inventory per policy
    li = log[log["item_id"] == item]
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                         subplot_titles=[f"{pol} ordering" for pol in POLICY_COLORS][::-1])
    for r, pol in enumerate(["Average based", "Forecast based"], start=1):
        d = li[li["policy"] == pol]
        color = POLICY_COLORS[pol]
        fig2.add_trace(go.Scatter(x=d["date"], y=d["on_hand"], name=f"Stock on hand ({pol.lower()})", line=dict(color=color, width=2, shape="hv"),
                                  fill="tozeroy", fillcolor=rgba(color, 0.12),
                                  hovertemplate="%{y:.0f} units"), row=r, col=1)
        so = d[d["lost"] > 0]
        fig2.add_trace(go.Scatter(x=so["date"], y=np.zeros(len(so)), mode="markers", name="Out of stock",
                                  marker=dict(symbol="x", size=9, color=CRITICAL, line=dict(width=1, color=CRITICAL)),
                                  customdata=so["lost"], hovertemplate="Out of stock: %{customdata:.0f} units lost",
                                  showlegend=r == 1, legendgroup="so"), row=r, col=1)
        ex = d[d["expired"] > 0]
        fig2.add_trace(go.Scatter(x=ex["date"], y=ex["on_hand"], mode="markers", name="Stock expired",
                                  marker=dict(symbol="triangle-down", size=12, color=WARNING, line=dict(width=1, color="#8a5a00")),
                                  customdata=ex["expired"], hovertemplate="Expired: %{customdata:.0f} units thrown away",
                                  showlegend=r == 1, legendgroup="ex"), row=r, col=1)
        fig2.update_yaxes(title_text="Units", rangemode="tozero", row=r, col=1)
    fig2.update_layout(title=dict(text="Stock on the shelf, day by day", font=dict(size=15, color=INK)))
    fig2.update_annotations(font=dict(size=13, color=INK_2))
    style_fig(fig2, 540).update_layout(legend=dict(yanchor="top", y=-0.1), margin=dict(b=70))
    st.plotly_chart(fig2, use_container_width=True)

    pi = per_item[per_item["item_id"] == item].set_index("policy")
    show = pd.DataFrame({
        "Days out of stock": pi["stockout_days"].astype(int),
        "Lost sales": pi["lost_sales_value"].map(money),
        "Expired units": pi["expired_units"].astype(int),
        "Expired value": pi["expired_value"].map(money),
        "Avg. cash in stock": pi["avg_inventory_value"].map(money),
        "Fill rate": pi["fill_rate"].map(lambda v: f"{v:.1%}"),
        "Total cost": pi["total_cost"].map(money),
    }).loc[["Average based", "Forecast based"]].rename_axis("Ordering")
    st.dataframe(show, use_container_width=True)

# ----------------------------------------------------------------------------- trade-off
with tab_trade:
    st.markdown(
        "Every ordering rule has a dial: keep more stock and you lose fewer sales but tie up more cash and risk more "
        "waste. A fair comparison looks at the whole curve, not one setting. **Lower and further left is better.**"
    )
    fr = run_frontier(lead_time, shelf_mult, expiry_aware)
    fig3 = go.Figure()
    for pol, color in POLICY_COLORS.items():
        d = fr[fr["policy"] == pol].sort_values("avg_inventory_value")
        labels = ([f"+{s:g} d" for s in d["setting"]] if pol == "Average based"
                  else [f"{s:.1%}".replace(".0%", "%") for s in d["setting"]])
        fig3.add_trace(go.Scatter(
            x=d["avg_inventory_value"], y=d["lost_sales_value"], mode="lines+markers+text", name=pol,
            text=labels, textposition="top right" if pol == "Average based" else "bottom left", textfont=dict(color=INK_2, size=11),
            line=dict(color=color, width=2), marker=dict(size=9, color=color, line=dict(width=2, color=SURFACE)),
            customdata=np.column_stack([d["expired_value"], d["fill_rate"] * 100, d["total_cost"]]),
            hovertemplate=("Cash in stock $%{x:,.0f}<br>Lost sales $%{y:,.0f}<br>Expired $%{customdata[0]:,.0f}"
                           "<br>Fill rate %{customdata[1]:.1f}%<br>Total cost $%{customdata[2]:,.0f}<extra>" + pol + "</extra>"),
        ))
    for pol in POLICY_COLORS:
        cur = summary.loc[pol]
        fig3.add_trace(go.Scatter(x=[cur["avg_inventory_value"]], y=[cur["lost_sales_value"]], mode="markers",
                                  marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(width=2, color=INK)),
                                  name="Current settings", showlegend=pol == "Forecast based", hoverinfo="skip"))
    fig3.update_layout(title=dict(text="Cash tied up vs. sales lost (labels: safety setting)", font=dict(size=15, color=INK)),
                       hovermode="closest")
    fig3.update_xaxes(title_text="Average cash tied up in stock ($)", tickprefix="$", showgrid=True, gridcolor=GRID)
    fig3.update_yaxes(title_text=f"Lost sales over {days} days ($)", tickprefix="$")
    st.plotly_chart(style_fig(fig3, 480), use_container_width=True)

    # Matched-service comparison: cheapest stock level reaching the forecast plan's fill rate.
    a = fr[fr["policy"] == "Average based"].sort_values("avg_inventory_value")
    target_fill = fc["fill_rate"]
    inv_needed = np.interp(target_fill, a["fill_rate"].cummax(), a["avg_inventory_value"]) if target_fill <= a["fill_rate"].max() else None
    if inv_needed is not None:
        st.markdown(
            f"To reach the forecast plan's **{target_fill:.1%} fill rate**, the average-based rule needs about "
            f"**{md_money(inv_needed)}** of stock on hand versus **{md_money(fc['avg_inventory_value'])}**: "
            f"**{1 - fc['avg_inventory_value'] / inv_needed:.0%} less cash** for the same service."
        )
    else:
        st.markdown(f"The average-based rule cannot reach the forecast plan's {target_fill:.1%} fill rate "
                    "at any of the tested cover levels.")

# ----------------------------------------------------------------------------- accuracy
with tab_acc:
    st.markdown(
        "Rolling-origin backtest: the model is retrained at the start of each 4-week window using only data "
        "available at that moment, then forecasts the next 28 days. Four windows cover the final 16 weeks."
    )
    ov = tables["overall"].iloc[0]
    rows = []
    for col, name in MODEL_COLUMNS.items():
        rows.append({"Method": name, "WAPE": ov[f"wape_{col}"], "Bias": ov[f"bias_{col}"]})
    acc = pd.DataFrame(rows)
    m1, m2, m3 = st.columns(3)
    best = acc.loc[acc["Method"] == MODEL_COLUMNS["lgbm"], "WAPE"].iloc[0]
    base = acc.loc[acc["Method"] == MODEL_COLUMNS["ma28"], "WAPE"].iloc[0]
    m1.metric("WAPE, LightGBM", f"{best:.1%}", f"{(best - base) * 100:+.1f} pts vs 28-day average", delta_color="inverse")
    m2.metric("Bias, LightGBM", f"{ov['bias_lgbm']:+.1%}", help="Positive = forecasts too high on average.")
    m3.metric("P10-P90 coverage (target 80%)", f"{ov['coverage_p10_p90']:.1%}",
              help="Share of actual days falling inside the 80% forecast range.")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Overall")
        st.dataframe(acc.style.format({"WAPE": "{:.1%}", "Bias": "{:+.1%}"}), hide_index=True, use_container_width=True)
        st.caption("WAPE = total absolute error / total units sold (lower is better). "
                   "Bias = (forecast - actual) / actual.")
        pf = tables["per_fold"]
        fig4 = go.Figure()
        for col, color in [("ma28", POLICY_COLORS["Average based"]), ("lgbm", POLICY_COLORS["Forecast based"])]:
            fig4.add_trace(go.Bar(x=[f"Window {i + 1}" for i in pf["fold"]], y=pf[f"wape_{col}"], name=MODEL_COLUMNS[col],
                                  marker=dict(color=color, line=dict(width=2, color=SURFACE)),
                                  hovertemplate="%{y:.1%}"))
        fig4.update_layout(title=dict(text="WAPE per backtest window", font=dict(size=15, color=INK)), barmode="group",
                           bargap=0.35)
        fig4.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_fig(fig4, 330), use_container_width=True)
    with c2:
        st.subheader("What drives the forecast")
        imp = tables["importance"].head(12).iloc[::-1]
        fig5 = go.Figure(go.Bar(x=imp["gain"], y=imp["feature"], orientation="h",
                                marker=dict(color=POLICY_COLORS["Forecast based"]), hovertemplate="%{x:.1%}<extra></extra>"))
        fig5.update_layout(title=dict(text="Feature importance (share of gain, mean model)", font=dict(size=15, color=INK)),
                           hovermode="closest")
        fig5.update_xaxes(tickformat=".0%", showgrid=True, gridcolor=GRID)
        st.plotly_chart(style_fig(fig5, 430), use_container_width=True)
    st.subheader("Per product")
    pit = tables["per_item"][["item_id", "actual_units", "wape_ma28", "wape_snaive", "wape_lgbm", "bias_lgbm", "coverage_p10_p90"]]
    pit = pit.rename(columns={"item_id": "Product", "actual_units": "Units sold", "wape_ma28": "WAPE 28-day avg",
                              "wape_snaive": "WAPE seasonal naive", "wape_lgbm": "WAPE LightGBM",
                              "bias_lgbm": "Bias LightGBM", "coverage_p10_p90": "P10-P90 coverage"})
    st.dataframe(pit.style.format({"Units sold": "{:,.0f}", "WAPE 28-day avg": "{:.1%}", "WAPE seasonal naive": "{:.1%}",
                                   "WAPE LightGBM": "{:.1%}", "Bias LightGBM": "{:+.1%}", "P10-P90 coverage": "{:.0%}"}),
                 hide_index=True, use_container_width=True, height=420)

# ----------------------------------------------------------------------------- how it works
with tab_how:
    st.markdown(f"""
**1. Forecast.** One LightGBM model learns from all {info['n_items']} products at once: recent sales, weekly patterns,
holidays, food-stamp (SNAP) days and price changes. It gives a most-likely value and a likely range (P10 to P90) for
each of the next 28 days, and is refreshed every day with the latest sales.

**2. Order.** Once a week each product is topped up to cover the {lead_time}-day lead time plus the next week:

* **Average based** (today's common practice): order up to *average daily sales over the last 28 days x
  ({lead_time} + 7 + {safety_days}) days*.
* **Forecast based**: order up to *forecast demand for the next {lead_time + 7} days + safety stock*. Safety stock
  is sized from the forecast's own uncertainty for the chosen service level, and is capped so the order can sell
  before it expires.

**3. Replay.** Both plans are replayed against what customers actually bought, day by day. Stock arrives after the
lead time, the soonest-expiring units are sold first, expired units are thrown away, and demand that finds an empty
shelf is lost.

**Honest caveats.** Shelf lives (30/60/90/180 days) and costs (unit cost = 70% of price, holding = 25% a year) are
simulated; M5 has no expiry data. Recorded sales are not true demand: when the store ran out, sales show zero. This is
one store and its 50 best sellers, mostly food, standing in for pharmacy products.
""")
