from statistics import NormalDist

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="DXB Predictive Six Sigma Dashboard", layout="wide")
CUTOFF = 0.553
NAVY, MUTED, RED = "#21244F", "#696E9F", "#FF5252"

# ---------- Data ----------
@st.cache_data
def load_data():
    df = pd.read_parquet("data/replay_2026-01-03_scored.parquet")
    rules = pd.read_csv("rules.csv")
    sched = pd.to_datetime(df["scheduledinblocktime"])
    actual = pd.to_datetime(df["actualinblocktime"])
    df["eta_min"] = sched.dt.hour * 60 + sched.dt.minute
    df["act_min"] = actual.dt.hour * 60 + actual.dt.minute
    df["eta_txt"] = sched.dt.strftime("%H:%M")
    df["risk_pct"] = (df["p_xgb"] * 100).round().astype(int)
    df["high"] = df["p_xgb"] >= CUTOFF
    # add SHAP values up into the reasons from the rules table
    for reason in rules["reason"].unique():
        drivers = rules.loc[rules["reason"] == reason, "driver"]
        df["r_" + reason] = df[["shap_" + d for d in drivers]].sum(axis=1)
    reason_cols = ["r_" + r for r in rules["reason"].unique()]
    df["main_reason"] = df[reason_cols].idxmax(axis=1).str[2:]
    return df, rules

flights, rules = load_data()

# ---------- Replay clock ----------
if "now" not in st.session_state:
    st.session_state.now = 10 * 60
if "tab" not in st.session_state:
    st.session_state.tab = "Today"

def fmt(m):
    return f"{int(m) // 60:02d}:{int(m) % 60:02d}"

def go_to(tab_name):
    st.session_state.tab = tab_name

now = st.session_state.now
flights["landed"] = flights["act_min"] <= now
flights["mins_to"] = flights["eta_min"] - now
to_come = flights[~flights["landed"]]

# ---------- Header ----------
left, right = st.columns([2, 3])
with left:
    st.markdown("### DXB Predictive Six Sigma Dashboard")
    st.caption(f"Ops Control Manager · Replay day: 3 Jan 2026 · Time now **{fmt(now)}**")
    c1, c2, c3 = st.columns(3)
    if c1.button("- 30 min"):
        st.session_state.now = max(0, now - 30)
        st.rerun()
    if c2.button("+ 30 min"):
        st.session_state.now = min(23 * 60 + 59, now + 30)
        st.rerun()
    if c3.button("Restart day"):
        st.session_state.now = 10 * 60
        st.rerun()
with right:
    st.radio("Tab", ["Today", "Flights", "Flight detail", "Warnings", "Control", "Try a flight"],
             key="tab", horizontal=True, label_visibility="collapsed")
st.divider()
tab = st.session_state.tab

# ---------- TODAY ----------
if tab == "Today":
    landed = flights[flights["landed"]]
    p_late = landed["delayed_15"].mean() if len(landed) else 0
    if 0 < p_late < 1:
        sigma = f"{NormalDist().inv_cdf(1 - p_late) + 1.5:.2f} σ"
    else:
        sigma = "n/a"
    need_action = to_come[to_come["high"] & (to_come["mins_to"] <= 60)]

    k1, k2, k3, k4 = st.columns(4)
    with k1.container(border=True):
        st.metric("Flights today", len(flights))
        st.caption(f"{len(landed)} landed · {len(to_come)} still to come")
    with k2.container(border=True):
        st.metric("High risk, still to come", int(to_come["high"].sum()))
        st.caption(f"Risk score at or above {CUTOFF:.0%}")
    with k3.container(border=True):
        st.metric("Need action now", len(need_action))
        st.caption("High risk, lands within 60 min")
    with k4.container(border=True):
        st.metric("Sigma so far today", sigma)
        st.caption(f"Since 00:00 · {p_late:.0%} of landed flights late · baseline 2.24 σ")

    g1, g2 = st.columns([2, 1])
    with g1.container(border=True):
        st.markdown("**Arrivals by hour**")
        flights["hr"] = flights["eta_min"] // 60
        by_hour = flights.groupby("hr").agg(total=("high", "size"), late=("high", "sum")).reindex(range(24), fill_value=0)
        fig = go.Figure()
        fig.add_bar(x=by_hour.index, y=by_hour["total"] - by_hour["late"], name="Predicted on time", marker_color=MUTED)
        fig.add_bar(x=by_hour.index, y=by_hour["late"], name="Predicted late", marker_color=RED)
        fig.add_vrect(x0=now // 60 - 0.5, x1=now // 60 + 0.5, line_width=2, line_color=NAVY, fillcolor="rgba(0,0,0,0)")
        fig.update_layout(barmode="stack", height=280, margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=-0.2), xaxis_title="Hour", yaxis_title="Flights")
        st.plotly_chart(fig, use_container_width=True)
    with g2.container(border=True):
        st.markdown("**Top reasons today**")
        st.caption("High-risk flights still to come, by main reason")
        top = to_come.loc[to_come["high"], "main_reason"].value_counts()
        fig2 = go.Figure(go.Bar(x=top.values[::-1], y=top.index[::-1], orientation="h", marker_color=NAVY))
        fig2.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with st.container(border=True):
        h1, h2 = st.columns([4, 1])
        h1.markdown("**Next 3 hours: top 5 highest risk**")
        h2.button("See all flights", on_click=go_to, args=("Flights",), use_container_width=True)
        nxt = to_come[to_come["mins_to"].between(-30, 180)].nlargest(5, "p_xgb")
        st.dataframe(
            nxt.rename(columns={"flightnumber": "Flight", "origin_iata": "From", "eta_txt": "ETA",
                                "risk_pct": "Risk %", "main_reason": "Main reason"})[["Flight", "From", "ETA", "Risk %", "Main reason"]],
            hide_index=True, use_container_width=True)

else:
    st.info(f"{tab} tab: coming next.")

st.caption("Student prototype (BUS922). Not affiliated with or endorsed by Dubai Airports. Open data from Dubai Pulse.")