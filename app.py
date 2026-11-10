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
    day_start = pd.Timestamp("2026-01-03")
    df["act_min"] = (actual - day_start).dt.total_seconds() / 60
    df["eta_txt"] = sched.dt.strftime("%H:%M")
    df["risk_pct"] = (df["p_xgb"] * 100).round().astype(int)
    df["high"] = df["p_xgb"] >= CUTOFF
    # add SHAP values up into the reasons from the rules table
    for reason in rules["reason"].unique():
        drivers = rules.loc[rules["reason"] == reason, "driver"]
        df["r_" + reason] = df[["shap_" + d for d in drivers]].sum(axis=1)
    reason_cols = ["r_" + r for r in rules["reason"].unique()]
    df["main_reason"] = df[reason_cols].idxmax(axis=1).str[2:]
    # suggested fixes = top 2 actionable reasons that push the risk up
    action = (rules[rules["actionable"] == 1].drop_duplicates("reason")
              .set_index("reason")[["fix", "owner"]].to_dict("index"))
    def fixes_for(row):
        pushes = sorted([(r, row["r_" + r]) for r in action if row["r_" + r] > 0], key=lambda x: -x[1])
        out = []
        for i, (r, _) in enumerate(pushes[:2]):
            key = f"{row['flightnumber'].replace(' ', '')}-{i + 1}"
            out.append({"key": key, "code": f"WRN-20260103-{key}", "reason": r,
                        "fix": action[r]["fix"], "owner": action[r]["owner"]})
        return out
    df["fixes"] = df.apply(fixes_for, axis=1)
    return df, rules

flights, rules = load_data()
flights = flights.copy()

# ---------- Memory between clicks ----------
for k, v in {"now": 10 * 60, "tab": "Today", "sel": None, "done": {}, "warn": {}, "log": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def fmt(m):
    return f"{int(m) // 60:02d}:{int(m) % 60:02d}"

def go_to(tab_name):
    st.session_state.tab = tab_name

def open_from(table_key, ids_key):
    rows = st.session_state[table_key].selection.rows
    if rows:
        st.session_state.sel = st.session_state[ids_key][rows[0]]
        st.session_state.tab = "Flight detail"

now = st.session_state.now
flights["landed"] = flights["act_min"] <= now
flights["mins_to"] = flights["eta_min"] - now

def fix_status(row):
    if row["landed"]:
        return "Landed"
    if not row["high"]:
        return "Low risk"
    all_done = all(st.session_state.done.get(f["key"]) for f in row["fixes"])
    return "Fix in place" if all_done else "Fix missing"

flights["fix_status"] = flights.apply(fix_status, axis=1)
to_come = flights[~flights["landed"]]

SHOW_COLS = {"flightnumber": "Flight", "origin_iata": "From", "eta_txt": "ETA",
             "risk_pct": "Risk %", "main_reason": "Main reason", "fix_status": "Fix status"}

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
    sigma = f"{NormalDist().inv_cdf(1 - p_late) + 1.5:.2f} σ" if 0 < p_late < 1 else "n/a"
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
        h1.markdown("**Next 3 hours: top 5 highest risk** (click a row to open it)")
        h2.button("See all flights", on_click=go_to, args=("Flights",), use_container_width=True)
        nxt = to_come[to_come["mins_to"].between(-30, 180)].nlargest(5, "p_xgb")
        st.session_state.today_ids = nxt.index.tolist()
        st.dataframe(nxt.rename(columns=SHOW_COLS)[list(SHOW_COLS.values())], hide_index=True,
                     use_container_width=True, key="today_table", on_select=lambda: open_from("today_table", "today_ids"),
                     selection_mode="single-row")

# ---------- FLIGHTS ----------
elif tab == "Flights":
    f1, f2, f3 = st.columns([3, 2, 2])
    show = f1.radio("Show", ["All flights", "High risk only", "Next 3 hours"], horizontal=True)
    fix_filter = f2.selectbox("Fix status", ["All", "Fix missing", "Fix in place", "Low risk", "Landed"])
    search = f3.text_input("Search flight", placeholder="e.g. EK 512")

    view = flights.sort_values("eta_min")
    if show == "High risk only":
        view = view[view["high"]]
    if show == "Next 3 hours":
        view = view[~view["landed"] & view["mins_to"].between(-30, 180)]
    if fix_filter != "All":
        view = view[view["fix_status"] == fix_filter]
    if search:
        view = view[view["flightnumber"].str.replace(" ", "").str.contains(search.replace(" ", ""), case=False)]

    s1, s2 = st.columns([4, 1])
    s1.caption(f"Showing {len(view)} flights · {int(view['high'].sum())} high risk · "
               f"{int((view['fix_status'] == 'Fix missing').sum())} with a fix missing · click a row to open it")
    table = view.rename(columns=SHOW_COLS)[list(SHOW_COLS.values())]
    s2.download_button("Download CSV", table.to_csv(index=False), "flights_3jan2026.csv", "text/csv",
                       use_container_width=True)
    st.session_state.flight_ids = view.index.tolist()
    st.dataframe(table, hide_index=True, use_container_width=True, height=560, key="flight_table",
                 on_select=lambda: open_from("flight_table", "flight_ids"), selection_mode="single-row")

else:
    st.info(f"{tab} tab: coming next.")

st.caption("Student prototype (BUS922). Not affiliated with or endorsed by Dubai Airports. Open data from Dubai Pulse.")