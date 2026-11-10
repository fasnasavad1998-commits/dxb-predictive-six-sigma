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
    df["act_txt"] = actual.dt.strftime("%H:%M")
    df["risk_pct"] = (df["p_xgb"] * 100).round().astype(int)
    df["high"] = df["p_xgb"] >= CUTOFF
    for reason in rules["reason"].unique():
        drivers = rules.loc[rules["reason"] == reason, "driver"]
        df["r_" + reason] = df[["shap_" + d for d in drivers]].sum(axis=1)
    reason_cols = ["r_" + r for r in rules["reason"].unique()]
    df["main_reason"] = df[reason_cols].idxmax(axis=1).str[2:]
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
REASONS = list(rules["reason"].unique())

# ---------- Memory between clicks ----------
for k, v in {"now": 10 * 60, "tab": "Today", "sel": None, "auto": True,
             "done": {}, "warn": {}, "log": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def fmt(m):
    return f"{int(m) // 60:02d}:{int(m) % 60:02d}"

def go_to(tab_name):
    st.session_state.tab = tab_name

def tab_changed():
    if st.session_state.tab == "Flight detail":      # opened from the tab bar -> auto-pick
        st.session_state.sel = None
        st.session_state.auto = True

def open_from(table_key, ids_key):
    rows = st.session_state[table_key].selection.rows
    if rows:
        st.session_state.sel = st.session_state[ids_key][rows[0]]
        st.session_state.auto = False
        st.session_state.tab = "Flight detail"

def record(fx, flight_no, action):
    st.session_state.log.append({"time": fmt(st.session_state.now), "warning_code": fx["code"], "flight": flight_no,
                                 "owner": fx["owner"], "fix": fx["fix"], "action": action})

def send_warning(fx, flight_no):
    if not st.session_state.done.get(fx["key"]) and not st.session_state.warn.get(fx["key"]):
        st.session_state.warn[fx["key"]] = "Sent"
        record(fx, flight_no, "Sent")

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

def actioned(row):
    return all(st.session_state.done.get(f["key"]) or st.session_state.warn.get(f["key"]) for f in row["fixes"])

flights["fix_status"] = flights.apply(fix_status, axis=1)
flights["actioned"] = flights.apply(actioned, axis=1)
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
             key="tab", horizontal=True, label_visibility="collapsed", on_change=tab_changed)
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

# ---------- FLIGHT DETAIL ----------
elif tab == "Flight detail":
    hi_up = to_come[to_come["high"]]
    n_actioned = int(hi_up["actioned"].sum())

    # pick the flight: clicked one, or the most urgent unactioned one (then keep it while the clock runs)
    if st.session_state.sel is None:
        pending = hi_up[~hi_up["actioned"]].sort_values("p_xgb", ascending=False)
        pool = pending if len(pending) else to_come.sort_values("p_xgb", ascending=False)
        st.session_state.sel = pool.index[0] if len(pool) else flights.index[0]
        st.session_state.auto = len(pending) > 0
    f = flights.loc[st.session_state.sel]

    t1, t2, t3 = st.columns([2, 2, 1])
    with t1.container(border=True):
        st.metric("Pending action", f"{len(hi_up) - n_actioned} flights")
        st.caption("High risk, a fix not in place and no warning sent")
    with t2.container(border=True):
        st.metric("Actioned", f"{n_actioned} flights")
        st.caption("Every fix in place or warned")
    t3.button("Back to flight list", on_click=go_to, args=("Flights",), use_container_width=True)

    if st.session_state.auto:
        st.info("Auto-picked the most urgent flight: highest risk with a fix still missing. "
                "It stays here while you work. Click the Flight detail tab again for the next one.")

    a1, a2 = st.columns([3, 1])
    with a1.container(border=True):
        st.markdown(f"#### {f['flightnumber']} from {f['originname']} ({f['origin_iata']})")
        when = f"landed {f['act_txt']}" if f["landed"] else f"in {int(f['mins_to'])} min"
        st.write(f"{f['airlinename']} · aircraft {f['aircraft_iata']} ({f['aircraftregistration']}) · "
                 f"terminal {f['aircraftterminal']} · scheduled {f['eta_txt']} ({when})")
        if f["landed"]:
            outcome = "late" if f["delayed_15"] else "on time"
            right_call = (f["high"] and f["delayed_15"]) or (not f["high"] and not f["delayed_15"])
            st.write(f"**Actual result:** {outcome} ({f['arr_delay_min']:.0f} min). "
                     f"Prediction was {'correct' if right_call else 'wrong'}.")
    with a2.container(border=True):
        band = "High" if f["p_xgb"] >= 0.70 else ("Medium" if f["high"] else "Low")
        st.metric("Risk score", f"{f['risk_pct']}%")
        st.caption(f"{band} risk · XGBoost · cut-off {CUTOFF:.0%}")

    b1, b2 = st.columns([2, 1])
    with b1.container(border=True):
        st.markdown("**Why is it risky?** Red pushes towards late, blue towards on time.")
        vals = pd.Series({r: f["r_" + r] for r in REASONS})
        vals = vals.reindex(vals.abs().sort_values(ascending=False).index)[:6][::-1]
        fig = go.Figure(go.Bar(x=vals.values, y=vals.index, orientation="h",
                               marker_color=[RED if v > 0 else MUTED for v in vals.values]))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="SHAP (log-odds)")
        st.plotly_chart(fig, use_container_width=True)
    with b2.container(border=True):
        st.markdown("**Key facts**")
        st.write(f"Inbound plane departure delay: **{f['knockon_dep_delay_min']:.0f} min**")
        st.write(f"Rotation gap: **{f['rotation_gap_h']:.1f} h**")
        st.write(f"Route late rate (7 days): **{f['route_late_rate_7d']:.0%}**")
        st.write(f"Arrivals in that hour: **{f['hourly_arrivals']:.0f}**")
        st.write(f"Wind **{f['wind_kt']:.0f} kt** · visibility **{f['vis_km']:.0f} km** · {f['temp_c']:.0f}°C")

    with st.container(border=True):
        h1, h2 = st.columns([4, 1])
        h1.markdown("**Suggested fixes**")
        fixes = f["fixes"]
        open_fixes = [x for x in fixes if not st.session_state.done.get(x["key"])]
        if not f["high"]:
            st.write("Below the cut-off. No action needed.")
        elif f["landed"]:
            st.write("Flight has landed. Fixes no longer needed.")
        else:
            all_sent = open_fixes and all(st.session_state.warn.get(x["key"]) for x in open_fixes)
            if not open_fixes:
                h2.button("All fixes in place", disabled=True, use_container_width=True)
            elif all_sent:
                h2.button("Warnings sent", disabled=True, use_container_width=True)
            elif h2.button("Send all warnings", type="primary", use_container_width=True):
                for x in open_fixes:
                    send_warning(x, f["flightnumber"])
                st.rerun()
            for x in fixes:
                done = st.session_state.done.get(x["key"])
                status = st.session_state.warn.get(x["key"])
                r1, r2, r3, r4 = st.columns([4, 2, 1.3, 1.3])
                r1.write(f"{x['fix']}  \n`{x['code']}` · reason: {x['reason']}")
                r2.write(x["owner"])
                r3.write("In place" if done else "Not in place")
                if done:
                    r4.button("Not needed", key="b" + x["key"], disabled=True, use_container_width=True)
                elif status:
                    r4.button(f"Warning {status.lower()}", key="b" + x["key"], disabled=True, use_container_width=True)
                elif r4.button("Send warning", key="b" + x["key"], use_container_width=True):
                    send_warning(x, f["flightnumber"])
                    st.rerun()

# ---------- WARNINGS ----------
elif tab == "Warnings":
    WINDOW = 60          # warning opens when a high-risk flight is this close (min)
    OVERDUE = 15         # sent but not acknowledged after this many min

    def open_flight(idx):
        st.session_state.sel = idx
        st.session_state.auto = False
        st.session_state.tab = "Flight detail"

    def due(m):
        return f"{int(m)} min to go" if m >= 0 else f"{int(-m)} min overdue"

    sent_at = {}
    for e in st.session_state.log:
        if e["action"] == "Sent" and e["warning_code"] not in sent_at:
            h, m = e["time"].split(":")
            sent_at[e["warning_code"]] = int(h) * 60 + int(m)

    items = []
    for idx, row in to_come[to_come["high"]].iterrows():
        for fx in row["fixes"]:
            status = st.session_state.warn.get(fx["key"])
            if st.session_state.done.get(fx["key"]):
                continue
            if row["mins_to"] > WINDOW and not status:
                continue
            status = status or "New"
            if status == "Sent" and now - sent_at.get(fx["code"], now) > OVERDUE:
                status = "Overdue"
            items.append({"idx": idx, "row": row, "fx": fx, "status": status})

    counts = pd.Series([i["status"] for i in items]).value_counts()
    k = st.columns(4)
    for col, name in zip(k, ["New", "Sent", "Acknowledged", "Overdue"]):
        with col.container(border=True):
            st.metric(name, int(counts.get(name, 0)))
    st.caption(f"A warning opens when a flight is high risk, a fix is not in place and it lands within {WINDOW} min. "
               f"Overdue = sent but not acknowledged within {OVERDUE} min. Mark fix done closes it.")

    w1, w2 = st.columns([4, 1])
    w1.markdown("**Open warnings by owner**")
    log = pd.DataFrame(st.session_state.log)
    w2.download_button(f"Download log (CSV) · {len(log)} entries",
                       log.to_csv(index=False) if len(log) else "time,warning_code,flight,owner,fix,action\n",
                       "warning_log_3jan2026.csv", "text/csv", use_container_width=True)

    owners = list(rules.loc[rules["actionable"] == 1, "owner"].unique())
    cols = st.columns(len(owners))
    for col, owner in zip(cols, owners):
        mine = sorted([i for i in items if i["fx"]["owner"] == owner],
                      key=lambda i: (i["row"]["mins_to"] < 0, abs(i["row"]["mins_to"])))
        with col.container(border=True):
            st.markdown(f"**{owner}**  \n{len(mine)} open")
            if not mine:
                st.caption("All clear")
            for i in mine:
                row, fx = i["row"], i["fx"]
                with st.container(border=True):
                    st.caption(fx["code"])
                    st.markdown(f"**{row['flightnumber']}** · scheduled {row['eta_txt']} · **{due(row['mins_to'])}**")
                    st.write(fx["fix"])
                    st.write(f"Status: **{i['status']}**")
                    b1, b2 = st.columns(2)
                    if i["status"] == "New" and b1.button("Send", key="s" + fx["key"], use_container_width=True):
                        send_warning(fx, row["flightnumber"])
                        st.rerun()
                    if i["status"] in ("Sent", "Overdue") and b1.button("Acknowledge", key="a" + fx["key"], use_container_width=True):
                        st.session_state.warn[fx["key"]] = "Acknowledged"
                        record(fx, row["flightnumber"], "Acknowledged")
                        st.rerun()
                    if b2.button("Mark fix done", key="d" + fx["key"], use_container_width=True):
                        st.session_state.done[fx["key"]] = True
                        record(fx, row["flightnumber"], "Fix done")
                        st.rerun()
                    st.button("Open flight", key="o" + fx["key"], on_click=open_flight, args=(i["idx"],),
                              use_container_width=True)
else:
    st.info(f"{tab} tab: coming next.")

st.caption("Student prototype (BUS922). Not affiliated with or endorsed by Dubai Airports. Open data from Dubai Pulse.")