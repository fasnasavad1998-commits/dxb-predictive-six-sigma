import json
from statistics import NormalDist

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from model_utils import load_model, predict

st.set_page_config(page_title="DXB Predictive Six Sigma Dashboard", layout="wide")
CUTOFF = 0.553

# ---------- Colours (from the Dubai Airports site palette) ----------
NAVY = "#2C2D65"      # header, Send, primary buttons
PURPLE = "#554CAA"    # links, on-time bars, Open flight
LAVENDER = "#D1D4FA"  # warning cards, info boxes
PAGE = "#EDEEFD"      # page background
YELLOW = "#FED100"    # Acknowledge, warning sent
GREEN_BG, GREEN_TX = "#C6DECA", "#3C3C64"   # Mark fix done, all clear
RED = "#B42318"       # late, high risk, out of limits
AMBER = "#B45309"     # medium risk
MUTED = NAVY          # "on time" / "lowers the risk" parts of charts

pio.templates["dxb"] = go.layout.Template(layout=dict(
    font=dict(family="Readex Pro, sans-serif", size=14, color=NAVY),
    paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
    xaxis=dict(gridcolor="#E6E8FA", zerolinecolor="#C9CCF5"),
    yaxis=dict(gridcolor="#E6E8FA", zerolinecolor="#C9CCF5")))
pio.templates.default = "plotly_white+dxb"


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
    df["risk_pct"] = (df["p_xgb"] * 100).astype(int)          # round down, so 69.5% shows as 69 (medium), not 70
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
    cells = st.session_state[table_key].selection.cells
    if cells and cells[0][1] == "Flight":          # only a click on the flight number opens it
        st.session_state.sel = st.session_state[ids_key][cells[0][0]]
        st.session_state.auto = False
        st.session_state.tab = "Flight detail"


def record(fx, flight_no, action):
    st.session_state.log.append({"time": fmt(st.session_state.now), "warning_code": fx["code"], "flight": flight_no,
                                 "owner": fx["owner"], "fix": fx["fix"], "action": action})


def send_warning(fx, flight_no):
    if not st.session_state.done.get(fx["key"]) and not st.session_state.warn.get(fx["key"]):
        st.session_state.warn[fx["key"]] = "Sent"
        record(fx, flight_no, "Sent")


def plural(n, word):
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


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


def style_table(t):
    def risk_style(v):
        if v >= 70:
            return f"color:{RED};font-weight:700"
        if v >= 55:
            return f"color:{AMBER};font-weight:700"
        return "color:#2E6B34;font-weight:700"
    status_colour = {"Fix missing": f"color:{RED};font-weight:600", "Fix in place": f"color:{GREEN_TX};font-weight:600",
                     "Low risk": f"color:{GREEN_TX}", "Landed": "color:#8A94A3"}
    return (t.style.map(lambda v: f"color:{NAVY};font-weight:700;text-decoration:underline", subset=["Flight"])
             .map(risk_style, subset=["Risk %"])
             .map(lambda v: status_colour.get(v, ""), subset=["Fix status"])
             .format({"Risk %": lambda v: ("● " if v >= CUTOFF * 100 else "○ ") + f"{v}%"}))


# ---------- Card rows (like the DXB flight list) ----------
BUBBLE = {"hi": ("#F2E1E1", "#862422"), "md": ("#FFF1DC", AMBER), "lo": (GREEN_BG, GREEN_TX), "nt": (PAGE, NAVY)}
STATUS_KIND = {"Fix missing": "hi", "Fix in place": "lo", "Low risk": "lo", "Landed": "nt"}
ROW_W = [1.1, 0.8, 0.8, 0.9, 1.8, 1.3, 0.35]


def bubble(text, kind):
    bg, fg = BUBBLE[kind]
    return (f"<span style='display:inline-block;background:{bg};color:{fg};border-radius:8px;"
            f"padding:3px 12px;font-weight:600'>{text}</span>")


def risk_kind(p):
    return "hi" if p >= 0.70 else ("md" if p >= CUTOFF else "lo")


def risk_block(p, title, size=52):
    """Big risk number in its colour bubble, with the band label underneath."""
    kind = risk_kind(p)
    bg, fg = BUBBLE[kind]
    band = {"hi": "High risk", "md": "Medium risk", "lo": "Low risk"}[kind]
    return (f"<div style='font-size:13px;color:{PURPLE};margin-bottom:8px'>{title}</div>"
            f"<div style='display:inline-block;background:{bg};color:{fg};border-radius:14px;padding:2px 18px;"
            f"font-size:{size}px;font-weight:700;line-height:1.2'>{p * 100:.1f}%</div>"
            f"<div style='margin-top:10px'><span style='display:inline-block;background:{bg};color:{fg};"
            f"border-radius:999px;padding:3px 14px;font-weight:700;font-size:14px'>{band}</span></div>")


def open_flight_idx(idx):
    st.session_state.sel = idx
    st.session_state.auto = False
    st.session_state.tab = "Flight detail"


def flight_rows(df, prefix):
    head = st.columns(ROW_W, vertical_alignment="center")
    for col, title in zip(head, ["Flight", "From", "ETA", "Risk %", "Main reason", "Fix status", ""]):
        col.markdown(f"<span style='font-weight:600;color:{NAVY}'>{title}</span>", unsafe_allow_html=True)
    for idx, r in df.iterrows():
        with st.container(key=f"row_{prefix}_{idx}"):
            c = st.columns(ROW_W, vertical_alignment="center")
            c[0].markdown(f"**{r['flightnumber']}**")
            c[1].markdown(r["origin_iata"])
            c[2].markdown(r["eta_txt"])
            c[3].markdown(bubble(f"{r['risk_pct']}%", risk_kind(r["p_xgb"])), unsafe_allow_html=True)
            c[4].markdown(r["main_reason"])
            c[5].markdown(bubble(r["fix_status"], STATUS_KIND[r["fix_status"]]), unsafe_allow_html=True)
            c[6].button("→", key=f"arrow_{prefix}_{idx}", on_click=open_flight_idx, args=(idx,),
                        help="Open flight detail")


# ---------- Theme (CSS) ----------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Readex+Pro:wght@400;500;600;700&display=swap');
html, body, .stApp, p, label, input, textarea, select, button, h1, h2, h3, h4, li, td, th {{ font-family: 'Readex Pro', sans-serif !important; }}
[data-testid="stIconMaterial"], span[class*="material-symbols"], .material-symbols-rounded,
[data-testid="stIconMaterial"] * {{ font-family: 'Material Symbols Rounded' !important; }}
.stApp {{ background: {PAGE}; color: {NAVY}; }}
header[data-testid="stHeader"], #MainMenu, footer {{ visibility: hidden; height: 0; }}
.block-container {{ padding: 1rem 2rem 2rem 2rem; max-width: 100% !important; }}
/* cards */
[data-testid="stVerticalBlockBorderWrapper"] {{ background: #FFFFFF; border-radius: 12px !important; border-color: #D9DEE5 !important; }}
[data-testid="stMetricValue"] {{ color: {NAVY}; font-weight: 700; }}
[data-testid="stMetricLabel"] p {{ color: {PURPLE}; font-size: 13px; }}
[data-testid="stAlert"] {{ background: {LAVENDER}; color: {NAVY}; border: none; border-radius: 10px; }}
/* buttons: all pills */
.stButton button, .stDownloadButton button, .stFormSubmitButton button {{ border-radius: 999px; font-weight: 600; }}
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"], .stDownloadButton button {{
    background: {NAVY}; border-color: {NAVY}; color: #FFFFFF; }}
[class*="st-key-send_"] button, [class*="st-key-dsend_"] button, [class*="st-key-tsend_"] button {{
    background: {NAVY} !important; color: #FFFFFF !important; border: none !important; }}
[class*="st-key-ack_"] button, [class*="st-key-dsent_"] button, [class*="st-key-tsent_"] button {{
    background: {YELLOW} !important; color: {NAVY} !important; border: none !important; opacity: 1 !important; }}
[class*="st-key-done_"] button, [class*="st-key-dneed_"] button, [class*="st-key-allok"] button {{
    background: {GREEN_BG} !important; color: {GREEN_TX} !important; border: none !important; opacity: 1 !important; }}
[class*="st-key-open_"] button {{ background: #FFFFFF !important; color: {PURPLE} !important; border: 1px solid {PURPLE} !important; }}
.stButton button:disabled p {{ color: inherit !important; }}
/* warning cards inside the owner columns */
[class*="st-key-card_"] {{ background: {LAVENDER} !important; border: none !important; border-radius: 10px; }}
/* flight rows as white rounded cards, arrow as a plain icon */
[class*="st-key-row_"] {{ background: #FFFFFF; border-radius: 12px; padding: 0 18px; gap: 0; min-height: 68px; justify-content: center; }}
[class*="st-key-row_"] [data-testid="stHorizontalBlock"] {{ min-height: 40px; align-items: center; }}
[class*="st-key-row_"] [data-testid="stElementContainer"], [class*="st-key-row_"] [data-testid="stLayoutWrapper"] {{ height: auto !important; }}
[class*="st-key-row_"] [data-testid="stColumn"] [data-testid="stVerticalBlock"] {{ height: auto !important; justify-content: center; }}
[class*="st-key-row_"] [data-testid="stMarkdownContainer"] {{ height: auto !important; margin-bottom: 0 !important; }}
[class*="st-key-row_"] p {{ margin: 0; line-height: 1.5; }}
[class*="st-key-arrow_"] button {{ background: transparent !important; border: none !important; box-shadow: none !important;
    padding: 0 !important; min-height: 0 !important; }}
[class*="st-key-arrow_"] button p {{ color: {NAVY} !important; font-size: 28px !important; line-height: 1 !important; }}
[class*="st-key-arrow_"] button:hover p {{ color: {PURPLE} !important; }}
[class*="st-key-plan_"] {{ background: #FFFFFF; border-radius: 12px; padding: 10px 16px; }}
/* login form */
[data-testid="stForm"] {{ background: #FFFFFF; border: none; border-radius: 12px; padding: 20px; }}
/* every radio becomes pills */
div[role="radiogroup"] {{ gap: 6px; flex-wrap: wrap; }}
div[role="radiogroup"] label {{ background: #FFFFFF; border: 1px solid #D9DEE5; border-radius: 999px; padding: 6px 14px; margin: 0; cursor: pointer; }}
label[data-testid="stRadioOption"] > div > div:not([data-testid="stMarkdownContainer"]),
div[role="radiogroup"] label > div:not(:has([data-testid="stMarkdownContainer"])) {{ display: none !important; }}
div[role="radiogroup"] label[data-selected="true"], div[role="radiogroup"] label:has(input:checked) {{ background: {NAVY}; border-color: {NAVY}; }}
div[role="radiogroup"] label[data-selected="true"] p, div[role="radiogroup"] label:has(input:checked) p {{ color: #FFFFFF; font-weight: 600; }}
/* main tabs: plain words like the DXB site menu, active one underlined in yellow */
.st-key-tabs div[role="radiogroup"] {{ justify-content: flex-end; gap: 26px; }}
.st-key-tabs div[role="radiogroup"] label {{ background: transparent !important; border: none !important; border-radius: 0; padding: 6px 2px; }}
.st-key-tabs div[role="radiogroup"] label p {{ color: {NAVY} !important; font-size: 17px; font-weight: 500; }}
.st-key-tabs div[role="radiogroup"] label:hover p {{ color: {PURPLE} !important; }}
.st-key-tabs div[role="radiogroup"] label[data-selected="true"], .st-key-tabs div[role="radiogroup"] label:has(input:checked) {{
    border-bottom: 3px solid {YELLOW} !important; }}
.st-key-tabs div[role="radiogroup"] label[data-selected="true"] p, .st-key-tabs div[role="radiogroup"] label:has(input:checked) p {{
    color: {NAVY} !important; font-weight: 700; }}
</style>
""", unsafe_allow_html=True)

# ---------- Login ----------
# Usernames and passwords live in .streamlit/secrets.toml (never on GitHub), e.g.
# [users.ops_manager]
# password = "..."
# name = "Ops Control Manager"
USERS = st.secrets.get("users", {})
if "auth" not in st.session_state:
    st.session_state.auth = False
    st.session_state.who = ""


def sign_out():
    st.session_state.auth = False
    st.session_state.who = ""


if not st.session_state.auth:
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown(f"""
<div style="background:{NAVY};border-radius:12px;padding:18px 22px;margin:40px 0 14px 0">
  <div style="color:#FFFFFF;font-size:22px;font-weight:700">DXB Predictive Six Sigma Dashboard</div>
  <div style="color:#C9CCF5;font-size:14px">Arrival delay prediction for the Operations Control Manager</div>
</div>""", unsafe_allow_html=True)
        with st.form("login"):
            st.markdown("**Sign in**")
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", type="primary", use_container_width=True):
                u = user.strip()
                if not USERS:
                    st.error("No users are set up yet. Add them to .streamlit/secrets.toml.")
                elif u in USERS and pwd == USERS[u]["password"]:
                    st.session_state.auth = True
                    st.session_state.who = USERS[u].get("name", u)
                    st.rerun()
                else:
                    st.error("Wrong username or password.")
        st.caption("Student prototype (BUS922). Not affiliated with or endorsed by Dubai Airports.")
    st.stop()

# ---------- Header ----------
st.markdown(f"""
<div style="background:{NAVY};border-radius:12px;padding:16px 22px;display:flex;flex-wrap:wrap;
            justify-content:space-between;align-items:center;gap:12px;margin-bottom:10px">
  <div>
    <div style="color:#FFFFFF;font-size:22px;font-weight:700">DXB Predictive Six Sigma Dashboard</div>
    <div style="color:#C9CCF5;font-size:13px">Signed in as {st.session_state.who} · Replay day: 3 Jan 2026 (stress-test day)</div>
  </div>
  <div style="display:flex;align-items:center;gap:10px">
    <span style="background:{RED};color:#FFFFFF;font-size:11px;font-weight:700;border-radius:999px;padding:3px 10px">REPLAY</span>
    <span style="color:#FFFFFF;font-size:28px;font-weight:700">{fmt(now)}</span>
  </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([2, 5])
with left:
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("− 30 min", use_container_width=True):
        st.session_state.now = max(0, now - 30)
        st.rerun()
    if c2.button("+ 30 min", use_container_width=True):
        st.session_state.now = min(23 * 60 + 59, now + 30)
        st.rerun()
    if c3.button("Restart", use_container_width=True):
        st.session_state.now = 10 * 60
        st.rerun()
    c4.button("Sign out", on_click=sign_out, use_container_width=True)
with right, st.container(key="tabs"):
    st.radio("Tab", ["Today", "Flights", "Flight detail", "Warnings", "Control", "Try a flight"],
             key="tab", horizontal=True, label_visibility="collapsed", on_change=tab_changed)
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
        st.markdown(":material/flight_land: **Arrivals by hour**")
        flights["hr"] = flights["eta_min"] // 60
        by_hour = flights.groupby("hr").agg(total=("high", "size"), late=("high", "sum")).reindex(range(24), fill_value=0)
        fig = go.Figure()
        fig.add_bar(x=by_hour.index, y=by_hour["total"] - by_hour["late"], name="Predicted on time", marker_color=MUTED)
        fig.add_bar(x=by_hour.index, y=by_hour["late"], name="Predicted late", marker_color=RED)
        fig.add_vrect(x0=now // 60 - 0.5, x1=now // 60 + 0.5, line_width=2, line_color=NAVY, fillcolor="rgba(0,0,0,0)")
        fig.update_layout(barmode="stack", height=280, margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=-0.2), xaxis_title="Hour", yaxis_title="Flights")
        st.plotly_chart(fig, use_container_width=True, theme=None)
    with g2.container(border=True):
        st.markdown("**Top reasons today**")
        st.caption("High-risk flights still to come, by main reason")
        top = to_come.loc[to_come["high"], "main_reason"].value_counts()
        fig2 = go.Figure(go.Bar(x=top.values[::-1], y=top.index[::-1], orientation="h", marker_color=RED))
        fig2.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True, theme=None)

    h1, h2 = st.columns([4, 1], vertical_alignment="center")
    h1.markdown("**Next 3 hours: top 5 highest risk** (press → to open a flight)")
    h2.button("See all flights", on_click=go_to, args=("Flights",), use_container_width=True, type="primary")
    nxt = to_come[to_come["mins_to"].between(-30, 180)].nlargest(5, "p_xgb")
    flight_rows(nxt, "today")

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

    PER_PAGE = 25
    pages = max(1, -(-len(view) // PER_PAGE))
    if "page" not in st.session_state:
        st.session_state.page = 1
    st.session_state.page = min(st.session_state.page, pages)

    s1, s2 = st.columns([4, 1], vertical_alignment="center")
    s1.caption(f"Showing {len(view)} flights · {int(view['high'].sum())} high risk · "
               f"{int((view['fix_status'] == 'Fix missing').sum())} with a fix missing · press → to open a flight")
    table = view.rename(columns=SHOW_COLS)[list(SHOW_COLS.values())]
    s2.download_button("Download CSV", table.to_csv(index=False), "flights_3jan2026.csv", "text/csv",
                       use_container_width=True)

    start = (st.session_state.page - 1) * PER_PAGE
    flight_rows(view.iloc[start:start + PER_PAGE], "list")

    p1, p2, p3 = st.columns([1, 2, 1], vertical_alignment="center")
    if p1.button("‹ Previous", use_container_width=True, disabled=st.session_state.page == 1):
        st.session_state.page -= 1
        st.rerun()
    p2.markdown(f"<div style='text-align:center'>Page {st.session_state.page} of {pages}</div>",
                unsafe_allow_html=True)
    if p3.button("Next ›", use_container_width=True, disabled=st.session_state.page == pages):
        st.session_state.page += 1
        st.rerun()

# ---------- FLIGHT DETAIL ----------
elif tab == "Flight detail":
    hi_up = to_come[to_come["high"]]
    n_actioned = int(hi_up["actioned"].sum())

    if st.session_state.sel is None:
        pending = hi_up[~hi_up["actioned"]].sort_values("p_xgb", ascending=False)
        pool = pending if len(pending) else to_come.sort_values("p_xgb", ascending=False)
        st.session_state.sel = pool.index[0] if len(pool) else flights.index[0]
        st.session_state.auto = len(pending) > 0
    f = flights.loc[st.session_state.sel]

    t1, t2, t3 = st.columns([2, 2, 1])
    with t1.container(border=True):
        st.metric("Pending action", plural(len(hi_up) - n_actioned, "flight"))
        st.caption("High risk, a fix not in place and no warning sent")
    with t2.container(border=True):
        st.metric("Actioned", plural(n_actioned, "flight"))
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
        st.markdown(risk_block(f["p_xgb"], "Risk score", 44), unsafe_allow_html=True)
        st.caption(f"XGBoost · cut-off {CUTOFF:.1%}")

    b1, b2 = st.columns([2, 1])
    with b1.container(border=True):
        st.markdown("**Why is it risky?** Red raises the risk, navy lowers it.")
        vals = pd.Series({r: f["r_" + r] for r in REASONS})
        vals = vals.reindex(vals.abs().sort_values(ascending=False).index)[:6][::-1]
        fig = go.Figure(go.Bar(x=vals.values, y=vals.index, orientation="h",
                               marker_color=[RED if v > 0 else MUTED for v in vals.values]))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="SHAP (log-odds)")
        st.plotly_chart(fig, use_container_width=True, theme=None)
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
            with h2:
                if not open_fixes:
                    st.button("All fixes in place", key="allok", disabled=True, use_container_width=True)
                elif all_sent:
                    st.button("Warnings sent", key="dsent_all", disabled=True, use_container_width=True)
                elif st.button("Send all warnings", key="dsend_all", use_container_width=True):
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
                with r4:
                    if done:
                        st.button("Not needed", key="dneed_" + x["key"], disabled=True, use_container_width=True)
                    elif status:
                        st.button(f"Warning {status.lower()}", key="dsent_" + x["key"], disabled=True,
                                  use_container_width=True)
                    elif st.button("Send warning", key="dsend_" + x["key"], use_container_width=True):
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

    counts = pd.Series([i["status"] for i in items], dtype=str).value_counts()
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
                st.markdown(f"<span style='color:{GREEN_TX};background:{GREEN_BG};border-radius:999px;"
                            f"padding:3px 12px;font-size:13px'>All clear</span>", unsafe_allow_html=True)
            for i in mine:
                row, fx = i["row"], i["fx"]
                with st.container(border=True, key="card_" + fx["key"]):
                    st.caption(fx["code"])
                    st.markdown(f"**{row['flightnumber']}** · scheduled {row['eta_txt']} · **{due(row['mins_to'])}**")
                    st.write(fx["fix"])
                    st.write(f"Status: **{i['status']}**")
                    b1, b2 = st.columns(2)
                    if i["status"] == "New" and b1.button("Send", key="send_" + fx["key"], use_container_width=True):
                        send_warning(fx, row["flightnumber"])
                        st.rerun()
                    if i["status"] in ("Sent", "Overdue") and b1.button("Acknowledge", key="ack_" + fx["key"],
                                                                        use_container_width=True):
                        st.session_state.warn[fx["key"]] = "Acknowledged"
                        record(fx, row["flightnumber"], "Acknowledged")
                        st.rerun()
                    if b2.button("Mark fix done", key="done_" + fx["key"], use_container_width=True):
                        st.session_state.done[fx["key"]] = True
                        record(fx, row["flightnumber"], "Fix done")
                        st.rerun()
                    st.button("Open flight", key="open_" + fx["key"], on_click=open_flight, args=(i["idx"],),
                              use_container_width=True)

# ---------- CONTROL ----------
elif tab == "Control":
    @st.cache_data
    def load_daily():
        d = pd.read_csv("data/daily_late.csv", parse_dates=["flight_date"])
        d["p"] = d["late"] / d["flights"]
        return d

    def sigma_of(p):
        return NormalDist().inv_cdf(1 - p) + 1.5 if 0 < p < 1 else float("nan")

    daily = load_daily()
    hist = daily[daily["flight_date"] < "2026-01-03"].copy()        # only days before the replay day
    landed = flights[flights["landed"]]

    c1, c2 = st.columns([3, 2])
    chart_type = c1.radio("Chart", ["Standard p-chart", "Laney p′ chart"], horizontal=True)
    window = c2.radio("Window", ["Last 30 days", "All days (May 2025 on)"], horizontal=True)
    view = hist.tail(30).copy() if window == "Last 30 days" else hist.copy()

    p_bar = view["late"].sum() / view["flights"].sum()
    view["s"] = (p_bar * (1 - p_bar) / view["flights"]) ** 0.5
    if chart_type == "Laney p′ chart":
        z = (view["p"] - p_bar) / view["s"]
        sigma_z = z.diff().abs().mean() / 1.128
        view["s"] = view["s"] * sigma_z
    view["ucl"] = p_bar + 3 * view["s"]
    view["lcl"] = (p_bar - 3 * view["s"]).clip(lower=0)
    view["out"] = (view["p"] > view["ucl"]) | (view["p"] < view["lcl"])

    sent_codes = {e["warning_code"] for e in st.session_state.log if e["action"] == "Sent"}
    done_codes = {e["warning_code"] for e in st.session_state.log if e["action"] == "Fix done"}
    jan = daily[(daily["flight_date"] >= "2026-01-01") & (daily["flight_date"] < "2026-01-03")]
    jan_late = jan["late"].sum() + landed["delayed_15"].sum()
    jan_n = jan["flights"].sum() + len(landed)

    k1, k2, k3, k4 = st.columns(4)
    with k1.container(border=True):
        st.metric("Days out of limits", f"{int(view['out'].sum())} of {len(view)}")
        st.caption(chart_type)
    with k2.container(border=True):
        rate = f"{len(done_codes & sent_codes) / len(sent_codes):.0%}" if sent_codes else "n/a"
        st.metric("Fix completion rate", rate)
        st.caption(f"{len(done_codes & sent_codes)} of {len(sent_codes)} warnings sent got a fix done")
    with k3.container(border=True):
        st.metric("Sigma this month", f"{sigma_of(jan_late / jan_n):.2f} σ")
        st.caption("January so far (incl. today) · baseline 2.24 σ")
    with k4.container(border=True):
        st.metric("Model AUC", "0.770")
        st.caption("Test Jan–Feb 2026 · target 0.75 or above · retrain if it drops")

    view["lbl"] = view["flight_date"].dt.strftime("%d %b %y")
    with st.container(border=True):
        st.markdown(f"**{chart_type}: daily share of late arrivals** (red = outside the limits)")
        fig = go.Figure()
        fig.add_scatter(x=view["lbl"], y=view["ucl"], name="UCL", line=dict(color=RED, dash="dash", shape="hv"))
        fig.add_scatter(x=view["lbl"], y=view["lcl"], name="LCL", line=dict(color=RED, dash="dash", shape="hv"))
        fig.add_hline(y=p_bar, line_color=MUTED, annotation_text=f"Mean {p_bar:.1%}")
        fig.add_scatter(x=view["lbl"], y=view["p"], mode="lines+markers", name="Daily late share",
                        line=dict(color=MUTED, width=1),
                        marker=dict(size=8, color=[RED if o else NAVY for o in view["out"]]))
        if len(landed):
            fig.add_scatter(x=["03 Jan 26 (today)"], y=[landed["delayed_15"].mean()], mode="markers",
                            name="Today so far", marker=dict(size=12, color="white", line=dict(color=NAVY, width=2)))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_tickformat=".0%",
                          xaxis=dict(type="category", nticks=12), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True, theme=None)
        st.caption("Days with no data in the source files are skipped. Standard limits assume days differ only by chance, "
                   "so with ~600 flights a day they are very tight. Laney p′ widens them for normal day-to-day swings, "
                   "so only truly unusual days stand out.")

    g1, g2 = st.columns(2)
    with g1.container(border=True):
        st.markdown("**Sigma level by month**")
        m = hist.groupby(hist["flight_date"].dt.to_period("M"))[["late", "flights"]].sum()
        m.loc[pd.Period("2026-01", "M")] = [jan_late, jan_n]
        m["sigma"] = (m["late"] / m["flights"]).apply(sigma_of)
        fig2 = go.Figure(go.Bar(x=m.index.strftime("%b %y"), y=m["sigma"], marker_color=NAVY))
        fig2.add_hline(y=2.24, line_dash="dash", line_color=RED, annotation_text="Baseline 2.24 σ")
        fig2.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True, theme=None)
    with g2:
        st.markdown("**Control plan**")
        plan = [("What we watch", "Target", "Owner", "If it breaks"),
                ("Daily late share", "Inside the limits", "Ops control manager", "Find the special cause"),
                ("Fix completion rate", "90% or above", "Shift lead", "Chase overdue owners"),
                ("Sigma level", "Above 2.24 σ", "Ops control manager", "Review the fix rules"),
                ("Model AUC", "0.75 or above", "Analytics team", "Retrain the model")]
        for n, row in enumerate(plan):
            box = st.container() if n == 0 else st.container(key=f"plan_{n}")
            with box:
                cc = st.columns([1.2, 1, 1.1, 1.3], vertical_alignment="center")
                for col, txt in zip(cc, row):
                    col.markdown(f"<span style='font-weight:600'>{txt}</span>" if n == 0 else txt,
                                 unsafe_allow_html=True)

# ---------- TRY A FLIGHT ----------
elif tab == "Try a flight":
    @st.cache_resource
    def get_model():
        booster, enc = load_model()
        with open("data/lookup.json") as fh:
            lk = json.load(fh)
        xd = pd.read_parquet("data/x_day.parquet")
        knock_fill = float(xd.loc[xd["knockon_missing"] == 1, "knockon_dep_delay_min"].iloc[0])
        rot_fill = float(xd["rotation_gap_h"].median())
        return booster, enc, lk, knock_fill, rot_fill

    booster, enc, lk, knock_fill, rot_fill = get_model()
    cats = enc["categories"]
    if "try_res" not in st.session_state:
        st.session_state.try_res = None
        st.session_state.try_sent = set()

    st.info("Live demo: type in any real flight and press Predict. With flight and weather APIs connected, "
            "these fields would fill in by themselves. Warnings here are demo only and are not logged.")
    left_col, right_col = st.columns([5, 7], gap="medium")

    with left_col, st.form("try_form"):
        h1, h2 = st.columns([3, 2])
        h1.markdown("**Enter a flight**")
        go_btn = h2.form_submit_button("Predict", type="primary", use_container_width=True)
        st.caption("FLIGHT")
        a, b, c = st.columns(3)
        date = a.date_input("Date", pd.Timestamp("2026-11-12"))
        sched = b.time_input("Arrival", pd.Timestamp("2026-11-12 10:40").time())
        airline = c.text_input("Airline", "EK").strip().upper()
        origin = a.text_input("Origin", "BOM").strip().upper()
        terminal = b.selectbox("Terminal", cats["aircraftterminal"])
        body = c.selectbox("Body", cats["body_type"])
        st.caption("PLANE AND ROUTE")
        a, b, c = st.columns(3)
        knock = a.number_input("Inbound delay (min)", 0, 600, 0)
        knock_known = a.checkbox("Delay known")
        rot = b.number_input("Hours since left DXB", 0.0, 72.0, 0.0, 0.5, help="0 = unknown")
        season = c.selectbox("Season", cats["season"],
                             index=cats["season"].index("off_peak") if "off_peak" in cats["season"] else 0)
        route_default = lk["route_rate"].get(origin, lk["avg_late_rate"]) * 100
        route = a.number_input("Route late 7d (%)", 0.0, 100.0, float(round(route_default, 1)))
        arrivals = b.number_input("Arrivals that hour", 0, 80, int(lk["hourly_arrivals"].get(str(sched.hour), 30)))
        codeshare = c.checkbox("Codeshare")
        stopover = c.checkbox("Stopover")
        st.caption("WEATHER")
        a, b, c = st.columns(3)
        wind = a.number_input("Wind (kt)", 0, 80, 10)
        vis = b.number_input("Visibility (km)", 0.0, 10.0, 10.0, 0.5)
        temp = c.number_input("Temp (°C)", 0, 50, 25)
        fog = a.checkbox("Fog or dust")
        storm = b.checkbox("Storm or rain")

    if go_btn:
        region = lk["origin_region"].get(origin)
        X = pd.DataFrame([{
            "hour": sched.hour, "day_of_week": pd.Timestamp(date).day_name(), "season": season,
            "hourly_arrivals": arrivals, "origin_region": region or "Sub-Saharan Africa",
            "route_late_rate_7d": route / 100, "has_stopover": int(stopover),
            "airline_group": airline if airline in lk["airlines"] else "Others",
            "body_type": body, "is_codeshare": int(codeshare), "aircraftterminal": str(terminal),
            "knockon_dep_delay_min": knock if knock_known else knock_fill,
            "rotation_gap_h": rot if rot > 0 else rot_fill,
            "vis_km": vis, "wind_kt": wind, "temp_c": temp, "fog_dust": int(fog), "severe_weather": int(storm),
            "knockon_missing": 0 if knock_known else 1,
            "route_rate_missing": 0 if origin in lk["route_rate"] else 1,
        }])
        p, shap = predict(booster, enc, X)
        vals = pd.Series({r: shap[rules.loc[rules["reason"] == r, "driver"]].sum(axis=1).iloc[0] for r in REASONS})
        st.session_state.try_res = {"risk": float(p[0]), "vals": vals, "unknown_origin": region is None,
                                    "label": f"{airline} from {origin}, {sched.strftime('%H:%M')}"}
        st.session_state.try_sent = set()

    with right_col:
        res = st.session_state.try_res
        if res is None:
            with st.container(border=True):
                st.markdown("**Risk score**")
                st.caption("Fill in the flight and press Predict.")
        else:
            risk, vals = res["risk"], res["vals"]
            colour = BUBBLE[risk_kind(risk)][1]
            main = vals.idxmax()
            with st.container(border=True):
                r1, r2 = st.columns([2, 3])
                r1.markdown(risk_block(risk, f"Risk score · {res['label']}", 56), unsafe_allow_html=True)
                r2.markdown("**Where it sits**")
                r2.markdown(
                    f"<div style='position:relative;height:18px;border-radius:9px;background:linear-gradient(90deg,"
                    f"{GREEN_BG} 0 {CUTOFF*100:.0f}%,#FDECEA {CUTOFF*100:.0f}% 100%)'>"
                    f"<div style='position:absolute;left:{CUTOFF*100:.1f}%;top:-4px;width:2px;height:26px;background:{NAVY}'></div>"
                    f"<div style='position:absolute;left:calc({risk*100:.0f}% - 7px);top:-1px;width:14px;height:18px;"
                    f"border-radius:4px;background:{colour}'></div></div>"
                    f"<div style='display:flex;justify-content:space-between;font-size:12px;margin-top:4px'>"
                    f"<span>0%</span><span>cut-off {CUTOFF:.1%}</span><span>100%</span></div>", unsafe_allow_html=True)
                r2.caption(f"Real XGBoost model, same as the replay tabs · main reason: **{main}**")
                if res["unknown_origin"]:
                    r2.warning("Origin not in the training data, so region and route history are guesses.")

            with st.container(border=True):
                st.markdown("**Why?** Red raises the risk, navy lowers it.")
                v = vals.reindex(vals.abs().sort_values(ascending=False).index)[:6][::-1]
                fig = go.Figure(go.Bar(x=v.values, y=v.index, orientation="h",
                                       marker_color=[RED if x > 0 else MUTED for x in v.values]))
                fig.update_layout(height=230, margin=dict(l=10, r=10, t=5, b=10), xaxis_title="SHAP (log-odds)")
                st.plotly_chart(fig, use_container_width=True, theme=None)

            with st.container(border=True):
                action = (rules[rules["actionable"] == 1].drop_duplicates("reason")
                          .set_index("reason")[["fix", "owner"]].to_dict("index"))
                todo = [r for r in vals.sort_values(ascending=False).index if r in action and vals[r] > 0][:2]
                w1, w2 = st.columns([3, 2])
                w1.markdown("**Who needs to act**")
                if risk < CUTOFF:
                    st.write("Below the cut-off. No action needed.")
                else:
                    with w2:
                        if set(todo) <= st.session_state.try_sent:
                            st.button("Warnings sent (demo)", key="tsent_all", disabled=True, use_container_width=True)
                        elif st.button("Send all warnings", key="tsend_all", use_container_width=True):
                            st.session_state.try_sent |= set(todo)
                            st.rerun()
                    cards = st.columns(len(todo)) if todo else []
                    for col, r in zip(cards, todo):
                        with col.container(border=True, key="card_try_" + r.replace(" ", "_")):
                            st.markdown(f"**{action[r]['owner']}**")
                            st.write(action[r]["fix"])
                            st.caption(f"Reason: {r}")
                            if r in st.session_state.try_sent:
                                st.button("Sent (demo)", key="tsent_" + r.replace(" ", "_"), disabled=True,
                                          use_container_width=True)
                            elif st.button("Send warning", key="tsend_" + r.replace(" ", "_"), use_container_width=True):
                                st.session_state.try_sent.add(r)
                                st.rerun()

st.caption("Student prototype (BUS922). Not affiliated with or endorsed by Dubai Airports. Open data from Dubai Pulse.")
