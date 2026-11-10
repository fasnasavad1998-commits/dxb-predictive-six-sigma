import pandas as pd
import streamlit as st

st.set_page_config(page_title="DXB Predictive Six Sigma Dashboard", layout="wide")

# ---------- Data ----------
@st.cache_data
def load_data():
    df = pd.read_parquet("data/replay_2026-01-03_scored.parquet")
    df["sched"] = pd.to_datetime(df["scheduledinblocktime"])
    df["eta_min"] = df["sched"].dt.hour * 60 + df["sched"].dt.minute
    rules = pd.read_csv("rules.csv")
    return df, rules

flights, rules = load_data()
CUTOFF = 0.553

# ---------- Replay clock (minutes after midnight) ----------
if "now" not in st.session_state:
    st.session_state.now = 10 * 60

def fmt(m):
    return f"{int(m) // 60:02d}:{int(m) % 60:02d}"

# ---------- Header: title left, tabs right ----------
left, right = st.columns([2, 3])
with left:
    st.markdown("### DXB Predictive Six Sigma Dashboard")
    st.caption(f"Ops Control Manager · Replay day: 3 Jan 2026 · Time now **{fmt(st.session_state.now)}**")
    c1, c2, c3 = st.columns(3)
    if c1.button("- 30 min"):
        st.session_state.now = max(0, st.session_state.now - 30)
        st.rerun()
    if c2.button("+ 30 min"):
        st.session_state.now = min(23 * 60 + 59, st.session_state.now + 30)
        st.rerun()
    if c3.button("Restart day"):
        st.session_state.now = 10 * 60
        st.rerun()
with right:
    tab = st.radio("Tab", ["Today", "Flights", "Flight detail", "Warnings", "Control"],
                   horizontal=True, label_visibility="collapsed")

st.divider()

# ---------- Tab content (filled in next steps) ----------
st.write(f"**{tab}** tab. {len(flights)} flights loaded, {len(rules)} rules loaded.")

st.caption("Student prototype (BUS922). Not affiliated with or endorsed by Dubai Airports. Open data from Dubai Pulse.")