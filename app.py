"""
ForgeMind AI – Autonomous Industrial Intelligence Platform  v2.0
Linear-inspired dashboard
"""

import io, os, sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.generate_data import generate_dataset
from engine.anomaly_detection import train_and_predict
from engine.risk_scoring import add_risk_columns
from engine.maintenance_engine import add_recommendations, simulate_maintenance_history
from engine.fleet_analytics import (
    add_oee, fleet_summary, efficiency_matrix,
    fault_distribution, estimate_mtbf,
)

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ForgeMind AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Linear-inspired CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & base ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main .block-container {
    background: #0f0f11 !important;
    color: #e2e2e5 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1400px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f0f11 !important;
    border-right: 1px solid #1e1e24 !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem !important; }

/* ── Sidebar text ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #9898a6 !important; font-size: 0.78rem !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid #1e1e24 !important;
    gap: 0 !important;
    background: transparent !important;
}
[data-testid="stTabs"] button[role="tab"] {
    color: #6b6b7b !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    padding: 8px 16px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    transition: color .15s;
}
[data-testid="stTabs"] button[role="tab"]:hover { color: #e2e2e5 !important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #e2e2e5 !important;
    border-bottom: 2px solid #7c5cfc !important;
    background: transparent !important;
}

/* ── Stat cards ── */
.stat-card {
    background: #13131a;
    border: 1px solid #1e1e24;
    border-radius: 8px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #7c5cfc44, transparent);
}
.stat-val  { font-size: 1.75rem; font-weight: 700; line-height: 1; margin: 0; letter-spacing: -0.02em; }
.stat-lbl  { font-size: 0.72rem; font-weight: 500; color: #6b6b7b;
             text-transform: uppercase; letter-spacing: 0.08em; margin-top: 6px; }
.stat-sub  { font-size: 0.7rem; color: #4a4a5a; margin-top: 3px; }

/* ── Section labels ── */
.section-label {
    font-size: 0.7rem; font-weight: 600; color: #6b6b7b;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 12px; padding-bottom: 8px;
    border-bottom: 1px solid #1e1e24;
}

/* ── Alert rows ── */
.alert-row {
    display: flex; align-items: flex-start; gap: 14px;
    padding: 12px 14px; border-radius: 6px;
    border: 1px solid #1e1e24;
    background: #13131a;
    margin-bottom: 6px;
    transition: border-color .15s;
}
.alert-row:hover { border-color: #2e2e3a; }
.alert-dot {
    width: 7px; height: 7px; border-radius: 50%;
    margin-top: 5px; flex-shrink: 0;
}
.alert-machine  { font-size: 0.82rem; font-weight: 600; color: #e2e2e5; }
.alert-meta     { font-size: 0.72rem; color: #6b6b7b; margin-top: 2px; }
.alert-rec      { font-size: 0.75rem; color: #9898a6; margin-top: 5px; line-height: 1.45; }
.alert-badge {
    font-size: 0.65rem; font-weight: 600; padding: 2px 7px;
    border-radius: 4px; white-space: nowrap; align-self: flex-start;
    flex-shrink: 0;
}

/* ── Priority colours ── */
.dot-critical { background: #e5534b; }
.dot-high     { background: #cc8833; }
.dot-medium   { background: #ae8b2e; }
.dot-low      { background: #3fb950; }
.badge-critical { background: #2d1212; color: #e5534b; border: 1px solid #3d1a1a; }
.badge-high     { background: #2b1900; color: #cc8833; border: 1px solid #3d2600; }
.badge-medium   { background: #282000; color: #ae8b2e; border: 1px solid #3a2f00; }
.badge-low      { background: #0d2018; color: #3fb950; border: 1px solid #1a3526; }

/* ── Divider ── */
hr { border: none; border-top: 1px solid #1e1e24 !important; margin: 1.2rem 0 !important; }

/* ── Metric widgets ── */
[data-testid="stMetric"] {
    background: #13131a;
    border: 1px solid #1e1e24;
    border-radius: 8px;
    padding: 12px 14px !important;
}
[data-testid="stMetricLabel"]  { font-size: 0.7rem !important; color: #6b6b7b !important; }
[data-testid="stMetricValue"]  { font-size: 1.15rem !important; font-weight: 600 !important; color: #e2e2e5 !important; }
[data-testid="stMetricDelta"]  { font-size: 0.7rem !important; }

/* ── Buttons ── */
.stButton > button, .stDownloadButton > button {
    background: #7c5cfc !important;
    color: #fff !important; border: none !important;
    border-radius: 6px !important; font-size: 0.78rem !important;
    font-weight: 500 !important; padding: 8px 16px !important;
    letter-spacing: 0.01em !important;
    transition: background .15s !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #6a4de0 !important;
}

/* ── Selects / sliders ── */
[data-testid="stMultiSelect"] > div > div,
[data-testid="stSelectbox"] > div > div {
    background: #13131a !important;
    border: 1px solid #1e1e24 !important;
    border-radius: 6px !important;
    color: #e2e2e5 !important;
    font-size: 0.78rem !important;
}
[data-testid="stSlider"] div[data-baseweb="slider"] div { background: #7c5cfc !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e1e24 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] table { background: #13131a !important; }
[data-testid="stDataFrame"] th {
    background: #13131a !important;
    color: #6b6b7b !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    border-bottom: 1px solid #1e1e24 !important;
}
[data-testid="stDataFrame"] td { color: #c8c8d0 !important; font-size: 0.78rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0f0f11; }
::-webkit-scrollbar-thumb { background: #2a2a35; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Chart theme ──────────────────────────────────────────────────────────────
BG      = "#0f0f11"
PAPER   = "#13131a"
GRID    = "#1e1e24"
TICK    = "#6b6b7b"
ACCENT  = "#7c5cfc"
RED     = "#e5534b"
ORANGE  = "#cc8833"
YELLOW  = "#ae8b2e"
GREEN   = "#3fb950"
BLUE    = "#4a9eff"
RISK_COLOURS = {"Low": GREEN, "Medium": YELLOW, "High": RED}
URG_COLOURS  = {"Critical": RED, "Soon": ORANGE, "Planned": YELLOW, "Healthy": GREEN}

def _theme(fig, height=360, margin=None):
    m = margin or dict(l=16, r=16, t=36, b=16)
    fig.update_layout(
        paper_bgcolor=PAPER, plot_bgcolor=BG,
        font=dict(family="Inter, sans-serif", color=TICK, size=11),
        height=height, margin=m,
        legend=dict(bgcolor=PAPER, bordercolor=GRID, borderwidth=1,
                    font=dict(size=10, color=TICK)),
        title_font=dict(size=12, color="#9898a6", family="Inter, sans-serif"),
        coloraxis_colorbar=dict(tickfont=dict(color=TICK), title_font=dict(color=TICK)),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID,
                     tickfont=dict(size=10), linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID,
                     tickfont=dict(size=10), linecolor=GRID)
    return fig

# ── Helpers ──────────────────────────────────────────────────────────────────
def stat(col, val, label, color=None, sub=""):
    c = color or "#e2e2e5"
    col.markdown(
        f"<div class='stat-card'>"
        f"<p class='stat-val' style='color:{c}'>{val}</p>"
        f"<p class='stat-lbl'>{label}</p>"
        f"{'<p class=stat-sub>'+sub+'</p>' if sub else ''}"
        f"</div>", unsafe_allow_html=True)

def section(title):
    st.markdown(f"<p class='section-label'>{title}</p>", unsafe_allow_html=True)

# ── Pipeline ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Running AI pipeline…")
def load_pipeline(cont=None):
    df = generate_dataset()
    df = train_and_predict(df, contamination=cont)
    df = add_risk_columns(df)
    df = add_recommendations(df)
    df = add_oee(df)
    return df

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='padding:0 0 20px'>"
        "<p style='font-size:0.72rem;font-weight:700;color:#6b6b7b;"
        "text-transform:uppercase;letter-spacing:0.12em;margin:0'>ForgeMind AI</p>"
        "<p style='font-size:0.68rem;color:#4a4a5a;margin:2px 0 0'>Industrial Intelligence · v2.0</p>"
        "</div>", unsafe_allow_html=True)

    _tmp = load_pipeline()
    st.markdown("<p style='font-size:0.7rem;font-weight:600;color:#6b6b7b;"
                "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px'>Filters</p>",
                unsafe_allow_html=True)
    selected_types = st.multiselect(
        "Machine type", options=sorted(_tmp["machine_type"].unique()),
        default=sorted(_tmp["machine_type"].unique()), label_visibility="collapsed",
    )
    risk_filter = st.multiselect(
        "Risk level", options=["Low","Medium","High"],
        default=["Low","Medium","High"], label_visibility="collapsed",
    )
    rul_filter = st.multiselect(
        "RUL urgency", options=["Healthy","Planned","Soon","Critical"],
        default=["Healthy","Planned","Soon","Critical"], label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.7rem;font-weight:600;color:#6b6b7b;"
                "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px'>AI Settings</p>",
                unsafe_allow_html=True)
    use_override = st.toggle("Override sensitivity", value=False)
    cont_val = None
    if use_override:
        cont_val = st.slider("Contamination", 0.05, 0.30, 0.12, 0.01)

    if st.button("Re-run pipeline", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.65rem;color:#3a3a4a'>Isolation Forest · LOF Ensemble<br>"
        "Per-type thresholds · RUL estimation<br>© 2026 ForgeMind Technologies</p>",
        unsafe_allow_html=True)

# ── Load & filter ─────────────────────────────────────────────────────────────
df_full = load_pipeline(cont=cont_val)
df_latest = (
    df_full.sort_values("timestamp")
    .groupby("machine_id", as_index=False).last()
)
mask = (
    df_latest["machine_type"].isin(selected_types) &
    df_latest["risk_level"].isin(risk_filter) &
    df_latest["rul_urgency"].isin(rul_filter)
)
dv = df_latest[mask].copy()
kpis = fleet_summary(dv)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-size:1.4rem;font-weight:700;color:#e2e2e5;"
    "letter-spacing:-0.02em;margin:0 0 2px'>Fleet Overview</h1>"
    "<p style='font-size:0.78rem;color:#6b6b7b;margin:0 0 20px'>"
    "Edge AI predictive maintenance · real-time sensor telemetry</p>",
    unsafe_allow_html=True)

# ── KPI strip ────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)
stat(c1, len(dv),                                 "Machines",         "#e2e2e5")
stat(c2, kpis.get("fleet_health_index","–"),      "Fleet health",     GREEN,  "avg score")
stat(c3, f"{kpis.get('avg_oee','–')}%",           "Avg OEE",          ACCENT, "avail × perf × qual")
stat(c4, kpis.get("critical_count","–"),          "Critical",         RED,    "immediate action")
stat(c5, kpis.get("machines_critical_rul","–"),   "Critical RUL",     ORANGE, "< 100 hr left")
stat(c6, f"${kpis.get('total_cost_exposure',0):,}","Cost exposure",   TICK,   "est. repair USD")

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
t1,t2,t3,t4,t5,t6 = st.tabs([
    "Fleet", "Analytics", "Alerts", "Machine", "Forecast", "Export"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Fleet
# ═══════════════════════════════════════════════════════════════════════════════
with t1:
    section("Machine status")

    # colour helpers for Styler.map (pandas 2.1+)
    def _c_risk(v):
        return f"color:{RISK_COLOURS.get(v,'#e2e2e5')};font-weight:600"
    def _c_health(v):
        c = GREEN if v>=70 else YELLOW if v>=40 else RED
        return f"color:{c};font-weight:600"
    def _c_trend(v):
        c = RED if v=="CRITICAL_ESCALATION" else ORANGE if v=="WORSENING" else GREEN
        return f"color:{c}"
    def _c_rul(v):
        return f"color:{URG_COLOURS.get(v,'#e2e2e5')};font-weight:500"

    cols = [
        "machine_id","machine_type","temperature","vibration","pressure",
        "power_kw","humidity","rpm","oil_temp","runtime_hours",
        "health_score","failure_prob","risk_level","is_anomaly",
        "anomaly_confidence","oee","rul_hours","rul_urgency","severity_trend",
    ]
    renames = {
        "machine_id":"ID","machine_type":"Type",
        "temperature":"Temp °C","vibration":"Vib","pressure":"Press",
        "power_kw":"kW","humidity":"Hum %","rpm":"RPM","oil_temp":"Oil °C",
        "runtime_hours":"Runtime h","health_score":"Health","failure_prob":"Fail %",
        "risk_level":"Risk","is_anomaly":"Anom","anomaly_confidence":"Conf",
        "oee":"OEE %","rul_hours":"RUL h","rul_urgency":"RUL status",
        "severity_trend":"Trend",
    }
    tbl = dv[[c for c in cols if c in dv.columns]].rename(columns=renames).sort_values("Health")

    styled = (
        tbl.style
        .map(_c_risk,   subset=["Risk"])
        .map(_c_health, subset=["Health"])
        .map(_c_trend,  subset=["Trend"])
        .map(_c_rul,    subset=["RUL status"])
        .format({
            "Temp °C":"{:.1f}","Vib":"{:.3f}","Press":"{:.2f}","kW":"{:.1f}",
            "Hum %":"{:.1f}","RPM":"{:.0f}","Oil °C":"{:.1f}","Runtime h":"{:.0f}",
            "Health":"{:.1f}","Fail %":"{:.1f}","Conf":"{:.0f}",
            "OEE %":"{:.1f}","RUL h":"{:.0f}",
        })
        .set_properties(**{"background-color":"#13131a","color":"#c8c8d0",
                           "font-size":"0.76rem","border-color":"#1e1e24"})
        .set_table_styles([{
            "selector":"th",
            "props":"background:#13131a;color:#6b6b7b;font-size:0.68rem;"
                    "text-transform:uppercase;letter-spacing:0.06em;"
                    "border-bottom:1px solid #1e1e24;"
        }])
    )
    st.dataframe(styled, use_container_width=True, height=500)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analytics
# ═══════════════════════════════════════════════════════════════════════════════
with t2:
    # Trend data: worst 12 machines
    sample_ids = dv.sort_values("health_score").head(12)["machine_id"].tolist()
    df_tr = df_full[df_full["machine_id"].isin(sample_ids)].copy()
    df_tr["timestamp"] = pd.to_datetime(df_tr["timestamp"])

    # row 1 — temp & vibration
    section("Sensor trends — bottom 12 machines by health")
    ca, cb = st.columns(2)
    with ca:
        fig = px.line(df_tr, x="timestamp", y="temperature", color="machine_id",
                      title="Temperature (°C)",
                      color_discrete_sequence=px.colors.qualitative.Bold)
        fig.add_hline(y=95,  line_dash="dot", line_color=RED,    line_width=1,
                      annotation_text="critical", annotation_font_color=RED,
                      annotation_font_size=9)
        fig.add_hline(y=78,  line_dash="dot", line_color=YELLOW, line_width=1,
                      annotation_text="warning",  annotation_font_color=YELLOW,
                      annotation_font_size=9)
        fig.update_layout(showlegend=False)
        st.plotly_chart(_theme(fig), use_container_width=True)
    with cb:
        fig = px.line(df_tr, x="timestamp", y="vibration", color="machine_id",
                      title="Vibration (mm/s)",
                      color_discrete_sequence=px.colors.qualitative.Bold)
        fig.add_hline(y=4.5, line_dash="dot", line_color=YELLOW, line_width=1)
        fig.add_hline(y=6.5, line_dash="dot", line_color=RED,    line_width=1)
        fig.update_layout(showlegend=False)
        st.plotly_chart(_theme(fig), use_container_width=True)

    # row 2 — oil temp & rpm
    cc, cd = st.columns(2)
    with cc:
        fig = px.line(df_tr, x="timestamp", y="oil_temp", color="machine_id",
                      title="Oil Temperature (°C)",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.add_hline(y=100, line_dash="dot", line_color=ORANGE, line_width=1)
        fig.update_layout(showlegend=False)
        st.plotly_chart(_theme(fig), use_container_width=True)
    with cd:
        fig = px.line(df_tr, x="timestamp", y="rpm", color="machine_id",
                      title="RPM",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(showlegend=False)
        st.plotly_chart(_theme(fig), use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # row 3 — health histogram & risk donut
    section("Distribution")
    ce, cf = st.columns(2)
    with ce:
        fig = px.histogram(dv, x="health_score", nbins=20,
                           title="Health score distribution",
                           color_discrete_sequence=[ACCENT])
        fig.add_vline(x=40, line_dash="dot", line_color=RED,    line_width=1)
        fig.add_vline(x=70, line_dash="dot", line_color=YELLOW, line_width=1)
        st.plotly_chart(_theme(fig), use_container_width=True)
    with cf:
        rc = dv["risk_level"].value_counts().reset_index()
        rc.columns = ["Risk","Count"]
        fig = px.pie(rc, values="Count", names="Risk", hole=0.6,
                     title="Risk breakdown",
                     color="Risk",
                     color_discrete_map=RISK_COLOURS)
        fig.update_traces(textfont_size=11, textinfo="percent+label",
                          marker=dict(line=dict(color=BG, width=2)))
        st.plotly_chart(_theme(fig), use_container_width=True)

    # row 4 — OEE by type & fault dist
    cg, ch = st.columns(2)
    with cg:
        eff = efficiency_matrix(dv)
        if not eff.empty and "avg_oee" in eff.columns:
            fig = px.bar(eff, x="machine_type", y="avg_oee",
                         color="avg_health", title="OEE by machine type (%)",
                         color_continuous_scale=[RED, YELLOW, GREEN],
                         range_color=[0,100])
            fig.add_hline(y=85, line_dash="dot", line_color=GREEN, line_width=1,
                          annotation_text="world-class", annotation_font_size=9)
            st.plotly_chart(_theme(fig), use_container_width=True)
    with ch:
        fd = fault_distribution(dv)
        if not fd.empty:
            fig = px.bar(fd, x="count", y="fault_type", orientation="h",
                         title="Fault type distribution",
                         color="count",
                         color_continuous_scale=[GREEN, YELLOW, RED])
            st.plotly_chart(_theme(fig), use_container_width=True)

    # row 5 — scatter
    st.markdown("<hr>", unsafe_allow_html=True)
    section("Correlation")
    fig = px.scatter(
        dv, x="temperature", y="vibration",
        color="risk_level", size="failure_prob",
        hover_data=["machine_id","health_score","machine_type","anomaly_confidence"],
        title="Temperature vs Vibration",
        color_discrete_map=RISK_COLOURS)
    fig.update_traces(marker=dict(line=dict(width=0.5, color=BG)))
    st.plotly_chart(_theme(fig, height=340), use_container_width=True)

    # row 6 — anomaly confidence
    if dv["is_anomaly"].any():
        fig = px.histogram(
            dv[dv["is_anomaly"]], x="anomaly_confidence", nbins=14,
            color="anomaly_category",
            title="Anomaly confidence distribution (flagged machines only)",
            color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(_theme(fig), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Alerts
# ═══════════════════════════════════════════════════════════════════════════════
with t3:
    # mini KPI row
    section("Summary")
    a1,a2,a3,a4 = st.columns(4)
    stat(a1, kpis.get("critical_count",0),   "Critical",           RED)
    stat(a2, kpis.get("high_count",0) - kpis.get("critical_count",0), "High", ORANGE)
    stat(a3, f"${kpis.get('total_cost_exposure',0):,}", "Repair exposure", YELLOW, "USD estimate")
    stat(a4, f"{kpis.get('total_downtime_exposure',0):.0f} hr", "Downtime exposure", TICK)
    st.markdown("<br>", unsafe_allow_html=True)

    section("Active alerts")
    prio_order = {"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3}
    df_al = dv[dv["alert_priority"].isin(prio_order)].copy()
    df_al["_s"] = df_al["alert_priority"].map(prio_order)
    df_al = df_al.sort_values(["_s","health_score"])

    trend_icon = {
        "CRITICAL_ESCALATION": "↑ escalating",
        "WORSENING":            "↗ worsening",
        "STABLE":               "→ stable",
    }
    dot_cls   = {"CRITICAL":"dot-critical","HIGH":"dot-high",
                 "MEDIUM":"dot-medium","LOW":"dot-low"}
    badge_cls = {"CRITICAL":"badge-critical","HIGH":"badge-high",
                 "MEDIUM":"badge-medium","LOW":"badge-low"}

    if df_al.empty:
        st.markdown(
            "<div style='padding:24px;border:1px solid #1e1e24;border-radius:8px;"
            "background:#13131a;text-align:center'>"
            "<p style='color:#6b6b7b;font-size:0.78rem;margin:0'>"
            "No active alerts — all machines nominal</p></div>",
            unsafe_allow_html=True)
    else:
        for _, r in df_al.iterrows():
            p         = r["alert_priority"]
            trend     = trend_icon.get(r.get("severity_trend","STABLE"), "→ stable")
            fault     = r.get("inferred_fault","–")
            cost      = r.get("est_cost_usd", 0)
            dtime     = r.get("est_downtime_hr", 0)
            conf      = r.get("anomaly_confidence", 0)
            rul       = r.get("rul_hours", 0)
            rec       = r.get("recommendation","")
            clean_rec = rec.replace("🔴","").replace("🟠","").replace("🟡","")\
                           .replace("🟢","").replace("✅","").strip()
            dot_c   = dot_cls.get(p, "dot-low")
            badge_c = badge_cls.get(p, "badge-low")
            st.markdown(
                f"<div class='alert-row'>"
                f"  <div class='alert-dot {dot_c}'></div>"
                f"  <div style='flex:1;min-width:0'>"
                f"    <div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap'>"
                f"      <span class='alert-machine'>{r['machine_id']}</span>"
                f"      <span style='font-size:0.7rem;color:#4a4a5a'>{r['machine_type']}</span>"
                f"      <span class='alert-badge {badge_c}'>{p}</span>"
                f"    </div>"
                f"    <div class='alert-meta'>"
                f"      Health {r['health_score']:.0f} &nbsp;·&nbsp; "
                f"      Fail {r['failure_prob']:.0f}% &nbsp;·&nbsp; "
                f"      Conf {conf:.0f} &nbsp;·&nbsp; "
                f"      RUL {rul:.0f} hr &nbsp;·&nbsp; "
                f"      {trend} &nbsp;·&nbsp; {fault}"
                f"    </div>"
                f"    <div class='alert-rec'>{clean_rec}</div>"
                f"    <div style='font-size:0.68rem;color:#4a4a5a;margin-top:4px'>"
                f"      Est. cost <strong style='color:#6b6b7b'>${cost:,}</strong>"
                f"      &nbsp;·&nbsp; downtime <strong style='color:#6b6b7b'>{dtime} hr</strong>"
                f"    </div>"
                f"  </div>"
                f"</div>",
                unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Machine detail
# ═══════════════════════════════════════════════════════════════════════════════
with t4:
    opts = sorted(dv["machine_id"].tolist())
    if not opts:
        st.warning("No machines match filters.")
    else:
        sel = st.selectbox("Machine", opts, label_visibility="collapsed")
        mr  = dv[dv["machine_id"]==sel].iloc[0]

        # ── top strip: gauge + metadata ──────────────────────────────────
        section(f"{sel}  ·  {mr['machine_type']}")
        gc_left, gc_right = st.columns([1, 2])

        with gc_left:
            h  = mr["health_score"]
            hc = GREEN if h>=70 else YELLOW if h>=40 else RED
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=h,
                title={"text":"Health", "font":{"color":TICK,"size":11}},
                number={"font":{"color":hc,"size":46,"family":"Inter"},"suffix":""},
                gauge={
                    "axis":    {"range":[0,100],"tickcolor":TICK,"tickwidth":1,
                                "ticklen":4,"nticks":6},
                    "bar":     {"color":hc,"thickness":0.18},
                    "bgcolor": BG,
                    "borderwidth":0,
                    "steps": [
                        {"range":[0,  40],"color":"#1a0d0d"},
                        {"range":[40, 70],"color":"#1a1500"},
                        {"range":[70,100],"color":"#0a1f0f"},
                    ],
                    "threshold":{"line":{"color":RED,"width":2},
                                 "thickness":0.6,"value":40},
                },
            ))
            fig_g.update_layout(
                paper_bgcolor=PAPER, font=dict(family="Inter",color=TICK),
                height=240, margin=dict(l=12,r=12,t=32,b=12))
            st.plotly_chart(fig_g, use_container_width=True)

        with gc_right:
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            r1c1.metric("OEE",      f"{mr.get('oee',0):.1f}%")
            r1c2.metric("Fail prob",f"{mr['failure_prob']:.1f}%")
            r1c3.metric("RUL",      f"{mr.get('rul_hours',0):.0f} hr")
            r1c4.metric("Runtime",  f"{mr['runtime_hours']:,.0f} hr")

            st.markdown("<br>", unsafe_allow_html=True)
            r2c1,r2c2,r2c3 = st.columns(3)
            r2c1.markdown(f"**Risk** &nbsp; `{mr['risk_level']}`")
            r2c2.markdown(f"**RUL status** &nbsp; `{mr.get('rul_urgency','–')}`")
            r2c3.markdown(f"**Trend** &nbsp; `{mr.get('severity_trend','–')}`")

            st.markdown(
                f"<div style='margin-top:10px;padding:10px 12px;background:#0f0f11;"
                f"border:1px solid #1e1e24;border-radius:6px;font-size:0.75rem;"
                f"color:#9898a6;line-height:1.5'>"
                f"<strong style='color:#6b6b7b'>Fault</strong>  "
                f"{mr.get('inferred_fault','–')}&emsp;"
                f"<strong style='color:#6b6b7b'>Anom. conf</strong>  "
                f"{mr.get('anomaly_confidence',0):.0f}<br>"
                f"<strong style='color:#6b6b7b'>Recommendation</strong><br>"
                f"{mr.get('recommendation','').replace('🔴','').replace('🟠','').replace('🟡','').replace('🟢','').replace('✅','').strip()}"
                f"</div>", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── sensor readings row ───────────────────────────────────────────
        section("Current readings")
        s1,s2,s3,s4,s5,s6,s7 = st.columns(7)
        s1.metric("Temp °C",     f"{mr['temperature']:.1f}")
        s2.metric("Vibration",   f"{mr['vibration']:.3f}")
        s3.metric("Pressure",    f"{mr['pressure']:.2f}")
        s4.metric("Power kW",    f"{mr['power_kw']:.1f}")
        s5.metric("Humidity %",  f"{mr['humidity']:.1f}")
        s6.metric("RPM",         f"{mr['rpm']:,.0f}")
        s7.metric("Oil °C",      f"{mr['oil_temp']:.1f}")

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── time-series subplots ──────────────────────────────────────────
        section("Sensor history")
        df_mts = df_full[df_full["machine_id"]==sel].copy()
        df_mts["timestamp"] = pd.to_datetime(df_mts["timestamp"])

        sensors   = ["temperature","vibration","pressure","power_kw","humidity","rpm","oil_temp"]
        subtitles = ["Temp °C","Vib","Press","kW","Hum %","RPM","Oil °C"]
        clrs      = [BLUE, ORANGE, GREEN, ACCENT, "#a371f7", "#39d353", YELLOW]
        positions = [(1,1),(1,2),(1,3),(1,4),(2,1),(2,2),(2,3)]

        fig_ts = make_subplots(rows=2, cols=4, subplot_titles=subtitles,
                               vertical_spacing=0.12, horizontal_spacing=0.06)
        for s,(r,c),cl in zip(sensors, positions, clrs):
            fig_ts.add_trace(go.Scatter(
                x=df_mts["timestamp"], y=df_mts[s],
                mode="lines", line=dict(color=cl,width=1.5), name=s,
                showlegend=False), row=r, col=c)
            anom = df_mts[df_mts["is_anomaly"]]
            if not anom.empty:
                fig_ts.add_trace(go.Scatter(
                    x=anom["timestamp"], y=anom[s], mode="markers",
                    marker=dict(color=RED,size=5,symbol="x"),
                    showlegend=False), row=r, col=c)

        fig_ts.update_layout(
            paper_bgcolor=PAPER, plot_bgcolor=BG,
            font=dict(family="Inter",color=TICK,size=10),
            height=420, margin=dict(l=12,r=12,t=36,b=12),
            title_text=None)
        fig_ts.update_xaxes(gridcolor=GRID, linecolor=GRID, tickfont_size=9)
        fig_ts.update_yaxes(gridcolor=GRID, linecolor=GRID, tickfont_size=9)
        for ann in fig_ts.layout.annotations:
            ann.font.update(size=10, color=TICK)
        st.plotly_chart(fig_ts, use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── maintenance history ───────────────────────────────────────────
        section("Maintenance history")
        hist = simulate_maintenance_history(sel)
        st.dataframe(
            hist.style.set_properties(**{
                "background-color":"#13131a","color":"#c8c8d0","font-size":"0.76rem"
            }),
            use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Forecast & RUL
# ═══════════════════════════════════════════════════════════════════════════════
with t5:
    section("Remaining useful life — all machines")

    rul_df = dv[["machine_id","machine_type","rul_hours","rul_urgency",
                 "health_score","risk_level"]].sort_values("rul_hours").copy()

    fig_rul = go.Figure()
    for urg, col in URG_COLOURS.items():
        sub = rul_df[rul_df["rul_urgency"]==urg]
        if sub.empty:
            continue
        fig_rul.add_trace(go.Bar(
            y=sub["machine_id"], x=sub["rul_hours"],
            name=urg, orientation="h", marker_color=col,
            customdata=sub[["machine_type","health_score","risk_level"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>%{customdata[0]}<br>"
                "RUL: %{x:.0f} hr &nbsp;·&nbsp; "
                "Health: %{customdata[1]:.0f}<extra></extra>"
            ),
        ))
    fig_rul.add_vline(x=100,  line_dash="dot", line_color=RED,    line_width=1)
    fig_rul.add_vline(x=500,  line_dash="dot", line_color=ORANGE, line_width=1)
    fig_rul.add_vline(x=2000, line_dash="dot", line_color=YELLOW, line_width=1)
    fig_rul.update_layout(
        barmode="stack",
        height=max(380, len(rul_df)*18),
        xaxis_title="RUL (hours)",
        paper_bgcolor=PAPER, plot_bgcolor=BG,
        font=dict(family="Inter",color=TICK,size=10),
        legend=dict(bgcolor=PAPER,bordercolor=GRID,borderwidth=1,
                    orientation="h",yanchor="bottom",y=1.02,
                    font=dict(size=10)),
        margin=dict(l=16,r=16,t=36,b=16),
    )
    fig_rul.update_xaxes(gridcolor=GRID, linecolor=GRID)
    fig_rul.update_yaxes(gridcolor=GRID, linecolor=GRID, tickfont_size=9)
    st.plotly_chart(fig_rul, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # urgency donut + MTBF
    fc1, fc2 = st.columns(2)
    with fc1:
        section("RUL urgency breakdown")
        rc2 = rul_df["rul_urgency"].value_counts().reset_index()
        rc2.columns = ["Urgency","Count"]
        fig_d = px.pie(rc2, values="Count", names="Urgency", hole=0.6,
                       color="Urgency", color_discrete_map=URG_COLOURS)
        fig_d.update_traces(textfont_size=10, textinfo="percent+label",
                            marker=dict(line=dict(color=BG,width=2)))
        st.plotly_chart(_theme(fig_d, height=300), use_container_width=True)

    with fc2:
        section("MTBF estimate per machine")
        mtbf_df = estimate_mtbf(df_full[df_full["machine_id"].isin(dv["machine_id"])])
        mtbf_df = mtbf_df.merge(dv[["machine_id","machine_type"]], on="machine_id", how="left")
        fig_m = px.bar(
            mtbf_df.sort_values("mtbf_hours"),
            x="mtbf_hours", y="machine_id", orientation="h",
            color="mtbf_hours", title=None,
            color_continuous_scale=[RED, YELLOW, GREEN])
        fig_m.update_layout(showlegend=False, coloraxis_showscale=False,
                            yaxis_tickfont_size=9,
                            height=max(300, len(mtbf_df)*18))
        st.plotly_chart(_theme(fig_m, height=max(300,len(mtbf_df)*18)),
                        use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # health trajectory
    section("Health score trajectory")
    sel2 = st.selectbox("Select machine", sorted(dv["machine_id"].tolist()),
                        key="forecast_sel")
    df_tj = df_full[df_full["machine_id"]==sel2].copy()
    df_tj["timestamp"] = pd.to_datetime(df_tj["timestamp"])

    if "health_score" in df_tj.columns:
        fig_tj = go.Figure()
        fig_tj.add_trace(go.Scatter(
            x=df_tj["timestamp"], y=df_tj["health_score"],
            mode="lines", name="Health",
            line=dict(color=ACCENT, width=2)))
        x_num = np.arange(len(df_tj))
        if len(x_num) >= 2:
            sl, ic = np.polyfit(x_num, df_tj["health_score"].values, 1)
            fig_tj.add_trace(go.Scatter(
                x=df_tj["timestamp"], y=sl*x_num+ic,
                mode="lines", name="Trend",
                line=dict(color=ORANGE, dash="dash", width=1.2)))
        fig_tj.add_hline(y=40, line_dash="dot", line_color=RED,    line_width=1)
        fig_tj.add_hline(y=70, line_dash="dot", line_color=YELLOW, line_width=1)
        fig_tj.update_layout(yaxis_range=[0,105], title=None)
        st.plotly_chart(_theme(fig_tj, height=320), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Export
# ═══════════════════════════════════════════════════════════════════════════════
with t6:
    section("Fleet snapshot")
    e1,e2,e3,e4,e5 = st.columns(5)
    stat(e1, len(dv),                              "Machines",      "#e2e2e5")
    stat(e2, kpis.get("fleet_health_index","–"),   "Avg health",    GREEN)
    stat(e3, f"{kpis.get('avg_oee','–')}%",        "Avg OEE",       ACCENT)
    stat(e4, kpis.get("critical_count","–"),       "Critical",      RED)
    stat(e5, f"${kpis.get('total_cost_exposure',0):,}", "Exposure", TICK)

    st.markdown("<br>", unsafe_allow_html=True)

    # CSV
    section("CSV download")
    export_cols = [
        "machine_id","machine_type","temperature","vibration","pressure",
        "power_kw","humidity","rpm","oil_temp","runtime_hours",
        "health_score","failure_prob","risk_level","is_anomaly",
        "anomaly_confidence","anomaly_category","oee","rul_hours","rul_urgency",
        "alert_priority","recommendation","est_cost_usd","est_downtime_hr",
        "inferred_fault","severity_trend",
    ]
    avail = [c for c in export_cols if c in dv.columns]
    csv_b = dv[avail].to_csv(index=False).encode()
    st.download_button(
        "Download fleet CSV", csv_b,
        "forgemind_fleet.csv", "text/csv",
        use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # PDF
    section("PDF report")

    def _build_pdf(df: pd.DataFrame, kpi_data: dict) -> bytes:
        try:
            from fpdf import FPDF
        except ImportError:
            return b""

        from datetime import datetime as _dt
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=14)
        pdf.add_page()

        # header
        pdf.set_font("Helvetica","B",18)
        pdf.set_text_color(124,92,252)
        pdf.cell(0, 10, "ForgeMind AI — Fleet Health Report", ln=True, align="C")
        pdf.set_font("Helvetica","",9)
        pdf.set_text_color(107,107,123)
        pdf.cell(0,5,"Industrial Intelligence Platform v2.0",ln=True,align="C")
        pdf.cell(0,5,f"Generated {_dt.now().strftime('%Y-%m-%d  %H:%M')}",ln=True,align="C")
        pdf.ln(6)

        # KPI table
        pdf.set_font("Helvetica","B",11)
        pdf.set_text_color(226,226,229)
        pdf.cell(0,7,"Fleet KPIs",ln=True)
        pdf.set_font("Helvetica","",9)
        rows_kpi = [
            ("Machines",             str(len(df))),
            ("Fleet health index",   str(kpi_data.get("fleet_health_index","–"))),
            ("Average OEE",          f"{kpi_data.get('avg_oee','–')}%"),
            ("Critical alerts",      str(kpi_data.get("critical_count","–"))),
            ("Critical RUL",         str(kpi_data.get("machines_critical_rul","–"))),
            ("Cost exposure",        f"${kpi_data.get('total_cost_exposure',0):,}"),
            ("Downtime exposure",    f"{kpi_data.get('total_downtime_exposure',0):.0f} hr"),
            ("Top fault",            str(kpi_data.get("top_fault","–"))),
        ]
        for lbl,val in rows_kpi:
            pdf.set_text_color(107,107,123); pdf.cell(72,5,lbl+":",ln=False)
            pdf.set_text_color(226,226,229); pdf.cell(0,5,val,ln=True)
        pdf.ln(5)

        # critical/high table
        ch = df[df["alert_priority"].isin(["CRITICAL","HIGH"])].sort_values("health_score")
        if not ch.empty:
            pdf.set_font("Helvetica","B",11)
            pdf.set_text_color(229,83,75)
            pdf.cell(0,7,f"Critical & High Alerts  ({len(ch)})",ln=True)
            pdf.set_font("Helvetica","B",7)
            pdf.set_text_color(226,226,229)
            hdrs = ["Machine","Type","Health","Fail%","RUL h","Priority","Fault","Cost"]
            wds  = [22,38,13,13,16,18,36,20]
            for h,w in zip(hdrs,wds):
                pdf.cell(w,5,h,border=1,align="C")
            pdf.ln()
            pdf.set_font("Helvetica","",7)
            for _,row in ch.iterrows():
                pdf.set_text_color(152,152,166)
                vals=[str(row["machine_id"]),str(row["machine_type"])[:20],
                      f"{row['health_score']:.1f}",f"{row['failure_prob']:.1f}",
                      f"{row.get('rul_hours',0):.0f}",str(row["alert_priority"]),
                      str(row.get('inferred_fault','–'))[:20],
                      f"${row.get('est_cost_usd',0):,}"]
                for v,w in zip(vals,wds):
                    pdf.cell(w,5,v,border=1,align="C")
                pdf.ln()
            pdf.ln(4)

        # full table on new page
        pdf.add_page()
        pdf.set_font("Helvetica","B",11)
        pdf.set_text_color(124,92,252)
        pdf.cell(0,7,"Full Fleet Status",ln=True)
        pdf.set_font("Helvetica","B",7)
        pdf.set_text_color(226,226,229)
        hdrs2 = ["Machine","Type","Health","OEE%","RUL h","Risk","Trend","Recommendation"]
        wds2  = [22,36,12,12,14,14,22,68]
        for h,w in zip(hdrs2,wds2):
            pdf.cell(w,5,h,border=1,align="C")
        pdf.ln()
        pdf.set_font("Helvetica","",6)
        for _,row in df.sort_values("health_score").iterrows():
            rec = str(row.get("recommendation","")).replace("🔴","").replace("🟠","")\
                     .replace("🟡","").replace("🟢","").replace("✅","").strip()[:60]
            pdf.set_text_color(152,152,166)
            vals2=[str(row["machine_id"]),str(row["machine_type"])[:20],
                   f"{row['health_score']:.1f}",f"{row.get('oee',0):.1f}",
                   f"{row.get('rul_hours',0):.0f}",str(row["risk_level"]),
                   str(row.get("severity_trend","–"))[:12],rec]
            for v,w in zip(vals2,wds2):
                pdf.cell(w,4,v,border=1,align="C")
            pdf.ln()

        # footer
        pdf.ln(6)
        pdf.set_font("Helvetica","I",7)
        pdf.set_text_color(58,58,74)
        pdf.cell(0,4,"ForgeMind AI v2.0 · IsolationForest + LOF · © 2026 ForgeMind Technologies",align="C")

        return bytes(pdf.output())

    if st.button("Generate PDF", use_container_width=True):
        with st.spinner("Building PDF…"):
            pdf_b = _build_pdf(dv, kpis)
        if pdf_b:
            st.download_button("Download PDF", pdf_b,
                               "forgemind_fleet.pdf","application/pdf",
                               use_container_width=True)
        else:
            st.error("fpdf2 not installed — run: pip install fpdf2")

    st.markdown("<br>", unsafe_allow_html=True)

    # tables
    section("Efficiency matrix by machine type")
    eff2 = efficiency_matrix(dv)
    if not eff2.empty:
        fmt = {"avg_oee":"{:.1f}","avg_health":"{:.1f}",
               "avg_failure_prob":"{:.1f}","anomaly_rate_pct":"{:.1f}"}
        st.dataframe(
            eff2.style.format(fmt).set_properties(**{
                "background-color":"#13131a","color":"#c8c8d0","font-size":"0.76rem"
            }),
            use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("Fault distribution")
    fd2 = fault_distribution(dv)
    if not fd2.empty:
        st.dataframe(
            fd2.style.format({"pct":"{:.1f}%"}).set_properties(**{
                "background-color":"#13131a","color":"#c8c8d0","font-size":"0.76rem"
            }),
            use_container_width=True)
