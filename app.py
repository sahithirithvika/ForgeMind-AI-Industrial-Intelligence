"""
ForgeMind AI – Autonomous Industrial Intelligence Platform
Main Streamlit Dashboard
"""

import io
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Path setup – allow running from any working directory
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.generate_data import generate_dataset
from engine.anomaly_detection import train_and_predict
from engine.risk_scoring import add_risk_columns
from engine.maintenance_engine import add_recommendations, simulate_maintenance_history

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ForgeMind AI – Industrial Intelligence Platform",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS – dark industrial theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Global ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: 'Segoe UI', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    /* ── KPI cards ── */
    .kpi-card {
        background: linear-gradient(135deg, #1c2a3a 0%, #162032 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 4px;
    }
    .kpi-value { font-size: 2.2rem; font-weight: 700; margin: 0; }
    .kpi-label { font-size: 0.82rem; color: #8b949e; text-transform: uppercase;
                 letter-spacing: 0.08em; margin-top: 4px; }
    /* ── Alert cards ── */
    .alert-critical { background:#2d0f0f; border-left:4px solid #f85149;
                      border-radius:8px; padding:12px 16px; margin-bottom:8px; }
    .alert-high     { background:#2b1a00; border-left:4px solid #f0883e;
                      border-radius:8px; padding:12px 16px; margin-bottom:8px; }
    .alert-medium   { background:#2b2200; border-left:4px solid #d29922;
                      border-radius:8px; padding:12px 16px; margin-bottom:8px; }
    .alert-info     { background:#0d2818; border-left:4px solid #3fb950;
                      border-radius:8px; padding:12px 16px; margin-bottom:8px; }
    /* ── Section headers ── */
    .section-header {
        font-size: 1.05rem; font-weight: 600; color: #58a6ff;
        text-transform: uppercase; letter-spacing: 0.1em;
        border-bottom: 1px solid #30363d; padding-bottom: 6px; margin-bottom: 16px;
    }
    /* ── Dataframe tweaks ── */
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
    /* ── Button ── */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #1f6feb, #388bfd);
        color: white; border: none; border-radius: 8px;
        padding: 10px 20px; font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data pipeline (cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner="Running AI pipeline…")
def load_pipeline() -> pd.DataFrame:
    df = generate_dataset()
    df = train_and_predict(df)
    df = add_risk_columns(df)
    df = add_recommendations(df)
    return df


df_full = load_pipeline()

# Latest snapshot per machine (last timestamp for each machine_id)
df_latest = (
    df_full.sort_values("timestamp")
    .groupby("machine_id", as_index=False)
    .last()
)

# ---------------------------------------------------------------------------
# Sidebar – filters & controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding:10px 0 20px'>"
        "<span style='font-size:2rem'>⚙️</span><br>"
        "<span style='font-size:1.2rem; font-weight:700; color:#58a6ff'>ForgeMind AI</span><br>"
        "<span style='font-size:0.75rem; color:#8b949e'>Industrial Intelligence Platform</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### 🔧 Filters")

    selected_types = st.multiselect(
        "Machine Type",
        options=sorted(df_latest["machine_type"].unique()),
        default=sorted(df_latest["machine_type"].unique()),
    )

    risk_filter = st.multiselect(
        "Risk Level",
        options=["Low", "Medium", "High"],
        default=["Low", "Medium", "High"],
    )

    st.markdown("---")
    st.markdown("### ⚙️ AI Settings")
    contamination = st.slider(
        "Anomaly Sensitivity",
        min_value=0.05,
        max_value=0.30,
        value=0.12,
        step=0.01,
        help="Isolation Forest contamination parameter. Higher = more anomalies flagged.",
    )
    if st.button("🔄 Re-run AI Pipeline", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem; color:#484f58; text-align:center'>"
        "ForgeMind AI v1.0.0<br>Powered by Edge AI · Scikit-Learn<br>"
        "© 2026 ForgeMind Technologies"
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------
mask = (
    df_latest["machine_type"].isin(selected_types)
    & df_latest["risk_level"].isin(risk_filter)
)
df_view = df_latest[mask].copy()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='color:#58a6ff; margin-bottom:4px'>⚙️ ForgeMind AI</h1>"
    "<p style='color:#8b949e; margin-top:0'>Autonomous Industrial Intelligence Platform · "
    "Edge AI Predictive Maintenance Dashboard</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ---------------------------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------------------------
total_machines   = len(df_view)
healthy          = (df_view["risk_level"] == "Low").sum()
at_risk          = (df_view["risk_level"].isin(["Medium", "High"])).sum()
active_alerts    = (df_view["is_anomaly"]).sum()
critical_alerts  = (df_view["alert_priority"] == "CRITICAL").sum()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        f"<div class='kpi-card'>"
        f"<p class='kpi-value' style='color:#58a6ff'>{total_machines}</p>"
        f"<p class='kpi-label'>Total Machines</p></div>",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"<div class='kpi-card'>"
        f"<p class='kpi-value' style='color:#3fb950'>{healthy}</p>"
        f"<p class='kpi-label'>Healthy Machines</p></div>",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"<div class='kpi-card'>"
        f"<p class='kpi-value' style='color:#d29922'>{at_risk}</p>"
        f"<p class='kpi-label'>Machines at Risk</p></div>",
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"<div class='kpi-card'>"
        f"<p class='kpi-value' style='color:#f0883e'>{active_alerts}</p>"
        f"<p class='kpi-label'>Active Alerts</p></div>",
        unsafe_allow_html=True,
    )
with col5:
    st.markdown(
        f"<div class='kpi-card'>"
        f"<p class='kpi-value' style='color:#f85149'>{critical_alerts}</p>"
        f"<p class='kpi-label'>Critical Alerts</p></div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ===========================================================================
# TAB LAYOUT
# ===========================================================================
tab_monitor, tab_charts, tab_alerts, tab_machine, tab_report = st.tabs(
    ["📊 Monitoring", "📈 Analytics", "🚨 Alerts", "🔩 Machine Detail", "📄 Report"]
)

# ---------------------------------------------------------------------------
# TAB 1 – Real-Time Monitoring Table
# ---------------------------------------------------------------------------
with tab_monitor:
    st.markdown("<div class='section-header'>Real-Time Machine Monitoring</div>",
                unsafe_allow_html=True)

    def colour_risk(val):
        colours = {"Low": "#3fb950", "Medium": "#d29922", "High": "#f85149"}
        c = colours.get(val, "white")
        return f"color: {c}; font-weight: 600"

    def colour_score(val):
        if val >= 70:   return "color: #3fb950; font-weight:600"
        if val >= 40:   return "color: #d29922; font-weight:600"
        return "color: #f85149; font-weight:600"

    table_cols = [
        "machine_id", "machine_type", "temperature", "vibration",
        "pressure", "power_kw", "runtime_hours",
        "health_score", "failure_prob", "risk_level", "is_anomaly",
    ]
    display_df = df_view[table_cols].rename(columns={
        "machine_id":    "Machine ID",
        "machine_type":  "Type",
        "temperature":   "Temp (°C)",
        "vibration":     "Vibration",
        "pressure":      "Pressure",
        "power_kw":      "Power (kW)",
        "runtime_hours": "Runtime (hr)",
        "health_score":  "Health Score",
        "failure_prob":  "Fail Prob (%)",
        "risk_level":    "Risk Level",
        "is_anomaly":    "Anomaly",
    }).sort_values("Health Score")

    styled = (
        display_df.style
        .applymap(colour_risk,    subset=["Risk Level"])
        .applymap(colour_score,   subset=["Health Score"])
        .format({
            "Temp (°C)":    "{:.1f}",
            "Vibration":    "{:.3f}",
            "Pressure":     "{:.2f}",
            "Power (kW)":   "{:.1f}",
            "Runtime (hr)": "{:.0f}",
            "Health Score": "{:.1f}",
            "Fail Prob (%)":"{:.1f}",
        })
        .set_properties(**{"background-color": "#161b22", "color": "#e6edf3"})
    )
    st.dataframe(styled, use_container_width=True, height=480)

# ---------------------------------------------------------------------------
# TAB 2 – Analytics Charts
# ---------------------------------------------------------------------------
with tab_charts:
    st.markdown("<div class='section-header'>Analytics & Trends</div>",
                unsafe_allow_html=True)

    CHART_BG   = "#0d1117"
    GRID_COLOR = "#21262d"
    TEXT_COLOR = "#8b949e"

    def apply_dark_theme(fig):
        fig.update_layout(
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
            font_color=TEXT_COLOR,
            legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        )
        fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
        fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
        return fig

    # ── Row 1: Temperature & Vibration trends (last 10 machines) ──────────
    col_l, col_r = st.columns(2)

    machine_sample = df_view.sort_values("health_score").head(10)["machine_id"].tolist()
    df_trend = df_full[df_full["machine_id"].isin(machine_sample)].copy()
    df_trend["timestamp"] = pd.to_datetime(df_trend["timestamp"])

    with col_l:
        fig_temp = px.line(
            df_trend, x="timestamp", y="temperature", color="machine_id",
            title="🌡️ Temperature Trends – At-Risk Machines",
            labels={"temperature": "Temp (°C)", "timestamp": "Time"},
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_temp.add_hline(y=90,  line_dash="dot", line_color="#f85149",
                           annotation_text="Critical Threshold")
        fig_temp.add_hline(y=75,  line_dash="dot", line_color="#d29922",
                           annotation_text="Warning Threshold")
        fig_temp = apply_dark_theme(fig_temp)
        st.plotly_chart(fig_temp, use_container_width=True)

    with col_r:
        fig_vib = px.line(
            df_trend, x="timestamp", y="vibration", color="machine_id",
            title="📳 Vibration Trends – At-Risk Machines",
            labels={"vibration": "Vibration (mm/s)", "timestamp": "Time"},
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_vib.add_hline(y=4.0, line_dash="dot", line_color="#d29922",
                          annotation_text="Warning")
        fig_vib.add_hline(y=6.0, line_dash="dot", line_color="#f85149",
                          annotation_text="Critical")
        fig_vib = apply_dark_theme(fig_vib)
        st.plotly_chart(fig_vib, use_container_width=True)

    # ── Row 2: Health distribution & Risk pie ─────────────────────────────
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        fig_hist = px.histogram(
            df_view, x="health_score", nbins=20,
            title="🏥 Machine Health Score Distribution",
            labels={"health_score": "Health Score (0–100)", "count": "Machines"},
            color_discrete_sequence=["#388bfd"],
        )
        fig_hist.add_vline(x=40, line_dash="dot", line_color="#f85149",
                           annotation_text="High Risk")
        fig_hist.add_vline(x=70, line_dash="dot", line_color="#d29922",
                           annotation_text="Medium Risk")
        fig_hist = apply_dark_theme(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_r2:
        risk_counts = df_view["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig_pie = px.pie(
            risk_counts, values="Count", names="Risk Level",
            title="⚠️ Risk Level Breakdown",
            color="Risk Level",
            color_discrete_map={"Low": "#3fb950", "Medium": "#d29922", "High": "#f85149"},
            hole=0.45,
        )
        fig_pie.update_traces(textfont_size=13, textinfo="label+percent+value")
        fig_pie = apply_dark_theme(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Row 3: Scatter – Temp vs Vibration, coloured by risk ──────────────
    fig_scatter = px.scatter(
        df_view,
        x="temperature", y="vibration",
        color="risk_level",
        size="failure_prob",
        hover_data=["machine_id", "health_score", "machine_type"],
        title="🔍 Temperature vs Vibration – Risk Correlation",
        labels={"temperature": "Temperature (°C)", "vibration": "Vibration (mm/s)"},
        color_discrete_map={"Low": "#3fb950", "Medium": "#d29922", "High": "#f85149"},
    )
    fig_scatter = apply_dark_theme(fig_scatter)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Row 4: Power consumption bar by machine type ──────────────────────
    power_by_type = (
        df_view.groupby("machine_type")[["power_kw", "health_score"]]
        .mean()
        .reset_index()
    )
    fig_bar = px.bar(
        power_by_type, x="machine_type", y="power_kw",
        color="health_score",
        title="⚡ Average Power Consumption by Machine Type",
        labels={"power_kw": "Avg Power (kW)", "machine_type": "Machine Type"},
        color_continuous_scale=["#f85149", "#d29922", "#3fb950"],
        range_color=[0, 100],
    )
    fig_bar = apply_dark_theme(fig_bar)
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3 – Alerts Panel
# ---------------------------------------------------------------------------
with tab_alerts:
    st.markdown("<div class='section-header'>Active Alerts & Maintenance Recommendations</div>",
                unsafe_allow_html=True)

    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
    df_alerts = (
        df_view[df_view["alert_priority"].isin(["CRITICAL", "HIGH", "MEDIUM", "LOW"])]
        .copy()
    )
    df_alerts["_sort"] = df_alerts["alert_priority"].map(priority_order)
    df_alerts = df_alerts.sort_values(["_sort", "health_score"])

    if df_alerts.empty:
        st.success("✅ No active alerts. All machines operating normally.")
    else:
        css_map = {
            "CRITICAL": "alert-critical",
            "HIGH":     "alert-high",
            "MEDIUM":   "alert-medium",
            "LOW":      "alert-info",
        }
        for _, row in df_alerts.iterrows():
            css = css_map.get(row["alert_priority"], "alert-info")
            st.markdown(
                f"<div class='{css}'>"
                f"<strong>{row['machine_id']}</strong> · {row['machine_type']}<br>"
                f"<span style='font-size:0.85rem; color:#8b949e'>"
                f"Health: <b>{row['health_score']}</b> | "
                f"Fail Prob: <b>{row['failure_prob']}%</b> | "
                f"Priority: <b>{row['alert_priority']}</b>"
                f"</span><br>"
                f"<span style='margin-top:6px; display:block'>{row['recommendation']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# TAB 4 – Machine Detail (gauge + history)
# ---------------------------------------------------------------------------
with tab_machine:
    st.markdown("<div class='section-header'>Machine Deep Dive</div>",
                unsafe_allow_html=True)

    machine_options = sorted(df_view["machine_id"].tolist())
    if not machine_options:
        st.warning("No machines match the current filters.")
    else:
        selected_machine = st.selectbox("Select Machine", machine_options)
        mrow = df_view[df_view["machine_id"] == selected_machine].iloc[0]

        # ── Gauge ─────────────────────────────────────────────────────────
        col_gauge, col_info = st.columns([1, 2])

        with col_gauge:
            health = mrow["health_score"]
            gauge_color = (
                "#3fb950" if health >= 70
                else "#d29922" if health >= 40
                else "#f85149"
            )
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=health,
                delta={"reference": 70, "decreasing": {"color": "#f85149"}},
                title={"text": f"Health Score<br><span style='font-size:0.8em;color:#8b949e'>{selected_machine}</span>",
                       "font": {"color": "#e6edf3"}},
                number={"font": {"color": gauge_color, "size": 52}},
                gauge={
                    "axis":       {"range": [0, 100], "tickcolor": "#8b949e"},
                    "bar":        {"color": gauge_color},
                    "bgcolor":    "#161b22",
                    "bordercolor":"#30363d",
                    "steps": [
                        {"range": [0,  40], "color": "#2d0f0f"},
                        {"range": [40, 70], "color": "#2b2200"},
                        {"range": [70, 100],"color": "#0d2818"},
                    ],
                    "threshold": {
                        "line": {"color": "#f85149", "width": 3},
                        "thickness": 0.75,
                        "value": 40,
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0d1117",
                height=320,
                font_color="#e6edf3",
                margin=dict(l=20, r=20, t=60, b=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_info:
            st.markdown(f"**Machine ID:** {mrow['machine_id']}")
            st.markdown(f"**Type:** {mrow['machine_type']}")
            st.markdown(f"**Risk Level:** {mrow['risk_level']}")
            st.markdown(f"**Failure Probability:** {mrow['failure_prob']}%")
            st.markdown(f"**Runtime Hours:** {mrow['runtime_hours']:,.0f} hr")
            st.markdown("---")
            st.markdown("**Current Sensor Readings:**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Temperature", f"{mrow['temperature']:.1f} °C")
            m2.metric("Vibration",   f"{mrow['vibration']:.3f}")
            m3.metric("Pressure",    f"{mrow['pressure']:.2f}")
            m4.metric("Power",       f"{mrow['power_kw']:.1f} kW")
            st.markdown("---")
            st.markdown(f"**Recommendation:**<br>{mrow['recommendation']}",
                        unsafe_allow_html=True)

        # ── Sensor time series for selected machine ────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Sensor Time Series</div>",
                    unsafe_allow_html=True)

        df_machine_ts = df_full[df_full["machine_id"] == selected_machine].copy()
        df_machine_ts["timestamp"] = pd.to_datetime(df_machine_ts["timestamp"])

        fig_ts = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Temperature (°C)", "Vibration (mm/s)",
                            "Pressure (bar)", "Power (kW)"),
        )
        colours = ["#388bfd", "#f0883e", "#3fb950", "#d29922"]
        sensors = ["temperature", "vibration", "pressure", "power_kw"]
        positions = [(1,1),(1,2),(2,1),(2,2)]

        for sensor, (r, c), colour in zip(sensors, positions, colours):
            fig_ts.add_trace(
                go.Scatter(
                    x=df_machine_ts["timestamp"],
                    y=df_machine_ts[sensor],
                    mode="lines",
                    name=sensor,
                    line=dict(color=colour, width=1.8),
                ),
                row=r, col=c,
            )
            anomaly_df = df_machine_ts[df_machine_ts["is_anomaly"]]
            fig_ts.add_trace(
                go.Scatter(
                    x=anomaly_df["timestamp"],
                    y=anomaly_df[sensor],
                    mode="markers",
                    name=f"{sensor} anomaly",
                    marker=dict(color="#f85149", size=7, symbol="x"),
                    showlegend=False,
                ),
                row=r, col=c,
            )

        fig_ts.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font_color="#8b949e",
            height=500,
            showlegend=False,
            title_text=f"Sensor History – {selected_machine}",
        )
        fig_ts.update_xaxes(gridcolor="#21262d")
        fig_ts.update_yaxes(gridcolor="#21262d")
        st.plotly_chart(fig_ts, use_container_width=True)

        # ── Maintenance history ────────────────────────────────────────────
        st.markdown("<div class='section-header'>Maintenance History</div>",
                    unsafe_allow_html=True)
        hist_df = simulate_maintenance_history(selected_machine)
        st.dataframe(
            hist_df.style.set_properties(
                **{"background-color": "#161b22", "color": "#e6edf3"}
            ),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# TAB 5 – Report Download
# ---------------------------------------------------------------------------
with tab_report:
    st.markdown("<div class='section-header'>Export & Reporting</div>",
                unsafe_allow_html=True)

    st.markdown("Generate and download a summary report of the current fleet status.")

    # ── CSV download ──────────────────────────────────────────────────────
    report_cols = [
        "machine_id", "machine_type", "temperature", "vibration", "pressure",
        "power_kw", "runtime_hours", "health_score", "failure_prob",
        "risk_level", "is_anomaly", "alert_priority", "recommendation",
    ]
    csv_data = df_view[report_cols].to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Fleet Status CSV",
        data=csv_data,
        file_name="forgemind_fleet_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Summary stats ─────────────────────────────────────────────────────
    st.markdown("### Fleet Summary Statistics")
    summary = df_view[["temperature", "vibration", "pressure", "power_kw", "health_score"]].describe().T
    summary.columns = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    st.dataframe(
        summary.style
        .format("{:.2f}")
        .set_properties(**{"background-color": "#161b22", "color": "#e6edf3"}),
        use_container_width=True,
    )

    # ── Alert summary ─────────────────────────────────────────────────────
    st.markdown("### Alert Priority Summary")
    alert_summary = (
        df_view["alert_priority"]
        .value_counts()
        .reset_index()
    )
    alert_summary.columns = ["Priority", "Count"]
    st.dataframe(
        alert_summary.style.set_properties(
            **{"background-color": "#161b22", "color": "#e6edf3"}
        ),
        use_container_width=True,
    )
