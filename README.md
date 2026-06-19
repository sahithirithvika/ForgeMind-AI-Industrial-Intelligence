# ⚙️ ForgeMind AI – Autonomous Industrial Intelligence Platform

> **Edge AI for Connected, Secure & Intelligent Industrial Systems**  
> Predictive Maintenance · Anomaly Detection · Real-Time Fleet Monitoring

---

## 📋 Project Overview

ForgeMind AI is a lightweight but production-realistic Proof of Concept (PoC) for an **Edge AI-based industrial monitoring platform**.  
It simulates a fleet of heavy industrial machines, runs an AI pipeline for anomaly detection and health scoring, and presents everything in a professional Streamlit dashboard — suitable for a Tata Technologies hackathon demo.

---

## 🏗️ Project Structure

```
ForgeMind-AI/
├── app.py                        # Main Streamlit dashboard
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── data/
│   ├── __init__.py
│   └── generate_data.py          # Synthetic sensor data generator (1 200 records)
│
└── engine/
    ├── __init__.py
    ├── anomaly_detection.py      # Isolation Forest anomaly detection
    ├── risk_scoring.py           # Machine health score (0–100) + risk levels
    └── maintenance_engine.py     # Recommendation rules + maintenance history sim
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9 or higher  
- pip

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the dashboard
```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

---

## 🤖 AI Pipeline

```
Raw Sensor Data (1 200 records, 20 machines)
        │
        ▼
Isolation Forest (Anomaly Detection)
  → anomaly_score, is_anomaly
        │
        ▼
Risk Scoring Engine
  → health_score (0–100), risk_level (Low/Medium/High), failure_prob (%)
        │
        ▼
Maintenance Recommendation Engine
  → alert_priority (CRITICAL/HIGH/MEDIUM/LOW/INFO)
  → recommendation text
        │
        ▼
Streamlit Dashboard
```

---

## 📊 Dashboard Tabs

| Tab | Contents |
|-----|----------|
| **📊 Monitoring** | Real-time machine table with colour-coded health scores and risk levels |
| **📈 Analytics** | Temperature & vibration trends, health distribution, risk pie chart, scatter plot, power bar chart |
| **🚨 Alerts** | Prioritised alert panel with maintenance recommendations |
| **🔩 Machine Detail** | Health gauge, sensor metrics, 4-panel time series, maintenance history log |
| **📄 Report** | Fleet summary stats, alert breakdown, CSV export |

---

## 🔧 Simulated Machine Types

| Machine | Temp Range | Vibration | Pressure | Power |
|---------|-----------|-----------|----------|-------|
| CNC Milling Machine | 55–75 °C | 0.5–2.0 | 4–6 bar | 15–30 kW |
| Hydraulic Press | 60–85 °C | 1.0–3.5 | 10–18 bar | 30–55 kW |
| Industrial Compressor | 70–95 °C | 1.5–4.0 | 8–14 bar | 40–70 kW |
| Conveyor Drive Motor | 45–65 °C | 0.3–1.5 | 2–4 bar | 8–20 kW |
| Injection Moulding Unit | 80–110 °C | 0.8–2.5 | 12–20 bar | 50–80 kW |

---

## ⚠️ Fault Injection

~15% of machines have a fault injected into the last 20% of their sensor timeline:

| Fault | Effect |
|-------|--------|
| `overheating` | Temperature +20–45 °C, Power +10–25 kW |
| `bearing_wear` | Vibration +3–6, Temperature +5–15 °C |
| `pressure_leak` | Pressure −2–5 bar, Vibration +1–3 |
| `power_surge` | Power +20–40 kW, Temperature +10–20 °C |

---

## 📦 Dependencies

```
streamlit==1.35.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
plotly==5.22.0
fpdf2==2.7.9
```

---

## 🎯 Dashboard Screenshot Description

**Header:** Dark (#0d1117) background, blue accent title "⚙️ ForgeMind AI"

**Sidebar:** Machine type / risk filters, anomaly sensitivity slider, re-run AI button

**KPI Row:** 5 gradient cards — Total Machines (blue), Healthy (green), At Risk (amber), Active Alerts (orange), Critical (red)

**Monitoring Tab:** Dark-themed dataframe with colour-coded risk and health columns

**Analytics Tab:** 6 interactive Plotly charts in a dark theme (line, histogram, donut, scatter, bar)

**Alerts Tab:** Colour-coded alert cards (red=CRITICAL, orange=HIGH, yellow=MEDIUM, green=LOW)

**Machine Detail Tab:** Plotly gauge with green/amber/red zones, st.metric sensor cards, 4-panel sensor time series with anomaly markers (red ✕), simulated maintenance history table

**Report Tab:** Download CSV button (blue gradient), summary statistics table, alert breakdown

---

## 💡 Bonus Features Included

- ✅ Machine Health Gauge (Plotly Indicator)
- ✅ Predictive Failure Probability (%)
- ✅ Download Report Button (CSV)
- ✅ Maintenance History Simulation

---

*Built for Tata Technologies Hackathon · ForgeMind AI v1.0.0 · © 2026*
