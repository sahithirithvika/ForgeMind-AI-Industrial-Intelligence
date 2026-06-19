# ⚙️ ForgeMind AI – Autonomous Industrial Intelligence Platform

> **Edge AI for Connected, Secure & Intelligent Industrial Systems**
> Predictive Maintenance · Anomaly Detection · Real-Time Fleet Monitoring

---

## 🔗 Project Resources

| Resource              | Link                  |
| --------------------- | --------------------- |
| 🚀 Live Demo          | https://sahithirithvika-forgemind-ai-industrial-intelligence-app-xxxx.streamlit.app |
| 🎥 Demo Video         | https://drive.google.com/file/d/1jNz-bE_ieMWNWcXRPYZp_JH1s0fDf2gA/view?usp=sharing |
| 📊 Presentation (PPT) | https://drive.google.com/file/d/1QWKnbLyhb1Y3wDlxHqAYkKZhyskBzBKX/view?usp=sharing     |
| drive link            |https://drive.google.com/drive/folders/1vA5NphjBqNW6HphrO5ix6wKoz5CqyJZ6?usp=sharing |


---

## 📸 Dashboard Preview

![ForgeMind Dashboard](images/dashboard.png)

---

## 📋 Project Overview

ForgeMind AI is a lightweight yet production-oriented **Edge AI Industrial Monitoring Platform** designed to demonstrate how artificial intelligence can improve operational efficiency, machine reliability, and predictive maintenance in modern industrial environments.

The platform simulates a fleet of industrial machines, analyzes sensor telemetry using machine learning, detects anomalies in real time, predicts equipment health degradation, and generates actionable maintenance recommendations through an interactive dashboard.

Built as a Proof of Concept (PoC), ForgeMind AI showcases the future of intelligent manufacturing and Industry 4.0 systems.

---

## ✨ Key Features

* 🔍 AI-Powered Anomaly Detection using Isolation Forest
* 📊 Real-Time Fleet Monitoring Dashboard
* ⚠️ Predictive Failure Probability Estimation
* 🏥 Machine Health Scoring (0–100)
* 🛠️ Automated Maintenance Recommendations
* 📈 Interactive Data Analytics & Visualizations
* 📄 Downloadable Fleet Reports
* 🌐 Edge AI Inspired Industrial Architecture

---

## 🏗️ System Architecture

```text
Industrial Machines
        │
        ▼
Synthetic Sensor Data Generator
        │
        ▼
Anomaly Detection Engine
(Isolation Forest)
        │
        ▼
Risk Scoring Engine
        │
        ▼
Maintenance Recommendation Engine
        │
        ▼
Interactive Streamlit Dashboard
```

---

## 🤖 AI Processing Pipeline

```text
Raw Sensor Data (1,200 Records | 20 Machines)
                │
                ▼
      Isolation Forest
      (Anomaly Detection)
                │
                ▼
     Risk Scoring Engine
                │
                ▼
 Failure Probability Estimation
                │
                ▼
Maintenance Recommendation Engine
                │
                ▼
      Streamlit Dashboard
```

### AI Outputs

| Component          | Output                         |
| ------------------ | ------------------------------ |
| Anomaly Detection  | anomaly_score, is_anomaly      |
| Risk Scoring       | health_score, risk_level       |
| Prediction         | failure_probability            |
| Maintenance Engine | alert_priority, recommendation |

---

## 🛠️ Tech Stack

| Category             | Technologies                 |
| -------------------- | ---------------------------- |
| Frontend             | Streamlit                    |
| Programming Language | Python                       |
| Data Processing      | Pandas, NumPy                |
| Machine Learning     | Scikit-Learn                 |
| Visualization        | Plotly                       |
| Reporting            | FPDF2                        |
| Data Generation      | Custom Synthetic Data Engine |

---

## 📂 Project Structure

```text
ForgeMind-AI/
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── __init__.py
│   └── generate_data.py
│
└── engine/
    ├── __init__.py
    ├── anomaly_detection.py
    ├── risk_scoring.py
    └── maintenance_engine.py
```

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd ForgeMind-AI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Application

```bash
streamlit run app.py
```

Application launches at:

```text
http://localhost:8501
```

---

## 📊 Dashboard Modules

### 📊 Monitoring

* Live machine status monitoring
* Health score visualization
* Risk level classification
* Anomaly indicators

### 📈 Analytics

* Temperature trend analysis
* Vibration monitoring
* Risk distribution charts
* Power consumption analytics
* Machine performance insights

### 🚨 Alerts

* Prioritized maintenance alerts
* Critical issue identification
* Automated recommendations

### 🔩 Machine Detail

* Health gauge visualization
* Sensor performance metrics
* Time-series analysis
* Maintenance history tracking

### 📄 Reports

* Fleet health summary
* Alert breakdown statistics
* CSV export functionality

---

## 🔧 Simulated Industrial Assets

| Machine Type            | Temperature | Vibration | Pressure  | Power    |
| ----------------------- | ----------- | --------- | --------- | -------- |
| CNC Milling Machine     | 55–75°C     | 0.5–2.0   | 4–6 bar   | 15–30 kW |
| Hydraulic Press         | 60–85°C     | 1.0–3.5   | 10–18 bar | 30–55 kW |
| Industrial Compressor   | 70–95°C     | 1.5–4.0   | 8–14 bar  | 40–70 kW |
| Conveyor Drive Motor    | 45–65°C     | 0.3–1.5   | 2–4 bar   | 8–20 kW  |
| Injection Moulding Unit | 80–110°C    | 0.8–2.5   | 12–20 bar | 50–80 kW |

---

## ⚠️ Fault Injection Simulation

Approximately **15% of machines** receive simulated faults to emulate real-world industrial failures.

| Fault Type    | Simulated Impact           |
| ------------- | -------------------------- |
| Overheating   | Temperature ↑, Power ↑     |
| Bearing Wear  | Vibration ↑, Temperature ↑ |
| Pressure Leak | Pressure ↓, Vibration ↑    |
| Power Surge   | Power ↑, Temperature ↑     |

---

## 📈 Performance Metrics

* 20 Simulated Industrial Machines
* 1,200 Sensor Records
* Multi-Sensor Monitoring
* Real-Time Risk Assessment
* Predictive Failure Estimation
* Automated Maintenance Insights

---

## 🎯 Industrial Impact

ForgeMind AI enables industries to transition from reactive maintenance to predictive maintenance by:

* Reducing unexpected machine downtime
* Improving equipment lifespan
* Lowering operational maintenance costs
* Enhancing workplace safety
* Increasing manufacturing efficiency
* Supporting data-driven decision making

---

## 🚀 Future Roadmap

* IoT Sensor Integration
* Real-Time MQTT Streaming
* Edge Deployment on NVIDIA Jetson
* Digital Twin Simulation
* LLM-Based Maintenance Assistant
* Remaining Useful Life (RUL) Prediction
* Enterprise Fleet Management Features

---

## 👥 Team Members

### Team ForgeMind

1. **Sahithi Rithvika Katakam**
2. **Sai Spoorthy Eturu**

---

## 📜 License

This project is developed for educational, research, and hackathon demonstration purposes.

---

**Built for Tata Technologies Hackathon 2026**
**ForgeMind AI v1.0.0**

