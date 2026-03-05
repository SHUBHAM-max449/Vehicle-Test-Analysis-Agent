<div align="center">

# 🏎️ Vehicle Test Analysis Agent

**Agentic AI pipeline for automotive sensor data analysis**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B35?style=for-the-badge)](https://trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

*An autonomous AI agent that detects sensor anomalies, retrieves similar past failures via semantic search, and generates structured engineering reports — end to end.*

<br/>

</div>

---

## 📌 What It Does

```
  Vehicle Sensor Data (CSV)
          │
          ▼
  ┌───────────────┐
  │ Anomaly       │  ← Z-score analysis across all sensor channels
  │ Detection     │     flags critical deviations automatically
  └───────┬───────┘
          │  anomalies found?
    ┌─────┴──────┐
   YES            NO
    │              │
    ▼              ▼
  ┌──────────┐  ┌──────────────┐
  │ Semantic │  │ Clean Report │
  │ Search   │  │   ✅ PASS    │
  └────┬─────┘  └──────────────┘
       │  ← Queries ChromaDB for similar
       │    past failures using embeddings
       ▼
  ┌──────────────────┐
  │ Report Generator │  ← GPT-4o-mini synthesizes findings
  │  📄 .txt output  │    into a structured engineering report
  └──────────────────┘
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LangGraph State Machine                      │
│                                                                      │
│   START                                                              │
│     │                                                                │
│     ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  load_data_node                             [nodes.py]       │    │
│  │  • Generates 500 rows of synthetic sensor data              │    │
│  │  • Sensors: speed | brake_pressure | engine_temp | torque   │    │
│  │  • Stores → state["raw_data"]                               │    │
│  └──────────────────────────┬──────────────────────────────────┘    │
│                             │                                        │
│                             ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  anomaly_detection_node                     [nodes.py]       │    │
│  │  • Calls detect_anomalies()      →          [tools.py]       │    │
│  │  • Z-score per sensor column (threshold = 3)                │    │
│  │  • Tags: severity (Medium / High / Critical)                │    │
│  │  • Stores → state["anomalies"]                              │    │
│  └──────────────────────────┬──────────────────────────────────┘    │
│                             │                                        │
│              ┌──────────────▼──────────────┐                        │
│              │   Conditional Edge           │   [graph.py]           │
│              │   should_search()            │                        │
│              └──────┬───────────────┬───────┘                        │
│                     │               │                                │
│              anomalies          no anomalies                        │
│              found                  │                                │
│                     │               ▼                                │
│                     │    ┌──────────────────┐                        │
│                     │    │ clean_report_node│                        │
│                     │    │ Status: ✅ PASS  │                        │
│                     │    └────────┬─────────┘                        │
│                     │             │                                  │
│                     ▼             │                                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  semantic_search_node                       [nodes.py]       │    │
│  │  • Calls query_knowledge_base()  →          [tools.py]       │    │
│  │  • Embeds anomaly description (text-embedding-3-small)      │    │
│  │  • Queries ChromaDB → top 2 similar past failures           │    │
│  │  • Knowledge base: 15 domain failure reports                │    │
│  │  • Stores → state["similar_failures"]                       │    │
│  └──────────────────────────┬──────────────────────────────────┘    │
│                             │                                        │
│                             ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  report_generation_node                     [nodes.py]       │    │
│  │  • Sends anomalies + matches → GPT-4o-mini                  │    │
│  │  • Structured report: Summary | Table | Root Causes |       │    │
│  │    Risk Assessment | Recommendations                        │    │
│  │  • Saves → reports/test_report_<timestamp>.txt              │    │
│  │  • Stores → state["final_report"]                           │    │
│  └──────────────────────────┬──────────────────────────────────┘    │
│                             │             │                          │
│                             └──────┬──────┘                          │
│                                    ▼                                 │
│                                   END                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
Vehicle-Test-Analysis-Agent/
│
├── 📂 data/
│   └── generate_data.py          # Synthetic sensor data (500 rows, 4 sensors)
│
├── 📂 knowledge_base/
│   ├── failure_reports.py        # 15 domain-specific past failure cases
│   └── setup_chroma.py           # Embeds reports → loads into ChromaDB
│
├── 📂 agent/
│   ├── state.py                  # AgentState TypedDict
│   ├── tools.py                  # detect_anomalies() + query_knowledge_base()
│   ├── nodes.py                  # All LangGraph node functions
│   └── graph.py                  # Graph definition + conditional routing
│
├── 📂 reports/                   # Auto-generated timestamped .txt reports
├── 📂 chroma_db/                 # Persisted vector store (auto-created)
│
├── main.py                       # Entry point
├── requirements.txt
├── .env.example                  # Template — copy to .env and add your key
└── README.md
```

---

## 🔄 State Flow

The `AgentState` TypedDict carries all data between nodes. Each node receives the full state and returns an updated copy — nothing is modified in place.

```
AgentState
│
├── raw_data          ← pd.DataFrame   set by: load_data_node
│                                      500 rows × 5 columns
│
├── anomalies         ← list[dict]     set by: anomaly_detection_node
│                                      [{sensor, value, z_score, severity, timestamp}, ...]
│
├── similar_failures  ← list[dict]     set by: semantic_search_node
│                                      [{anomaly: {...}, similar_failures: [str, str]}, ...]
│
└── final_report      ← str            set by: report_generation_node
                                       Full structured engineering report
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| 🧠 Orchestration | LangGraph | State machine, node routing, conditional edges |
| 💬 LLM | OpenAI GPT-4o-mini | Report generation |
| 🔢 Embeddings | text-embedding-3-small | Semantic anomaly-to-failure matching |
| 🗄️ Vector Store | ChromaDB | Persistent knowledge base storage |
| 📊 Data Processing | Pandas + NumPy + SciPy | Sensor data generation + z-score analysis |
| 🔐 Config | python-dotenv | API key management |

---

## ⚙️ Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/SHUBHAM-max449/Vehicle-Test-Analysis-Agent.git
cd Vehicle-Test-Analysis-Agent
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment**
```bash
cp .env.example .env
# Open .env and add your OpenAI API key
```

**5. Set up knowledge base** *(first time only)*
```bash
python -m knowledge_base.setup_chroma
```

**6. Run the agent**
```bash
python main.py
```

---

## 📊 Sensor Data Overview

| Sensor | Normal Range | Injected Anomaly | Row |
|--------|-------------|-----------------|-----|
| `speed_kmh` | ~120 ± 15 km/h | — | — |
| `brake_pressure_bar` | ~30 ± 3 bar | **72 bar** 🔴 | 150 |
| `engine_temp_c` | ~90 ± 5 °C | **140°C** 🔴 | 300 |
| `torque_nm` | ~300 ± 20 Nm | **600 Nm** 🔴 | 420 |

---

## 📄 Sample Terminal Output

```
============================================================
PORSCHE ENGINEERING — VEHICLE TEST ANALYSIS AGENT
============================================================
>> Loading sensor data...
   Loaded 500 rows, 5 sensors.

>> Running anomaly detection...
   Found 3 anomalies.
   - brake_pressure_bar = 72.0 at 2024-01-01 00:02:30 (z=14.2, Critical)
   - engine_temp_c      = 140.0 at 2024-01-01 00:05:00 (z=10.1, Critical)
   - torque_nm          = 600.0 at 2024-01-01 00:07:00 (z=15.3, Critical)

>> Querying knowledge base for similar past failures...
   Matched 2 past failures for brake_pressure_bar
   Matched 2 past failures for engine_temp_c
   Matched 2 past failures for torque_nm

>> Generating engineering report...
   Report generated.

>> Report saved to reports/test_report_20240101_120000.txt
============================================================
```

---

## 📋 Sample Report Output

```
============================================================
PORSCHE ENGINEERING
VEHICLE TEST ANALYSIS REPORT
Generated: 2024-01-01 12:00:00
============================================================

1. EXECUTIVE SUMMARY
--------------------
Three critical anomalies detected during test session.
All findings match known failure patterns in the knowledge base.
Immediate engineering review recommended.

2. ANOMALIES DETECTED
---------------------
Sensor               | Value    | Severity | Timestamp
---------------------|----------|----------|--------------------
brake_pressure_bar   | 72.0 bar | Critical | 2024-01-01 00:02:30
engine_temp_c        | 140.0 °C | Critical | 2024-01-01 00:05:00
torque_nm            | 600.0 Nm | Critical | 2024-01-01 00:07:00

3. SIMILAR PAST FAILURES & ROOT CAUSES
---------------------------------------
[brake_pressure_bar]
→ Brake pressure spike above 60 bar during cornering.
  Root cause: caliper seal failure. Severity: Critical.

[engine_temp_c]
→ Engine temperature exceeded 130°C during track test.
  Root cause: coolant pump degradation. Severity: High.

4. RISK ASSESSMENT
------------------
All three anomalies classified Critical.
Risk of component failure under continued test conditions: HIGH.

5. RECOMMENDATIONS
------------------
• Inspect brake calipers for seal integrity before next run
• Check coolant pump and radiator for blockage
• Inspect drivetrain coupling for wear
============================================================
```

---

## 🧠 Key Design Decisions

**Why LangGraph over plain LangChain AgentExecutor?**
> LangGraph gives explicit control over node execution and conditional routing. The branching logic — skip semantic search if no anomalies found — is clean as a graph. LangChain's executor requires workarounds for this kind of stateful conditional flow.

**Why z-score for anomaly detection?**
> Simple, interpretable, requires no training data. Every detected anomaly can be explained as *"X standard deviations from normal"* — critical in QA engineering where explainability matters. For production, Isolation Forest or LSTM-based detection would better handle complex time series patterns.

**Why separate RAG from report generation?**
> Single responsibility per node — semantic_search_node only retrieves, report_generation_node only synthesizes. Each is independently testable and swappable without touching the other.

---

## 🚀 Future Improvements

- [ ] Replace synthetic data with real CAN bus data ingestion
- [ ] Add streaming support for real-time test drive monitoring
- [ ] Expand knowledge base with real historical failure reports
- [ ] Add LangSmith observability for production agent tracing
- [ ] Implement per-subsystem specialist agents (brakes, engine, drivetrain)
- [ ] Wrap agent in FastAPI as a REST endpoint
- [ ] Full Docker Compose containerization

---

## 👤 Author

**Shubham Chitaguppe**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/SHUBHAM-max449)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-FF6B35?style=flat-square)](https://your-portfolio.com)

---

<div align="center">

*Built as part of preparation for Agentic AI roles in automotive engineering.*

⭐ **Star this repo if you found it useful**

</div>