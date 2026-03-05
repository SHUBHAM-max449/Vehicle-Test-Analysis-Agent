# 🚗 Vehicle Test Analysis Agent
### Agentic AI Pipeline for Automotive Sensor Data — Built with LangGraph, OpenAI & ChromaDB

---

## Overview

An end-to-end agentic AI system that autonomously analyzes vehicle sensor data from test drives, detects anomalies using statistical methods, retrieves semantically similar past failure cases from a knowledge base, and generates a structured engineering report — mimicking a real automotive QA workflow.

Built as a demonstration of **agentic tool use**, **RAG-based semantic search**, and **modular LangGraph orchestration** applied to engineering data analysis.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (Entry Point)                    │
│                  Loads .env → Builds Graph → Invokes            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LangGraph State Machine                      │
│                                                                 │
│   ┌──────────────┐     ┌────────────────────┐                  │
│   │ load_data    │────▶│ anomaly_detection  │                  │
│   │ node         │     │ node               │                  │
│   │              │     │                    │                  │
│   │ generate_    │     │ z-score per sensor │                  │
│   │ sensor_data()│     │ flags outliers > 3 │                  │
│   └──────────────┘     └─────────┬──────────┘                  │
│                                  │                              │
│                    ┌─────────────▼─────────────┐               │
│                    │   Conditional Edge         │               │
│                    │   should_search()          │               │
│                    └──────┬────────────┬────────┘               │
│                           │            │                        │
│                    anomalies       no anomalies                 │
│                    found               │                        │
│                           │            ▼                        │
│                           │    ┌───────────────┐               │
│                           │    │ clean_report  │               │
│                           │    │ node          │               │
│                           │    │               │               │
│                           │    │ Status: PASS  │               │
│                           │    └───────┬───────┘               │
│                           │            │                        │
│                           ▼            │                        │
│                  ┌─────────────────┐   │                        │
│                  │ semantic_search │   │                        │
│                  │ node            │   │                        │
│                  │                 │   │                        │
│                  │ embed anomaly   │   │                        │
│                  │ query ChromaDB  │   │                        │
│                  │ top 2 matches   │   │                        │
│                  └────────┬────────┘   │                        │
│                           │            │                        │
│                           ▼            │                        │
│                  ┌─────────────────┐   │                        │
│                  │ report_         │   │                        │
│                  │ generation_node │   │                        │
│                  │                 │   │                        │
│                  │ GPT-4o-mini     │   │                        │
│                  │ structured      │   │                        │
│                  │ report → .txt   │   │                        │
│                  └────────┬────────┘   │                        │
│                           │            │                        │
│                           └─────┬──────┘                        │
│                                 ▼                               │
│                               END                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## State Flow

```
AgentState
│
├── raw_data          → set by load_data_node (pandas DataFrame)
├── anomalies         → set by anomaly_detection_node (list of dicts)
├── similar_failures  → set by semantic_search_node (matched past failures)
└── final_report      → set by report_generation_node (string)
```

State is immutable between nodes — each node receives the full state and returns an updated copy. No node modifies state directly.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Orchestration | LangGraph |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Database | ChromaDB |
| Data Processing | Pandas, NumPy, SciPy |
| Environment | Python 3.11+, python-dotenv |

---

## Project Structure

```
vehicle-agent/
│
├── data/
│   └── generate_data.py          # Synthetic sensor data generation
│
├── knowledge_base/
│   ├── failure_reports.py        # 15 domain-specific past failure cases
│   └── setup_chroma.py           # Embed reports and load into ChromaDB
│
├── agent/
│   ├── state.py                  # AgentState TypedDict definition
│   ├── tools.py                  # detect_anomalies() + query_knowledge_base()
│   ├── nodes.py                  # All LangGraph node functions
│   └── graph.py                  # Graph definition, edges, conditional routing
│
├── reports/                      # Auto-generated timestamped report outputs
│
├── main.py                       # Entry point
├── requirements.txt
└── .env                          # OPENAI_API_KEY (not committed)
```

---

## How It Works

### 1. Synthetic Sensor Data
Generates 500 rows of vehicle test data sampled at 1-second intervals:

```
Sensors: speed_kmh | brake_pressure_bar | engine_temp_c | torque_nm
Normal ranges: ~120 km/h | ~30 bar | ~90°C | ~300 Nm
```

Three anomalies are injected to simulate real failure events:
- Row 150: brake pressure spike → 72 bar (normal ~30)
- Row 300: engine overheating → 140°C (normal ~90)
- Row 420: torque spike → 600 Nm (normal ~300)

### 2. Anomaly Detection
Z-score analysis per sensor column. Any reading exceeding 3 standard deviations from the mean is flagged with:
- Sensor name
- Exact value and timestamp
- Z-score
- Severity (Medium / High / Critical)

### 3. Semantic Search (RAG)
For each detected anomaly, a natural language description is built and embedded using OpenAI embeddings. ChromaDB retrieves the top 2 most semantically similar past failure cases from a knowledge base of 15 domain-specific Porsche engineering failure reports.

### 4. Report Generation
GPT-4o-mini synthesizes all findings into a structured engineering report:
1. Executive Summary
2. Anomalies Detected (plain text table)
3. Similar Past Failures & Root Causes
4. Risk Assessment
5. Recommendations for Engineering Team

Report is saved to `reports/test_report_<timestamp>.txt`.

---

## Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/vehicle-test-analysis-agent.git
cd vehicle-test-analysis-agent
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

**4. Set up environment variables**
```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

**.env file:**
```
OPENAI_API_KEY=your_key_here
```

**5. Set up the knowledge base (first time only)**
```bash
python -m knowledge_base.setup_chroma
```

**6. Run the agent**
```bash
python main.py
```

---

## Sample Output

```
============================================================
PORSCHE ENGINEERING — VEHICLE TEST ANALYSIS AGENT
============================================================
>> Loading sensor data...
   Loaded 500 rows, 5 sensors.
>> Running anomaly detection...
   Found 3 anomalies.
   - brake_pressure_bar = 72.0 at 2024-01-01 00:02:30 (z=14.2, Critical)
   - engine_temp_c = 140.0 at 2024-01-01 00:05:00 (z=10.1, Critical)
   - torque_nm = 600.0 at 2024-01-01 00:07:00 (z=15.3, Critical)
>> Querying knowledge base for similar past failures...
   Matched 2 past failures for brake_pressure_bar
   Matched 2 past failures for engine_temp_c
   Matched 2 past failures for torque_nm
>> Generating engineering report...
   Report generated.
>> Report saved to reports/test_report_20240101_120000.txt
```

---

## Key Design Decisions

**Why LangGraph over plain LangChain AgentExecutor?**
LangGraph gives explicit control over node execution and conditional routing. The branching logic — skip semantic search entirely if no anomalies are found — is clean and readable as a graph. LangChain's executor would require workarounds for this.

**Why z-score for anomaly detection?**
Simple, interpretable, requires no training data. Every detected anomaly can be explained to an engineer as "X standard deviations from normal" — which matters in a QA engineering context where explainability is required. For production, Isolation Forest or LSTM-based detection would be more appropriate for complex time series patterns.

**Why separate the RAG step from report generation?**
Separation of concerns — the semantic search node is purely retrieval, the report node is purely synthesis. This makes each node independently testable and replaceable. You can swap ChromaDB for another vector store or swap the report LLM without touching the retrieval logic.

---

## Relevance to Engineering Workflows

This project demonstrates:
- **Agentic tool use** — LLM-orchestrated multi-step analysis pipeline
- **Semantic analysis** — embedding-based retrieval over engineering knowledge base
- **Modular agent design** — each node has a single responsibility
- **Decision algorithms** — conditional routing based on data analysis results
- **Engineering data types** — time series sensor data with domain-specific anomaly thresholds

---

## Future Improvements

- [ ] Replace synthetic data with real CAN bus data ingestion
- [ ] Add streaming support for real-time test drive monitoring
- [ ] Expand knowledge base with real historical failure reports
- [ ] Add LangSmith observability for agent tracing in production
- [ ] Implement per-subsystem specialist agents (brakes, engine, drivetrain)
- [ ] Add FastAPI wrapper to serve the agent as a REST endpoint
- [ ] Containerize with Docker Compose

---

## Author

**Shubham Chitaguppe**
- LinkedIn: [your-linkedin]
- GitHub: [your-github]
- Portfolio: [your-portfolio]

---

*Built as part of interview preparation for Agentic AI roles in automotive engineering.*