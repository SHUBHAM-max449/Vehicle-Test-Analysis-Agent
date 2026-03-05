from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from agent.state import AgentState
from agent.tools import detect_anomalies, query_knowledge_base
from data.generate_data import generate_sensor_data

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def load_data_node(state: AgentState) -> AgentState:
    print(">> Loading sensor data...")
    df = generate_sensor_data()
    print(f"   Loaded {len(df)} rows, {len(df.columns)} sensors.")
    return {**state, "raw_data": df}


def anomaly_detection_node(state: AgentState) -> AgentState:
    print(">> Running anomaly detection...")
    df = state["raw_data"]
    anomalies = detect_anomalies(df)
    print(f"   Found {len(anomalies)} anomalies.")
    for a in anomalies:
        print(f"   - {a['sensor']} = {a['value']} at {a['timestamp']} (z={a['z_score']}, {a['severity']})")
    return {**state, "anomalies": anomalies}


def semantic_search_node(state: AgentState, vectorstore) -> AgentState:
    print(">> Querying knowledge base for similar past failures...")
    anomalies = state["anomalies"]
    all_matches = []

    for anomaly in anomalies:
        description = f"{anomaly['sensor']} reading of {anomaly['value']} detected. Severity: {anomaly['severity']}."
        matches = query_knowledge_base(description, vectorstore)
        all_matches.append({
            "anomaly": anomaly,
            "similar_failures": matches
        })
        print(f"   Matched {len(matches)} past failures for {anomaly['sensor']}")

    return {**state, "similar_failures": all_matches}


def report_generation_node(state: AgentState) -> AgentState:
    print(">> Generating engineering report...")

    anomalies = state["anomalies"]
    similar_failures = state["similar_failures"]

    prompt = f"""
You are an expert automotive test engineer at Porsche Engineering.
Generate a structured vehicle test analysis report. 
Format it cleanly for a text file — use clear section headers with dashes underneath, 
spacing between sections, and plain text tables using dashes and pipes.

ANOMALIES DETECTED:
{state["anomalies"]}

SIMILAR PAST FAILURES:
{state["similar_failures"]}

Structure:
1. Executive Summary
2. Anomalies Detected (plain text table: Sensor | Value | Severity | Timestamp)
3. Similar Past Failures & Root Causes
4. Risk Assessment
5. Recommendations for Engineering Team
"""

    response = llm.invoke(prompt)
    report = response.content
    print("   Report generated.")
    return {**state, "final_report": report}


def clean_report_node(state: AgentState) -> AgentState:
    print(">> No anomalies detected. Generating clean report...")
    report = "VEHICLE TEST REPORT\n\nStatus: PASS\nNo anomalies detected across all sensor channels.\nAll readings within expected operational parameters."
    return {**state, "final_report": report}