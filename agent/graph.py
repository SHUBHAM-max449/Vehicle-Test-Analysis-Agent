from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import load_data_node,anomaly_detection_node,semantic_search_node,report_generation_node,clean_report_node
from knowledge_base.setup_chroma import load_knowledge_base
from functools import partial


def should_search(state: AgentState) -> str:
    """Conditional edge — route based on whether anomalies were found."""
    if state["anomalies"] and len(state["anomalies"]) > 0:
        return "anomalies_found"
    return "no_anomalies"


def build_graph():
    vectorstore = load_knowledge_base()

    # Bind vectorstore to semantic_search_node
    search_node = partial(semantic_search_node, vectorstore=vectorstore)

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("load_data", load_data_node)
    graph.add_node("anomaly_detection", anomaly_detection_node)
    graph.add_node("semantic_search", search_node)
    graph.add_node("report_generation", report_generation_node)
    graph.add_node("clean_report", clean_report_node)

    # Add edges
    graph.set_entry_point("load_data")
    graph.add_edge("load_data", "anomaly_detection")

    # Conditional edge after anomaly detection
    graph.add_conditional_edges(
        "anomaly_detection",
        should_search,
        {
            "anomalies_found": "semantic_search",
            "no_anomalies": "clean_report"
        }
    )

    graph.add_edge("semantic_search", "report_generation")
    graph.add_edge("report_generation", END)
    graph.add_edge("clean_report", END)

    return graph.compile()