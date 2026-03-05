import pandas as pd
import numpy as np
from scipy import stats

def detect_anomalies(df: pd.DataFrame, threshold: float = 3.0) -> list:
    """
    Z-score based anomaly detection on numeric sensor columns.
    Returns list of dicts with anomaly details.
    """
    anomalies = []
    numeric_cols = ["speed_kmh", "brake_pressure_bar", "engine_temp_c", "torque_nm"]

    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col]))


        for idx in anomaly_indices:
            anomalies.append({
                "timestamp": str(df.loc[idx, "timestamp"]),
                "sensor": col,
                "value": round(df.loc[idx, col], 2),
                "z_score": round(z_scores[idx], 2),
                "severity": "Critical" if z_scores[idx] > 5 else "High" if z_scores[idx] > 4 else "Medium"
            })

    return anomalies


def query_knowledge_base(anomaly_description: str, vectorstore, k: int = 2) -> list:
    """
    Semantic search over past failure reports given an anomaly description.
    Returns top k matching past failures.
    """
    results = vectorstore.similarity_search(anomaly_description, k=k)
    return [doc.page_content for doc in results]