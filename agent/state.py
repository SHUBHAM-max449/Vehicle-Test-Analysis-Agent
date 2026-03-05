from typing import TypedDict, Optional
import pandas as pd

class AgentState(TypedDict):
    raw_data: Optional[pd.DataFrame]
    anomalies: Optional[list]
    similar_failures: Optional[list]
    final_report: Optional[str]