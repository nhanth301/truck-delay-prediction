from langgraph.graph import StateGraph
from typing import TypedDict, List, Annotated, Dict, Any
import pandas as pd

class State(TypedDict, total=False):
    config: Dict[str, Any]
    constant: Dict[str, Any]
    should_continue: bool
    new_data: Dict[str, pd.DataFrame]
    new_data_status: Dict[str, bool]
    new_data_quality: Dict[str, bool]   
    update_status: Dict[str, str]
    db_conn: Any
    feature_store: Any
    feature_groups_data: Dict[str, pd.DataFrame]
    data_drift: Dict[str, Any]
    model_drift: Dict[str, Any]
    final_data: pd.DataFrame
