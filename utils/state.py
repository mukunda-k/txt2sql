from typing import Dict, List, Any, Optional, TypedDict
import pandas as pd
from .models import Qwen, Meta
from langchain.tools import BaseTool

class SQLAgentState(TypedDict):
    user_query: str
    generation_model: Qwen
    evaluation_model: Meta
    tables: Dict[str, pd.DataFrame]       # ← NEW
    schema_json: str                      # ← NEW JSON schema
    schema_pretty: str                    # ← human-readable schema
    quality_threshold: float              # ← from slider
    sql_query: Optional[str]
    query_rating: float
    query_attempts: List[Dict[str, Any]]
    best_query: Optional[Dict[str, Any]]
    result: Optional[Any]
    error: Optional[str]
    retry_count: int
    max_retries: int
