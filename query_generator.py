"""
Core LangGraph workflow:   get_schema → generate_sql → rate → maybe_retry → execute
Designed for multiple tables received from utils/schema_tools.
"""
from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import pandasql as ps
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from utils.state import SQLAgentState
from utils.schema_tools import json_schema, pretty_schema
from utils.models import Qwen, Meta


# ─────────────────── Nodes ───────────────────
def get_schema(state: SQLAgentState) -> SQLAgentState:
    return {
        **state,
        "schema_json": json_schema(state["tables"]),
        "schema_pretty": pretty_schema(state["tables"]),
    }


def generate_sql(state: SQLAgentState) -> SQLAgentState:
    system_prompt = f"""
You are an expert SQL assistant.

DATABASE SCHEMA (JSON):
{state['schema_json']}

Rules:
1. Use ONLY the tables / columns shown above.
2. Standard SQL-92 syntax.
3. Return ONLY the SQL string (no markdown or explanations).

User: {state['user_query']}
"""
    llm = state["generation_model"]
    resp = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=state["user_query"])]
    )
    sql = resp.content.strip().removeprefix("``````").strip()
    return {**state, "sql_query": sql}


def rate_sql(state: SQLAgentState) -> SQLAgentState:
    """Very lightweight LLM rater -> float in [0,1]"""
    eval_llm = state["evaluation_model"]
    prompt = f"""
Rate 0-1: Is the following SQL valid for the user question?

SQL: {state['sql_query']}
Question: {state['user_query']}
Just answer with a number between 0 and 1."""
    score = float(
        eval_llm.invoke([HumanMessage(content=prompt)]).content.strip().split()[0]
    )
    attempt = {
        "sql": state["sql_query"],
        "rating": score,
        "attempt_no": state["retry_count"] + 1,
    }
    attempts = state["query_attempts"] + [attempt]
    best = max(attempts, key=lambda a: a["rating"])
    return {**state, "query_rating": score, "query_attempts": attempts, "best_query": best}


def execute_sql(state: SQLAgentState) -> SQLAgentState:
    sql = state["best_query"]["sql"]
    try:
        result = ps.sqldf(sql, state["tables"])
    except Exception as e:  # noqa: BLE001
        return {**state, "error": str(e), "result": None}

    return {**state, "result": result.to_markdown(index=False)}


# ─────────────────── Edge logic ───────────────────
def need_retry(state: SQLAgentState) -> str:
    if state["query_rating"] >= state["quality_threshold"]:
        return "execute"
    if state["retry_count"] >= state["max_retries"]:
        return "execute"
    return "retry"


def inc_retry(state: SQLAgentState) -> SQLAgentState:
    return {**state, "retry_count": state["retry_count"] + 1}


# ─────────────────── Graph compile ───────────────────
def compile_graph() -> StateGraph:
    g = StateGraph(SQLAgentState)
    g.add_node("get_schema", get_schema)
    g.add_node("gen", generate_sql)
    g.add_node("rate", rate_sql)
    g.add_node("exec", execute_sql)
    g.add_node("inc", inc_retry)

    g.set_entry_point("get_schema")
    g.add_edge("get_schema", "gen")
    g.add_edge("gen", "rate")
    g.add_conditional_edges("rate", need_retry, {"execute": "exec", "retry": "inc"})
    g.add_edge("inc", "gen")
    g.add_edge("exec", END)
    return g.compile()


GRAPH = compile_graph()


def run_agent(user_query: str, tables: Dict[str, pd.DataFrame], threshold: float, max_r: int = 3) -> SQLAgentState:
    init = SQLAgentState(
        user_query=user_query,
        generation_model=Qwen(),
        evaluation_model=Meta(),
        tables=tables,
        schema_json="",
        schema_pretty="",
        quality_threshold=threshold,
        sql_query=None,
        query_rating=0.0,
        query_attempts=[],
        best_query=None,
        result=None,
        error=None,
        retry_count=0,
        max_retries=max_r,
    )
    return GRAPH.invoke(init)
