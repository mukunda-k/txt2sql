from typing import TypedDict, List, Any, Optional, Dict
from models import Qwen, Meta
from langchain.tools import BaseTool
from IPython.display import Image, display
import os
import pandas as pd
import pandasql as ps
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import re
import traceback
# Enhanced State definition with clean table name support
class SQLAgentState(TypedDict):
    """Enhanced state for SQL agent with clean table name support"""
    user_query: str
    generation_model: Qwen
    evaluation_model: Meta
    schema_tool: BaseTool
    execute_tool: Optional[BaseTool]
    csv_file_path: Optional[str]
    clean_table_name: str  # Clean table name without timestamp
    schema: Optional[str]
    sql_query: Optional[str]
    query_rating: float
    evaluation_reason: Optional[str]
    result: Optional[Any]
    error: Optional[str]
    need_schema: bool
    retry_count: int
    max_retries: int
    query_attempts: List[Dict[str, Any]]
    best_query: Optional[Dict[str, Any]]
    execution_threshold: float


def get_schema_node(state: SQLAgentState) -> SQLAgentState:
    """Get database schema using the schema tool"""
    try:
        schema_tool = state["schema_tool"]
        schema = schema_tool._run() if hasattr(schema_tool, '_run') else schema_tool()
        
        return {
            **state,
            "schema": schema,
            "need_schema": False,
            "error": None
        }
    except Exception as e:
        return {
            **state,
            "error": f"Error getting schema: {str(e)}",
            "need_schema": True
        }


def generate_sql_node(state: SQLAgentState) -> SQLAgentState:
    """Generate SQL query using clean table name"""
    try:
        generation_model = state["generation_model"]
        user_query = state["user_query"]
        schema = state["schema"]
        retry_count = state.get("retry_count", 0)
        query_attempts = state.get("query_attempts", [])
        csv_file_path = state.get("csv_file_path")
        clean_table_name = state.get("clean_table_name", "your_table")
        
        # Get valid columns from CSV
        valid_columns = set()
        if csv_file_path:
            try:
                df = pd.read_csv(csv_file_path)
                valid_columns = set(df.columns)
            except Exception as e:
                return {
                    **state,
                    "error": f"Error reading CSV file: {str(e)}",
                    "sql_query": None
                }
        
        # Build context from previous attempts
        context_from_attempts = ""
        if retry_count > 0 and query_attempts:
            context_from_attempts = "\n=== PREVIOUS ATTEMPTS AND FEEDBACK ===\n"
            for i, attempt in enumerate(query_attempts):
                context_from_attempts += f"\nAttempt {i+1} (Rating: {attempt['rating']}):\n"
                context_from_attempts += f"Query: {attempt['query']}\n"
                context_from_attempts += f"Feedback: {attempt['reason']}\n"
            context_from_attempts += "\n=== END PREVIOUS ATTEMPTS ===\n"
            context_from_attempts += "\nIMPORTANT: Generate a DIFFERENT and IMPROVED query based on the feedback above.\n"

        system_prompt = f"""You are an expert SQL query generator. Generate ONLY the SQL query as a clean string.

DATABASE SCHEMA:
{schema}

CRITICAL REQUIREMENTS:
1. Table name MUST be '{clean_table_name}' (use this exact name)
2. Use ONLY these valid columns: {', '.join(valid_columns)}
3. Generate syntactically correct SQL
4. Return ONLY the SQL query, no explanations or markdown formatting
5. Do not add any prefixes, suffixes, or extra text

{context_from_attempts}

User Request: {user_query}

Generate a SQL query using table '{clean_table_name}' with columns: {', '.join(valid_columns)}

IMPORTANT: Return ONLY the SQL query, nothing else."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate SQL for: {user_query}")
        ]
        
        response = generation_model.invoke(messages)
        
        if not response or not hasattr(response, 'content') or not response.content:
            return {
                **state,
                "error": "Model returned empty or invalid response",
                "sql_query": None
            }
        
        sql_query = response.content.strip()
        
        # Minimal cleaning - just remove obvious markdown
        if sql_query.startswith("```sql"):
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        elif sql_query.startswith("```"):
            sql_query = sql_query.replace("```", "").strip()
        
        if not sql_query:
            return {
                **state,
                "error": "Unable to extract valid SQL from model response",
                "sql_query": None
            }
        
        # Validate column names if we have valid columns
        if valid_columns:
            used_columns = extract_columns_from_query(sql_query)
            invalid_columns = used_columns - valid_columns
            if invalid_columns and len(invalid_columns) > 0:
                return {
                    **state,
                    "error": f"Query contains invalid columns: {', '.join(invalid_columns)}. Valid columns: {', '.join(valid_columns)}",
                    "sql_query": sql_query
                }
        
        # Validate table name is used correctly
        if clean_table_name and clean_table_name.lower() not in sql_query.lower():
            return {
                **state,
                "error": f"Query must use table name '{clean_table_name}'",
                "sql_query": sql_query
            }
        
        return {
            **state,
            "sql_query": sql_query,
            "error": None
        }
        
    except Exception as e:
        return {
            **state,
            "error": f"Error generating SQL: {str(e)}",
            "sql_query": None
        }


def evaluate_query_node(state: SQLAgentState) -> SQLAgentState:
    """Evaluate SQL query and provide rating between 0 and 1"""
    try:
        evaluation_model = state["evaluation_model"]
        sql_query = state.get("sql_query", "")
        user_query = state["user_query"]
        schema = state["schema"]
        clean_table_name = state.get("clean_table_name", "your_table")
        
        if not sql_query:
            # Still track this as an attempt even if no query was generated
            query_attempts = state.get("query_attempts", [])
            current_attempt = {
                "query": "No query generated",
                "rating": 0.0,
                "reason": "No SQL query generated to evaluate",
                "attempt_number": len(query_attempts) + 1
            }
            query_attempts.append(current_attempt)
            
            return {
                **state,
                "query_rating": 0.0,
                "evaluation_reason": "No SQL query generated to evaluate",
                "query_attempts": query_attempts,
                "best_query": state.get("best_query")  # Keep existing best query if any
            }
        
        # Check if query uses correct table name
        rating_adjustment = 0.0
        table_usage_feedback = ""
        
        if clean_table_name and clean_table_name.lower() not in sql_query.lower():
            rating_adjustment = -0.4
            table_usage_feedback = f" CRITICAL: Query uses wrong table name. Should use '{clean_table_name}'."
        
        system_prompt = f"""You are an expert SQL query evaluator. Rate the SQL query on a scale of 0.0 to 1.0 based on:

1. SQL syntax correctness (0.3 weight)
2. Relevance to user request (0.3 weight)  
3. Proper use of schema elements (0.2 weight)
4. Query logic and efficiency (0.2 weight)

DATABASE SCHEMA:
{schema}

IMPORTANT: The table name must be '{clean_table_name}'

USER REQUEST: {user_query}
SQL QUERY TO EVALUATE: {sql_query}

RESPONSE FORMAT (EXACTLY):
RATING: [0.0 to 1.0]
REASON: [Detailed explanation of rating covering syntax, relevance, schema usage, and logic]

Examples:
- Perfect query: RATING: 1.0
- Good query with minor issues: RATING: 0.8
- Decent query with some problems: RATING: 0.6  
- Poor query with major issues: RATING: 0.3
- Completely wrong query: RATING: 0.1"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Evaluate: {sql_query}")
        ]
        
        response = evaluation_model.invoke(messages)
        evaluation_text = response.content.strip() if response and response.content else ""
        
        # Parse rating and reason
        rating, reason = parse_evaluation_response(evaluation_text, sql_query, schema)
        
        # Apply table name adjustment
        rating = max(0.0, min(1.0, rating + rating_adjustment))
        reason = reason + table_usage_feedback
        
        # Update query attempts - FIXED: Always increment attempts
        query_attempts = state.get("query_attempts", [])
        current_attempt = {
            "query": sql_query,
            "rating": rating,
            "reason": reason,
            "attempt_number": len(query_attempts) + 1
        }
        query_attempts.append(current_attempt)
        
        # Update best query if current is better
        best_query = state.get("best_query")
        if not best_query or rating > best_query["rating"]:
            best_query = current_attempt
        
        return {
            **state,
            "query_rating": rating,
            "evaluation_reason": reason,
            "query_attempts": query_attempts,
            "best_query": best_query,
            "error": None
        }
    except Exception as e:
        # Still track this as an attempt even if evaluation failed
        query_attempts = state.get("query_attempts", [])
        current_attempt = {
            "query": state.get("sql_query", "Unknown query"),
            "rating": 0.0,
            "reason": f"Error during evaluation: {str(e)}",
            "attempt_number": len(query_attempts) + 1
        }
        query_attempts.append(current_attempt)
        
        return {
            **state,
            "query_rating": 0.0,
            "evaluation_reason": f"Error during evaluation: {str(e)}",
            "query_attempts": query_attempts,
            "best_query": state.get("best_query"),
            "error": f"Error evaluating query: {str(e)}"
        }


def execute_query_node(state: SQLAgentState) -> SQLAgentState:
    """Execute the SQL query using pandasql with clean table name"""
    try:
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        query_rating = state.get("query_rating", 0.0)
        execution_threshold = state.get("execution_threshold", 0.8)
        best_query = state.get("best_query")
        query_attempts = state.get("query_attempts", [])
        
        # Choose which query to execute
        chosen_query = None
        execution_note = ""
        
        if query_rating >= execution_threshold:
            chosen_query = state.get("sql_query", "")
            execution_note = f"Executed query with rating: {query_rating}"
        elif retry_count >= max_retries and best_query:
            chosen_query = best_query["query"]
            execution_note = f"Executed best query (Rating: {best_query['rating']}) after {max_retries} attempts"
        elif retry_count < max_retries:
            return {
                **state,
                "result": f"Query rating ({query_rating}) below threshold ({execution_threshold}). Will retry.",
                "error": None
            }
        else:
            # Fallback: use current query even if below threshold
            chosen_query = state.get("sql_query", "")
            execution_note = f"Executed query after max retries with rating: {query_rating}"
        
        if not chosen_query or chosen_query == "No query generated":
            return {
                **state,
                "result": "No valid SQL query available for execution",
                "error": "Missing SQL query"
            }
        
        # Execute the query
        csv_file_path = state.get("csv_file_path")
        clean_table_name = state.get("clean_table_name", "your_table")
        
        if not csv_file_path:
            return {
                **state,
                "result": "No CSV file path provided for execution",
                "error": "Missing CSV file path"
            }
        
        try:
            # Load the CSV data
            df = pd.read_csv(csv_file_path)
            
            # Create a dictionary with the clean table name
            table_dict = {clean_table_name: df}
            
            # Execute the SQL query
            result = ps.sqldf(chosen_query, table_dict)
            
            # Format result for display - preserve multi-line query formatting
            if result is not None and not result.empty:
                result_summary = f"Query executed successfully!\n"
                result_summary += f"Rows returned: {len(result)}\n"
                result_summary += f"Query:\n{chosen_query}\n\n"
                result_summary += f"Results:\n{result.to_string(index=False)}\n"
                result_summary += f"\n{execution_note}"
            else:
                result_summary = f"Query executed successfully but returned no results.\n"
                result_summary += f"Query:\n{chosen_query}\n"
                result_summary += f"{execution_note}"
            
            return {
                **state,
                "result": result_summary,
                "error": None
            }
            
        except Exception as e:
            error_msg = f"SQL execution error: {str(e)}"
            return {
                **state,
                "result": f"Query failed to execute: {error_msg}\nQuery:\n{chosen_query}",
                "error": error_msg
            }
        
    except Exception as e:
        return {
            **state,
            "result": f"Execution node error: {str(e)}",
            "error": f"Error in execution node: {str(e)}"
        }

def should_retry(state: SQLAgentState) -> str:
    """Determine if we should retry or end"""
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    query_rating = state.get("query_rating", 0.0)
    execution_threshold = state.get("execution_threshold", 0.8)
    error = state.get("error")
    sql_query = state.get("sql_query")
    
    # If there's a critical error (no SQL query generated), still allow retries
    if error and not sql_query and retry_count < max_retries:
        return "retry"
    
    # End if we've exceeded max retries
    if retry_count >= max_retries:
        return "execute"
    
    # End if query rating is above threshold
    if query_rating >= execution_threshold:
        return "execute"
    
    # Otherwise, retry
    return "retry"


def increment_retry_count(state: SQLAgentState) -> SQLAgentState:
    """Increment retry count and continue"""
    return {
        **state,
        "retry_count": state.get("retry_count", 0) + 1,
        "error": None
    }


def extract_columns_from_query(sql_query: str) -> set:
    """Extract column names from SQL query"""
    try:
        # Simple regex to find potential column names
        # This is a basic implementation and might need refinement
        sql_lower = sql_query.lower()
        
        # Remove SQL keywords and operators
        keywords_to_remove = [
            'select', 'from', 'where', 'group by', 'order by', 'having',
            'inner join', 'left join', 'right join', 'full join', 'join',
            'on', 'and', 'or', 'not', 'in', 'like', 'between', 'is', 'null',
            'count', 'sum', 'avg', 'max', 'min', 'distinct', 'as',
            'limit', 'offset', 'union', 'intersect', 'except'
        ]
        
        # Find potential column references
        columns = set()
        
        # Look for patterns like column_name, table.column_name
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\b'
        matches = re.findall(pattern, sql_query)
        
        for match in matches:
            # Skip SQL keywords
            if match.lower() not in keywords_to_remove:
                # If it contains a dot, take the part after the dot
                if '.' in match:
                    column_name = match.split('.')[-1]
                else:
                    column_name = match
                columns.add(column_name)
        
        return columns
    except Exception:
        return set()


def parse_evaluation_response(evaluation_text: str, sql_query: str, schema: str) -> tuple:
    """Parse evaluation response to extract rating and reason"""
    try:
        lines = evaluation_text.split('\n')
        rating = 0.0
        reason = "No evaluation reason provided"
        
        for line in lines:
            line = line.strip()
            if line.startswith("RATING:"):
                try:
                    rating_str = line.replace("RATING:", "").strip()
                    rating = float(rating_str)
                    rating = max(0.0, min(1.0, rating))  # Clamp between 0 and 1
                except ValueError:
                    rating = 0.0
            elif line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()
        
        # If no reason found, look for any text after rating
        if reason == "No evaluation reason provided":
            reason_lines = []
            found_reason = False
            for line in lines:
                if found_reason:
                    reason_lines.append(line.strip())
                elif line.startswith("REASON:"):
                    found_reason = True
                    reason_lines.append(line.replace("REASON:", "").strip())
            
            if reason_lines:
                reason = " ".join(reason_lines)
        
        return rating, reason
        
    except Exception as e:
        return 0.0, f"Error parsing evaluation: {str(e)}"


def get_csv_schema(csv_path: str) -> str:
    """Get schema information from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        
        schema_info = f"Columns: {', '.join(df.columns)}\n"
        schema_info += f"Shape: {df.shape}\n"
        schema_info += f"Data types:\n{df.dtypes.to_string()}\n\n"
        schema_info += f"Sample data (first 3 rows):\n{df.head(3).to_string(index=False)}"
        return schema_info
    except Exception as e:
        return f"Error reading CSV: {str(e)}"


def create_sql_agent_graph() -> StateGraph:
    """Create the SQL agent state graph"""
    workflow = StateGraph(SQLAgentState)
    
    # Add nodes
    workflow.add_node("get_schema", get_schema_node)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("evaluate_query", evaluate_query_node)
    workflow.add_node("execute_query", execute_query_node)
    workflow.add_node("increment_retry", increment_retry_count)
    
    # Add edges
    workflow.set_entry_point("get_schema")
    workflow.add_edge("get_schema", "generate_sql")
    workflow.add_edge("generate_sql", "evaluate_query")
    
    # Conditional edges
    workflow.add_conditional_edges(
        "evaluate_query",
        should_retry,
        {
            "retry": "increment_retry",
            "execute": "execute_query",
            "end": END
        }
    )
    
    workflow.add_edge("increment_retry", "generate_sql")
    workflow.add_edge("execute_query", END)
    return workflow.compile()

create_sql_agent_graph()
def run_sql_agent(
    user_query: str,
    schema_tool: BaseTool,
    csv_file_path: str,
    clean_table_name: str,
    max_retries: int = 3,
    execution_threshold: float = 0.8,
    return_full_state: bool = False
) -> Any:
    """
    Run the SQL agent with the given parameters
    
    Args:
        user_query: Natural language query from user
        schema_tool: Tool to get database schema
        csv_file_path: Path to CSV file
        clean_table_name: Clean table name to use
        max_retries: Maximum number of retry attempts
        execution_threshold: Minimum rating threshold for execution
        return_full_state: Whether to return full state or just result
    
    Returns:
        Result string or full state based on return_full_state parameter
    """
    try:
        # Initialize models
        generation_model = Qwen()
        evaluation_model = Meta()
        
        # Create initial state
        initial_state = SQLAgentState(
            user_query=user_query,
            generation_model=generation_model,
            evaluation_model=evaluation_model,
            schema_tool=schema_tool,
            execute_tool=None,
            csv_file_path=csv_file_path,
            clean_table_name=clean_table_name,
            schema=None,
            sql_query=None,
            query_rating=0.0,
            evaluation_reason=None,
            result=None,
            error=None,
            need_schema=True,
            retry_count=0,
            max_retries=max_retries,
            query_attempts=[],
            best_query=None,
            execution_threshold=execution_threshold
        )
        
        # Create and run the agent
        agent = create_sql_agent_graph()
        final_state = agent.invoke(initial_state)
        
        if return_full_state:
            return final_state
        
        # Format the result for display - FIXED: Ensure attempts are always shown
        result = final_state.get("result", "No result generated")
        query_attempts = final_state.get("query_attempts", [])
        best_query = final_state.get("best_query")
        retry_count = final_state.get("retry_count", 0)
        
        output = f"{result}\n\n"
        output += f"=== SUMMARY ===\n"
        output += f"Total Attempts: {len(query_attempts)}\n"
        output += f"Total Retries: {retry_count}\n"
        
        if best_query:
            output += f"Best Query (Rating: {best_query['rating']}):\n{best_query['query']}\n"
        
        if query_attempts:
            output += f"\n=== ALL ATTEMPTS ===\n"
            for i, attempt in enumerate(query_attempts, 1):
                output += f"\nAttempt {i} (Rating: {attempt['rating']}):\n"
                output += f"Query:\n{attempt['query']}\n"
                output += f"Feedback: {attempt['reason']}\n"
        else:
            output += f"\nWARNING: No query attempts were recorded. This indicates a potential issue in the agent workflow.\n"
        
        return output
        
    except Exception as e:
        error_msg = f"Error running SQL agent: {str(e)}"
        if return_full_state:
            return {"error": error_msg, "result": error_msg}
        return error_msg