from typing import TypedDict, List, Any, Optional, Dict
from models import Qwen, Meta
from langchain.tools import BaseTool
import os
import pandas as pd
import pandasql as ps
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Enhanced State definition with rating system
class SQLAgentState(TypedDict):
    """Enhanced state for SQL agent with rating-based evaluation system"""
    user_query: str
    generation_model: Qwen
    evaluation_model: Meta
    schema_tool: BaseTool
    execute_tool: Optional[BaseTool]
    csv_file_path: Optional[str]
    schema: Optional[str]
    sql_query: Optional[str]
    query_rating: float  # Rating between 0 and 1
    evaluation_reason: Optional[str]  # Reason for the rating
    result: Optional[Any]
    error: Optional[str]
    need_schema: bool
    retry_count: int
    max_retries: int
    query_attempts: List[Dict[str, Any]]  # Track all attempts with ratings
    best_query: Optional[Dict[str, Any]]  # Best query so far
    execution_threshold: float  # Threshold for direct execution (default 0.8)


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
    """Generate SQL query with improved feedback integration"""
    try:
        generation_model = state["generation_model"]
        user_query = state["user_query"]
        schema = state["schema"]
        retry_count = state.get("retry_count", 0)
        query_attempts = state.get("query_attempts", [])
        csv_file_path = state.get("csv_file_path")
        
        # Get correct table name from CSV file
        table_name = "your_table"  # default
        if csv_file_path:
            table_name = os.path.splitext(os.path.basename(csv_file_path))[0]
        
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

CRITICAL: The table name MUST be '{table_name}' (this is the CSV filename without extension).

{context_from_attempts}

RULES:
1. Generate only valid, executable SQL
2. Use EXACTLY the table name '{table_name}' in your query
3. Use proper column names from the schema
4. Return ONLY the SQL query, no explanations or formatting
5. Optimize for performance and correctness
6. Handle edge cases appropriately

User Request: {user_query}

REMEMBER: Table name is '{table_name}' - use this exact name in your SQL query."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ]
        
        response = generation_model.invoke(messages)
        sql_query = response.content.strip()
        
        # Clean up SQL query formatting
        sql_query = clean_sql_query(sql_query)
        
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
        csv_file_path = state.get("csv_file_path")
        
        if not sql_query:
            return {
                **state,
                "query_rating": 0.0,
                "evaluation_reason": "No SQL query generated to evaluate"
            }
        
        # Get correct table name from CSV file
        table_name = None
        if csv_file_path:
            table_name = os.path.splitext(os.path.basename(csv_file_path))[0]
        
        # Check if query uses correct table name
        rating_adjustment = 0.0
        table_usage_feedback = ""
        
        if table_name and table_name.lower() not in sql_query.lower():
            # Major penalty for wrong table name
            rating_adjustment = -0.4
            table_usage_feedback = f" CRITICAL: Query uses wrong table name. Should use '{table_name}' (CSV filename without extension)."
        
        # Create evaluation prompt for rating
        system_prompt = f"""You are an expert SQL query evaluator. Rate the SQL query on a scale of 0.0 to 1.0 based on:

1. SQL syntax correctness (0.3 weight)
2. Relevance to user request (0.3 weight)  
3. Proper use of schema elements (0.2 weight)
4. Query logic and efficiency (0.2 weight)

DATABASE SCHEMA:
{schema}

IMPORTANT: For CSV files, the table name must be the CSV filename without extension: '{table_name}'

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
        evaluation_text = response.content.strip()
        
        # Parse rating and reason
        rating, reason = parse_evaluation_response(evaluation_text, sql_query, schema)
        
        # Apply table name adjustment
        rating = max(0.0, min(1.0, rating + rating_adjustment))
        reason = reason + table_usage_feedback
        
        # Update query attempts
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
        return {
            **state,
            "query_rating": 0.0,
            "evaluation_reason": f"Error during evaluation: {str(e)}",
            "error": f"Error evaluating query: {str(e)}"
        }


def execute_query_node(state: SQLAgentState) -> SQLAgentState:
    """Execute the SQL query using pandasql or execute_tool"""
    try:
        # Determine which query to execute
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        query_rating = state.get("query_rating", 0.0)
        execution_threshold = state.get("execution_threshold", 0.8)
        best_query = state.get("best_query")
        
        # Choose query to execute
        if query_rating >= execution_threshold or retry_count >= max_retries:
            if retry_count >= max_retries and best_query:
                # Use best query after max retries
                sql_query = best_query["query"]
                execution_note = f"Executed best query (Rating: {best_query['rating']}) after {max_retries} attempts"
            else:
                # Use current query
                sql_query = state.get("sql_query", "")
                execution_note = f"Executed query with rating: {query_rating}"
        else:
            return {
                **state,
                "result": f"Query rating ({query_rating}) below threshold ({execution_threshold}). Will retry.",
                "error": None
            }
        
        if not sql_query:
            return {
                **state,
                "result": "No SQL query available for execution",
                "error": "Missing SQL query"
            }
        
        # Execute the query
        csv_file_path = state.get("csv_file_path")
        execute_tool = state.get("execute_tool")
        
        if execute_tool:
            result = execute_tool._run(sql_query)
        elif csv_file_path:
            result = execute_csv_query_direct(sql_query, csv_file_path)
        else:
            result = "No execution method available (no CSV file path or execute tool)"
        
        final_result = f"{execution_note}\n\nSQL Query Executed:\n{sql_query}\n\nResult:\n{result}"
        
        return {
            **state,
            "result": final_result,
            "error": None
        }
    except Exception as e:
        return {
            **state,
            "result": f"Error executing query: {str(e)}",
            "error": f"Error executing query: {str(e)}"
        }


def should_continue(state: SQLAgentState) -> str:
    """Determine next step based on rating and retry logic"""
    error = state.get("error")
    if error and "schema" in error.lower():
        return END
    
    query_rating = state.get("query_rating", 0.0)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    execution_threshold = state.get("execution_threshold", 0.8)
    
    # Execute if rating is good enough or max retries reached
    if query_rating >= execution_threshold or retry_count >= max_retries:
        return "execute_query"
    
    # Retry if rating is below threshold and retries available
    if query_rating < execution_threshold and retry_count < max_retries:
        return "retry_generation"
    
    return END


def retry_generation_node(state: SQLAgentState) -> SQLAgentState:
    """Prepare state for retry with incremented counter"""
    retry_count = state.get("retry_count", 0)
    
    return {
        **state,
        "retry_count": retry_count + 1,
        "sql_query": None,  # Clear for regeneration
        "query_rating": 0.0,  # Reset rating
        "evaluation_reason": None  # Reset reason
    }


def clean_sql_query(sql_query: str) -> str:
    """Clean SQL query from markdown formatting"""
    sql_query = sql_query.strip()
    if sql_query.startswith("```sql"):
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
    elif sql_query.startswith("```"):
        sql_query = sql_query.replace("```", "").strip()
    return sql_query


def parse_evaluation_response(evaluation_text: str, sql_query: str, schema: str) -> tuple[float, str]:
    """Parse evaluation response to extract rating and reason"""
    try:
        lines = evaluation_text.split('\n')
        rating = 0.0
        reason = "Unable to parse evaluation response"
        
        for line in lines:
            if line.startswith("RATING:"):
                rating_str = line.replace("RATING:", "").strip()
                try:
                    rating = float(rating_str)
                    rating = max(0.0, min(1.0, rating))  # Clamp between 0 and 1
                except ValueError:
                    rating = 0.0
            elif line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()
        
        # If no structured response, try to extract from content
        if rating == 0.0 and reason == "Unable to parse evaluation response":
            if "good" in evaluation_text.lower() or "correct" in evaluation_text.lower():
                rating = 0.7
            elif "perfect" in evaluation_text.lower() or "excellent" in evaluation_text.lower():
                rating = 0.9
            elif "poor" in evaluation_text.lower() or "wrong" in evaluation_text.lower():
                rating = 0.3
            else:
                rating = 0.5
            reason = evaluation_text[:200] + "..." if len(evaluation_text) > 200 else evaluation_text
        
        return rating, reason
    except Exception:
        return 0.0, "Error parsing evaluation response"


def execute_csv_query_direct(sql_query: str, csv_file_path: str) -> str:
    """Execute SQL query on CSV data using pandasql with proper DataFrame loading"""
    try:
        # Load CSV data into DataFrame
        df = pd.read_csv(csv_file_path)
        
        # Get table name from CSV filename (without extension)
        table_name = os.path.splitext(os.path.basename(csv_file_path))[0]
        
        # Create locals dictionary for pandasql
        # The key must match the table name used in the SQL query
        locals_dict = {table_name: df}
        
        # Execute the SQL query using pandasql
        result_df = ps.sqldf(sql_query, locals_dict)
        
        # Format and return results
        if len(result_df) == 0:
            return "Query executed successfully but returned no results."
        else:
            # Format the results nicely
            result_str = f"Query executed successfully!\n\n"
            result_str += f"Results:\n{result_df.to_string(index=False)}\n\n"
            result_str += f"Total rows returned: {len(result_df)}\n"
            result_str += f"Columns: {list(result_df.columns)}"
            return result_str
            
    except Exception as e:
        # Provide detailed error information
        error_msg = f"Error executing SQL query on CSV data:\n"
        error_msg += f"Error: {str(e)}\n"
        error_msg += f"CSV file: {csv_file_path}\n"
        error_msg += f"Table name used: {os.path.splitext(os.path.basename(csv_file_path))[0]}\n"
        error_msg += f"SQL query: {sql_query}"
        return error_msg


def create_sql_agent_graph() -> StateGraph:
    """Create the SQL agent workflow graph with rating system"""
    workflow = StateGraph(SQLAgentState)
    
    # Add nodes
    workflow.add_node("get_schema", get_schema_node)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("evaluate_query", evaluate_query_node)
    workflow.add_node("execute_query", execute_query_node)
    workflow.add_node("retry_generation", retry_generation_node)
    
    # Define edges
    workflow.add_edge("get_schema", "generate_sql")
    workflow.add_edge("generate_sql", "evaluate_query")
    workflow.add_edge("retry_generation", "generate_sql")
    
    workflow.add_conditional_edges(
        "evaluate_query",
        should_continue,
        {
            "execute_query": "execute_query",
            "retry_generation": "retry_generation",
            END: END
        }
    )
    
    workflow.add_edge("execute_query", END)
    workflow.set_entry_point("get_schema")
    
    return workflow.compile()


@tool
def get_csv_schema(csv_path: str = None) -> str:
    """Get schema information from CSV file"""
    try:
        file_path = csv_path if csv_path else r'c:\Users\mukun\Documents\ml\datasets\placement.csv'
        df = pd.read_csv(file_path)
        table_name = os.path.splitext(os.path.basename(file_path))[0]
        
        schema_info = f"Table: {table_name}\n"
        schema_info += f"Columns: {', '.join(df.columns)}\n"
        schema_info += f"Shape: {df.shape}\n"
        schema_info += f"Data types:\n{df.dtypes.to_string()}\n\n"
        schema_info += f"Sample data (first 3 rows):\n{df.head(3).to_string(index=False)}"
        return schema_info
    except Exception as e:
        return f"Error reading CSV: {str(e)}"


def run_sql_agent(
    user_query: str, 
    schema_tool: BaseTool, 
    execute_tool: Optional[BaseTool] = None,
    csv_file_path: Optional[str] = None,
    max_retries: int = 3,
    execution_threshold: float = 0.8,
    return_full_state: bool = False
) -> str | Dict[str, Any]:
    """
    Run the SQL agent with rating-based evaluation system
    
    Args:
        user_query: Natural language query from user
        schema_tool: Tool to get database schema
        execute_tool: Tool to execute SQL queries (optional)
        csv_file_path: Path to CSV file for pandasql integration
        max_retries: Maximum number of retry attempts
        execution_threshold: Rating threshold for direct execution (default 0.8)
        return_full_state: If True, returns full state dict
        
    Returns:
        Final result (str) or full state dict
    """
    generation_model = Qwen()
    evaluation_model = Meta()
    
    initial_state = SQLAgentState(
        user_query=user_query,
        generation_model=generation_model,
        evaluation_model=evaluation_model,
        schema_tool=schema_tool,
        execute_tool=execute_tool,
        csv_file_path=csv_file_path,
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
    
    graph = create_sql_agent_graph()
    final_state = graph.invoke(initial_state)
    
    if return_full_state:
        return final_state
    else:
        # Format output with attempt details
        result = final_state.get("result", "No result available")
        error = final_state.get("error")
        query_attempts = final_state.get("query_attempts", [])
        
        output = f"=== SQL AGENT RESULT ===\n"
        output += f"Total Attempts: {len(query_attempts)}\n\n"
        
        # Show all attempts with ratings
        for attempt in query_attempts:
            output += f"Attempt {attempt['attempt_number']} (Rating: {attempt['rating']}):\n"
            output += f"Query: {attempt['query']}\n"
            output += f"Feedback: {attempt['reason']}\n\n"
        
        if error:
            output += f"Error: {error}\n\n"
        
        output += f"Final Result:\n{result}"
        
        return output


if __name__ == "__main__":
    csv_file_path = r'c:\Users\mukun\Documents\ml\datasets\placement.csv'
    
    result = run_sql_agent(
        user_query="select the top 5 students with highest cgpa and find the average cgpa of those students",
        schema_tool=lambda: get_csv_schema._run(csv_file_path),
        csv_file_path=csv_file_path,
        max_retries=3,
        execution_threshold=0.8,
        return_full_state=False
    )
    
    print(result)