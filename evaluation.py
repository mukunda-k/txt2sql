# evaluation.py

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
from functools import partial
import uuid
from datetime import datetime

# Third-party imports
try:
    from langsmith import Client
    from langsmith.evaluation import evaluate, LangChainStringEvaluator
    from langsmith.schemas import Run, Example
    import pandasql as ps
except ImportError as e:
    print(f"Import Error: {e}. Please install required packages: pip install langsmith pandasql pandas")
    exit()

# Local application imports
from query_generator_evaluator import run_sql_agent, get_csv_schema
from app import extract_sql_from_results

# --- Environment Setup ---
# For this script to run, you must set the following environment variables:
# os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"
# os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"

# --- LangSmith Client Initialization ---
try:
    client = Client(
        api_key='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',
        api_url="https://api.smith.langchain.com"
    )
except Exception as e:
    print(f"Error initializing LangSmith client: {e}")
    print("Please ensure your LANGCHAIN_API_KEY is set as an environment variable.")
    client = None

# --- Helper Function ---
def create_schema_tool(csv_path: str, table_name: str):
    """
    Creates a schema tool for the given CSV path.
    This is a local implementation to avoid dependencies on the Streamlit app.
    """
    class CSVSchemaTool:
        def __init__(self, csv_path: str, table_name: str):
            self.csv_path = csv_path
            self.table_name = table_name

        def _run(self) -> str:
            schema_info = get_csv_schema(self.csv_path)
            # Prepend the table name to the schema
            return f"Table: {self.table_name}\n{schema_info}"

        def __call__(self) -> str:
            return self._run()

    return CSVSchemaTool(csv_path, table_name)


class SQLQueryEvaluator:
    """
    A class to create datasets, run evaluations, and generate reports
    for the SQL query generation agent using LangSmith.
    """
    def __init__(self, project_name: str = "sql-query-generator-eval"):
        self.client = client
        self.project_name = project_name

    def create_evaluation_dataset(self, csv_file_path: str, table_name: str) -> str:
        """Create a comprehensive evaluation dataset in LangSmith."""
        df = pd.read_csv(csv_file_path)
        columns = df.columns.tolist()
        
        # Define a diverse set of test cases with more specific patterns
        test_cases = [
            {
                "query": "Show me all records", 
                "expected_sql_pattern": f"SELECT * FROM {table_name}", 
                "difficulty": "easy", 
                "category": "basic_select"
            },
            {
                "query": f"Get the first 10 rows of {columns[0]} and {columns[1]}", 
                "expected_sql_pattern": f"SELECT {columns[0]}, {columns[1]} FROM {table_name} LIMIT 10", 
                "difficulty": "easy", 
                "category": "basic_select"
            },
            {
                "query": "What is the total count of records?", 
                "expected_sql_pattern": f"SELECT COUNT(*) FROM {table_name}", 
                "difficulty": "medium", 
                "category": "aggregation"
            },
            {
                "query": "Find the average salary", 
                "expected_sql_pattern": f"SELECT AVG(salary) FROM {table_name}", 
                "difficulty": "medium", 
                "category": "aggregation"
            },
            {
                "query": "Show records for the Engineering department", 
                "expected_sql_pattern": f"WHERE department = 'Engineering'", 
                "difficulty": "medium", 
                "category": "filtering"
            },
            {
                "query": "Sort the data by salary in descending order", 
                "expected_sql_pattern": f"ORDER BY salary DESC", 
                "difficulty": "medium", 
                "category": "sorting"
            },
            {
                "query": "Group by department and count the number of employees in each", 
                "expected_sql_pattern": f"SELECT department, COUNT(*) FROM {table_name} GROUP BY department", 
                "difficulty": "hard", 
                "category": "grouping"
            },
        ]

        dataset_name = f"{self.project_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Step 1: Create an empty dataset first
        dataset = self.client.create_dataset(dataset_name=dataset_name)

        # Prepare lists of inputs and outputs for bulk-adding
        inputs = []
        outputs = []
        
        for case in test_cases:
            inputs.append({"user_query": case["query"]})
            outputs.append({
                "expected_pattern": case["expected_sql_pattern"],
                "difficulty": case["difficulty"],
                "category": case["category"]
            })

        # Step 2: Add examples to the dataset you just created
        self.client.create_examples(
            inputs=inputs,
            outputs=outputs,
            dataset_id=dataset.id,
            metadata=[{"table_name": table_name} for _ in test_cases]
        )
        return dataset_name

    def sql_correctness_evaluator(self, run: Run, example: Example, csv_file_path: str) -> Dict[str, Any]:
        """Custom evaluator for SQL correctness, syntax, and execution."""
        try:
            # Extract output from run
            output = run.outputs.get("result", "") if run.outputs else ""
            
            # Extract SQL from results - handle case where extract_sql_from_results might not exist
            try:
                generated_sql = extract_sql_from_results(output)
            except:
                # Fallback: simple extraction
                generated_sql = self._extract_sql_fallback(output)
            
            # Get table name from metadata or use default
            table_name = "employees_data"
            if hasattr(example, 'metadata') and example.metadata:
                table_name = example.metadata.get("table_name", "employees_data")
            
            syntax_score = self.check_sql_syntax(generated_sql, table_name)
            pattern_score = self.check_sql_pattern_match(generated_sql, example.outputs.get("expected_pattern", ""))
            execution_score = self.test_sql_execution(generated_sql, table_name, csv_file_path)
            
            overall_score = (syntax_score * 0.3) + (pattern_score * 0.3) + (execution_score * 0.4)
            
            feedback = self.generate_feedback(generated_sql, syntax_score, pattern_score, execution_score)
            
            return {
                "key": "sql_correctness",
                "score": overall_score, 
                "value": overall_score,
                "comment": feedback,
                "correction": None
            }
        except Exception as e:
            return {
                "key": "sql_correctness",
                "score": 0.0, 
                "value": 0.0,
                "comment": f"Evaluation failed: {e}",
                "correction": None
            }

    def _extract_sql_fallback(self, output: str) -> str:
        """Fallback SQL extraction method"""
        if not output:
            return ""
        
        # Look for SQL patterns in the output
        import re
        
        # Try to find SQL queries
        sql_patterns = [
            r'```sql\s*(.*?)\s*```',
            r'Query:\s*(.*?)(?:\n|$)',
            r'SELECT.*?FROM.*?(?:\n|$)',
        ]
        
        for pattern in sql_patterns:
            matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        # If no pattern found, return the output as is
        return output.strip()

    def check_sql_syntax(self, sql: str, table_name: str) -> float:
        """Basic SQL syntax and structure validation."""
        if not sql: 
            return 0.0
        
        score = 0.0
        sql_upper = sql.upper()
        
        # Check for basic SQL keywords
        if any(keyword in sql_upper for keyword in ['SELECT', 'FROM']):
            score += 0.5
        
        # Check for table name usage
        if table_name.lower() in sql.lower():
            score += 0.5
        
        return min(score, 1.0)

    def check_sql_pattern_match(self, generated_sql: str, expected_pattern: str) -> float:
        """Check if generated SQL contains key elements from the expected pattern."""
        if not generated_sql or not expected_pattern:
            return 0.0
        
        score = 0.0
        generated_upper = generated_sql.upper()
        expected_upper = expected_pattern.upper()
        
        # Split expected pattern into key components
        expected_parts = expected_upper.split()
        matched_parts = 0
        
        for part in expected_parts:
            if len(part) > 2 and part in generated_upper:  # Skip very short parts
                matched_parts += 1
        
        if expected_parts:
            score = matched_parts / len(expected_parts)
        
        return min(score, 1.0)

    def test_sql_execution(self, sql: str, table_name: str, csv_file_path: str) -> float:
        """Test if the SQL query can be executed against the CSV file using pandasql."""
        try:
            if not sql:
                return 0.0
            
            # Load CSV data
            df = pd.read_csv(csv_file_path)
            
            # Create table dictionary
            table_dict = {table_name: df}
            
            # Execute SQL query
            result = ps.sqldf(sql, table_dict)
            
            # If we get here without exception, execution was successful
            return 1.0
            
        except Exception as e:
            return 0.0

    def generate_feedback(self, sql: str, syntax_score: float, pattern_score: float, execution_score: float) -> str:
        """Generate human-readable feedback based on scores."""
        if execution_score < 1.0:
            return "SQL failed to execute successfully."
        if syntax_score < 0.8:
            return "SQL syntax or table name usage is incorrect."
        if pattern_score < 0.6:
            return "SQL does not seem to match the expected query pattern."
        return "Good query generated successfully."

    def run_evaluation(self, dataset_name: str, sql_agent_function, csv_file_path: str):
        """Run the evaluation on the specified dataset."""
        # Create a wrapper for the evaluator that includes csv_file_path
        def wrapped_evaluator(run, example):
            return self.sql_correctness_evaluator(run=run, example=example, csv_file_path=csv_file_path)
        
        results = evaluate(
            sql_agent_function,
            data=dataset_name,
            evaluators=[wrapped_evaluator],
            experiment_prefix=self.project_name,
            metadata={"version": "1.0", "model_under_test": "qwen-2.5-coder"}
        )
        return results

    def generate_evaluation_report(self, results) -> Dict[str, Any]:
        """Generate a summary report from the evaluation results - FIXED VERSION."""
        try:
            # NEW: Access experiment results data directly
            experiment_results = results
            
            # Get experiment data
            experiment_name = experiment_results.experiment_name if hasattr(experiment_results, 'experiment_name') else "Unknown"
            
            # NEW: Get all runs from the experiment
            runs = list(self.client.list_runs(
                project_name=experiment_name,
                execution_order=1,
                is_root=True
            ))
            
            if not runs:
                return {"error": "No runs found in the experiment."}
            
            # Extract scores and feedback from runs
            scores = []
            feedbacks = []
            detailed_results = []
            
            for i, run in enumerate(runs):
                try:
                    # Get feedback data from run
                    feedback_data = run.feedback_stats if hasattr(run, 'feedback_stats') else {}
                    
                    # Extract score - check multiple possible locations
                    score = None
                    feedback = "No feedback available"
                    
                    # Check if feedback_stats exists and has our evaluator results
                    if feedback_data and 'sql_correctness' in feedback_data:
                        correctness_data = feedback_data['sql_correctness']
                        if isinstance(correctness_data, dict):
                            score = correctness_data.get('avg', 0.0)
                            feedback = f"Avg score: {score}"
                        else:
                            score = float(correctness_data) if correctness_data else 0.0
                    
                    # Fallback: check run outputs for any score information
                    if score is None and hasattr(run, 'outputs') and run.outputs:
                        output_str = str(run.outputs)
                        # Try to extract any score information from outputs
                        import re
                        score_match = re.search(r'score["\']?\s*:\s*([0-9.]+)', output_str, re.IGNORECASE)
                        if score_match:
                            score = float(score_match.group(1))
                    
                    # Default score if none found
                    if score is None:
                        score = 0.0
                    
                    scores.append(score)
                    feedbacks.append(feedback)
                    
                    # Create detailed result entry
                    detailed_result = {
                        "test_case": i + 1,
                        "run_id": str(run.id),
                        "score": score,
                        "feedback": feedback,
                        "status": run.status if hasattr(run, 'status') else "completed"
                    }
                    
                    # Add input/output information if available
                    if hasattr(run, 'inputs') and run.inputs:
                        detailed_result["input"] = run.inputs.get("user_query", "Unknown query")
                    if hasattr(run, 'outputs') and run.outputs:
                        detailed_result["output"] = str(run.outputs)[:200] + "..." if len(str(run.outputs)) > 200 else str(run.outputs)
                    
                    detailed_results.append(detailed_result)
                    
                except Exception as e:
                    print(f"Error processing run {i}: {e}")
                    # Add a default entry for failed processing
                    scores.append(0.0)
                    feedbacks.append(f"Error processing run: {e}")
                    detailed_results.append({
                        "test_case": i + 1,
                        "run_id": str(run.id) if hasattr(run, 'id') else "unknown",
                        "score": 0.0,
                        "feedback": f"Error processing run: {e}",
                        "status": "error"
                    })
            
            if not scores:
                return {"error": "No scores could be extracted from runs."}

            # Calculate statistics
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            
            # Calculate pass/fail counts (assuming 0.6 as pass threshold)
            pass_threshold = 0.6
            passed_tests = sum(1 for score in scores if score >= pass_threshold)
            failed_tests = len(scores) - passed_tests
            
            return {
                "experiment_name": experiment_name,
                "overall_average_score": avg_score,
                "max_score": max_score,
                "min_score": min_score,
                "total_tests_run": len(scores),
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "pass_rate": passed_tests / len(scores) if scores else 0.0,
                "detailed_results": detailed_results,
                "summary": {
                    "total_queries_tested": len(scores),
                    "average_score": round(avg_score, 3),
                    "score_distribution": {
                        "excellent (>= 0.9)": sum(1 for s in scores if s >= 0.9),
                        "good (0.7-0.89)": sum(1 for s in scores if 0.7 <= s < 0.9),
                        "fair (0.5-0.69)": sum(1 for s in scores if 0.5 <= s < 0.7),
                        "poor (< 0.5)": sum(1 for s in scores if s < 0.5)
                    }
                }
            }
            
        except Exception as e:
            return {"error": f"Error generating report: {str(e)}"}


def setup_and_run_evaluation(csv_file_path: str):
    """Setup the environment and run the full evaluation process."""
    if not client:
        print("LangSmith client not available. Aborting evaluation.")
        return None

    table_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    print(f"Starting evaluation for table: '{table_name}' using file: '{csv_file_path}'")
    
    evaluator = SQLQueryEvaluator(project_name=f"sql-gen-eval-{table_name}")
    
    print("Creating evaluation dataset in LangSmith...")
    dataset_name = evaluator.create_evaluation_dataset(csv_file_path=csv_file_path, table_name=table_name)
    print(f"Dataset '{dataset_name}' created successfully.")
    
    def sql_agent_wrapper(inputs: dict) -> dict:
        """Wraps the agent to be compatible with LangSmith's evaluate function."""
        user_query = inputs["user_query"]
        print(f"\nRunning agent for query: '{user_query}'")
        
        schema_tool = create_schema_tool(csv_file_path, table_name)
        
        try:
            result_string = run_sql_agent(
                user_query=user_query,
                schema_tool=schema_tool,
                csv_file_path=csv_file_path,
                clean_table_name=table_name,
                max_retries=1,
                execution_threshold=0.7,
                return_full_state=False
            )
            return {"result": result_string}
        except Exception as e:
            return {"result": f"Error running agent: {str(e)}"}

    print("\nRunning evaluation against the dataset...")
    results = evaluator.run_evaluation(dataset_name, sql_agent_wrapper, csv_file_path)
    
    print("\nGenerating evaluation report...")
    report = evaluator.generate_evaluation_report(results)
    
    print("\nEvaluation Complete!")
    return report


if __name__ == "__main__":
    # Create a dummy CSV file for a runnable demonstration
    csv_data = """employee_id,first_name,last_name,department,salary,hire_date
101,John,Doe,Engineering,90000,2022-01-15
102,Jane,Smith,Marketing,75000,2021-11-20
103,Peter,Jones,Engineering,92000,2022-03-10
104,Mary,Johnson,Sales,85000,2023-01-05
"""
    dummy_csv_path = "employees_data.csv"
    with open(dummy_csv_path, "w") as f:
        f.write(csv_data)
    print(f"Created a dummy CSV file: '{dummy_csv_path}'")

    # Check if API keys are available (commented out the actual check)
    api_keys_available = True  # Set to True to run the evaluation
    
    if not api_keys_available:
        print("\n" + "="*60)
        print("WARNING: API KEYS NOT FOUND")
        print("Please set the following environment variables to run:")
        print(" - LANGCHAIN_API_KEY (for LangSmith reporting)")
        print(" - OPENROUTER_API_KEY (for Qwen/Meta models via OpenRouter)")
        print("="*60 + "\n")
    else:
        try:
            evaluation_report = setup_and_run_evaluation(csv_file_path=dummy_csv_path)
            if evaluation_report:
                print("\n\n" + "="*25 + " EVALUATION REPORT " + "="*25)
                
                # Check if there's an error in the report
                if "error" in evaluation_report:
                    print(f"Error in evaluation: {evaluation_report['error']}")
                else:
                    # Print detailed summary
                    print(f"Experiment: {evaluation_report.get('experiment_name', 'Unknown')}")
                    print(f"Overall Average Score: {evaluation_report.get('overall_average_score', 'N/A'):.3f}")
                    print(f"Total Tests Run: {evaluation_report.get('total_tests_run', 'N/A')}")
                    print(f"Passed Tests: {evaluation_report.get('passed_tests', 'N/A')}")
                    print(f"Failed Tests: {evaluation_report.get('failed_tests', 'N/A')}")
                    print(f"Pass Rate: {evaluation_report.get('pass_rate', 'N/A'):.1%}")
                    
                    # Print score distribution
                    if 'summary' in evaluation_report and 'score_distribution' in evaluation_report['summary']:
                        print("\nScore Distribution:")
                        for category, count in evaluation_report['summary']['score_distribution'].items():
                            print(f"  {category}: {count}")
                
                print("="*70)
                
                # Optionally, save the full report to a file
                with open("evaluation_report.json", "w") as f:
                    json.dump(evaluation_report, f, indent=2, default=str)
                print("Full report saved to 'evaluation_report.json'")
            else:
                print("Evaluation failed to generate a report.")
        except Exception as e:
            print(f"Evaluation failed with error: {str(e)}")
            import traceback
            traceback.print_exc()

    # Clean up the dummy file
    try:
        os.remove(dummy_csv_path)
        print(f"\nCleaned up dummy CSV file: '{dummy_csv_path}'")
    except:
        print(f"\nWarning: Could not clean up dummy CSV file: '{dummy_csv_path}'")