import streamlit as st
import pandas as pd
import os
import json
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback
import re
from query_generator_evaluator import run_sql_agent

# Check for required dependencies
try:
    import pandasql as ps
except ImportError:
    st.error("Missing Required Package: pandasql. Please install with: pip install pandasql")
    st.stop()

# Import modules with error handling
try:
    from models import Qwen, Meta
    from query_generator_evaluator import run_sql_agent, get_csv_schema, SQLAgentState
except ImportError as e:
    st.error(f"Import Error: {str(e)}. Please ensure all files are in the correct directory.")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="SQL Query Generator",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
def initialize_session_state():
    default_values = {
        'prompt_history': [],
        'uploaded_file_path': None,
        'uploaded_file_name': None,
        'clean_file_name': None,  # Added for clean filename
        'schema_info': None,
        'query_results': None,
        'saved_prompts': [],
        'temp_dir': tempfile.mkdtemp(),
        'current_prompt': '',
        'processing': False
    }
    
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

def extract_clean_filename(filename):
    """Extract clean filename by removing timestamp prefix"""
    # Remove timestamp pattern like "20250703_044021_"
    import re
    # Pattern to match timestamp prefix: YYYYMMDD_HHMMSS_
    pattern = r'^\d{8}_\d{6}_'
    clean_name = re.sub(pattern, '', filename)
    # Remove file extension
    clean_name = os.path.splitext(clean_name)[0]
    return clean_name

def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Save uploaded file and return both paths"""
    try:
        # Extract clean filename without timestamp
        clean_name = extract_clean_filename(uploaded_file.name)
        st.session_state.clean_file_name = clean_name
        
        # Create timestamped filename for storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{timestamp}_{uploaded_file.name}"
        file_path = os.path.join(st.session_state.temp_dir, file_name)
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    except Exception as e:
        st.error(f"Error saving uploaded file: {str(e)}")
        return None

def load_saved_prompts() -> List[Dict[str, Any]]:
    """Load saved prompts from local storage"""
    try:
        if os.path.exists("saved_prompts.json"):
            with open("saved_prompts.json", "r") as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Could not load saved prompts: {str(e)}")
    return []

def save_prompts_to_file(prompts: List[Dict[str, Any]]):
    """Save prompts to local storage"""
    try:
        with open("saved_prompts.json", "w") as f:
            json.dump(prompts, f, indent=2, default=str)
    except Exception as e:
        st.warning(f"Could not save prompts: {str(e)}")

def create_schema_tool(csv_path: str, clean_table_name: str):
    """Create a schema tool for the given CSV path with clean table name"""
    class CSVSchemaTool:
        def __init__(self, csv_path: str, table_name: str):
            self.csv_path = csv_path
            self.table_name = table_name
        
        def _run(self) -> str:
            return get_csv_schema_with_clean_name(self.csv_path, self.table_name)
        
        def __call__(self) -> str:
            return self._run()
    
    return CSVSchemaTool(csv_path, clean_table_name)

def get_csv_schema_with_clean_name(csv_path: str, table_name: str) -> str:
    """Get schema information from CSV file with custom table name"""
    try:
        df = pd.read_csv(csv_path)
        
        schema_info = f"Table: {table_name}\n"
        schema_info += f"Columns: {', '.join(df.columns)}\n"
        schema_info += f"Shape: {df.shape}\n"
        schema_info += f"Data types:\n{df.dtypes.to_string()}\n\n"
        schema_info += f"Sample data (first 3 rows):\n{df.head(3).to_string(index=False)}"
        return schema_info
    except Exception as e:
        return f"Error reading CSV: {str(e)}"

def extract_sql_from_results(results_text: str) -> str:
    """Extract SQL query from results text with improved parsing"""
    try:
        lines = results_text.split('\n')
        
        # Look for different query patterns
        sql_query = None
        
        # Method 1: Look for "Query:" followed by SQL
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Pattern: "Query:" followed by SQL
            if line_stripped.startswith("Query:"):
                sql_part = line_stripped.replace("Query:", "").strip()
                if sql_part:
                    sql_query = sql_part
                    # Check if query continues on next lines until we hit "Results:" or another section
                    for j in range(i+1, len(lines)):
                        next_line = lines[j].strip()
                        if not next_line or next_line.startswith("===") or next_line.startswith("Results:") or next_line.startswith("Total Attempts:"):
                            break
                        sql_query += " " + next_line
                    break
        
        # Method 2: Look for "Best Query" pattern
        if not sql_query:
            in_best_query_section = False
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                
                if "Best Query (Rating:" in line_stripped:
                    in_best_query_section = True
                    continue
                elif in_best_query_section and line_stripped:
                    if line_stripped.lower().startswith(('select', 'insert', 'update', 'delete', 'with')):
                        sql_query = line_stripped
                        # Continue collecting query lines
                        for j in range(i+1, len(lines)):
                            next_line = lines[j].strip()
                            if not next_line or next_line.startswith("===") or next_line.startswith("Feedback:"):
                                break
                            if not next_line.lower().startswith(('select', 'insert', 'update', 'delete', 'with')):
                                sql_query += " " + next_line
                            else:
                                break
                        break
                elif in_best_query_section and line_stripped.startswith("==="):
                    in_best_query_section = False
        
        # Method 3: Look for any SQL statement in the results
        if not sql_query:
            for line in lines:
                line_stripped = line.strip()
                if line_stripped.lower().startswith(('select', 'insert', 'update', 'delete', 'with')):
                    sql_query = line_stripped
                    break
        
        # Method 4: Extract from "Attempt X" sections
        if not sql_query:
            attempt_queries = []
            in_attempt_section = False
            current_query = ""
            
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                
                if line_stripped.startswith("Attempt") and "Rating:" in line_stripped:
                    in_attempt_section = True
                    if current_query:
                        attempt_queries.append(current_query.strip())
                        current_query = ""
                    continue
                elif line_stripped.startswith("Query:") and in_attempt_section:
                    query_part = line_stripped.replace("Query:", "").strip()
                    current_query = query_part
                    # Continue collecting query lines
                    for j in range(i+1, len(lines)):
                        next_line = lines[j].strip()
                        if not next_line or next_line.startswith("Feedback:") or next_line.startswith("Attempt"):
                            break
                        current_query += " " + next_line
                elif line_stripped.startswith("Feedback:") and in_attempt_section:
                    if current_query:
                        attempt_queries.append(current_query.strip())
                        current_query = ""
                    in_attempt_section = False
            
            # Add the last query if exists
            if current_query:
                attempt_queries.append(current_query.strip())
            
            # Use the last (most recent) attempt query
            if attempt_queries:
                sql_query = attempt_queries[-1]
        
        # Clean up the extracted query
        if sql_query:
            # Remove common prefixes/suffixes
            sql_query = sql_query.strip()
            
            # Remove markdown code blocks if present
            if sql_query.startswith("```sql"):
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            elif sql_query.startswith("```"):
                sql_query = sql_query.replace("```", "").strip()
            
            # Remove trailing semicolon and whitespace
            sql_query = sql_query.rstrip(';').strip()
            
            # Ensure it's a valid SQL statement
            if sql_query and any(sql_query.lower().startswith(keyword) for keyword in ['select', 'insert', 'update', 'delete', 'with']):
                return sql_query
        
        return ""
    except Exception as e:
        st.warning(f"Error extracting SQL: {str(e)}")
        return ""

def main():
    """Main application function"""
    initialize_session_state()
    
    # Load saved prompts
    if not st.session_state.saved_prompts:
        st.session_state.saved_prompts = load_saved_prompts()
    
    # Header
    st.title("🔍 SQL Query Generator")
    st.write("Transform your natural language questions into SQL queries!")
    st.divider()
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("📁 File Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file to generate SQL queries"
        )
        
        if uploaded_file is not None:
            if not st.session_state.processing:
                file_path = save_uploaded_file(uploaded_file)
                if file_path:
                    st.session_state.uploaded_file_path = file_path
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.success("File uploaded successfully!")
                    
                    # Display file info
                    try:
                        df = pd.read_csv(file_path)
                        st.info(f"**File Info:**\n- Rows: {len(df):,}\n- Columns: {len(df.columns)}\n- Table Name: {st.session_state.clean_file_name}")
                        
                        # Show column names
                        with st.expander("Column Names"):
                            for i, col in enumerate(df.columns, 1):
                                st.write(f"{i}. {col}")
                                
                    except Exception as e:
                        st.warning(f"Could not read file info: {str(e)}")
        
        st.divider()
        
        # Settings
        st.header("⚙️ Settings")
        
        max_retries = st.slider("Max Retries", min_value=1, max_value=5, value=3)
        execution_threshold = st.slider("Execution Threshold", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
        
        # Saved prompts
        if st.session_state.saved_prompts:
            st.divider()
            st.header("💾 Saved Prompts")
            
            selected_prompt = st.selectbox(
                "Load saved prompt",
                options=[""] + [p["name"] for p in st.session_state.saved_prompts]
            )
            
            if selected_prompt:
                prompt_data = next((p for p in st.session_state.saved_prompts if p["name"] == selected_prompt), None)
                if prompt_data:
                    st.text_area("Preview:", value=prompt_data["query"], height=100, disabled=True)
                    if st.button("Load This Prompt"):
                        st.session_state.current_prompt = prompt_data["query"]
                        st.success("Prompt loaded!")
                        st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Enter Your Query")
        
        # Display current file info
        if st.session_state.uploaded_file_name:
            st.info(f"📄 **Current File:** {st.session_state.uploaded_file_name}")
            st.info(f"🏷️ **Table Name:** {st.session_state.clean_file_name}")
        else:
            st.warning("Please upload a CSV file first")
        
        # Query input
        default_query = st.session_state.get('current_prompt', '')
        
        user_query = st.text_area(
            "Enter your question about the data:",
            value=default_query,
            height=120,
            placeholder="e.g., Show me the top 5 customers by total sales amount"
        )
        
        # Clear current prompt after using it
        if 'current_prompt' in st.session_state and st.session_state.current_prompt:
            st.session_state.current_prompt = ''
        
        # Action buttons
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn1:
            generate_clicked = st.button("🚀 Generate SQL", type="primary", use_container_width=True)
        
        with col_btn2:
            if user_query.strip():
                save_name = st.text_input("Save as:", placeholder="My Query")
                if st.button("💾 Save Prompt", use_container_width=True) and save_name:
                    new_prompt = {
                        "name": save_name,
                        "query": user_query,
                        "saved_at": datetime.now().isoformat()
                    }
                    st.session_state.saved_prompts.append(new_prompt)
                    save_prompts_to_file(st.session_state.saved_prompts)
                    st.success("Prompt saved!")
                    st.rerun()
        
        with col_btn3:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.query_results = None
                st.session_state.schema_info = None
                st.session_state.prompt_history = []
                st.success("Cleared!")
                st.rerun()
    
    with col2:
        # Display additional info
        if st.session_state.schema_info:
            st.header("📋 Schema Info")
            with st.expander("View Schema"):
                st.text(st.session_state.schema_info)
    
    # Process query generation
    if generate_clicked:
        if not user_query.strip():
            st.error("Please enter a query!")
            return
        
        if not st.session_state.uploaded_file_path or not st.session_state.clean_file_name:
            st.error("Please upload a CSV file first!")
            return
        
        # Set processing flag
        st.session_state.processing = True
        
        # Show progress
        with st.spinner("Processing your query..."):
            try:
                # Create schema tool with clean table name
                schema_tool = create_schema_tool(st.session_state.uploaded_file_path, st.session_state.clean_file_name)
                
                # Get schema info for display
                st.session_state.schema_info = schema_tool._run()
                
                # Run SQL agent
                results = run_sql_agent(
                    user_query=user_query,
                    schema_tool=schema_tool,
                    csv_file_path=st.session_state.uploaded_file_path,
                    clean_table_name=st.session_state.clean_file_name,
                    max_retries=max_retries,
                    execution_threshold=execution_threshold,
                    return_full_state=False
                )
                
                st.session_state.query_results = results
                
                # Add to history
                history_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "query": user_query,
                    "file": st.session_state.uploaded_file_name,
                    "table_name": st.session_state.clean_file_name,
                    "results": results
                }
                st.session_state.prompt_history.append(history_entry)
                
                st.success("SQL query generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating SQL: {str(e)}")
                st.error(f"Debug info: {traceback.format_exc()}")
            finally:
                st.session_state.processing = False
    
    # Display results
    if st.session_state.query_results:
        st.divider()
        st.header("📊 Results")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["🎯 Query Results", "📋 Schema Info", "📈 Analysis"])
        
        with tab1:
            st.text(st.session_state.query_results)
            
            # Extract SQL query for copying with improved parsing
            sql_query = extract_sql_from_results(st.session_state.query_results)
            if sql_query:
                st.subheader("📝 Generated SQL Query")
                print(sql_query)
                st.code(sql_query, language="sql")
                
                # Add copy button functionality
                st.write("**Copy the SQL query above to use in your database client**")
            else:
                st.warning("Could not extract SQL query from results. Please check the full results above.")
        
        with tab2:
            if st.session_state.schema_info:
                st.text(st.session_state.schema_info)
        
        with tab3:
            # Parse and display analysis
            results_text = st.session_state.query_results
            
            if "Total Attempts:" in results_text:
                lines = results_text.split('\n')
                attempts = 0
                ratings = []
                
                for line in lines:
                    if line.startswith("Total Attempts:"):
                        attempts = int(line.split(":")[1].strip())
                    elif "Rating:" in line and ")" in line:
                        try:
                            rating_part = line.split("Rating: ")[1].split(")")[0]
                            ratings.append(float(rating_part))
                        except:
                            pass
                
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Total Attempts", attempts)
                with col_m2:
                    if ratings:
                        st.metric("Best Rating", f"{max(ratings):.2f}")
                with col_m3:
                    if ratings:
                        st.metric("Avg Rating", f"{sum(ratings)/len(ratings):.2f}")
            
            st.text(results_text)
    
    # Query history - FIXED: No nested expanders
    if st.session_state.prompt_history:
        st.divider()
        st.header("📚 Query History")
        
        with st.expander(f"Recent Queries ({len(st.session_state.prompt_history)})", expanded=False):
            for i, entry in enumerate(reversed(st.session_state.prompt_history[-5:])):
                st.write(f"**Query {len(st.session_state.prompt_history)-i}:** {entry['query']}")
                st.write(f"**File:** {entry['file']} (Table: {entry['table_name']})")
                st.write(f"**Time:** {entry['timestamp'][:19].replace('T', ' ')}")
                
                # Use a toggle button instead of nested expander
                show_results_key = f"show_results_{len(st.session_state.prompt_history)-i}"
                if show_results_key not in st.session_state:
                    st.session_state[show_results_key] = False
                
                if st.button(f"🔍 {'Hide' if st.session_state[show_results_key] else 'Show'} Results", 
                           key=f"toggle_{len(st.session_state.prompt_history)-i}"):
                    st.session_state[show_results_key] = not st.session_state[show_results_key]
                
                # Show results if toggled on
                if st.session_state[show_results_key]:
                    st.text(entry['results'])
                    
                    # Extract and show SQL from history entry
                    historical_sql = extract_sql_from_results(entry['results'])
                    if historical_sql:
                        st.code(historical_sql, language="sql")
                
                if i < len(st.session_state.prompt_history[-5:]) - 1:
                    st.divider()

if __name__ == "__main__":
    main()