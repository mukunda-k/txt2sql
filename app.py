import streamlit as st
import pandas as pd
import os
import json
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback
import sys

# Check for required dependencies and provide helpful error messages
try:
    import pandasql as ps
except ImportError:
    st.error("""
    ❌ **Missing Required Package: pandasql**
    
    Please install the missing package by running:
    ```bash
    pip install pandasql
    ```
    
    Or install all requirements:
    ```bash
    pip install -r requirements.txt
    ```
    """)
    st.stop()

# Import your existing modules with error handling
try:
    from models import Qwen, Meta
    from qeury_generator_evaluator import run_sql_agent, get_csv_schema, SQLAgentState
except ImportError as e:
    st.error(f"""
    ❌ **Import Error**
    
    Could not import required modules: {str(e)}
    
    Please ensure:
    1. `models.py` and `qg2.py` are in the same directory as this app
    2. All dependencies are installed: `pip install -r requirements.txt`
    3. No syntax errors in the imported files
    """)
    st.stop()
except Exception as e:
    st.error(f"""
    ❌ **Module Loading Error**
    
    Error loading modules: {str(e)}
    
    Please check:
    1. All required packages are installed
    2. API keys are properly configured
    3. No circular imports exist
    
    Full error: {traceback.format_exc()}
    """)
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="SQL Query Generator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling and copy functionality
st.markdown("""
<style>
.copy-button {
    position: relative;
    display: inline-block;
    margin-top: 5px;
}

.copy-text {
    background-color: #f0f2f6;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 10px;
    font-family: monospace;
    white-space: pre-wrap;
    word-wrap: break-word;
    max-height: 200px;
    overflow-y: auto;
}

.stAlert > div {
    padding: 1rem;
}

.schema-info {
    background-color: #f8f9fa;
    border-left: 4px solid #007bff;
    padding: 10px;
    margin: 10px 0;
}

.query-result {
    background-color: #e8f5e8;
    border: 1px solid #4caf50;
    border-radius: 4px;
    padding: 10px;
    margin: 10px 0;
}

.error-result {
    background-color: #ffebee;
    border: 1px solid #f44336;
    border-radius: 4px;
    padding: 10px;
    margin: 10px 0;
}
</style>

<script>
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        // Show success message
        const button = event.target;
        const originalText = button.innerHTML;
        button.innerHTML = '✅ Copied!';
        setTimeout(() => {
            button.innerHTML = originalText;
        }, 2000);
    });
}
</script>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'prompt_history' not in st.session_state:
        st.session_state.prompt_history = []
    if 'uploaded_file_path' not in st.session_state:
        st.session_state.uploaded_file_path = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'schema_info' not in st.session_state:
        st.session_state.schema_info = None
    if 'query_results' not in st.session_state:
        st.session_state.query_results = None
    if 'saved_prompts' not in st.session_state:
        st.session_state.saved_prompts = load_saved_prompts()
    if 'current_dataframe' not in st.session_state:
        st.session_state.current_dataframe = None
    if 'detailed_schema' not in st.session_state:
        st.session_state.detailed_schema = None

def load_saved_prompts() -> List[Dict]:
    """Load saved prompts from file"""
    try:
        if os.path.exists('saved_prompts.json'):
            with open('saved_prompts.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading saved prompts: {str(e)}")
    return []

def save_prompts_to_file(prompts: List[Dict]):
    """Save prompts to file"""
    try:
        with open('saved_prompts.json', 'w') as f:
            json.dump(prompts, f, indent=2)
    except Exception as e:
        st.error(f"Error saving prompts: {str(e)}")

def add_to_history(prompt: str, result: str, sql_query: str = None, rating: float = None, error: str = None):
    """Add prompt and result to history"""
    history_item = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'prompt': prompt,
        'result': result,
        'sql_query': sql_query,
        'rating': rating,
        'error': error
    }
    
    st.session_state.prompt_history.insert(0, history_item)
    # Keep only latest 3 prompts
    if len(st.session_state.prompt_history) > 3:
        st.session_state.prompt_history = st.session_state.prompt_history[:3]

def save_prompt(prompt: str, result: str, sql_query: str = None):
    """Save a prompt to permanent storage"""
    saved_item = {
        'id': len(st.session_state.saved_prompts) + 1,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'prompt': prompt,
        'result': result,
        'sql_query': sql_query,
        'file_name': st.session_state.uploaded_file_name
    }
    
    st.session_state.saved_prompts.append(saved_item)
    save_prompts_to_file(st.session_state.saved_prompts)
    st.success("Prompt saved successfully!")

def get_detailed_schema_info(df: pd.DataFrame) -> str:
    """Generate detailed schema information with examples"""
    table_name = os.path.splitext(st.session_state.uploaded_file_name)[0] if st.session_state.uploaded_file_name else "data"
    
    schema_info = f"**Table Name:** {table_name}\n\n"
    schema_info += f"**Total Rows:** {len(df)}\n"
    schema_info += f"**Total Columns:** {len(df.columns)}\n\n"
    
    schema_info += "**Column Details:**\n"
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()
        null_count = df[col].isnull().sum()
        
        schema_info += f"- **{col}** ({dtype})\n"
        schema_info += f"  - Unique values: {unique_count}\n"
        schema_info += f"  - Null values: {null_count}\n"
        
        # Add sample values
        if dtype in ['object', 'string']:
            sample_values = df[col].dropna().unique()[:3]
            schema_info += f"  - Sample values: {', '.join(map(str, sample_values))}\n"
        elif dtype in ['int64', 'float64', 'int32', 'float32']:
            min_val, max_val = df[col].min(), df[col].max()
            schema_info += f"  - Range: {min_val} to {max_val}\n"
        schema_info += "\n"
    
    return schema_info

def process_csv_upload(uploaded_file):
    """Process uploaded CSV file with enhanced schema analysis"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Update session state
        st.session_state.uploaded_file_path = tmp_file_path
        st.session_state.uploaded_file_name = uploaded_file.name
        
        # Load and analyze the data
        df = pd.read_csv(tmp_file_path)
        st.session_state.current_dataframe = df
        
        # Generate detailed schema
        detailed_schema = get_detailed_schema_info(df)
        st.session_state.detailed_schema = detailed_schema
        
        # Basic schema info for display
        table_name = os.path.splitext(uploaded_file.name)[0]
        schema_info = f"**Table:** {table_name}\n"
        schema_info += f"**Columns:** {', '.join(df.columns)}\n"
        schema_info += f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns\n"
        
        st.session_state.schema_info = schema_info
        
        return df, schema_info, tmp_file_path
        
    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")
        return None, None, None

def execute_sql_with_retry(user_query: str, csv_file_path: str, max_attempts: int = 3) -> tuple:
    """Execute SQL query with automatic retry on errors"""
    attempt = 0
    last_error = None
    query_history = []
    
    while attempt < max_attempts:
        try:
            attempt += 1
            st.info(f"🔄 Attempt {attempt}/{max_attempts}: Generating SQL query...")
            
            with st.spinner("Generating and executing SQL query..."):
                # Create enhanced schema tool
                def enhanced_schema_tool():
                    if st.session_state.detailed_schema:
                        return st.session_state.detailed_schema
                    return get_csv_schema._run(csv_file_path)
                
                # Add error context if this is a retry attempt
                query_with_context = user_query
                if attempt > 1 and last_error:
                    query_with_context += f"\n\nPREVIOUS ERROR TO FIX: {last_error}"
                    query_with_context += f"\nPREVIOUS FAILED QUERIES: {', '.join(query_history)}"
                    query_with_context += "\nPlease generate a corrected SQL query that fixes these issues."
                
                # Run the SQL agent
                result = run_sql_agent(
                    user_query=query_with_context,
                    schema_tool=enhanced_schema_tool,
                    csv_file_path=csv_file_path,
                    max_retries=2,  # Internal retries for rating
                    execution_threshold=0.7,  # Lower threshold for faster execution
                    return_full_state=True
                )
                
                # Extract information from result
                final_result = result.get("result", "No result available")
                error = result.get("error")
                query_attempts = result.get("query_attempts", [])
                best_query = result.get("best_query")
                
                # Get the executed query
                if best_query:
                    sql_query = best_query.get("query", "No SQL query generated")
                    rating = best_query.get("rating", 0.0)
                else:
                    sql_query = result.get("sql_query", "No SQL query generated")
                    rating = result.get("query_rating", 0.0)
                
                # Test execute the query to catch runtime errors
                if sql_query and sql_query != "No SQL query generated":
                    test_result = test_execute_query(sql_query, csv_file_path)
                    if "Error" in test_result:
                        last_error = test_result
                        query_history.append(sql_query)
                        if attempt < max_attempts:
                            st.warning(f"⚠️ Attempt {attempt} failed: {test_result}")
                            continue
                        else:
                            return f"❌ All attempts failed. Last error: {test_result}", sql_query, rating
                    else:
                        # Success - return the working query and result
                        success_msg = f"✅ Query successful on attempt {attempt}!\n\n{test_result}"
                        return success_msg, sql_query, rating
                
                # If no query generated, treat as error
                if not sql_query or sql_query == "No SQL query generated":
                    last_error = "No valid SQL query was generated"
                    if attempt < max_attempts:
                        st.warning(f"⚠️ Attempt {attempt} failed: No query generated")
                        continue
                    else:
                        return "❌ Failed to generate a valid SQL query", None, 0.0
                
                # Return the result if no errors
                return final_result, sql_query, rating
                
        except Exception as e:
            last_error = str(e)
            if attempt < max_attempts:
                st.warning(f"⚠️ Attempt {attempt} failed with error: {str(e)}")
            else:
                error_msg = f"❌ All attempts failed. Final error: {str(e)}\n"
                error_msg += f"Traceback: {traceback.format_exc()}"
                return error_msg, None, 0.0
    
    return f"❌ Maximum attempts ({max_attempts}) reached. Last error: {last_error}", None, 0.0

def test_execute_query(sql_query: str, csv_file_path: str) -> str:
    """Test execute a SQL query to catch errors"""
    try:
        # Load CSV data
        df = pd.read_csv(csv_file_path)
        table_name = os.path.splitext(os.path.basename(csv_file_path))[0]
        
        # Create locals dictionary
        locals_dict = {table_name: df}
        
        # Execute query
        result_df = ps.sqldf(sql_query, locals_dict)
        
        # Format results
        if len(result_df) == 0:
            return "✅ Query executed successfully but returned no results."
        else:
            result_str = f"✅ Query executed successfully!\n\n"
            result_str += f"**Results Preview:**\n{result_df.head(10).to_string(index=False)}\n\n"
            result_str += f"📊 **Summary:** {len(result_df)} rows returned\n"
            result_str += f"📋 **Columns:** {', '.join(result_df.columns)}"
            return result_str
            
    except Exception as e:
        return f"Error: {str(e)}"

def render_copy_button(text: str, button_text: str = "📋 Copy"):
    """Render a copy button for text"""
    button_id = f"copy_btn_{hash(text) % 10000}"
    
    st.markdown(f"""
    <div class="copy-button">
        <button onclick="navigator.clipboard.writeText(`{text.replace('`', '\\`')}`).then(() => {{
            const btn = document.getElementById('{button_id}');
            const originalText = btn.innerHTML;
            btn.innerHTML = '✅ Copied!';
            setTimeout(() => {{ btn.innerHTML = originalText; }}, 2000);
        }})" 
        id="{button_id}"
        style="
            background-color: #007bff;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        ">
            {button_text}
        </button>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.title("🔍 Enhanced SQL Query Generator")
    st.markdown("Upload your CSV data and ask questions in natural language with improved error handling!")
    
    # Sidebar for file upload and schema info
    with st.sidebar:
        st.header("📁 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file to analyze"
        )
        
        if uploaded_file is not None:
            if (st.session_state.uploaded_file_name != uploaded_file.name or 
                st.session_state.uploaded_file_path is None):
                
                df, schema_info, file_path = process_csv_upload(uploaded_file)
                
                if df is not None:
                    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
                    
                    # Show data preview
                    st.subheader("📊 Data Preview")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Show basic schema info
                    st.subheader("📋 Schema Summary")
                    st.markdown(schema_info)
                    
                    # Show detailed schema in expander
                    with st.expander("🔍 Detailed Schema Information"):
                        st.markdown(st.session_state.detailed_schema)
        
        elif st.session_state.uploaded_file_path:
            st.info("📄 Current file: " + st.session_state.uploaded_file_name)
            if st.session_state.schema_info:
                st.subheader("📋 Schema Summary")
                st.markdown(st.session_state.schema_info)
                
                with st.expander("🔍 Detailed Schema"):
                    st.markdown(st.session_state.detailed_schema)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Your Question")
        
        # Query input
        user_query = st.text_area(
            "Enter your question in natural language:",
            placeholder="e.g., Show me the top 5 customers by sales amount, or Calculate the average age of employees",
            height=100
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            max_attempts = st.slider("Maximum retry attempts", 1, 5, 3)
            show_attempts = st.checkbox("Show all query attempts", value=False)
        
        # Execute button
        col_btn1, col_btn2 = st.columns([1, 1])
        
        with col_btn1:
            execute_btn = st.button(
                "🚀 Generate & Execute SQL",
                type="primary",
                disabled=not (user_query and st.session_state.uploaded_file_path)
            )
        
        with col_btn2:
            if st.session_state.query_results:
                save_btn = st.button("💾 Save This Query", type="secondary")
                if save_btn and st.session_state.query_results:
                    last_history = st.session_state.prompt_history[0] if st.session_state.prompt_history else None
                    if last_history:
                        save_prompt(
                            last_history['prompt'],
                            last_history['result'],
                            last_history.get('sql_query')
                        )
        
        # Execute query with retry logic
        if execute_btn and user_query and st.session_state.uploaded_file_path:
            result, sql_query, rating = execute_sql_with_retry(
                user_query, 
                st.session_state.uploaded_file_path,
                max_attempts
            )
            
            # Add to history
            error = "Error" in result if result else None
            add_to_history(user_query, result, sql_query, rating, error)
            st.session_state.query_results = result
            
            # Display results
            st.subheader("📊 Query Results")
            
            # Show SQL query with copy button
            if sql_query and sql_query != "No SQL query generated":
                st.subheader(f"🔧 Generated SQL Query")
                if rating:
                    st.info(f"⭐ Query Rating: {rating:.2f}/1.0")
                
                # Display SQL in code block
                st.code(sql_query, language='sql')
                
                # Add copy button
                render_copy_button(sql_query, "📋 Copy SQL Query")
            
            # Show results
            st.subheader("📈 Execution Results")
            if "❌" in result or "Error" in result:
                st.error(result)
            else:
                st.success("Query executed successfully!")
                st.markdown(result)
    
    with col2:
        st.header("📚 Query History")
        
        # Recent prompts (latest 3)
        if st.session_state.prompt_history:
            st.subheader("🕐 Recent Queries")
            
            for i, item in enumerate(st.session_state.prompt_history):
                status_icon = "❌" if item.get('error') else "✅"
                with st.expander(f"{status_icon} Query {i+1} - {item['timestamp']}", expanded=(i==0)):
                    st.markdown(f"**Question:** {item['prompt']}")
                    
                    if item.get('sql_query') and item['sql_query'] != "No SQL query generated":
                        st.markdown("**SQL Query:**")
                        st.code(item['sql_query'], language='sql')
                        render_copy_button(item['sql_query'], "📋 Copy")
                        
                        if item.get('rating') is not None:
                            st.markdown(f"**Rating:** {item['rating']:.2f}/1.0")
                    
                    st.markdown("**Result:**")
                    result_preview = item['result'][:300] + "..." if len(item['result']) > 300 else item['result']
                    
                    if item.get('error') or "❌" in item['result']:
                        st.error(result_preview)
                    else:
                        st.success(result_preview)
        else:
            st.info("No recent queries yet. Execute a query to see history here.")
    
    # Saved prompts section
    st.header("💾 Saved Queries")
    
    if st.session_state.saved_prompts:
        # Create tabs for better organization
        tab1, tab2 = st.tabs(["📋 All Saved Queries", "🗑️ Manage Queries"])
        
        with tab1:
            for i, item in enumerate(reversed(st.session_state.saved_prompts)):
                with st.expander(f"Saved Query #{item['id']} - {item['timestamp']}"):
                    st.markdown(f"**File:** {item.get('file_name', 'Unknown')}")
                    st.markdown(f"**Question:** {item['prompt']}")
                    
                    if item.get('sql_query') and item['sql_query'] != "No SQL query generated":
                        st.markdown("**SQL Query:**")
                        st.code(item['sql_query'], language='sql')
                        render_copy_button(item['sql_query'], "📋 Copy Query")
                    
                    st.markdown("**Result:**")
                    if "Error" in item['result'] or "❌" in item['result']:
                        st.error(item['result'])
                    else:
                        st.text(item['result'])
        
        with tab2:
            st.subheader("🗑️ Delete Saved Queries")
            
            if st.button("Clear All Saved Queries", type="secondary"):
                if st.session_state.saved_prompts:
                    st.session_state.saved_prompts = []
                    save_prompts_to_file([])
                    st.success("All saved queries cleared!")
                    st.rerun()
            
            st.markdown(f"**Total saved queries:** {len(st.session_state.saved_prompts)}")
    else:
        st.info("No saved queries yet. Execute and save a query to see it here.")
    
    # Footer with tips
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🔍 Enhanced SQL Query Generator | 
            💡 <strong>Tips:</strong> Be specific in your questions | 
            🔄 Auto-retry on errors | 
            📋 Copy queries to clipboard</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Help section
    with st.expander("❓ Help & Tips"):
        st.markdown("""
        ### 🎯 **How to get better results:**
        
        1. **Be specific:** Instead of "show data", try "show top 10 customers by sales"
        2. **Use column names:** Check the schema and use exact column names in your questions
        3. **Ask for calculations:** "calculate average", "find maximum", "count records"
        4. **Filter data:** "show records where age > 25", "find customers from specific city"
        
        ### 🔧 **Features:**
        - **Auto-retry:** Failed queries are automatically regenerated with error context
        - **Copy queries:** Click the copy button to copy SQL queries to clipboard  
        - **Detailed schema:** Expanded schema shows column types and sample values
        - **Query history:** Recent queries are saved for reference
        
        ### 💡 **Example questions:**
        - "Show me the top 5 products by sales amount"
        - "Calculate the average age of customers by city"
        - "Find all orders placed in the last month"
        - "Count how many employees work in each department"
        """)

if __name__ == "__main__":
    main()