import streamlit as st
import os
import tempfile
from datetime import datetime
from utils.schema_tools import load_tables, pretty_schema
from query_generator import run_agent

st.set_page_config(page_title="Multi-Table SQL Assistant", page_icon="🗄️", layout="wide")

# ───── Sidebar: upload + settings ─────
st.sidebar.header("📁  Load CSV tables")
uploads = st.sidebar.file_uploader("CSV files", type="csv", accept_multiple_files=True)
threshold = st.sidebar.slider("Quality threshold", 0.1, 1.0, 0.8, 0.05)

if uploads:
    tmp_dir = tempfile.mkdtemp()
    table_paths = []
    for up in uploads:
        name = os.path.splitext(up.name)[0]
        path = os.path.join(tmp_dir, up.name)
        with open(path, "wb") as f:
            f.write(up.getbuffer())
        table_paths.append((name, path))

    tables = load_tables(table_paths)
    st.sidebar.success(f"Loaded {len(tables)} table(s)")
    st.sidebar.code(pretty_schema(tables))

    # ───── Main: query box ─────
    st.title("🗄️  Natural-Language SQL over Multiple Tables")
    q = st.text_area("Ask a question about the data", height=110, placeholder="e.g. List the top 5 students by marks")
    if st.button("Run Query 🚀", type="primary") and q.strip():
        with st.spinner("Thinking…"):
            state = run_agent(q, tables, threshold)
        if state["error"]:
            st.error(state["error"])
        else:
            st.subheader("Generated SQL")
            st.code(state["best_query"]["sql"], language="sql")
            st.subheader("Result")
            st.markdown(state["result"] or "_(no rows returned)_")
            st.caption(f"Rating {state['best_query']['rating']:.2f} after {len(state['query_attempts'])} attempt(s)")
else:
    st.info("Upload at least one CSV file to begin.")
