"""Streamlit dashboard for Continual Counsel.

Never imports from src.offline. Reads only SQLite and JSON files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add repository root to Python path so 'src' can be imported
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import requests
import streamlit as st

from src.audit.log import AuditLog
from src.audit.registry import AdapterRegistry
from src.eval.report import load_all_reports, reports_to_dataframe

# Configuration
st.set_page_config(page_title="Continual Counsel", layout="wide")

DB_PATH = "continual_counsel.db"
RUNS_DIR = "runs"
EXPORTS_DIR = "exports"

@st.cache_data
def get_reports():
    return load_all_reports(RUNS_DIR)

@st.cache_resource
def get_db():
    registry = AdapterRegistry(DB_PATH)
    audit_log = AuditLog(DB_PATH)
    return registry, audit_log

st.title("Continual Counsel — Compliance Dashboard")

registry, audit_log = get_db()
adapters = registry.list_adapters()

if not adapters:
    st.info("Run `verify_export_bundle.sh` to register your first adapter.")
    st.stop()

# Sidebar
st.sidebar.header("Filters")
adapter_options = [a.adapter_version_hash for a in adapters]
selected_adapter = st.sidebar.selectbox("Select Adapter Version", ["All"] + adapter_options)

# Tabs
tab_chat, tab_metrics, tab_queries, tab_lineage, tab_confusion = st.tabs([
    "Query Assistant", "Metrics", "Audit Queries", "Lineage", "Confusion Set"
])

with tab_chat:
    st.header("Query the Compliance Assistant")
    st.write("This communicates with the local FastAPI inference service (`http://localhost:8000/query`).")
    
    user_query = st.text_area("Enter your regulatory question:")
    if st.button("Submit Query"):
        if not user_query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Querying inference service..."):
                try:
                    resp = requests.post(
                        "http://localhost:8000/query",
                        json={"query": user_query},
                        timeout=120
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("status") == "escalate":
                            st.error(f"🚨 Escalation Triggered: {data.get('message')}")
                        else:
                            st.success("Answer Generated Successfully")
                            st.markdown(f"**Answer:**\n\n{data.get('answer')}")
                            with st.expander("View Query Metadata"):
                                st.json(data)
                    else:
                        st.error(f"API Error {resp.status_code}: {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error(
                        "Failed to connect to the inference API. Make sure the FastAPI service is running in another terminal:\n\n"
                        "`uvicorn src.online.api:app --host 0.0.0.0 --port 8000`"
                    )

with tab_metrics:
    st.header("Training & Evaluation Metrics")
    reports = get_reports()
    if not reports:
        st.write("No evaluation reports found in runs directory.")
    else:
        df = reports_to_dataframe(reports)
        if df is not None and not df.empty:
            st.subheader("Accuracy over Time")
            acc_cols = [c for c in df.columns if c.startswith("acc_")]
            if acc_cols:
                chart_data = df.set_index("timestamp")[acc_cols]
                st.line_chart(chart_data)
                
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("BWT & FWT")
                st.line_chart(df.set_index("timestamp")[["bwt", "fwt"]])
            with col2:
                st.subheader("Adapter Footprint (MB)")
                st.line_chart(df.set_index("timestamp")[["footprint_mb"]])
                
            st.subheader("Basis Rank (Sawtooth)")
            st.line_chart(df.set_index("timestamp")[["basis_rank"]])
            
            st.subheader("Hallucination Rate")
            st.line_chart(df.set_index("timestamp")[["hallucination_rate"]])


with tab_queries:
    st.header("Audit Log Queries")
    # Read from sqlite directly using pandas
    try:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        query_filter = ""
        if selected_adapter != "All":
            query_filter = f"WHERE adapter_version_hash = '{selected_adapter}'"
            
        audit_df = pd.read_sql_query(f"SELECT * FROM audit_log {query_filter} ORDER BY timestamp DESC LIMIT 100", conn)
        
        if audit_df.empty:
            st.write("No queries found.")
        else:
            st.dataframe(audit_df[['query_id', 'timestamp', 'adapter_version_hash', 'verification_verdict', 'regime_tags', 'query_text']])
            
            # Show details
            st.subheader("Query Details")
            selected_qid = st.selectbox("Select Query ID", audit_df['query_id'].tolist())
            if selected_qid:
                record = audit_log.get_record(selected_qid)
                if record:
                    st.json({
                        "query": record.query_text,
                        "answer": record.final_answer,
                        "chunks": json.loads(record.retrieved_chunk_ids),
                        "verification": record.verification_verdict,
                        "confidence": json.loads(record.confidence_scores)
                    })
    except Exception as e:
        st.error(f"Could not load audit log: {e}")

with tab_lineage:
    st.header("Adapter Lineage")
    if selected_adapter != "All":
        lineage = registry.get_lineage(selected_adapter)
        for idx, rec in enumerate(lineage):
            st.markdown(f"**Level {idx}: {rec.adapter_version_hash}**")
            st.write(f"- Regime: {rec.regime}")
            st.write(f"- Training data: {rec.training_data_hash}")
            st.write(f"- Export manifest: {rec.export_manifest_hash}")
            st.write(f"- Commit: {rec.repo_commit_hash}")
            st.write("---")
    else:
        st.write("Select a specific adapter in the sidebar to view lineage.")

with tab_confusion:
    st.header("Cross-Regime Confusion Performance")
    reports = get_reports()
    if reports:
        df = reports_to_dataframe(reports)
        if df is not None and not df.empty and "confusion_score" in df.columns:
            st.line_chart(df.set_index("timestamp")[["confusion_score"]])
    st.write("In-depth cross-regime adversarial test performance helps verify that models distinguish regimes with overlapping terminology.")
