import os
import sys
import time

import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph.agent_graph import build_graph
from memory.semantic_memory import SemanticMemory
from tools.data_query_tool import DataQueryTool

st.set_page_config(page_title="RosterIQ Intelligence Agent", layout="wide")
st.title("RosterIQ Intelligence Agent")
st.caption("Memory-driven provider roster diagnostics for pipeline health and root-cause analysis.")


def _advance_progress(progress_bar, start, end, label, step_delay=0.04):

    for value in range(start, end + 1, 5):
        progress_bar.progress(min(value, 100), text=label)
        time.sleep(step_delay)


@st.cache_resource
def _get_data_tool():

    return DataQueryTool()


@st.cache_resource
def _get_semantic_memory():

    return SemanticMemory()


def _render_sidebar():

    st.sidebar.header("Demo Prompts")
    prompt_options = [
        "Show stuck ROs in CA with a chart",
        "Run market health report for TX",
        "Have we investigated CA rejection issues before?",
        "Update record_quality_audit to include SKIP_REC_CNT",
        "Give me a full operational report for KS",
    ]
    for prompt in prompt_options:
        if st.sidebar.button(prompt, use_container_width=True):
            st.session_state.pending_query = prompt

    st.sidebar.divider()
    st.sidebar.header("System Status")
    llm_ready = "Yes" if os.getenv("OPENROUTER_API_KEY") else "No"
    web_ready = "Yes" if os.getenv("TAVILY_API_KEY") else "Fallback only"
    st.sidebar.caption(f"LLM enabled: {llm_ready}")
    st.sidebar.caption(f"Web search live: {web_ready}")
    st.sidebar.caption("Memory: episodic + procedural + semantic")


def _render_kpis():

    data = _get_data_tool()
    pipeline = data.pipeline.copy()
    market = data.market.copy()

    stuck_count = int(pd.to_numeric(pipeline.get("IS_STUCK", 0), errors="coerce").fillna(0).astype(int).sum())
    failed_count = int(pd.to_numeric(pipeline.get("IS_FAILED", 0), errors="coerce").fillna(0).astype(int).sum())

    market["SCS_PERCENT"] = pd.to_numeric(market["SCS_PERCENT"], errors="coerce")
    latest_market = market.sort_values("SCS_PERCENT", ascending=True).iloc[0] if not market.empty else None
    latest_month_row = market.iloc[-1] if not market.empty else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stuck ROs", stuck_count)
    col2.metric("Failed ROs", failed_count)
    col3.metric(
        "Latest Success Rate",
        f"{float(latest_month_row['SCS_PERCENT']):.2f}%" if latest_month_row is not None else "N/A",
        latest_month_row["MARKET"] if latest_month_row is not None else None,
    )
    col4.metric(
        "Worst Market Snapshot",
        f"{float(latest_market['SCS_PERCENT']):.2f}%" if latest_market is not None else "N/A",
        latest_market["MARKET"] if latest_market is not None else None,
    )


def _render_memory_panel(result):

    if not result:
        st.info("Memory and reasoning details will appear here after the first investigation.")
        return

    brief = result.get("investigation_brief", {})
    history = result.get("history", []) or []
    procedure_names = result.get("plan", []) or []
    semantic_memory = _get_semantic_memory()
    semantic_hits = semantic_memory.query_hybrid(result.get("query", ""), alpha=0.5, limit=3)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Episodic Memory**")
        if history:
            latest = history[0]
            st.caption(f"Matched {len(history)} past investigation(s)")
            st.write(f"{latest.get('timestamp', 'Unknown time')}")
            st.write(latest.get("metadata", {}).get("investigation_summary") or latest.get("response", ""))
        else:
            st.caption("No prior investigation recalled")

    with col2:
        st.markdown("**Procedural Memory**")
        if procedure_names:
            for item in procedure_names[:3]:
                st.write(f"- `{item}`")
        else:
            st.caption("No named procedure executed")
        if brief.get("is_procedure_update"):
            st.caption("Procedure update applied in this turn")

    with col3:
        st.markdown("**Semantic Memory**")
        if semantic_hits:
            for item in semantic_hits[:3]:
                if isinstance(item, dict):
                    st.write(f"- `{item.get('name', 'unknown')}`")
                else:
                    st.write(f"- `{str(item)}`")
        else:
            st.caption("No semantic concept recall available")


def _initial_state(query):

    return {
        "query": query,
        "market": None,
        "history": [],
        "plan": [],
        "evidence": [],
        "procedure_results": [],
        "investigation_brief": {},
        "query_embedding": None,
        "llm_status": "",
        "visualizations": {},
        "web_context": [],
        "report": "",
    }


def _initialize_graph_with_progress():

    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    status = status_placeholder.status("Initializing RosterIQ...", expanded=True)
    progress = progress_placeholder.progress(0, text="Booting the interface")

    _advance_progress(progress, 0, 20, "Loading agent workflow")
    status.write("Building the multi-agent investigation graph.")

    _advance_progress(progress, 25, 55, "Preparing memory and analytics components")
    status.write("Initializing memory stores, tools, and routing logic.")
    graph = build_graph()
    st.session_state.graph = graph

    _advance_progress(progress, 60, 85, "Warming up the chat workspace")
    status.write("Memory stores, procedures, and analysis tools are ready.")

    _advance_progress(progress, 90, 100, "RosterIQ is ready", step_delay=0.03)
    status.update(label="RosterIQ ready", state="complete", expanded=False)
    progress_placeholder.empty()
    status_placeholder.empty()

if "graph" not in st.session_state:
    _initialize_graph_with_progress()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

_render_sidebar()
_render_kpis()
st.divider()
st.subheader("Memory View")
_render_memory_panel(st.session_state.last_result)
st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask about market health, stuck rosters, or failure trends")
if not query and st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    result = st.session_state.graph.invoke(_initial_state(query))
    st.session_state.last_result = result

    with st.chat_message("assistant"):
        if result.get("llm_status"):
            st.caption(f"LLM status: {result['llm_status']}")
        st.markdown(result["response"])
        if result.get("visualizations"):
            chart_items = list(result["visualizations"].items())
            for start in range(0, len(chart_items), 2):
                col1, col2 = st.columns(2)
                left = chart_items[start]
                with col1:
                    st.plotly_chart(left[1], use_container_width=True)
                if start + 1 < len(chart_items):
                    right = chart_items[start + 1]
                    with col2:
                        st.plotly_chart(right[1], use_container_width=True)

        if result.get("web_context"):
            with st.expander("External context"):
                for item in result["web_context"]:
                    st.markdown(f"**{item['title']}**")
                    st.markdown(item["snippet"])
                    st.markdown(item["url"])

        if result.get("report"):
            with st.expander("Structured report"):
                st.markdown(result["report"])

        with st.expander("Memory usage"):
            _render_memory_panel(result)

        with st.expander("Investigation details"):
            st.markdown(f"**Detected market:** {result.get('market') or 'Not inferred'}")
            brief = result.get("investigation_brief", {})
            if brief:
                st.markdown(f"**Detected intents:** {', '.join(brief.get('intents', [])) or 'None'}")
                st.markdown(f"**Requested chart types:** {', '.join(brief.get('chart_preferences', [])) or 'None'}")
            st.markdown(f"**LLM status:** {result.get('llm_status') or 'Unknown'}")
            st.markdown(f"**Plan:** {', '.join(result.get('plan', [])) or 'None'}")
            st.markdown("**Evidence**")
            for item in result.get("evidence", []):
                st.markdown(f"- {item}")
            if result.get("history"):
                st.markdown("**Related past investigations**")
                for entry in result["history"]:
                    st.markdown(f"- {entry['timestamp']}: {entry['response']}")

    st.session_state.messages.append({"role": "assistant", "content": result["response"]})
