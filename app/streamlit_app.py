import os
import sys
import time

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph.agent_graph import build_graph

PAGE_TITLE = "RosterIQ Intelligence Agent"
CHAT_PROMPT = "Ask about market health, stuck rosters, or failure trends"
STARTER_PROMPTS = [
    "Why CA success rate is dropping?",
    "Show the biggest failure trends in TX.",
    "Give me a full operational report for FL.",
]
STARTUP_STEPS = [
    ("Preparing workspace", "Wiring up the page shell and session state.", 20),
    ("Loading agent graph", "Building the investigation graph and its dependencies.", 68),
    ("Restoring session", "Recovering your chat history for this browser session.", 88),
    ("Finalizing", "Making the workspace ready for investigation.", 100),
]


def default_query_state(query):

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


def build_chart_key(message_key, chart_name, position):

    safe_name = str(chart_name).replace(" ", "_").lower()
    return f"{message_key}_plotly_{position}_{safe_name}"


def render_assistant_details(message, message_key="assistant"):

    if message.get("runtime_seconds") is not None:
        st.caption(f"Completed in {message['runtime_seconds']:.1f}s")

    if message.get("llm_status"):
        st.caption(f"LLM status: {message['llm_status']}")

    visualizations = message.get("visualizations") or {}
    if visualizations:
        chart_items = list(visualizations.items())
        for start in range(0, len(chart_items), 2):
            col1, col2 = st.columns(2)
            left = chart_items[start]
            with col1:
                st.plotly_chart(
                    left[1],
                    use_container_width=True,
                    key=build_chart_key(message_key, left[0], f"{start}_left"),
                )
            if start + 1 < len(chart_items):
                right = chart_items[start + 1]
                with col2:
                    st.plotly_chart(
                        right[1],
                        use_container_width=True,
                        key=build_chart_key(message_key, right[0], f"{start + 1}_right"),
                    )

    if message.get("web_context"):
        with st.expander("External context"):
            for item in message["web_context"]:
                st.markdown(f"**{item.get('purpose', item.get('title', 'External context'))}**")
                st.markdown(item.get("search_answer") or item.get("snippet", ""))
                if item.get("url"):
                    st.markdown(item["url"])

    if message.get("report"):
        with st.expander("Structured report"):
            st.markdown(message["report"])


def render_message(message, message_key=None):

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] != "assistant":
            return

        render_assistant_details(message, message_key=message_key or "assistant")


def render_sidebar():

    queued_query = None
    with st.sidebar:
        st.subheader("Session")
        st.caption("Use a starter prompt, then let the agent investigate the pipeline for you.")

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []

        st.markdown("**Starter prompts**")
        for index, prompt in enumerate(STARTER_PROMPTS):
            if st.button(prompt, key=f"starter_prompt_{index}", use_container_width=True):
                queued_query = prompt

        st.markdown("---")
        st.caption(f"{len(st.session_state.messages)} messages in this session")

    return queued_query


def invoke_graph(query):

    return st.session_state.graph.invoke(default_query_state(query))


def build_assistant_message(result, runtime_seconds):

    return {
        "role": "assistant",
        "content": result.get("response") or result.get("report") or "No response generated.",
        "llm_status": result.get("llm_status"),
        "visualizations": result.get("visualizations") or {},
        "web_context": result.get("web_context") or [],
        "report": result.get("report") or "",
        "runtime_seconds": runtime_seconds,
    }


def render_loading_status():

    if hasattr(st, "status"):
        status = st.status("Investigating your request...", expanded=True)
        status.write("Reviewing prior memory and market context.")
        status.write("Running diagnostics and collecting evidence.")
        status.write("Preparing the answer, report, and any charts.")
        return status

    return None


def advance_progress(progress_bar, current_value, target_value, label):

    if target_value <= current_value:
        return current_value

    for value in range(current_value + 1, target_value + 1):
        progress_bar.progress(value, text=label)
        time.sleep(0.01)

    return target_value


def update_startup_status(status_box, status_placeholder, label, description, state="running"):

    if status_box is not None:
        if state == "running":
            status_box.write(f"**{label}**")
            status_box.write(description)
        else:
            status_box.update(label=label, state=state, expanded=state != "complete")
        return

    if status_placeholder is None:
        return

    message = f"**{label}**\n\n{description}" if description else label
    if state == "complete":
        status_placeholder.success(message)
    elif state == "error":
        status_placeholder.error(message)
    else:
        status_placeholder.info(message)


def initialize_app():

    needs_startup = (
        not st.session_state.get("startup_complete", False)
        or "graph" not in st.session_state
        or "messages" not in st.session_state
    )
    if not needs_startup:
        return

    loading_shell = st.empty()
    status_box = None
    status_placeholder = None
    progress_value = 0

    with loading_shell.container():
        st.info("Preparing the RosterIQ workspace. The chat will appear as soon as startup completes.")
        if hasattr(st, "status"):
            status_box = st.status("Starting RosterIQ...", expanded=True)
        else:
            status_placeholder = st.empty()
        progress_bar = st.progress(0, text="Starting RosterIQ...")

        try:
            update_startup_status(
                status_box,
                status_placeholder,
                "Step 1 of 4: Preparing workspace",
                STARTUP_STEPS[0][1],
            )
            progress_value = advance_progress(
                progress_bar,
                progress_value,
                STARTUP_STEPS[0][2],
                "Step 1 of 4: Preparing workspace",
            )

            if "messages" not in st.session_state:
                st.session_state.messages = []

            update_startup_status(
                status_box,
                status_placeholder,
                "Step 2 of 4: Loading agent graph",
                STARTUP_STEPS[1][1],
            )
            progress_value = advance_progress(
                progress_bar,
                progress_value,
                STARTUP_STEPS[1][2] - 12,
                "Step 2 of 4: Loading agent graph",
            )
            if "graph" not in st.session_state:
                st.session_state.graph = build_graph()
            progress_value = advance_progress(
                progress_bar,
                progress_value,
                STARTUP_STEPS[1][2],
                "Step 2 of 4: Loading agent graph",
            )

            update_startup_status(
                status_box,
                status_placeholder,
                "Step 3 of 4: Restoring session",
                STARTUP_STEPS[2][1],
            )
            progress_value = advance_progress(
                progress_bar,
                progress_value,
                STARTUP_STEPS[2][2],
                "Step 3 of 4: Restoring session",
            )

            update_startup_status(
                status_box,
                status_placeholder,
                "Step 4 of 4: Finalizing",
                STARTUP_STEPS[3][1],
            )
            progress_value = advance_progress(
                progress_bar,
                progress_value,
                STARTUP_STEPS[3][2],
                "Step 4 of 4: Finalizing",
            )
        except Exception:
            update_startup_status(
                status_box,
                status_placeholder,
                "Startup failed",
                "RosterIQ could not finish initialization.",
                state="error",
            )
            raise

        update_startup_status(
            status_box,
            status_placeholder,
            "RosterIQ is ready",
            "Initialization complete. Launching the chat workspace.",
            state="complete",
        )
        time.sleep(0.2)

    loading_shell.empty()
    st.session_state.startup_complete = True


st.set_page_config(page_title="RosterIQ Intelligence Agent", layout="wide")
st.title(PAGE_TITLE)
st.caption("Memory-driven provider roster diagnostics for pipeline health and root-cause analysis.")

initialize_app()

starter_query = render_sidebar()

for index, message in enumerate(st.session_state.messages):
    render_message(message, message_key=f"message_{index}")

if not st.session_state.messages:
    st.info("Start with a question in the chat box or use one of the starter prompts in the sidebar.")

typed_query = st.chat_input(CHAT_PROMPT)
query = starter_query or typed_query

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    render_message({"role": "user", "content": query})

    with st.chat_message("assistant"):
        start_time = time.perf_counter()
        status = render_loading_status()

        try:
            if status is None:
                with st.spinner("Investigating roster memory, pipeline evidence, and report output..."):
                    result = invoke_graph(query)
            else:
                result = invoke_graph(query)
        except Exception as exc:
            if status is not None:
                status.update(label="Analysis failed", state="error", expanded=True)

            assistant_message = {
                "role": "assistant",
                "content": (
                    "I hit an error while generating the response.\n\n"
                    f"`{type(exc).__name__}: {exc}`"
                ),
                "llm_status": "error",
                "visualizations": {},
                "web_context": [],
                "report": "",
                "runtime_seconds": time.perf_counter() - start_time,
            }
        else:
            runtime_seconds = time.perf_counter() - start_time
            if status is not None:
                status.update(
                    label=f"Analysis complete in {runtime_seconds:.1f}s",
                    state="complete",
                    expanded=False,
                )

            assistant_message = build_assistant_message(result, runtime_seconds)

        st.markdown(assistant_message["content"])
        render_assistant_details(assistant_message, message_key="live_response")

    st.session_state.messages.append(assistant_message)
