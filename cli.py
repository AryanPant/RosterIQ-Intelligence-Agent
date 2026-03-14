"""
Command-line interface for RosterIQ.

Examples:
  python -m rosterIQ "Why did CA drop?"
  python -m rosterIQ --use-graph "Show stuck ROs in MO"
  python -m rosterIQ --json "Give me a full operational report for TX"
  python rosterIQ/cli.py --monitor --monitor-market CA
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

for path in (PACKAGE_ROOT, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

DEFAULT_AGGREGATED = PACKAGE_ROOT / "data" / "aggregated_operational_metrics.csv"
DEFAULT_ROSTER = PACKAGE_ROOT / "data" / "roster_processing_details.csv"
DEFAULT_ARTIFACTS = PACKAGE_ROOT / "artifacts" / "charts"
DEFAULT_EPISODIC = PACKAGE_ROOT / "memory" / "episodic_memory_store.db"


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="RosterIQ – memory-driven provider roster intelligence agent"
    )
    parser.add_argument("query", nargs="?", default=None, help="Investigation question")
    parser.add_argument("--aggregated-path", default=str(DEFAULT_AGGREGATED))
    parser.add_argument("--roster-path", default=str(DEFAULT_ROSTER))
    parser.add_argument("--episodic-memory", default=str(DEFAULT_EPISODIC))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS))
    parser.add_argument(
        "--web-provider",
        default=None,
        help="Web search provider. Current live support is tavily; others fall back gracefully.",
    )
    parser.add_argument("--disable-web-search", action="store_true")
    parser.add_argument(
        "--use-extended",
        action="store_true",
        help="Compatibility flag. Direct mode already uses the extended agent pipeline.",
    )
    parser.add_argument(
        "--use-graph",
        action="store_true",
        help="Run via LangGraph StateGraph (requires the langgraph package).",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Start the MonitoringEngine daemon (blocking; run in a separate terminal).",
    )
    parser.add_argument("--monitor-market", default=None, help="Market to monitor (optional)")
    parser.add_argument("--monitor-interval", type=int, default=5, help="Polling interval in minutes")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of markdown/text")
    return parser


def configure_environment(args: argparse.Namespace) -> None:

    os.environ["ROSTERIQ_AGGREGATED_PATH"] = str(Path(args.aggregated_path).expanduser().resolve())
    os.environ["ROSTERIQ_ROSTER_PATH"] = str(Path(args.roster_path).expanduser().resolve())
    os.environ["ROSTERIQ_EPISODIC_MEMORY_PATH"] = str(Path(args.episodic_memory).expanduser().resolve())
    os.environ["ROSTERIQ_DISABLE_WEB_SEARCH"] = "true" if args.disable_web_search else "false"
    if args.web_provider:
        os.environ["ROSTERIQ_WEB_PROVIDER"] = args.web_provider


def initial_state(query: str) -> dict:

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


def run_direct_pipeline(query: str) -> dict:

    from agents.answer_agent import AnswerAgent
    from agents.pipeline_health_agent import PipelineHealthAgent
    from agents.planner_agent import PlannerAgent
    from agents.record_quality_agent import RecordQualityAgent
    from agents.supervisor_agent import SupervisorAgent
    from engines.procedure_runner import ProcedureRunner
    from engines.root_cause_engine import RootCauseEngine

    state = initial_state(query)
    supervisor = SupervisorAgent()
    planner = PlannerAgent()
    pipeline = PipelineHealthAgent()
    record = RecordQualityAgent()
    runner = ProcedureRunner()
    root_cause = RootCauseEngine()
    answer = AnswerAgent()

    state = supervisor.run(state)
    state["plan"] = planner.plan(state)
    state = pipeline.run(state)
    state = record.run(state)
    state = runner.run(state["plan"], state)
    state["root_cause"] = root_cause.trace(
        state.get("market"),
        scope=state.get("investigation_brief", {}).get("query_scope", {}),
    )
    return answer.run(state)


def run_graph_pipeline(query: str) -> dict:

    from graph.agent_graph import run_graph

    return run_graph(query=query)


def save_visualizations(result: dict, artifacts_dir: Path) -> dict[str, str]:

    chart_paths = {}
    figures = result.get("visualizations") or {}
    if not figures:
        return chart_paths

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for chart_name, figure in figures.items():
        output_path = artifacts_dir / f"{chart_name}.html"
        figure.write_html(str(output_path), include_plotlyjs="cdn")
        chart_paths[chart_name] = str(output_path)
    return chart_paths


def printable_payload(result: dict, chart_paths: dict[str, str]) -> dict:

    brief = result.get("investigation_brief", {})
    scope = brief.get("query_scope", {})
    return {
        "query": result.get("query"),
        "response": result.get("response"),
        "report": result.get("report"),
        "market": result.get("market"),
        "plan": result.get("plan", []),
        "evidence": result.get("evidence", []),
        "llm_status": result.get("llm_status"),
        "web_context": result.get("web_context", []),
        "chart_artifacts": chart_paths,
        "investigation_brief": {
            "intents": brief.get("intents", []),
            "topics": brief.get("topics", []),
            "chart_preferences": brief.get("chart_preferences", []),
            "tool_requests": brief.get("tool_requests", []),
            "desired_outputs": brief.get("desired_outputs", []),
            "query_scope": {
                "market": scope.get("market"),
                "org_name": scope.get("org_name"),
                "time_window": scope.get("time_window"),
                "is_full_operational_report": scope.get("is_full_operational_report"),
            },
        },
    }


def render_text_output(result: dict, chart_paths: dict[str, str]) -> str:

    chunks = []
    if result.get("report"):
        chunks.append(result["report"])
    elif result.get("response"):
        chunks.append(result["response"])
    else:
        chunks.append(str(result))

    if chart_paths:
        chunks.append("## Visualization Artifacts")
        for chart_name, chart_path in chart_paths.items():
            chunks.append(f"- {chart_name}: {chart_path}")

    if result.get("web_context"):
        chunks.append("## External Context")
        for item in result["web_context"]:
            detail = item.get("search_answer") or item.get("snippet", "")
            chunks.append(
                f"- {item.get('purpose', item.get('title'))}: {detail} ({item.get('url') or 'Tavily summary'})"
            )

    return "\n".join(chunks).strip()


def main() -> None:

    parser = build_parser()
    args = parser.parse_args()
    configure_environment(args)

    if args.monitor:
        from engines.monitoring_engine import MonitoringEngine

        engine = MonitoringEngine(interval_minutes=args.monitor_interval)
        print(
            f"Starting MonitoringEngine "
            f"(interval={args.monitor_interval} min, market={args.monitor_market or 'ALL'})"
        )
        engine.start(market=args.monitor_market)
        return

    if not args.query:
        parser.print_help()
        return

    result = run_graph_pipeline(args.query) if args.use_graph else run_direct_pipeline(args.query)
    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    chart_paths = save_visualizations(result, artifacts_dir)
    payload = printable_payload(result, chart_paths)

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return

    print(render_text_output(result, chart_paths))


if __name__ == "__main__":
    main()
