from engines.procedure_runner import ProcedureRunner
from memory.episodic_memory import EpisodicMemory
from memory.procedural_memory import ProceduralMemory
from memory.semantic_memory import SemanticMemory
from tools.report_generator import ReportGenerator
from tools.visualization_tool import VisualizationTool
from tools.web_search_tool import WebSearchTool
from tools.data_query_tool import DataQueryTool
from utils.openrouter_client import OpenRouterClient


class AnswerAgent:

    def __init__(self):

        self.llm = OpenRouterClient()
        self.semantic_memory = SemanticMemory()
        self.procedures = ProceduralMemory()
        self.reports = ReportGenerator()
        self.memory = EpisodicMemory()
        self.visuals = VisualizationTool()
        self.web = WebSearchTool()
        self.data = DataQueryTool()

    def _build_visualizations(self, state):

        brief = state.get("investigation_brief", {})
        scope = brief.get("query_scope") or self.data.extract_visualization_scope(
            state.get("query"),
            requested_charts=brief.get("chart_preferences", []),
            market=state.get("market"),
        )
        chart_preferences = scope.get("requested_charts") or brief.get("chart_preferences", [])
        pipeline_df = state.get("pipeline_health", {}).get("pipeline_df")
        duration_anomalies = state.get("pipeline_health", {}).get("duration_anomalies")
        record_pipeline_df = state.get("record_quality", {}).get("pipeline_df")
        market_history_df = state.get("record_quality", {}).get("market_history_df")
        market = scope.get("market") or state.get("market")
        stages = scope.get("stages")
        ratio_columns = scope.get("ratio_columns")
        health_flags = scope.get("health_flags")
        max_items = scope.get("max_items")

        builders = {
            "pipeline_stage_health_heatmap": lambda: self.visuals.pipeline_stage_health_heatmap(
                pipeline_df,
                stages=stages,
                health_flags=health_flags,
                max_orgs=max_items or 25,
                market=market,
            ),
            "record_quality_breakdown": lambda: self.visuals.record_quality_breakdown(
                record_pipeline_df,
                ratio_columns=ratio_columns,
                max_files=max_items or 20,
                market=market,
            ),
            "duration_anomaly_chart": lambda: self.visuals.duration_anomaly_chart(
                duration_anomalies,
                stages=stages,
                max_rows=max_items or 25,
                market=market,
            ),
            "market_scs_percent_trend": lambda: self.visuals.market_scs_percent_trend(market_history_df, market=market),
            "retry_lift_chart": lambda: self.visuals.retry_lift_chart(self.data.market.copy(), market=market),
            "stuck_ro_tracker": lambda: self.visuals.stuck_ro_tracker(
                pipeline_df,
                stages=stages,
                health_flags=health_flags,
                max_rows=max_items or 30,
                market=market,
            ),
        }

        figures = {}
        for chart_name in chart_preferences:
            builder = builders.get(chart_name)
            if builder is None:
                continue
            figure = builder()
            if figure is not None:
                figures[chart_name] = figure

        return figures

    def _build_investigation_summary(self, state):

        metrics = state.get("record_quality", {}).get("market_metrics", {})
        root_cause = state.get("root_cause", {})
        parts = []

        if metrics:
            delta = metrics.get("success_rate_delta", 0)
            direction = "decreased" if delta < 0 else "increased"
            market_label = metrics["market"] if metrics["market"] != "ALL" else "All tracked states"
            parts.append(
                f"{market_label} {metrics['month']} success rate is {metrics['success_rate']:.2f}%, "
                f"which {direction} by {abs(delta):.2f} points versus the previous month."
            )

        if root_cause.get("top_failure_status"):
            parts.append(
                f"Leading failure status: {root_cause['top_failure_status']['status']} "
                f"({root_cause['top_failure_status']['count']})."
            )

        if root_cause.get("primary_stage"):
            parts.append(f"Primary pipeline stage concentration: {root_cause['primary_stage']}.")

        if root_cause.get("top_impacted_org"):
            parts.append(
                f"Most impacted organization: {root_cause['top_impacted_org']['org']} "
                f"({root_cause['top_impacted_org']['count']})."
            )

        return " ".join(parts).strip()

    def _build_trend_response(self, state):

        metrics = state.get("record_quality", {}).get("market_metrics", {})
        root_cause = state.get("root_cause", {})
        history = state.get("history", [])
        query = (state.get("query") or "").lower()
        brief = state.get("investigation_brief", {})
        market_history = state.get("record_quality", {}).get("market_history_df")
        aggregated_market_history = state.get("record_quality", {}).get("aggregated_market_history_df")

        if not metrics:
            return "I could not find enough market metrics to answer that question."

        if "historical_data" in brief.get("topics", []):
            if metrics.get("market") == "ALL" and aggregated_market_history is not None:
                market_history = aggregated_market_history
            if market_history is None or market_history.empty:
                return "I could not find enough historical market data to answer that question."

            lowest = market_history.loc[market_history["SCS_PERCENT"].idxmin()]
            latest = market_history.iloc[-1]
            market_label = latest["MARKET"] if latest["MARKET"] != "ALL" else "all tracked states"
            return (
                f"Here is the available history for {market_label} record quality across the tracked months. "
                f"The latest success rate is {latest['SCS_PERCENT']:.2f}% in {latest['MONTH']}, and the lowest point in the available history is "
                f"{lowest['SCS_PERCENT']:.2f}% in {lowest['MONTH']}."
            )

        market = metrics["market"]
        market_label = market if market != "ALL" else "Across all tracked states"
        month = metrics["month"]
        current_rate = metrics["success_rate"]
        previous_rate = metrics.get("previous_success_rate", current_rate)
        delta = metrics.get("success_rate_delta", 0)

        if "drop" in query and delta >= 0:
            response = (
                f"{market_label} did not drop in the latest month. It moved from "
                f"{previous_rate:.2f}% in the previous month to {current_rate:.2f}% in {month}, "
                f"an increase of {delta:.2f} points."
            )
        elif delta < 0:
            response = (
                f"{market_label} success rate declined to {current_rate:.2f}% in {month}, down "
                f"{abs(delta):.2f} points from {previous_rate:.2f}% in the previous month."
            )
        else:
            response = (
                f"{market_label} success rate is {current_rate:.2f}% in {month}, up "
                f"{delta:.2f} points from {previous_rate:.2f}% in the previous month."
            )

        if root_cause.get("top_failure_status"):
            response += (
                f" The leading failure status is {root_cause['top_failure_status']['status']} "
                f"({root_cause['top_failure_status']['count']} records)."
            )

        if root_cause.get("top_impacted_org"):
            response += f" The most impacted organization is {root_cause['top_impacted_org']['org']}."

        if history:
            response += " A similar investigation exists in episodic memory."

        return response

    def _build_fallback(self, state):

        root_cause = state.get("root_cause", {})
        metrics = state.get("record_quality", {}).get("market_metrics", {})
        history = state.get("history", [])
        brief = state.get("investigation_brief", {})

        if brief.get("is_memory_query"):
            if history:
                latest_entry = history[0]
                stored_summary = latest_entry.get("metadata", {}).get("investigation_summary") or latest_entry["response"]
                return (
                    f"Yes. We previously investigated a similar {latest_entry.get('metadata', {}).get('market') or state.get('market') or ''} issue "
                    f"on {latest_entry['timestamp']}. The stored conclusion was: {stored_summary}"
                ).strip()
            return "No similar prior investigation was found in episodic memory."

        return self._build_trend_response(state)

    def _build_memory_prompt(self, state):

        history = state.get("history", [])
        market = state.get("market") or "the requested market"

        if history:
            latest_entry = history[0]
            stored_summary = latest_entry.get("metadata", {}).get("investigation_summary") or latest_entry["response"]
            return f"""
You are a healthcare roster operations analyst.
User question: {state['query']}
Matched past investigation market: {latest_entry.get('metadata', {}).get('market') or market}
Matched investigation timestamp: {latest_entry['timestamp']}
Stored investigation summary: {stored_summary}

Answer in 2 short sentences:
1. Confirm whether a similar investigation exists.
2. Summarize the past conclusion clearly without repeating the same sentence structure.
""".strip()

        return f"""
You are a healthcare roster operations analyst.
User question: {state['query']}
No similar investigation was found in episodic memory for market {market}.

Answer in 1 short sentence.
""".strip()

    @staticmethod
    def _display_formula(function_text, evaluated_metric=None):

        if evaluated_metric and evaluated_metric.get("expression"):
            return evaluated_metric["expression"]
        if function_text and "=" in function_text:
            return function_text.split("=", 1)[1].strip()
        return function_text or "Not defined."

    def _format_procedure_update_response(
        self,
        update_result,
        before_formula,
        after_formula,
        logic_lines,
        computed_metric_line,
    ):

        procedure_name = update_result["procedure"]

        if not update_result.get("updated"):
            return (
                f"**Procedure Update**\n"
                f"`{procedure_name}` was not updated.\n\n"
                f"**Reason**\n"
                f"{update_result['confirmation']}"
            )

        status_line = "updated successfully" if update_result.get("before") else "stored successfully"
        response = (
            f"**Procedure Update**\n"
            f"`{procedure_name}` was {status_line}.\n\n"
            f"**Formula Change**\n"
            f"Previous: `{before_formula}`\n"
            f"Current: `{after_formula}`\n\n"
            f"**What Changed**\n"
            f"{update_result['confirmation']}"
        )

        if computed_metric_line:
            response += f"\n\n**Current Computed Result**\n{computed_metric_line}"

        if logic_lines:
            response += f"\n\n**Current Logic**\n{logic_lines}"

        return response

    def _build_procedure_execution_response(self, state):

        procedure_results = state.get("procedure_results", [])
        if not procedure_results:
            return "I could not run the requested procedure because no procedure result was produced."

        result = procedure_results[0]
        details = result.get("details", {}) if isinstance(result.get("details"), dict) else {}
        scope_labels = details.get("scope_labels") or state.get("query_scope", {}).get("labels", [])
        scope_text = ", ".join(scope_labels) if scope_labels else "all matching files"

        if result.get("procedure") == "record_quality_audit":
            evaluated_metric = details.get("evaluated_metric")
            file_count = int(details.get("file_count", 0))
            flagged_file_count = int(details.get("flagged_file_count", 0))
            threshold = float(details.get("audit_threshold_percent", 85))
            success_rate = float(details.get("success_rate_percent", 0))
            top_failure_status = details.get("top_failure_status")
            top_impacted_org = details.get("top_impacted_org")

            metric_line = "Computed metric unavailable."
            if evaluated_metric:
                display_formula = self._display_formula(
                    details.get("stored_function", ""),
                    evaluated_metric=evaluated_metric,
                )
                metric_line = (
                    f"The current record-quality issue rate is `{evaluated_metric['value'] * 100:.2f}%`, "
                    f"computed with `{display_formula}`."
                )

            response = (
                f"**Record Quality Audit**\n"
                f"Scope: {scope_text}\n\n"
                f"**Direct Answer**\n"
                f"I audited {file_count} files in scope. {metric_line}\n\n"
                f"**Audit Outcome**\n"
                f"- Average success rate: `{success_rate:.2f}%`\n"
                f"- Files below `{threshold:.0f}%` SCS_PCT: `{flagged_file_count}`\n"
            )

            if top_failure_status:
                response += (
                    f"- Most common failure status: `{top_failure_status['status']}` "
                    f"({top_failure_status['count']})\n"
                )
            if top_impacted_org:
                response += (
                    f"- Most impacted organization among flagged files: `{top_impacted_org['org']}` "
                    f"({top_impacted_org['count']} files)\n"
                )

            response += (
                "\n**Procedure Summary**\n"
                f"{result.get('summary', 'No summary available.')}"
            )
            return response

        if result.get("procedure") == "lob_rejection_breakdown":
            rows = details.get("rows", [])
            if not rows:
                return "I could not find any Line of Business rejection data for that scope."

            lines = [
                "**LOB Rejection Breakdown**",
                f"Scope: {scope_text}",
                "",
                result.get("summary", "No summary available."),
                "",
                "**Top LOBs**",
            ]
            for row in rows[:5]:
                lines.append(
                    f"- {row.get('LOB_ITEM', 'Unknown')}: `{float(row.get('rejection_rate', 0)) * 100:.2f}%` "
                    f"rejection rate over `{int(row.get('total_records', 0))}` records"
                )
            return "\n".join(lines)

        return (
            f"**Procedure Result**\n"
            f"Procedure: `{result.get('procedure', 'unknown')}`\n\n"
            f"{result.get('summary', 'No summary available.')}"
        )

    def _handle_procedure_update(self, state):

        brief = state.get("investigation_brief", {})
        target = brief.get("procedure_target")

        if target and self.procedures.get(target):
            update_result = self.procedures.improve(target, state["query"])
        else:
            update_result = self.procedures.upsert_from_query(state["query"])

        before = update_result.get("before", {})
        after = update_result.get("after", {})
        before_function = before.get("function", "Not previously defined.")
        after_function = after.get("function", "Not defined.")
        logic_lines = "\n".join(f"- {item}" for item in after.get("logic", []))
        execution_result = None

        if update_result.get("updated"):
            runner = ProcedureRunner()
            execution_result = runner.execute_defined_procedure(
                update_result["procedure"],
                market=state.get("market"),
                scope=brief.get("query_scope", {}),
            )

        computed_metric_line = ""
        if execution_result:
            details = execution_result.get("details", {})
            evaluated_metric = details.get("evaluated_metric") if isinstance(details, dict) else None
            if evaluated_metric:
                display_formula = self._display_formula(
                    details.get("stored_function", after_function),
                    evaluated_metric=evaluated_metric,
                )
                computed_metric_line = (
                    f"The current record-quality issue rate is "
                    f"`{evaluated_metric['value'] * 100:.2f}%`, computed with `{display_formula}`."
                )
            elif execution_result.get("summary"):
                computed_metric_line = execution_result["summary"]

        before_formula = self._display_formula(before_function)
        after_formula = self._display_formula(after_function)
        response = self._format_procedure_update_response(
            update_result=update_result,
            before_formula=before_formula,
            after_formula=after_formula,
            logic_lines=logic_lines,
            computed_metric_line=computed_metric_line,
        )

        if self.procedures.llm.last_status == "success":
            state["llm_status"] = f"openrouter ({self.procedures.llm.model})"
        else:
            error_detail = self.procedures.llm.last_error or "Unknown failure"
            state["llm_status"] = f"fallback ({error_detail})"

        state["procedure_update_result"] = update_result
        if execution_result:
            state["procedure_results"] = [execution_result]
        state["response"] = response
        state["report"] = ""
        state["visualizations"] = {}
        state["web_context"] = []
        state["investigation_summary"] = ""
        if update_result.get("updated"):
            state["evidence"].append(f"Stored procedural-memory update for {update_result['procedure']}.")
            if execution_result and execution_result.get("summary"):
                state["evidence"].append(execution_result["summary"])
        else:
            state["evidence"].append(f"Procedural-memory update for {update_result['procedure']} was not applied.")
        return state

    @staticmethod
    def _format_web_context(items, limit=6):

        lines = []
        for item in (items or [])[:limit]:
            lines.append(
                f"- [{item.get('category', 'external_context')}] {item.get('title', 'Untitled')} :: "
                f"{item.get('snippet', '')} ({item.get('url', '')})"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_query_keywords(scope):

        if not scope:
            return "None"
        lines = []
        if scope.get("market"):
            lines.append(f"- market: {scope['market']}")
        if scope.get("org_name"):
            lines.append(f"- organization: {scope['org_name']}")
        if scope.get("lob_terms"):
            lines.append(f"- lob: {', '.join(scope['lob_terms'])}")
        if scope.get("regulatory_terms"):
            lines.append(f"- regulatory_terms: {', '.join(scope['regulatory_terms'])}")
        if scope.get("time_window"):
            lines.append(f"- time_window: {scope['time_window'].get('label')}")
        if scope.get("query_keywords"):
            lines.append(f"- combined_keywords: {', '.join(scope['query_keywords'])}")
        return "\n".join(lines) or "None"

    def _augment_response_with_web_context(self, response, web_context):

        if not web_context:
            return response

        highlights = []
        for item in web_context[:3]:
            highlights.append(
                f"{item.get('category', 'external_context')}: {item.get('title', 'Untitled')} "
                f"({item.get('url', '')})"
            )

        external_block = " External context highlights: " + "; ".join(highlights) + "."
        if external_block.strip() in (response or ""):
            return response
        return (response or "").rstrip() + external_block

    def run(self, state):

        brief = state.get("investigation_brief", {})
        if brief.get("is_procedure_update"):
            return self._handle_procedure_update(state)
        if brief.get("is_procedure_execution"):
            state["llm_status"] = "procedure execution (rule-based)"
            tools = set(brief.get("tool_requests", []))
            desired_outputs = set(brief.get("desired_outputs", []))
            if "visualization" in tools and "visualization" in desired_outputs:
                state["visualizations"] = self._build_visualizations(state)
            else:
                state["visualizations"] = {}
            state["web_context"] = []
            state["report"] = ""
            state["response"] = self._build_procedure_execution_response(state)
            state["investigation_summary"] = ""
            return state

        is_memory_query = brief.get("is_memory_query")
        tools = set(brief.get("tool_requests", []))
        desired_outputs = set(brief.get("desired_outputs", []))
        scope = brief.get("query_scope", {})
        web_context = []

        if not is_memory_query and "web_search" in tools and "external_context" in desired_outputs:
            web_context = self.web.search_external_context(state, max_results_per_query=2)
        state["web_context"] = web_context

        evidence = "\n".join(f"- {item}" for item in state["evidence"])
        root_cause = state.get("root_cause", {})
        episodic_context = self.memory.format_for_prompt(state.get("history", []), limit=3)
        semantic_context = self.semantic_memory.semantic_recall(state["query"], alpha=0.5, limit=6)
        procedure_context = "\n".join(
            f"{name}: {self.procedures.get(name).get('description', '')}"
            for name in state.get("plan", [])
        )
        semantic_hints = {
            key: self.semantic_memory.explain(key)
            for key in ["FAIL_REC_CNT", "REJ_REC_CNT", "SCS_PERCENT"]
        }

        system_prompt = f"""
You are a healthcare roster operations analyst.
Use the following memory context when answering:

EPISODIC MEMORY:
{episodic_context or 'None'}

SEMANTIC MEMORY:
{semantic_context or 'None'}

PROCEDURAL MEMORY:
{procedure_context or 'None'}

QUERY KEYWORDS:
{self._format_query_keywords(scope)}

EXTERNAL CONTEXT:
{self._format_web_context(web_context) or 'None'}

If external context is present and relevant, explicitly incorporate it into the answer.
""".strip()

        prompt = f"""
You are a healthcare roster operations analyst.
User question: {state['query']}
Market: {state.get('market') or 'not specified'}
Evidence:
{evidence}

Root cause summary:
{root_cause}

Metric glossary:
{semantic_hints}

External context:
{self._format_web_context(web_context) or 'None'}

Write a concise explanation with:
1. The direct answer
2. Key evidence
3. Recommended next action
""".strip()

        if is_memory_query:
            response = self.llm.generate(self._build_memory_prompt(state), system_prompt=system_prompt)
        else:
            response = self.llm.generate(prompt, system_prompt=system_prompt)
        if response:
            state["llm_status"] = f"openrouter ({self.llm.model})"
        else:
            error_detail = self.llm.last_error or "Unknown failure"
            state["llm_status"] = f"fallback ({error_detail})"
        if not response:
            response = self._build_fallback(state)
        response = self._augment_response_with_web_context(response, web_context)

        if is_memory_query:
            state["visualizations"] = {}
            report = ""
        else:
            if "visualization" in tools and "visualization" in desired_outputs:
                state["visualizations"] = self._build_visualizations(state)
            else:
                state["visualizations"] = {}

            if "report_generator" in tools and "report" in desired_outputs:
                report = self.reports.generate(summary=response, state=state)
            else:
                report = ""

        state["response"] = response
        state["report"] = report
        investigation_summary = self._build_investigation_summary(state)
        state["investigation_summary"] = investigation_summary
        if not is_memory_query and investigation_summary:
            self.memory.store(
                state["query"],
                investigation_summary,
                metadata={
                    "market": state.get("market"),
                    "plan": state.get("plan", []),
                    "intents": brief.get("intents", []),
                    "topics": brief.get("topics", []),
                    "root_cause": root_cause,
                    "investigation_summary": investigation_summary,
                    "memory_kind": "investigation_summary",
                },
            )
        return state
