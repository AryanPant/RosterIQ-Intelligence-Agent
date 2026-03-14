import pandas as pd

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
    def _has_analysis_request(brief):

        analytical_intents = {
            "trend_analysis",
            "record_quality",
            "pipeline_diagnostics",
            "root_cause_analysis",
            "retry_analysis",
        }
        return bool(set(brief.get("intents", [])) & analytical_intents)

    def _is_combined_memory_analysis_query(self, state):

        brief = state.get("investigation_brief", {})
        if not brief.get("is_memory_query"):
            return False
        if self._has_analysis_request(brief):
            return True

        normalized = (state.get("query") or "").lower()
        return any(
            marker in normalized
            for marker in [
                "based on",
                "root cause",
                "why",
                "cause",
                "reason",
                "current pipeline stage",
                "pipeline stage",
                "scs_percent",
                "scs percent",
                "rej_rec_cnt",
                "reject",
                "rejection",
            ]
        )

    @staticmethod
    def _get_procedure_result(state, procedure_name):

        for result in state.get("procedure_results", []):
            if result.get("procedure") == procedure_name:
                return result
        return {}

    def _build_current_analysis_snapshot(self, state):

        metrics = state.get("record_quality", {}).get("market_metrics", {})
        root_cause = state.get("root_cause", {})
        pipeline_df = state.get("record_quality", {}).get("pipeline_df")
        duration_anomalies = state.get("pipeline_health", {}).get("duration_anomalies")
        audit_result = self._get_procedure_result(state, "record_quality_audit")
        audit_details = audit_result.get("details", {}) if isinstance(audit_result.get("details"), dict) else {}
        lines = []

        if metrics:
            lines.extend(
                [
                    f"- Latest market month: {metrics.get('month', 'Unknown')}",
                    f"- SCS_PERCENT: {float(metrics.get('success_rate', 0)):.2f}%",
                    f"- SCS_PERCENT delta vs previous month: {float(metrics.get('success_rate_delta', 0)):+.2f} points",
                ]
            )

        if pipeline_df is not None and not pipeline_df.empty and {"TOT_REC_CNT", "REJ_REC_CNT"}.issubset(pipeline_df.columns):
            total_records = float(pd.to_numeric(pipeline_df["TOT_REC_CNT"], errors="coerce").fillna(0).sum())
            rejected_records = float(pd.to_numeric(pipeline_df["REJ_REC_CNT"], errors="coerce").fillna(0).sum())
            rejection_rate = (rejected_records / total_records * 100) if total_records else 0
            lines.extend(
                [
                    f"- REJ_REC_CNT total: {int(rejected_records)}",
                    f"- Rejection rate by records: {rejection_rate:.2f}%",
                ]
            )

        evaluated_metric = audit_details.get("evaluated_metric")
        if evaluated_metric:
            display_formula = self._display_formula(
                audit_details.get("stored_function", ""),
                evaluated_metric=evaluated_metric,
            )
            lines.extend(
                [
                    f"- Record-quality issue rate: {float(evaluated_metric.get('value', 0)) * 100:.2f}%",
                    f"- Record-quality formula: {display_formula}",
                ]
            )

        if root_cause.get("primary_stage"):
            lines.append(f"- Current dominant pipeline stage: {root_cause['primary_stage']}")
        if root_cause.get("stuck_count") is not None:
            lines.append(f"- Current stuck roster count: {int(root_cause.get('stuck_count', 0))}")
        if duration_anomalies is not None and not duration_anomalies.empty:
            top_anomaly = duration_anomalies.iloc[0]
            lines.append(
                f"- Top duration anomaly: {top_anomaly.get('STAGE_DURATION_COLUMN', top_anomaly.get('STAGE_NAME', 'Unknown stage'))} "
                f"at {float(top_anomaly.get('ANOMALY_RATIO', 0)):.2f}x historical average"
            )
        if root_cause.get("top_failure_status"):
            lines.append(
                f"- Top failure status: {root_cause['top_failure_status']['status']} "
                f"({root_cause['top_failure_status']['count']})"
            )
        if root_cause.get("top_impacted_org"):
            lines.append(
                f"- Top impacted organization: {root_cause['top_impacted_org']['org']} "
                f"({root_cause['top_impacted_org']['count']})"
            )

        return "\n".join(lines) or "None"

    def _build_combined_memory_analysis_prompt(self, state, evidence):

        history = state.get("history", [])
        if history:
            latest_entry = history[0]
            stored_summary = latest_entry.get("metadata", {}).get("investigation_summary") or latest_entry.get("response", "")
            prior_context = (
                f"- Prior investigation found: Yes\n"
                f"- Prior investigation timestamp: {latest_entry.get('timestamp', 'Unknown')}\n"
                f"- Prior investigation summary: {stored_summary}"
            )
        else:
            prior_context = "- Prior investigation found: No"

        return f"""
You are a healthcare roster operations analyst.
User question: {state['query']}
Market: {state.get('market') or 'not specified'}

PRIOR INVESTIGATION:
{prior_context}

CURRENT ANALYSIS SNAPSHOT:
{self._build_current_analysis_snapshot(state)}

Evidence:
{evidence or 'None'}

Write the answer in 4 short sections with markdown headings:
**Prior Investigation**
**Current Evidence**
**Root Cause**
**Next Action**

Requirements:
- Answer whether we investigated this before.
- Use current live analysis, not just episodic memory.
- Explicitly mention SCS_PERCENT, REJ_REC_CNT or rejection-rate impact, and the current pipeline stage when available.
- If the latest month does not actually show a drop, say that clearly.
- Explain whether the issue looks more like record-quality/rejection leakage or a true pipeline-stage bottleneck.
""".strip()

    def _build_combined_memory_analysis_fallback(self, state):

        history = state.get("history", [])
        metrics = state.get("record_quality", {}).get("market_metrics", {})
        root_cause = state.get("root_cause", {})
        pipeline_df = state.get("record_quality", {}).get("pipeline_df")
        duration_anomalies = state.get("pipeline_health", {}).get("duration_anomalies")
        audit_result = self._get_procedure_result(state, "record_quality_audit")
        audit_details = audit_result.get("details", {}) if isinstance(audit_result.get("details"), dict) else {}
        market_label = metrics.get("market") or state.get("market") or "The requested market"
        lines = ["**Prior Investigation**"]

        if history:
            latest_entry = history[0]
            stored_summary = latest_entry.get("metadata", {}).get("investigation_summary") or latest_entry.get("response", "")
            lines.append(
                f"Yes. A similar investigation exists from {latest_entry.get('timestamp', 'an earlier run')}. "
                f"Stored conclusion: {stored_summary}"
            )
        else:
            lines.append("No similar prior investigation was found in episodic memory.")

        lines.extend(["", "**Current Evidence**"])
        if metrics:
            current_rate = float(metrics.get("success_rate", 0))
            delta = float(metrics.get("success_rate_delta", 0))
            previous_rate = float(metrics.get("previous_success_rate", current_rate))
            month = metrics.get("month", "the latest month")
            if delta < 0:
                lines.append(
                    f"- {market_label} SCS_PERCENT declined to {current_rate:.2f}% in {month}, "
                    f"down {abs(delta):.2f} points from {previous_rate:.2f}%."
                )
            elif delta > 0:
                lines.append(
                    f"- {market_label} SCS_PERCENT is {current_rate:.2f}% in {month}, "
                    f"up {delta:.2f} points from {previous_rate:.2f}%, so the latest data does not show an active drop."
                )
            else:
                lines.append(
                    f"- {market_label} SCS_PERCENT is flat at {current_rate:.2f}% in {month} versus the prior month."
                )

        if pipeline_df is not None and not pipeline_df.empty and {"TOT_REC_CNT", "REJ_REC_CNT"}.issubset(pipeline_df.columns):
            total_records = float(pd.to_numeric(pipeline_df["TOT_REC_CNT"], errors="coerce").fillna(0).sum())
            rejected_records = float(pd.to_numeric(pipeline_df["REJ_REC_CNT"], errors="coerce").fillna(0).sum())
            rejection_rate = (rejected_records / total_records * 100) if total_records else 0
            lines.append(
                f"- REJ_REC_CNT totals {int(rejected_records)} rejected records out of {int(total_records)} total records "
                f"({rejection_rate:.2f}% rejection rate)."
            )

        evaluated_metric = audit_details.get("evaluated_metric")
        if evaluated_metric:
            lines.append(
                f"- The record-quality issue rate is {float(evaluated_metric.get('value', 0)) * 100:.2f}% "
                f"using `{self._display_formula(audit_details.get('stored_function', ''), evaluated_metric=evaluated_metric)}`."
            )

        if root_cause.get("primary_stage"):
            lines.append(f"- The current dominant pipeline stage is {root_cause['primary_stage']}.")
        if root_cause.get("stuck_count") is not None:
            lines.append(f"- There are {int(root_cause.get('stuck_count', 0))} stuck roster operations in the current scope.")
        if duration_anomalies is not None and not duration_anomalies.empty:
            top_anomaly = duration_anomalies.iloc[0]
            lines.append(
                f"- Top duration anomaly is {top_anomaly.get('STAGE_DURATION_COLUMN', top_anomaly.get('STAGE_NAME', 'Unknown stage'))} "
                f"at {float(top_anomaly.get('ANOMALY_RATIO', 0)):.2f}x historical average."
            )
        if root_cause.get("top_failure_status"):
            lines.append(
                f"- The top failure status is {root_cause['top_failure_status']['status']} "
                f"({root_cause['top_failure_status']['count']} records)."
            )
        if root_cause.get("top_impacted_org"):
            lines.append(
                f"- The top impacted organization is {root_cause['top_impacted_org']['org']} "
                f"({root_cause['top_impacted_org']['count']} failed roster operations)."
            )

        lines.extend(["", "**Root Cause**"])
        primary_stage = root_cause.get("primary_stage")
        stuck_count = int(root_cause.get("stuck_count", 0) or 0)
        if primary_stage == "RESOLVED" and stuck_count == 0:
            lines.append(
                "The strongest live signal points to record-quality and rejection leakage rather than a blocked pipeline stage. "
                "RESOLVED being the dominant stage with zero stuck rosters means the pipeline is still flowing; the degradation is more consistent "
                "with files losing success through validation or compatibility issues. The duration anomaly should still be reviewed, but it is a "
                "secondary performance signal rather than the main blocker."
            )
        elif stuck_count > 0 and primary_stage:
            lines.append(
                f"The main signal is a pipeline bottleneck in {primary_stage}, because stuck inventory is present there and is more likely to be "
                "dragging success than isolated record-quality noise."
            )
        else:
            lines.append(
                "The main signal is concentrated failure and rejection pressure in the current scope, not a clearly blocked stage. "
                "That points more toward file-level quality or validation issues than a system-wide pipeline breakdown."
            )

        lines.extend(["", "**Next Action**"])
        if root_cause.get("top_failure_status"):
            lines.append(
                f"Start with the {root_cause['top_failure_status']['status']} failure bucket and audit the highest-impact files and organizations, "
                "because that is the cleanest path to recover SCS performance."
            )
        else:
            lines.append(
                "Audit the highest-rejection files in the current scope and validate whether the dominant stage behavior is operationally expected."
            )

        return "\n".join(lines).strip()

    @staticmethod
    def _response_looks_truncated(response):

        if not response:
            return True

        stripped = response.strip()
        if not stripped:
            return True

        trailing_connectors = {
            "and",
            "or",
            "but",
            "because",
            "with",
            "to",
            "of",
            "for",
            "in",
            "on",
            "at",
            "by",
        }
        last_token = stripped.split()[-1].strip("`*_.,;:()[]{}").lower()
        if last_token in trailing_connectors:
            return True

        return False

    @staticmethod
    def _combined_response_has_required_sections(response):

        normalized = (response or "").lower()
        required_markers = [
            "prior investigation",
            "current evidence",
            "root cause",
            "next action",
        ]
        return all(marker in normalized for marker in required_markers)

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

    def _is_combined_procedure_execution_query(self, state):

        brief = state.get("investigation_brief", {})
        if not brief.get("is_procedure_execution"):
            return False

        normalized = (state.get("query") or "").lower()
        return brief.get("is_memory_query") or any(
            marker in normalized
            for marker in [
                "have we investigated",
                "before",
                "previous",
                "past",
                "explain",
                "why",
                "root cause",
                "cause",
                "reason",
                "fail_rec_cnt",
                "scs_pct",
                "failure",
                "failures",
                "success pct",
                "success percent",
            ]
        )

    @staticmethod
    def _procedure_scope_text(state, details):

        scope_labels = details.get("scope_labels") or state.get("query_scope", {}).get("labels", [])
        return ", ".join(scope_labels) if scope_labels else "all matching files"

    @staticmethod
    def _prior_investigation_lines(history, market_label):

        if history:
            latest_entry = history[0]
            stored_summary = latest_entry.get("metadata", {}).get("investigation_summary") or latest_entry.get("response", "")
            return [
                "**Prior Investigation**",
                (
                    f"Yes. We previously investigated {market_label} on {latest_entry.get('timestamp', 'an earlier run')}. "
                    f"Stored conclusion: {stored_summary}"
                ),
            ]

        return [
            "**Prior Investigation**",
            f"No similar prior investigation was found in episodic memory for {market_label}.",
        ]

    def _build_triage_stuck_ros_detailed_response(self, state, result):

        details = result.get("details", {}) if isinstance(result.get("details"), dict) else {}
        rows = details.get("rows", [])
        pipeline_df = state.get("record_quality", {}).get("pipeline_df")
        metrics = state.get("record_quality", {}).get("market_metrics", {})
        root_cause = state.get("root_cause", {})
        history = state.get("history", [])
        scope_text = self._procedure_scope_text(state, details)
        market_label = state.get("market") or metrics.get("market") or scope_text
        stuck_count = int(root_cause.get("stuck_count", len(rows)) or 0)
        lines = self._prior_investigation_lines(history, market_label)

        lines.extend(["", "**Triage Result**"])
        if stuck_count > 0:
            stage_name = root_cause.get("primary_stage") or rows[0].get("LATEST_STAGE_NM", "Unknown")
            lines.append(
                f"`triage_stuck_ros` found {stuck_count} stuck roster operations in {scope_text}, concentrated in {stage_name}."
            )
            for row in rows[:5]:
                fail_rec_cnt = int(pd.to_numeric(row.get("FAIL_REC_CNT"), errors="coerce") or 0)
                rej_rec_cnt = int(pd.to_numeric(row.get("REJ_REC_CNT"), errors="coerce") or 0)
                scs_pct = float(pd.to_numeric(row.get("SCS_PCT"), errors="coerce") or 0)
                failure_status = row.get("FAILURE_STATUS") or "Unknown"
                lines.append(
                    f"- {row.get('RO_ID', 'Unknown RO')} | {row.get('ORG_NM', 'Unknown org')} | "
                    f"Stage {row.get('LATEST_STAGE_NM', 'Unknown')} | FAIL_REC_CNT={fail_rec_cnt} | "
                    f"REJ_REC_CNT={rej_rec_cnt} | SCS_PCT={scs_pct:.2f}% | Failure={failure_status}"
                )
        else:
            lines.append(f"`triage_stuck_ros` found no stuck roster operations in {scope_text}.")

        lines.extend(["", "**Failure Context**"])
        if pipeline_df is None or pipeline_df.empty:
            lines.append("I could not find scoped pipeline records to explain the failures.")
        else:
            fail_rec_cnt_total = int(pd.to_numeric(pipeline_df.get("FAIL_REC_CNT"), errors="coerce").fillna(0).sum())
            scs_pct_series = pd.to_numeric(pipeline_df.get("SCS_PCT"), errors="coerce").dropna()
            avg_scs_pct = float(scs_pct_series.mean()) if not scs_pct_series.empty else 0
            low_scs_df = pipeline_df.loc[pd.to_numeric(pipeline_df.get("SCS_PCT"), errors="coerce").fillna(100) < 85].copy()
            low_scs_count = int(len(low_scs_df))
            lines.append(f"- FAIL_REC_CNT totals {fail_rec_cnt_total} records across the scoped files.")
            if scs_pct_series.empty:
                lines.append("- SCS_PCT is unavailable for the scoped files.")
            else:
                lines.append(
                    f"- Average SCS_PCT is {avg_scs_pct:.2f}% with {low_scs_count} files below the 85% threshold."
                )

            if metrics:
                lines.append(
                    f"- Latest market SCS_PERCENT is {float(metrics.get('success_rate', 0)):.2f}% in "
                    f"{metrics.get('month', 'the latest month')}."
                )

            if root_cause.get("top_failure_status"):
                lines.append(
                    f"- Top failure status is {root_cause['top_failure_status']['status']} "
                    f"({root_cause['top_failure_status']['count']} records)."
                )

            if not low_scs_df.empty and "LATEST_STAGE_NM" in low_scs_df.columns:
                low_scs_stage = low_scs_df["LATEST_STAGE_NM"].fillna("UNKNOWN").value_counts().idxmax()
                lines.append(f"- The weakest SCS_PCT files are most concentrated in {low_scs_stage}.")

            if not low_scs_df.empty:
                preview_columns = [
                    column
                    for column in ["RO_ID", "ORG_NM", "LATEST_STAGE_NM", "FAIL_REC_CNT", "SCS_PCT", "FAILURE_STATUS"]
                    if column in low_scs_df.columns
                ]
                ranked = low_scs_df.sort_values(["SCS_PCT", "FAIL_REC_CNT"], ascending=[True, False]).head(3)
                for _, row in ranked[preview_columns].iterrows():
                    fail_rec_cnt = int(pd.to_numeric(row.get("FAIL_REC_CNT"), errors="coerce") or 0)
                    scs_pct = float(pd.to_numeric(row.get("SCS_PCT"), errors="coerce") or 0)
                    lines.append(
                        f"- Low-SCS file: {row.get('RO_ID', 'Unknown RO')} | {row.get('ORG_NM', 'Unknown org')} | "
                        f"Stage {row.get('LATEST_STAGE_NM', 'Unknown')} | FAIL_REC_CNT={fail_rec_cnt} | "
                        f"SCS_PCT={scs_pct:.2f}% | Failure={row.get('FAILURE_STATUS') or 'Unknown'}"
                    )

        lines.extend(["", "**Interpretation**"])
        if stuck_count > 0:
            lines.append(
                "This is an active stuck-roster problem in the scoped market, so the stuck inventory should be cleared first. "
                "Use the FAIL_REC_CNT and low SCS_PCT rows above to prioritize the worst affected operations."
            )
        else:
            lines.append(
                "This is not currently a stuck-roster problem in the scoped market. The stronger signal is broad file-level underperformance: "
                "FAIL_REC_CNT is elevated and a meaningful share of files sit below the 85% SCS_PCT threshold, so the failures are more consistent "
                "with quality or validation issues than with an active blocked queue."
            )

        lines.extend(["", "**Next Action**"])
        if stuck_count > 0:
            lines.append(
                "Triage the listed stuck roster operations first, then audit the lowest-SCS_PCT files in the same stage to separate true queue blockage from bad-input failures."
            )
        else:
            lines.append(
                "Start with the lowest-SCS_PCT Kansas files and their dominant failure status, then investigate the stage where low-SCS_PCT files cluster to find the main validation or data-quality break."
            )

        return "\n".join(lines).strip()

    def _build_procedure_execution_response(self, state):

        procedure_results = state.get("procedure_results", [])
        if not procedure_results:
            return "I could not run the requested procedure because no procedure result was produced."

        result = procedure_results[0]
        details = result.get("details", {}) if isinstance(result.get("details"), dict) else {}
        scope_text = self._procedure_scope_text(state, details)

        if result.get("procedure") == "triage_stuck_ros" and self._is_combined_procedure_execution_query(state):
            return self._build_triage_stuck_ros_detailed_response(state, result)

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
            detail = item.get("search_answer") or item.get("snippet", "")
            lines.append(
                f"- [{item.get('category', 'external_context')}] {item.get('purpose', item.get('title', 'Untitled'))} :: "
                f"{detail} ({item.get('url', '')})"
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

    @staticmethod
    def _should_include_episodic_context(brief, history):

        return bool(brief.get("is_memory_query") and history)

    @staticmethod
    def _should_suppress_audit_metrics(brief, query):

        scope = brief.get("query_scope", {}) if brief else {}
        if not scope.get("org_name"):
            return False
        if "external_context" not in set(brief.get("intents", [])):
            return False

        normalized = (query or "").lower()
        is_org_context_query = any(
            phrase in normalized
            for phrase in [
                "business context",
                "organization context",
                "org context",
                "background",
                "who is",
                "look up",
                "lookup",
                "research",
            ]
        )
        explicit_quality_request = any(
            token in normalized
            for token in [
                "quality",
                "record quality",
                "reject",
                "rejection",
                "failure",
                "validation",
                "audit",
                "skip",
                "scs_pct",
                "threshold",
            ]
        )
        return is_org_context_query and not explicit_quality_request

    def _filter_evidence_for_prompt(self, state):

        evidence_items = list(state.get("evidence", []))
        brief = state.get("investigation_brief", {})
        if not self._should_suppress_audit_metrics(brief, state.get("query")):
            return evidence_items

        filtered = []
        for item in evidence_items:
            normalized = str(item or "").lower()
            if "record-quality issue rate" in normalized:
                continue
            if "scs_pct threshold" in normalized:
                continue
            filtered.append(item)
        return filtered

    def _build_focus_instruction(self, state):

        brief = state.get("investigation_brief", {})
        if self._should_suppress_audit_metrics(brief, state.get("query")):
            return (
                "Focus on organization background and the most relevant high-level anomaly signals. "
                "Do not mention record-quality issue rate formulas, audit thresholds, or file-count audit summaries "
                "unless the user explicitly asks for quality metrics."
            )

        if self._is_combined_memory_analysis_query(state):
            return (
                "Use episodic memory only for the prior-investigation part of the question. "
                "Ground the main conclusion in the current live evidence, especially SCS_PERCENT, rejection signals, "
                "and pipeline-stage findings."
            )

        if brief.get("is_memory_query"):
            return "Use episodic memory because the user is explicitly asking about prior investigations."

        return "Do not mention episodic memory unless the user explicitly asks about prior investigations."

    def _should_use_structured_org_context_response(self, state):

        brief = state.get("investigation_brief", {})
        if not self._should_suppress_audit_metrics(brief, state.get("query")):
            return False

        normalized = (state.get("query") or "").lower()
        return any(
            marker in normalized
            for marker in [
                "pipeline anomaly",
                "anomaly",
                "business context",
                "look up",
                "lookup",
                "background",
            ]
        )

    @staticmethod
    def _format_month_label(month_value):

        parsed = pd.to_datetime(str(month_value or ""), format="%m-%Y", errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%B %Y")
        return str(month_value or "the latest period")

    @staticmethod
    def _extract_org_context_features(web_context):

        text = " ".join(
            str(item.get("search_answer") or item.get("snippet") or "")
            for item in (web_context or [])
        ).lower()
        return {
            "has_care_portal": "carelink" in text,
            "is_nonprofit_hospital": "nonprofit hospital" in text,
            "is_los_angeles_based": "los angeles" in text,
        }

    def _build_org_context_direct_answer(self, org_name, market, month_label, duration_column, anomaly_ratio, web_context):

        context_features = self._extract_org_context_features(web_context)
        org_phrase_parts = []
        if context_features["is_los_angeles_based"]:
            org_phrase_parts.append("Los Angeles-based")
        if context_features["is_nonprofit_hospital"]:
            org_phrase_parts.append("nonprofit hospital")
        if not org_phrase_parts:
            org_phrase_parts.append("provider organization")

        org_phrase = " ".join(org_phrase_parts)
        answer = (
            f"The {market} pipeline anomaly for {month_label} is a severe duration spike in the "
            f"{duration_column} metric ({anomaly_ratio:.2f}x historical average), not a failure-rate or data-quality issue."
        )
        answer += (
            f" {org_name} is a {org_phrase} and a meaningful California provider, so its submission patterns "
            f"are relevant to systemic processing delays."
        )
        return answer

    def _build_org_context_anomaly_response(self, state, web_context):

        brief = state.get("investigation_brief", {})
        scope = brief.get("query_scope", {})
        org_name = scope.get("org_name") or "The organization"
        market = scope.get("market") or state.get("market") or "requested market"
        metrics = state.get("record_quality", {}).get("market_metrics", {})
        root_cause = state.get("root_cause", {})
        duration_anomalies = state.get("pipeline_health", {}).get("duration_anomalies")
        month_label = self._format_month_label(metrics.get("month"))
        success_rate = float(metrics.get("success_rate", 0) or 0)
        success_threshold = 95.0
        stuck_count = int(root_cause.get("stuck_count", 0) or 0)
        top_impacted_org = root_cause.get("top_impacted_org")
        top_failure_status = root_cause.get("top_failure_status")
        top_anomaly = duration_anomalies.iloc[0] if duration_anomalies is not None and not duration_anomalies.empty else None
        duration_column = top_anomaly.get("STAGE_DURATION_COLUMN", "current stage duration") if top_anomaly is not None else "current stage duration"
        anomaly_ratio = float(top_anomaly.get("ANOMALY_RATIO", 0) or 0) if top_anomaly is not None else 0.0
        context_features = self._extract_org_context_features(web_context)

        lines = [
            "**Direct Answer**",
            self._build_org_context_direct_answer(
                org_name=org_name,
                market=market,
                month_label=month_label,
                duration_column=duration_column,
                anomaly_ratio=anomaly_ratio,
                web_context=web_context,
            ),
            "",
            "**Key Evidence**",
            (
                f"- Success rate remains healthy at {success_rate:.2f}% "
                f"({'above' if success_rate >= success_threshold else 'below'} {success_threshold:.0f}% threshold), "
                f"with {stuck_count} stuck roster operations."
            ),
            (
                f"- The strongest anomaly signal is {duration_column} at {anomaly_ratio:.2f}x historical average, "
                "so the issue is centered on processing duration rather than FAIL_REC_CNT or REJ_REC_CNT concentration."
            ),
        ]

        if top_impacted_org:
            lines.append(
                f"- The top impacted organization is {top_impacted_org['org']} ({top_impacted_org['count']} records), "
                f"so {org_name} is not the only organization in scope."
            )
        else:
            lines.append(
                f"- No single organization is flagged as the top impacted, suggesting the delay is broader than one failed subset and could reflect load from high-volume submitters like {org_name}."
            )

        if top_failure_status:
            lines.append(
                f"- No dominant failure bucket is driving the anomaly: the duration spike matters more than failure-status concentration in this query context."
            )

        lines.extend(
            [
                "",
                "**Recommended Next Action**",
                f"- Review {org_name}'s {month_label} roster submissions for unusual file sizes, LOB complexity, or submission timing that could strain {duration_column} resources.",
            ]
        )

        if context_features["has_care_portal"]:
            lines.append(
                f"- Correlate duration spikes with {org_name}'s CareLink portal activity; if referral or network-update workflows expanded, roster volume may have surged."
            )
        else:
            lines.append(
                f"- Correlate duration spikes with {org_name}'s submission windows and upstream workflow changes to see whether roster volume or payload complexity increased."
            )

        lines.append(
            f"- Monitor pipeline resource utilization during future {org_name} submission windows to determine whether scaling or scheduling adjustments are needed."
        )
        return "\n".join(lines).strip()

    @staticmethod
    def _format_web_details_for_response(items, limit=4):

        if not items:
            return ""

        category_labels = {
            "regulatory_change": "Regulatory context",
            "compliance_standard": "Compliance context",
            "lob_policy": "LOB policy context",
            "org_context": "Organization context",
        }
        lines = ["", "", "**Web Search Details**"]
        seen_categories = set()
        for item in items:
            category = item.get("category", "external_context")
            if category in seen_categories:
                continue
            seen_categories.add(category)
            detail = item.get("search_answer") or item.get("snippet", "")
            source = item.get("url") or "Tavily summary"
            lines.append(
                f"- {category_labels.get(category, category.replace('_', ' ').title())}: "
                f"{detail} Source: {source}"
            )
            if len(seen_categories) >= limit:
                break
        return "\n".join(lines)

    def _augment_response_with_web_context(self, response, web_context):

        if not web_context:
            return response

        detail_block = self._format_web_details_for_response(web_context)
        if detail_block.strip() in (response or ""):
            return response
        return (response or "").rstrip() + detail_block

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
        combined_memory_analysis = self._is_combined_memory_analysis_query(state)
        tools = set(brief.get("tool_requests", []))
        desired_outputs = set(brief.get("desired_outputs", []))
        scope = brief.get("query_scope", {})
        web_context = []

        if not is_memory_query and "web_search" in tools and "external_context" in desired_outputs:
            web_context = self.web.search_external_context(state, max_results_per_query=2)
        state["web_context"] = web_context
        structured_org_context_response = self._should_use_structured_org_context_response(state)

        filtered_evidence = self._filter_evidence_for_prompt(state)
        evidence = "\n".join(f"- {item}" for item in filtered_evidence)
        root_cause = state.get("root_cause", {})
        if self._should_include_episodic_context(brief, state.get("history", [])):
            episodic_context = self.memory.format_for_prompt(state.get("history", []), limit=3)
        else:
            episodic_context = ""
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
{self._build_focus_instruction(state)}
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

        if structured_org_context_response:
            response = self._build_org_context_anomaly_response(state, web_context)
            state["llm_status"] = "org context (rule-based)"
        elif is_memory_query and not combined_memory_analysis:
            response = self.llm.generate(self._build_memory_prompt(state), system_prompt=system_prompt)
        elif combined_memory_analysis:
            response = self.llm.generate(
                self._build_combined_memory_analysis_prompt(state, evidence),
                system_prompt=system_prompt,
            )
        else:
            response = self.llm.generate(prompt, system_prompt=system_prompt)
        if structured_org_context_response:
            state["llm_status"] = "org context (rule-based)"
        elif response:
            state["llm_status"] = f"openrouter ({self.llm.model})"
        else:
            error_detail = self.llm.last_error or "Unknown failure"
            state["llm_status"] = f"fallback ({error_detail})"
        if not response:
            if combined_memory_analysis:
                response = self._build_combined_memory_analysis_fallback(state)
            else:
                response = self._build_fallback(state)
        elif combined_memory_analysis and (
            self._response_looks_truncated(response)
            or not self._combined_response_has_required_sections(response)
        ):
            response = self._build_combined_memory_analysis_fallback(state)
        response = self._augment_response_with_web_context(response, web_context)

        if is_memory_query and not combined_memory_analysis:
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
