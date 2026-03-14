from memory.episodic_memory import EpisodicMemory
from memory.procedural_memory import ProceduralMemory
from tools.data_query_tool import DataQueryTool
from utils.openrouter_client import OpenRouterClient
import json


class SupervisorAgent:

    def __init__(self):

        self.memory = EpisodicMemory()
        self.procedures = ProceduralMemory()
        self.data = DataQueryTool()
        self.llm = OpenRouterClient()

    def _infer_market_from_text(self, query):

        return self.data.infer_market_from_text(query)

    @staticmethod
    def _coerce_bool(value, default=False):

        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1"}:
                return True
            if normalized in {"false", "no", "0", ""}:
                return False
        if value is None:
            return default
        return bool(value)

    @staticmethod
    def _should_request_external_context(normalized_query, scope):

        external_context_patterns = [
            "cms",
            "medicaid",
            "medicare",
            "compliance",
            "regulation",
            "regulatory",
            "policy",
            "payer",
            "validation",
            "complete validation failure",
            "data standard",
            "provider directory",
            "provider roster",
            "roster compliance",
            "submission requirement",
            "submission requirements",
            "external context",
            "business context",
        ]
        rejection_spike_signal = (
            bool(scope.get("market"))
            and any(token in normalized_query for token in ["spike", "surge", "jump", "drop", "decline"])
            and any(token in normalized_query for token in ["reject", "rejection", "failure", "validation"])
        )
        org_context_signal = scope.get("org_name") and any(
            token in normalized_query
            for token in ["context", "background", "who is", "look up", "research", "anomaly", "failure", "pipeline", "spike", "stuck"]
        )
        return any(token in normalized_query for token in external_context_patterns) or rejection_spike_signal or org_context_signal

    @staticmethod
    def _is_org_business_context_query(normalized_query, scope):

        if not scope.get("org_name"):
            return False

        return any(
            phrase in normalized_query
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

    @staticmethod
    def _has_explicit_quality_request(normalized_query):

        return any(
            token in normalized_query
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

    @staticmethod
    def _should_show_default_visualizations(normalized_query, scope):

        if scope.get("requested_charts"):
            return True
        if scope.get("is_full_operational_report"):
            return False
        if scope.get("market"):
            return True
        if scope.get("org_name"):
            return False

        context_only_markers = [
            "business context",
            "organization context",
            "org context",
            "background",
            "who is",
            "look up",
            "lookup",
            "research",
            "external context",
        ]
        if any(marker in normalized_query for marker in context_only_markers):
            return False

        all_state_markers = [
            "all states",
            "across all states",
            "all markets",
            "across markets",
            "overall",
            "portfolio",
            "system wide",
            "system-wide",
            "enterprise",
            "nationwide",
        ]
        analytic_markers = [
            "pipeline",
            "health",
            "stage",
            "stuck",
            "delay",
            "blocked",
            "rejection",
            "reject",
            "quality",
            "failure",
            "trend",
            "rate",
            "success",
            "anomaly",
            "anomalies",
            "duration",
            "outlier",
        ]
        return any(marker in normalized_query for marker in all_state_markers) or any(
            marker in normalized_query for marker in analytic_markers
        )

    def _fallback_brief(self, query, market):

        normalized = (query or "").lower()
        procedure_target = self.procedures.infer_target_procedure(query)
        is_procedure_update = self.procedures.is_update_query(query)
        is_procedure_execution = (
            not is_procedure_update and self._is_procedure_execution_query(query, procedure_target)
        )
        scope = self.data.extract_visualization_scope(query, market=market)
        market = None if is_procedure_update else (scope.get("market") or market)
        intents = []
        topics = []
        chart_preferences = list(scope.get("requested_charts", []))
        tool_requests = []
        desired_outputs = ["answer"]
        memory_lookup_patterns = [
            "have we investigated",
            "before",
            "previous",
            "past",
            "last time",
            "last session",
            "since my last session",
            "what changed since",
        ]
        historical_data_patterns = [
            "history of",
            "across runs",
            "show me the history",
            "trend over time",
        ]

        if is_procedure_update:
            intents.append("procedure_update")
            topics.append("procedure_memory")
            tool_requests.append("procedural_memory")
            desired_outputs.append("procedure_update_confirmation")
        elif is_procedure_execution:
            intents.append("procedure_execution")
            topics.append("procedure_memory")
            tool_requests.append("data_query")
            if chart_preferences:
                tool_requests.append("visualization")
                desired_outputs.append("visualization")

        if any(token in normalized for token in memory_lookup_patterns) and not is_procedure_update:
            intents.append("memory_lookup")
            topics.append("past_responses")
            tool_requests.append("episodic_memory")
            desired_outputs.append("memory_summary")

        if any(token in normalized for token in historical_data_patterns) and not is_procedure_update and not is_procedure_execution:
            intents.append("trend_analysis")
            topics.append("historical_data")
            if "market_scs_percent_trend" not in chart_preferences:
                chart_preferences.append("market_scs_percent_trend")
            tool_requests.append("visualization")
            desired_outputs.append("visualization")

        if not is_procedure_update and not is_procedure_execution and (
            any(
                chart in chart_preferences
                for chart in ["pipeline_stage_health_heatmap", "stuck_ro_tracker", "duration_anomaly_chart"]
            )
            or any(token in normalized for token in ["pipeline", "stage", "stuck", "delay", "blocked"])
        ):
            intents.append("pipeline_diagnostics")
            topics.append("pipeline")

        if not is_procedure_update and not is_procedure_execution and (
            "market_scs_percent_trend" in chart_preferences
            or any(token in normalized for token in ["drop", "trend", "rate", "market", "success"])
        ):
            intents.append("trend_analysis")
            topics.append("market_performance")

        if not is_procedure_update and not is_procedure_execution and (
            "record_quality_breakdown" in chart_preferences
            or any(token in normalized for token in ["quality", "reject", "rejection", "failure", "validation", "audit"])
        ):
            intents.append("record_quality")
            topics.append("quality")

        if not is_procedure_update and not is_procedure_execution and (
            "retry_lift_chart" in chart_preferences
            or any(token in normalized for token in ["retry", "rerun", "iteration"])
        ):
            intents.append("retry_analysis")
            topics.append("retry")

        if not is_procedure_update and not is_procedure_execution and any(
            token in normalized for token in ["why", "cause", "root cause", "reason"]
        ):
            intents.append("root_cause_analysis")
            topics.append("root_cause")

        if not is_procedure_update and not is_procedure_execution and self._should_request_external_context(normalized, scope):
            intents.append("external_context")
            topics.append("compliance")
            tool_requests.append("web_search")
            desired_outputs.append("external_context")

        if any(token in normalized for token in ["show", "plot", "chart", "graph", "visual"]) and not is_procedure_update:
            tool_requests.append("visualization")
            desired_outputs.append("visualization")

        if scope.get("is_full_operational_report") and not is_procedure_update and not is_procedure_execution:
            tool_requests.append("report_generator")
            desired_outputs.append("report")
            topics.append("operational_report")

        if (
            not intents
            and not is_procedure_update
            and not is_procedure_execution
            and self._should_show_default_visualizations(normalized, scope)
        ):
            intents = ["trend_analysis", "pipeline_diagnostics", "record_quality"]
            topics.extend(["market_performance", "pipeline", "quality"])
            if not chart_preferences and not scope.get("is_full_operational_report"):
                chart_preferences.extend(
                    ["market_scs_percent_trend", "pipeline_stage_health_heatmap", "record_quality_breakdown"]
                )

        if chart_preferences and "visualization" not in tool_requests and not is_procedure_update:
            tool_requests.append("visualization")
        if chart_preferences and "visualization" not in desired_outputs and not is_procedure_update:
            desired_outputs.append("visualization")
        if not is_procedure_update and "data_query" not in tool_requests:
            tool_requests.append("data_query")

        return {
            "market": market,
            "intents": list(dict.fromkeys(intents)),
            "topics": list(dict.fromkeys(topics)),
            "chart_preferences": list(dict.fromkeys(chart_preferences)),
            "tool_requests": list(dict.fromkeys(tool_requests)),
            "desired_outputs": list(dict.fromkeys(desired_outputs)),
            "is_memory_query": "memory_lookup" in intents,
            "use_memory_retrieval": "memory_lookup" in intents and "historical_data" not in topics,
            "is_procedure_update": is_procedure_update,
            "is_procedure_execution": is_procedure_execution,
            "procedure_target": procedure_target,
            "llm_routed": False,
            "query_scope": scope,
        }

    def _has_memory_lookup_intent(self, query):

        normalized = (query or "").lower()
        memory_lookup_patterns = [
            "have we investigated",
            "before",
            "previous",
            "past",
            "last time",
            "last session",
            "since my last session",
            "what changed since",
        ]
        return any(token in normalized for token in memory_lookup_patterns)

    def _is_procedure_execution_query(self, query, procedure_target):

        if not procedure_target:
            return False

        normalized = (query or "").lower()
        normalized_name_phrase = procedure_target.replace("_", " ").lower()

        if procedure_target.lower() in normalized or normalized_name_phrase in normalized:
            return any(marker in normalized for marker in ["run", "execute", "perform", "show", "analyze"])

        if procedure_target == "record_quality_audit" and "audit" in normalized:
            return True
        if procedure_target == "triage_stuck_ros" and "triage" in normalized:
            return True
        if procedure_target == "retry_effectiveness_analysis" and any(
            marker in normalized for marker in ["analyze", "analysis"]
        ):
            return True

        return False

    def _build_investigation_brief(self, query, market):

        fallback = self._fallback_brief(query, market)
        if fallback.get("is_procedure_update") or fallback.get("is_procedure_execution"):
            return fallback

        prompt = f"""
You are routing a healthcare operations user request.
Return JSON only with keys:
intents, topics, chart_preferences, tool_requests, desired_outputs, use_memory_retrieval, is_memory_query, is_procedure_update, is_procedure_execution, procedure_target.

Allowed intents:
memory_lookup, pipeline_diagnostics, trend_analysis, record_quality, retry_analysis, root_cause_analysis, external_context, procedure_update, procedure_execution

Allowed chart_preferences:
pipeline_stage_health_heatmap, record_quality_breakdown, duration_anomaly_chart, market_scs_percent_trend, retry_lift_chart, stuck_ro_tracker

Allowed tool_requests:
data_query, episodic_memory, procedural_memory, visualization, web_search, report_generator

Allowed desired_outputs:
answer, memory_summary, visualization, external_context, report, procedure_update_confirmation

User query: {query}
Detected market: {market or "unknown"}
""".strip()

        raw = self.llm.generate(prompt)
        if not raw:
            return fallback

        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(cleaned)
        except Exception:
            return fallback

        brief = {
            "market": market,
            "intents": parsed.get("intents", fallback["intents"]),
            "topics": parsed.get("topics", fallback["topics"]),
            "chart_preferences": parsed.get("chart_preferences", fallback["chart_preferences"]),
            "tool_requests": parsed.get("tool_requests", fallback["tool_requests"]),
            "desired_outputs": parsed.get("desired_outputs", fallback["desired_outputs"]),
            "use_memory_retrieval": self._coerce_bool(
                parsed.get("use_memory_retrieval", fallback["use_memory_retrieval"]),
                default=fallback["use_memory_retrieval"],
            ),
            "is_memory_query": self._coerce_bool(
                parsed.get("is_memory_query", fallback["is_memory_query"]),
                default=fallback["is_memory_query"],
            ),
            "is_procedure_update": self._coerce_bool(
                parsed.get("is_procedure_update", fallback["is_procedure_update"]),
                default=fallback["is_procedure_update"],
            ),
            "is_procedure_execution": self._coerce_bool(
                parsed.get("is_procedure_execution", fallback["is_procedure_execution"]),
                default=fallback["is_procedure_execution"],
            ),
            "procedure_target": parsed.get("procedure_target") or fallback["procedure_target"],
            "llm_routed": True,
            "query_scope": fallback.get("query_scope", {}),
        }

        if "data_query" not in brief["tool_requests"] and not brief["is_procedure_update"]:
            brief["tool_requests"].append("data_query")
        if "answer" not in brief["desired_outputs"]:
            brief["desired_outputs"].append("answer")

        # Keep only supported values.
        allowed_intents = {
            "memory_lookup",
            "pipeline_diagnostics",
            "trend_analysis",
            "record_quality",
            "retry_analysis",
            "root_cause_analysis",
            "external_context",
            "procedure_update",
            "procedure_execution",
        }
        allowed_charts = {
            "pipeline_stage_health_heatmap",
            "record_quality_breakdown",
            "duration_anomaly_chart",
            "market_scs_percent_trend",
            "retry_lift_chart",
            "stuck_ro_tracker",
        }
        allowed_tools = {"data_query", "episodic_memory", "procedural_memory", "visualization", "web_search", "report_generator"}
        allowed_outputs = {"answer", "memory_summary", "visualization", "external_context", "report", "procedure_update_confirmation"}
        brief["intents"] = [item for item in brief["intents"] if item in allowed_intents] or fallback["intents"]
        brief["chart_preferences"] = [item for item in brief["chart_preferences"] if item in allowed_charts]
        brief["tool_requests"] = [item for item in brief["tool_requests"] if item in allowed_tools] or fallback["tool_requests"]
        brief["desired_outputs"] = [item for item in brief["desired_outputs"] if item in allowed_outputs] or fallback["desired_outputs"]
        brief["topics"] = brief["topics"] or fallback["topics"]
        scope = self.data.extract_visualization_scope(query, requested_charts=brief["chart_preferences"], market=market)
        brief["query_scope"] = scope
        brief["chart_preferences"] = scope["requested_charts"] or fallback["chart_preferences"]
        brief["market"] = scope.get("market") or market
        normalized = (query or "").lower()
        if self._should_request_external_context(normalized, scope):
            if "web_search" not in brief["tool_requests"]:
                brief["tool_requests"].append("web_search")
            if "external_context" not in brief["desired_outputs"]:
                brief["desired_outputs"].append("external_context")
            if "external_context" not in brief["intents"]:
                brief["intents"].append("external_context")
            if "compliance" not in brief["topics"]:
                brief["topics"].append("compliance")

        if self._is_org_business_context_query(normalized, scope) and not self._has_explicit_quality_request(normalized):
            brief["intents"] = [item for item in brief["intents"] if item != "record_quality"]
            brief["chart_preferences"] = [
                item for item in brief["chart_preferences"] if item != "record_quality_breakdown"
            ]
        if not self._should_show_default_visualizations(normalized, scope):
            brief["chart_preferences"] = scope["requested_charts"]
        if scope.get("is_full_operational_report") and not brief["is_procedure_update"] and not brief["is_procedure_execution"]:
            if "report_generator" not in brief["tool_requests"]:
                brief["tool_requests"].append("report_generator")
            if "report" not in brief["desired_outputs"]:
                brief["desired_outputs"].append("report")
            if "operational_report" not in brief["topics"]:
                brief["topics"].append("operational_report")
        else:
            brief["tool_requests"] = [item for item in brief["tool_requests"] if item != "report_generator"]
            brief["desired_outputs"] = [item for item in brief["desired_outputs"] if item != "report"]

        if brief["chart_preferences"] and "visualization" not in brief["tool_requests"] and not brief["is_procedure_update"]:
            brief["tool_requests"].append("visualization")
        if brief["chart_preferences"] and "visualization" not in brief["desired_outputs"] and not brief["is_procedure_update"]:
            brief["desired_outputs"].append("visualization")
        if not brief["chart_preferences"] and not brief["is_procedure_execution"]:
            brief["tool_requests"] = [item for item in brief["tool_requests"] if item != "visualization"]
            brief["desired_outputs"] = [item for item in brief["desired_outputs"] if item != "visualization"]

        if not self._has_memory_lookup_intent(query):
            brief["use_memory_retrieval"] = False
            brief["is_memory_query"] = False
            brief["tool_requests"] = [item for item in brief["tool_requests"] if item != "episodic_memory"]
            brief["desired_outputs"] = [item for item in brief["desired_outputs"] if item != "memory_summary"]

        if brief["is_memory_query"]:
            brief["use_memory_retrieval"] = True
            if "episodic_memory" not in brief["tool_requests"]:
                brief["tool_requests"].append("episodic_memory")

        if "historical_data" in brief["topics"]:
            brief["is_memory_query"] = False
            brief["use_memory_retrieval"] = False

        if brief["is_procedure_update"]:
            brief["is_memory_query"] = False
            brief["use_memory_retrieval"] = False
            brief["chart_preferences"] = []
            brief["tool_requests"] = [tool for tool in brief["tool_requests"] if tool != "data_query"]
            if "procedural_memory" not in brief["tool_requests"]:
                brief["tool_requests"].append("procedural_memory")
            if "procedure_update_confirmation" not in brief["desired_outputs"]:
                brief["desired_outputs"].append("procedure_update_confirmation")

        if brief["is_procedure_execution"]:
            memory_lookup_requested = self._has_memory_lookup_intent(query)
            brief["is_memory_query"] = memory_lookup_requested
            brief["use_memory_retrieval"] = memory_lookup_requested
            brief["tool_requests"] = ["data_query"]
            brief["desired_outputs"] = ["answer"]
            if memory_lookup_requested:
                brief["tool_requests"].append("episodic_memory")
                brief["desired_outputs"].append("memory_summary")
            if brief["chart_preferences"]:
                brief["tool_requests"].append("visualization")
                brief["desired_outputs"].append("visualization")

        return brief

    def run(self, state):

        query = state["query"]
        market = state.get("market") or self._infer_market_from_text(query)
        brief = self._build_investigation_brief(query, market)
        market = None if brief.get("is_procedure_update") else (brief.get("market") or market)
        history = []
        query_embedding = None

        if brief.get("use_memory_retrieval"):
            query_embedding = self.memory.embed_query(query)
            history = self.memory.search_similar_responses(query, query_profile=brief)

        state["history"] = history
        state["market"] = market
        state["investigation_brief"] = brief
        state["query_scope"] = brief.get("query_scope", {})
        state["query_embedding"] = query_embedding

        if brief.get("is_procedure_update"):
            target = brief.get("procedure_target") or "the requested procedure"
            state["evidence"].append(f"User is refining procedural memory for {target}.")
        elif brief.get("is_procedure_execution"):
            target = brief.get("procedure_target") or "the requested procedure"
            labels = ", ".join(brief.get("query_scope", {}).get("labels", [])) or "all matching records"
            state["evidence"].append(f"User requested execution of {target} for {labels}.")

        if history:
            latest_entry = history[0]
            state["evidence"].append(
                f"Retrieved similar historical investigation from {latest_entry['timestamp']}."
            )

        if brief["use_memory_retrieval"]:
            state["evidence"].append("User is asking about prior investigations, so episodic memory should be prioritized.")

        return state
