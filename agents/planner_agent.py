from memory.procedural_memory import ProceduralMemory


class PlannerAgent:

    def __init__(self):

        self.proc = ProceduralMemory()

    def plan(self, state):

        brief = state.get("investigation_brief", {})
        if brief.get("is_procedure_update"):
            return []
        if brief.get("is_procedure_execution"):
            target = brief.get("procedure_target")
            return [target] if target in set(self.proc.list()) else []

        intents = set(brief.get("intents", []))
        history = state.get("history", [])
        tools = set(brief.get("tool_requests", []))
        plan = []

        if "data_query" not in tools:
            return plan

        if "trend_analysis" in intents:
            plan.append("market_health_report")

        if "record_quality" in intents or "root_cause_analysis" in intents:
            plan.append("record_quality_audit")

        if "pipeline_diagnostics" in intents or "root_cause_analysis" in intents:
            plan.append("triage_stuck_ros")

        if "retry_analysis" in intents:
            plan.append("retry_effectiveness_analysis")

        if brief.get("is_memory_query") and history and plan:
            plan = ["market_health_report"] + [procedure for procedure in plan if procedure != "market_health_report"]

        if brief.get("is_memory_query") and history and not plan:
            plan = []

        if not plan and not brief.get("is_memory_query"):
            plan = ["market_health_report", "record_quality_audit", "triage_stuck_ros"]

        available = set(self.proc.list())
        return [procedure for procedure in dict.fromkeys(plan) if procedure in available]
