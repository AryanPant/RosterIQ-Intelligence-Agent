from langgraph.graph import StateGraph

from agents.answer_agent import AnswerAgent
from agents.pipeline_health_agent import PipelineHealthAgent
from agents.planner_agent import PlannerAgent
from agents.record_quality_agent import RecordQualityAgent
from agents.supervisor_agent import SupervisorAgent
from engines.procedure_runner import ProcedureRunner
from engines.root_cause_engine import RootCauseEngine


def build_graph():

    supervisor = SupervisorAgent()
    planner = PlannerAgent()
    pipeline = PipelineHealthAgent()
    record = RecordQualityAgent()
    runner = ProcedureRunner()
    root_cause = RootCauseEngine()
    answer = AnswerAgent()

    def supervisor_node(state):
        return supervisor.run(state)

    def planner_node(state):
        state["plan"] = planner.plan(state)
        return state

    def pipeline_node(state):
        return pipeline.run(state)

    def record_node(state):
        return record.run(state)

    def procedure_node(state):
        return runner.run(state["plan"], state)

    def root_cause_node(state):
        state["root_cause"] = root_cause.trace(
            state.get("market"),
            scope=state.get("investigation_brief", {}).get("query_scope", {}),
        )
        return state

    def answer_node(state):
        return answer.run(state)

    graph = StateGraph(dict)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("planner", planner_node)
    graph.add_node("pipeline", pipeline_node)
    graph.add_node("record", record_node)
    graph.add_node("procedure", procedure_node)
    graph.add_node("root_cause", root_cause_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "planner")
    graph.add_edge("planner", "pipeline")
    graph.add_edge("pipeline", "record")
    graph.add_edge("record", "procedure")
    graph.add_edge("procedure", "root_cause")
    graph.add_edge("root_cause", "answer")
    graph.set_finish_point("answer")

    return graph.compile()


def run_graph(query):

    graph = build_graph()
    return graph.invoke(
        {
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
    )
