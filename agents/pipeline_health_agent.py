import pandas as pd

from tools.data_query_tool import DataQueryTool


class PipelineHealthAgent:

    def __init__(self):

        self.data = DataQueryTool()

    def run(self, state):

        scope = state.get("investigation_brief", {}).get("query_scope", {})
        if state.get("investigation_brief", {}).get("is_procedure_update"):
            state["pipeline_health"] = {
                "pipeline_df": pd.DataFrame(),
                "stuck_df": pd.DataFrame(),
                "stage_counts": [],
                "duration_anomalies": pd.DataFrame(),
            }
            return state

        market = scope.get("market") or state.get("market")
        stages = scope.get("stages")
        pipeline = self.data.get_scoped_dataset("pipeline", scope=scope)

        stuck = pipeline[pd.to_numeric(pipeline.get("IS_STUCK", 0), errors="coerce").fillna(0).astype(int) == 1]
        stage_counts = (
            pipeline["LATEST_STAGE_NM"]
            .fillna("UNKNOWN")
            .value_counts()
            .head(5)
            .reset_index()
        )
        stage_counts.columns = ["stage", "count"]

        filters = {"CNT_STATE": market} if market else None
        if scope.get("org_name"):
            filters = dict(filters or {})
            filters["ORG_NM"] = scope["org_name"]
        duration_anomalies = self.data.detect_stage_duration_anomalies(filters=filters, stages=stages, scope=scope)

        state["pipeline_health"] = {
            "pipeline_df": pipeline,
            "stuck_df": stuck,
            "stage_counts": stage_counts.to_dict("records"),
            "duration_anomalies": duration_anomalies,
        }

        if not stuck.empty:
            stage_name = stuck["LATEST_STAGE_NM"].fillna("UNKNOWN").value_counts().idxmax()
            state["evidence"].append(
                f"{len(stuck)} roster operations are stuck, most often in {stage_name}."
            )
        elif not stage_counts.empty:
            state["evidence"].append(
                f"No stuck roster operations found; the most common stage is {stage_counts.iloc[0]['stage']}."
            )

        if not duration_anomalies.empty:
            slowest = duration_anomalies.iloc[0]
            state["evidence"].append(
                f"Duration anomalies are concentrated in {slowest['STAGE_DURATION_COLUMN']} "
                f"at {slowest['ANOMALY_RATIO']:.2f}x historical average."
            )

        return state
