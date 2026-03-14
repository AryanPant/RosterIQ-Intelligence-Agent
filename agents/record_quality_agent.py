import pandas as pd

from tools.data_query_tool import DataQueryTool


class RecordQualityAgent:

    def __init__(self):

        self.data = DataQueryTool()

    def run(self, state):

        scope = state.get("investigation_brief", {}).get("query_scope", {})
        if state.get("investigation_brief", {}).get("is_procedure_update"):
            state["record_quality"] = {
                "market_metrics": {},
                "market_history_df": pd.DataFrame(),
                "aggregated_market_history_df": pd.DataFrame(),
                "pipeline_df": pd.DataFrame(),
                "failure_statuses": [],
                "top_failure_orgs": [],
            }
            return state

        market = scope.get("market") or state.get("market")
        market_df = self.data.get_scoped_dataset("market", scope=scope, sort_by="PERIOD", ascending=True)
        pipeline_df = self.data.get_scoped_dataset("pipeline", scope=scope)
        metrics_market_df = market_df if market else self.data.aggregate_market_history(market_df)

        latest_metrics = {}
        if not metrics_market_df.empty:
            latest = metrics_market_df.iloc[-1]
            previous = metrics_market_df.iloc[-2] if len(metrics_market_df) > 1 else latest
            latest_metrics = {
                "market": latest.get("MARKET", market or "ALL"),
                "month": latest.get("MONTH"),
                "success_rate": float(latest.get("SCS_PERCENT", 0)),
                "previous_success_rate": float(previous.get("SCS_PERCENT", latest.get("SCS_PERCENT", 0))),
                "success_rate_delta": float(latest.get("SCS_PERCENT", 0)) - float(previous.get("SCS_PERCENT", latest.get("SCS_PERCENT", 0))),
                "overall_success_count": int(latest.get("OVERALL_SCS_CNT", 0)),
                "overall_failure_count": int(latest.get("OVERALL_FAIL_CNT", 0)),
                "first_iteration_success_count": int(latest.get("FIRST_ITER_SCS_CNT", 0)),
                "retry_success_count": int(latest.get("NEXT_ITER_SCS_CNT", 0)),
            }

        failure_statuses = []
        top_failure_orgs = []
        if not pipeline_df.empty:
            failed_df = pipeline_df[pd.to_numeric(pipeline_df.get("IS_FAILED", 0), errors="coerce").fillna(0).astype(int) == 1]
            if not failed_df.empty:
                failure_statuses = (
                    failed_df["FAILURE_STATUS"]
                    .fillna("Unknown")
                    .value_counts()
                    .head(5)
                    .reset_index()
                )
                failure_statuses.columns = ["status", "count"]
                top_failure_orgs = (
                    failed_df["ORG_NM"]
                    .fillna("Unknown")
                    .replace("", "Unknown")
                    .value_counts()
                    .head(5)
                    .reset_index()
                )
                top_failure_orgs.columns = ["org", "count"]

        state["record_quality"] = {
            "market_metrics": latest_metrics,
            "market_history_df": market_df,
            "aggregated_market_history_df": metrics_market_df,
            "pipeline_df": pipeline_df,
            "failure_statuses": failure_statuses.to_dict("records") if isinstance(failure_statuses, pd.DataFrame) else [],
            "top_failure_orgs": top_failure_orgs.to_dict("records") if isinstance(top_failure_orgs, pd.DataFrame) else [],
        }

        if latest_metrics:
            failure_rate = 100 - latest_metrics["success_rate"]
            market_label = latest_metrics["market"] if latest_metrics["market"] != "ALL" else "All tracked states"
            state["evidence"].append(
                f"{market_label} {latest_metrics['month']} failure rate is {failure_rate:.2f}% "
                f"({latest_metrics['success_rate_delta']:+.2f} point success-rate change vs previous month)."
            )

        if isinstance(failure_statuses, pd.DataFrame) and not failure_statuses.empty:
            top_status = failure_statuses.iloc[0]
            state["evidence"].append(
                f"Most common failure status is {top_status['status']} ({int(top_status['count'])} records)."
            )

        if isinstance(top_failure_orgs, pd.DataFrame) and not top_failure_orgs.empty:
            top_org = top_failure_orgs.iloc[0]
            state["evidence"].append(
                f"Top impacted organization is {top_org['org']} ({int(top_org['count'])} failed roster operations)."
            )

        return state
