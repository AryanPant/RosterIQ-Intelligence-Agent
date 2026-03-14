import pandas as pd

from tools.data_query_tool import DataQueryTool


class RootCauseEngine:

    def __init__(self):

        self.data = DataQueryTool()

    def trace(self, market=None, scope=None):

        scope = scope or ({"market": market} if market else {})
        pipeline = self.data.get_scoped_dataset("pipeline", scope=scope)
        market_df = self.data.get_scoped_dataset("market", scope=scope, sort_by="PERIOD", ascending=True)
        metrics_market_df = market_df if market else self.data.aggregate_market_history(market_df)

        latest_metrics = {}
        if not metrics_market_df.empty:
            latest = metrics_market_df.iloc[-1]
            previous = metrics_market_df.iloc[-2] if len(metrics_market_df) > 1 else latest
            latest_metrics = {
                "market": latest.get("MARKET", market or "ALL"),
                "month": latest.get("MONTH"),
                "success_rate": float(latest.get("SCS_PERCENT", 0)),
                "success_rate_delta": float(latest.get("SCS_PERCENT", 0)) - float(previous.get("SCS_PERCENT", latest.get("SCS_PERCENT", 0))),
            }

        failed = pipeline[pd.to_numeric(pipeline.get("IS_FAILED", 0), errors="coerce").fillna(0).astype(int) == 1] if not pipeline.empty else pipeline
        stuck = pipeline[pd.to_numeric(pipeline.get("IS_STUCK", 0), errors="coerce").fillna(0).astype(int) == 1] if not pipeline.empty else pipeline

        top_failure_status = None
        if failed is not None and not failed.empty:
            status_counts = failed["FAILURE_STATUS"].fillna("Unknown").value_counts()
            top_failure_status = {"status": status_counts.index[0], "count": int(status_counts.iloc[0])}

        top_impacted_org = None
        if failed is not None and not failed.empty:
            org_counts = failed["ORG_NM"].fillna("Unknown").replace("", "Unknown").value_counts()
            top_impacted_org = {"org": org_counts.index[0], "count": int(org_counts.iloc[0])}

        primary_stage = None
        source_df = stuck if not stuck.empty else pipeline
        if source_df is not None and not source_df.empty:
            stage_counts = source_df["LATEST_STAGE_NM"].fillna("UNKNOWN").value_counts()
            primary_stage = stage_counts.index[0]

        return {
            "market": latest_metrics.get("market", market or "ALL"),
            "month": latest_metrics.get("month"),
            "success_rate": latest_metrics.get("success_rate"),
            "success_rate_delta": latest_metrics.get("success_rate_delta"),
            "primary_stage": primary_stage,
            "top_failure_status": top_failure_status,
            "top_impacted_org": top_impacted_org,
            "stuck_count": int(len(stuck)) if stuck is not None else 0,
        }
