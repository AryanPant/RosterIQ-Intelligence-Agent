import ast
import operator as op

import pandas as pd

from engines.anomaly_detector import AnomalyDetector
from memory.procedural_memory import ProceduralMemory
from tools.data_query_tool import DataQueryTool


class ProcedureRunner:

    _ALLOWED_OPERATORS = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
        ast.UAdd: op.pos,
    }

    def __init__(self):

        self.data = DataQueryTool()
        self.detector = AnomalyDetector()
        self.procedures = ProceduralMemory()

    def _safe_eval(self, expression, values):

        def _evaluate(node):
            if isinstance(node, ast.Expression):
                return _evaluate(node.body)
            if isinstance(node, ast.Constant):
                return float(node.value)
            if isinstance(node, ast.Num):
                return float(node.n)
            if isinstance(node, ast.Name):
                return float(values.get(node.id, 0))
            if isinstance(node, ast.BinOp) and type(node.op) in self._ALLOWED_OPERATORS:
                left = _evaluate(node.left)
                right = _evaluate(node.right)
                if isinstance(node.op, ast.Div) and right == 0:
                    return 0.0
                return self._ALLOWED_OPERATORS[type(node.op)](left, right)
            if isinstance(node, ast.UnaryOp) and type(node.op) in self._ALLOWED_OPERATORS:
                return self._ALLOWED_OPERATORS[type(node.op)](_evaluate(node.operand))
            raise ValueError("Unsupported expression.")

        parsed = ast.parse(expression, mode="eval")
        return float(_evaluate(parsed))

    def _evaluate_stored_function(self, function_text, values):

        if not function_text:
            return None

        metric_name = "computed_metric"
        expression = function_text.strip()
        if "=" in function_text:
            metric_name, expression = function_text.split("=", 1)
            metric_name = metric_name.strip() or "computed_metric"
            expression = expression.strip()

        if not expression:
            return None

        try:
            result = self._safe_eval(expression, values)
        except Exception:
            return None

        return {
            "metric_name": metric_name,
            "expression": expression,
            "value": result,
        }

    @staticmethod
    def _display_formula(function_text, evaluated_metric=None):

        if evaluated_metric and evaluated_metric.get("expression"):
            return evaluated_metric["expression"]
        if function_text and "=" in function_text:
            return function_text.split("=", 1)[1].strip()
        return function_text or ""

    @staticmethod
    def _scope_labels(scope):

        if not scope:
            return []

        labels = list(scope.get("labels", []))
        if labels:
            return labels

        derived = []
        if scope.get("market"):
            derived.append(str(scope["market"]))
        if scope.get("org_name"):
            derived.append(str(scope["org_name"]))
        for lob_term in scope.get("lob_terms", []) or []:
            derived.append(f"{lob_term} LOB")
        time_window = scope.get("time_window")
        if isinstance(time_window, dict) and time_window.get("label"):
            derived.append(time_window["label"])
        return derived

    @staticmethod
    def _split_lob_values(series):

        values = []
        for raw_value in series.fillna("").astype(str):
            parts = [part.strip() for part in raw_value.split(",") if part.strip()]
            values.extend(parts or ["Unknown"])
        return values

    def execute_defined_procedure(self, procedure_name, market=None, scope=None):

        return self._run_procedure(procedure_name, market, scope=scope)

    def _run_procedure(self, procedure_name, market, scope=None):

        scope = dict(scope or {})
        if market and not scope.get("market"):
            scope["market"] = market

        if procedure_name == "triage_stuck_ros":
            pipeline = self.data.get_scoped_dataset("pipeline", scope=scope)
            stuck = pipeline[pipeline["IS_STUCK"].fillna(0).astype(int) == 1]
            detail_columns = [
                "RO_ID",
                "ORG_NM",
                "LATEST_STAGE_NM",
                "IS_STUCK",
                "FAIL_REC_CNT",
                "REJ_REC_CNT",
                "SCS_PCT",
                "FAILURE_STATUS",
            ]
            available_columns = [column for column in detail_columns if column in stuck.columns]
            details = (
                stuck[available_columns].head(10).to_dict("records")
                if not stuck.empty
                else []
            )
            return {
                "procedure": procedure_name,
                "summary": f"Found {len(stuck)} stuck roster operations.",
                "details": {
                    "rows": details,
                    "scope_labels": self._scope_labels(scope),
                },
            }

        if procedure_name == "record_quality_audit":
            pipeline = self.data.get_scoped_dataset("pipeline", scope=scope)
            if pipeline.empty:
                return {"procedure": procedure_name, "summary": "No pipeline records found.", "details": {}}

            count_columns = ["TOT_REC_CNT", "SCS_REC_CNT", "FAIL_REC_CNT", "SKIP_REC_CNT", "REJ_REC_CNT"]
            available_count_columns = [column for column in count_columns if column in pipeline.columns]
            totals = pipeline[available_count_columns].apply(
                lambda series: pd.to_numeric(series, errors="coerce").fillna(0).sum()
            )
            numeric_totals = {key: float(value) for key, value in totals.to_dict().items()}
            total_records = float(numeric_totals.get("TOT_REC_CNT", 0)) or 0
            success_rate = ((float(numeric_totals.get("SCS_REC_CNT", 0)) / total_records) * 100) if total_records else 0
            procedure_definition = self.procedures.get(procedure_name)
            evaluated_metric = self._evaluate_stored_function(
                procedure_definition.get("function", ""),
                numeric_totals,
            )
            audit_threshold = 85.0
            file_count = int(len(pipeline))
            flagged_files = (
                pipeline[pipeline["SCS_PCT"].fillna(0).astype(float) < audit_threshold].copy()
                if "SCS_PCT" in pipeline.columns
                else pipeline.iloc[0:0].copy()
            )
            flagged_file_count = int(len(flagged_files))

            top_failure_status = None
            if "FAILURE_STATUS" in pipeline.columns:
                failure_counts = pipeline["FAILURE_STATUS"].fillna("Unknown").value_counts()
                if not failure_counts.empty:
                    top_failure_status = {
                        "status": str(failure_counts.index[0]),
                        "count": int(failure_counts.iloc[0]),
                    }

            top_impacted_org = None
            if flagged_file_count and "ORG_NM" in flagged_files.columns:
                org_counts = flagged_files["ORG_NM"].fillna("Unknown").replace("", "Unknown").value_counts()
                if not org_counts.empty:
                    top_impacted_org = {
                        "org": str(org_counts.index[0]),
                        "count": int(org_counts.iloc[0]),
                    }

            if evaluated_metric is not None:
                metric_percent = evaluated_metric["value"] * 100
                display_formula = self._display_formula(
                    procedure_definition.get("function", ""),
                    evaluated_metric=evaluated_metric,
                )
                summary = (
                    f"Record-quality issue rate is {metric_percent:.2f}% using "
                    f"`{display_formula}` across {file_count} files and {int(total_records)} records; "
                    f"{flagged_file_count} files are below the {audit_threshold:.0f}% SCS_PCT threshold."
                )
            else:
                summary = (
                    f"File-level success rate is {success_rate:.2f}% across {file_count} files and "
                    f"{int(total_records)} records; {flagged_file_count} files are below the {audit_threshold:.0f}% threshold."
                )

            return {
                "procedure": procedure_name,
                "summary": summary,
                "details": {
                    **totals.to_dict(),
                    "stored_function": procedure_definition.get("function", ""),
                    "evaluated_metric": evaluated_metric,
                    "success_rate_percent": success_rate,
                    "file_count": file_count,
                    "flagged_file_count": flagged_file_count,
                    "audit_threshold_percent": audit_threshold,
                    "scope_labels": self._scope_labels(scope),
                    "top_failure_status": top_failure_status,
                    "top_impacted_org": top_impacted_org,
                },
            }

        if procedure_name == "market_health_report":
            market_df = self.data.get_scoped_dataset("market", scope=scope, sort_by="PERIOD", ascending=True)
            metrics_market_df = market_df if market else self.data.aggregate_market_history(market_df)
            latest = metrics_market_df.iloc[-1].to_dict() if not metrics_market_df.empty else {}
            correlated = self.data.correlate_state_period_metrics()
            if scope.get("market") and not correlated.empty:
                correlated = correlated[
                    correlated["STATE"].astype(str).str.upper() == str(scope["market"]).upper()
                ]
            if scope.get("time_window") and not correlated.empty:
                correlated["MONTH_DT"] = pd.to_datetime(correlated["PERIOD"], format="%m-%Y", errors="coerce")
                correlated = correlated[
                    correlated["MONTH_DT"].dt.normalize().between(
                        scope["time_window"]["start"].replace(day=1),
                        scope["time_window"]["end"].replace(day=1),
                    )
                ]
            anomalies = self.detector.detect(market)
            market_label = latest.get("MARKET", market or "ALL")
            if market_label == "ALL":
                market_label = "All tracked states"
            return {
                "procedure": procedure_name,
                "summary": (
                    f"{market_label} {latest.get('MONTH', 'latest')} success rate is "
                    f"{float(latest.get('SCS_PERCENT', 0)):.2f}%."
                ),
                "details": {
                    "correlated_rows": correlated.head(10).to_dict("records") if not correlated.empty else [],
                    "anomalies": anomalies,
                    "scope_labels": self._scope_labels(scope),
                },
            }

        if procedure_name == "retry_effectiveness_analysis":
            retry = self.data.analyze_retry_quality(scope=scope)
            retry_run_count = int(retry["retry_run_count"].sum()) if not retry.empty else 0
            avg_change = float(retry["rejection_rate_change"].mean()) if not retry.empty else 0
            return {
                "procedure": procedure_name,
                "summary": f"Observed {retry_run_count} retry runs with average rejection-rate change of {avg_change:+.4f}.",
                "details": {
                    "rows": retry.head(10).to_dict("records") if not retry.empty else [],
                    "scope_labels": self._scope_labels(scope),
                },
            }

        if procedure_name == "lob_rejection_breakdown":
            pipeline = self.data.get_scoped_dataset("pipeline", scope=scope)
            if pipeline.empty or "LOB" not in pipeline.columns:
                return {
                    "procedure": procedure_name,
                    "summary": "No Line of Business data matched the requested scope.",
                    "details": {"rows": [], "scope_labels": self._scope_labels(scope)},
                }

            exploded = pipeline.copy()
            exploded["LOB_ITEM"] = exploded["LOB"].fillna("").astype(str).apply(
                lambda value: [item.strip() for item in value.split(",") if item.strip()] or ["Unknown"]
            )
            exploded = exploded.explode("LOB_ITEM")
            grouped = (
                exploded.groupby("LOB_ITEM", dropna=False)
                .agg(
                    total_records=("TOT_REC_CNT", lambda series: pd.to_numeric(series, errors="coerce").sum()),
                    rejected_records=("REJ_REC_CNT", lambda series: pd.to_numeric(series, errors="coerce").sum()),
                )
                .reset_index()
            )
            grouped["rejection_rate"] = (
                grouped["rejected_records"] / grouped["total_records"].replace(0, pd.NA)
            ).fillna(0)
            grouped = grouped.sort_values("rejection_rate", ascending=False).reset_index(drop=True)
            top_row = grouped.iloc[0] if not grouped.empty else None
            summary = "No Line of Business data matched the requested scope."
            if top_row is not None:
                summary = (
                    f"Highest rejection rate is {top_row['rejection_rate'] * 100:.2f}% "
                    f"for {top_row['LOB_ITEM']} across {int(top_row['total_records'])} records."
                )
            return {
                "procedure": procedure_name,
                "summary": summary,
                "details": {
                    "rows": grouped.head(10).to_dict("records"),
                    "scope_labels": self._scope_labels(scope),
                    "stored_function": self.procedures.get(procedure_name).get("function", ""),
                },
            }

        return {
            "procedure": procedure_name,
            "summary": "Procedure is defined but has no executor yet.",
            "details": {"scope_labels": self._scope_labels(scope)},
        }

    def run(self, procedures, state):

        if state.get("investigation_brief", {}).get("is_procedure_update"):
            state["procedure_results"] = []
            return state

        market = state.get("market")
        scope = state.get("investigation_brief", {}).get("query_scope", {})
        results = [self._run_procedure(procedure, market, scope=scope) for procedure in procedures]
        state["procedure_results"] = results

        for result in results:
            state["evidence"].append(result["summary"])

        return state
