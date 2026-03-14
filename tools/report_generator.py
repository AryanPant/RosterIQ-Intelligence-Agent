import pandas as pd

from tools.data_query_tool import DataQueryTool


class ReportGenerator:

    def generate(self, summary, state):

        scope = state.get("investigation_brief", {}).get("query_scope", {})
        if not scope.get("is_full_operational_report"):
            return ""

        pipeline_df = state.get("record_quality", {}).get("pipeline_df")
        market_df = state.get("record_quality", {}).get("market_history_df")
        summary_market_df = state.get("record_quality", {}).get("aggregated_market_history_df")
        if summary_market_df is None:
            summary_market_df = market_df
        duration_anomalies = state.get("pipeline_health", {}).get("duration_anomalies")
        stage_counts = state.get("pipeline_health", {}).get("stage_counts", [])
        root_cause = state.get("root_cause", {})

        lines = [
            "# Pipeline & Quality Health Report",
            "",
            "## Scope",
            f"- User request: {state.get('query', '')}",
            f"- State: {scope.get('market') or 'Not specified'}",
            f"- Organization: {scope.get('org_name') or 'All organizations in scope'}",
            f"- Time window: {self._time_window_label(scope.get('time_window'))}",
            f"- Procedures executed: {', '.join(state.get('plan', [])) or 'None'}",
            "",
            "## Executive Summary",
            summary or "No summary was generated.",
        ]

        if pipeline_df is None or pipeline_df.empty:
            lines.extend(
                [
                    "",
                    "## Summary Statistics",
                    "- No pipeline records matched the requested scope.",
                ]
            )
            if summary_market_df is None or summary_market_df.empty:
                lines.extend(["", "## Market SCS_PERCENT", "- No market trend data matched the requested scope."])
            return "\n".join(lines).strip()

        lines.extend(
            [
                "",
                "## Summary Statistics",
                f"- Total roster operations: {self._unique_count(pipeline_df, 'RO_ID')}",
                f"- Total processing runs: {len(pipeline_df)}",
                f"- Organizations covered: {self._unique_count(pipeline_df, 'ORG_NM')}",
                f"- Failed roster operations: {self._flag_count(pipeline_df, 'IS_FAILED')}",
                f"- Stuck roster operations: {self._flag_count(pipeline_df, 'IS_STUCK')}",
                f"- Average file success percent: {self._average_percent(pipeline_df, 'SCS_PCT')}",
            ]
        )

        lines.extend(
            [
                "",
                "## Flagged Roster Operations",
            ]
        )
        flagged_lines = self._build_flagged_ros(pipeline_df, duration_anomalies)
        lines.extend(flagged_lines or ["- No flagged roster operations were found in the current scope."])

        lines.extend(["", "## Stage Bottlenecks"])
        bottleneck_lines = self._build_stage_bottlenecks(stage_counts, duration_anomalies, root_cause)
        lines.extend(bottleneck_lines or ["- No stage bottlenecks were identified in the current scope."])

        lines.extend(["", "## Record Quality Breakdown"])
        lines.extend(self._build_record_quality_section(pipeline_df))

        lines.extend(["", "## Market SCS_PERCENT"])
        lines.extend(self._build_market_section(summary_market_df))

        lines.extend(["", "## Recommended Actions"])
        lines.extend(self._build_recommendations(pipeline_df, duration_anomalies, root_cause, summary_market_df))

        web_context = state.get("web_context", [])
        if web_context:
            lines.extend(["", "## External Context"])
            for item in web_context:
                detail = item.get("search_answer") or item.get("snippet", "")
                lines.append(
                    f"- [{item.get('category', 'external_context')}] {item.get('purpose', item.get('title', 'Untitled'))}: "
                    f"{detail} ({item.get('url', '')})"
                )

        return "\n".join(lines).strip()

    @staticmethod
    def _time_window_label(time_window):

        if not time_window:
            return "All available dates"
        return time_window.get("label") or "Custom window"

    @staticmethod
    def _unique_count(df, column):

        if column not in df.columns:
            return 0
        return int(df[column].dropna().astype(str).nunique())

    @staticmethod
    def _flag_count(df, column):

        if column not in df.columns:
            return 0
        flags = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
        if "RO_ID" in df.columns:
            return int(df.loc[flags == 1, "RO_ID"].astype(str).nunique())
        return int(flags.sum())

    @staticmethod
    def _average_percent(df, column):

        if column not in df.columns:
            return "N/A"
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            return "N/A"
        return f"{values.mean():.2f}%"

    def _build_flagged_ros(self, pipeline_df, duration_anomalies, limit=8):

        frame = pipeline_df.copy()
        flagged_rows = []

        if "FILE_REJECTION_RATE" not in frame.columns and {"TOT_REC_CNT", "REJ_REC_CNT"}.issubset(frame.columns):
            totals = pd.to_numeric(frame["TOT_REC_CNT"], errors="coerce").replace(0, pd.NA)
            frame["FILE_REJECTION_RATE"] = (
                pd.to_numeric(frame["REJ_REC_CNT"], errors="coerce") / totals
            ).fillna(0)

        frame["IS_FAILED_NUM"] = pd.to_numeric(frame.get("IS_FAILED", 0), errors="coerce").fillna(0).astype(int)
        frame["IS_STUCK_NUM"] = pd.to_numeric(frame.get("IS_STUCK", 0), errors="coerce").fillna(0).astype(int)
        frame["LATEST_STAGE_DURATION"] = frame.apply(self._latest_stage_duration, axis=1)
        if "FILE_REJECTION_RATE" in frame.columns:
            sort_columns = ["IS_STUCK_NUM", "IS_FAILED_NUM", "FILE_REJECTION_RATE", "LATEST_STAGE_DURATION"]
        else:
            sort_columns = ["IS_STUCK_NUM", "IS_FAILED_NUM", "LATEST_STAGE_DURATION"]
        frame = frame.sort_values(sort_columns, ascending=False).head(limit)

        for _, row in frame.iterrows():
            issues = []
            if int(row.get("IS_STUCK_NUM", 0)) == 1:
                issues.append("stuck")
            if int(row.get("IS_FAILED_NUM", 0)) == 1:
                issues.append("failed")
            rejection_rate = row.get("FILE_REJECTION_RATE")
            if pd.notna(rejection_rate) and float(rejection_rate) >= 0.30:
                issues.append(f"{float(rejection_rate) * 100:.1f}% rejected")
            latest_duration = row.get("LATEST_STAGE_DURATION")
            if pd.notna(latest_duration) and float(latest_duration) > 0:
                issues.append(f"{float(latest_duration):.1f} mins in current stage")
            issue_text = ", ".join(issues) if issues else "flagged by report scope"
            flagged_rows.append(
                f"- {row.get('RO_ID', 'Unknown RO')} | {row.get('ORG_NM', 'Unknown org')} | "
                f"Stage {row.get('LATEST_STAGE_NM', 'Unknown')} | {issue_text}"
            )

        if duration_anomalies is not None and not duration_anomalies.empty:
            top_anomaly = duration_anomalies.iloc[0]
            flagged_rows.append(
                f"- Top duration outlier: {top_anomaly.get('RO_ID', 'Unknown RO')} | "
                f"{top_anomaly.get('STAGE_NAME', top_anomaly.get('STAGE_DURATION_COLUMN', 'Unknown stage'))} at "
                f"{float(top_anomaly.get('ANOMALY_RATIO', 0)):.2f}x historical average"
            )

        return flagged_rows[: limit + 1]

    def _latest_stage_duration(self, row):

        stage_name = str(row.get("LATEST_STAGE_NM") or "").upper()
        duration_column = DataQueryTool.STAGE_METADATA.get(stage_name, {}).get("duration_column")
        if not duration_column:
            return pd.NA
        return pd.to_numeric(row.get(duration_column), errors="coerce")

    @staticmethod
    def _build_stage_bottlenecks(stage_counts, duration_anomalies, root_cause):

        lines = []
        for item in stage_counts[:5]:
            lines.append(f"- {item.get('stage', 'Unknown')}: {item.get('count', 0)} roster operations")

        if root_cause.get("primary_stage"):
            lines.append(f"- Primary concentration stage: {root_cause.get('primary_stage')}")

        if duration_anomalies is not None and not duration_anomalies.empty:
            grouped = (
                duration_anomalies.groupby("STAGE_NAME", dropna=False)["ANOMALY_RATIO"]
                .max()
                .sort_values(ascending=False)
                .head(3)
            )
            for stage_name, ratio in grouped.items():
                lines.append(f"- Duration bottleneck: {stage_name} peaks at {ratio:.2f}x historical average")
        return lines

    @staticmethod
    def _build_record_quality_section(pipeline_df):

        if not {"TOT_REC_CNT", "SCS_REC_CNT", "FAIL_REC_CNT", "SKIP_REC_CNT", "REJ_REC_CNT"}.issubset(pipeline_df.columns):
            return ["- Record count fields are unavailable for the current scope."]

        totals = {
            "Total records": pd.to_numeric(pipeline_df["TOT_REC_CNT"], errors="coerce").sum(),
            "Successful records": pd.to_numeric(pipeline_df["SCS_REC_CNT"], errors="coerce").sum(),
            "Failed records": pd.to_numeric(pipeline_df["FAIL_REC_CNT"], errors="coerce").sum(),
            "Skipped records": pd.to_numeric(pipeline_df["SKIP_REC_CNT"], errors="coerce").sum(),
            "Rejected records": pd.to_numeric(pipeline_df["REJ_REC_CNT"], errors="coerce").sum(),
        }
        total_records = totals["Total records"] or 0
        lines = [f"- {name}: {int(value)}" for name, value in totals.items()]
        if total_records > 0:
            lines.extend(
                [
                    f"- Success rate by records: {totals['Successful records'] / total_records * 100:.2f}%",
                    f"- Failure rate by records: {totals['Failed records'] / total_records * 100:.2f}%",
                    f"- Skip rate by records: {totals['Skipped records'] / total_records * 100:.2f}%",
                    f"- Rejection rate by records: {totals['Rejected records'] / total_records * 100:.2f}%",
                ]
            )
        return lines

    @staticmethod
    def _build_market_section(market_df):

        if market_df is None or market_df.empty:
            return ["- No market SCS_PERCENT data matched the requested scope."]

        latest = market_df.iloc[-1]
        previous = market_df.iloc[-2] if len(market_df) > 1 else latest
        current_percent = float(pd.to_numeric(latest.get("SCS_PERCENT"), errors="coerce") or 0)
        previous_percent = float(pd.to_numeric(previous.get("SCS_PERCENT"), errors="coerce") or 0)
        delta = current_percent - previous_percent
        return [
            f"- Latest market: {latest.get('MARKET', 'Unknown') if latest.get('MARKET') != 'ALL' else 'All tracked states'}",
            f"- Latest month: {latest.get('MONTH', 'Unknown')}",
            f"- SCS_PERCENT: {current_percent:.2f}%",
            f"- Change vs previous period: {delta:+.2f} points",
            f"- Overall success count: {int(pd.to_numeric(latest.get('OVERALL_SCS_CNT'), errors='coerce') or 0)}",
            f"- Overall failure count: {int(pd.to_numeric(latest.get('OVERALL_FAIL_CNT'), errors='coerce') or 0)}",
        ]

    def _build_recommendations(self, pipeline_df, duration_anomalies, root_cause, market_df):

        recommendations = []
        stuck_count = self._flag_count(pipeline_df, "IS_STUCK")
        failed_count = self._flag_count(pipeline_df, "IS_FAILED")
        rejection_rate = self._aggregate_rejection_rate(pipeline_df)

        if stuck_count:
            recommendations.append(
                f"- Prioritize stuck roster operations in {root_cause.get('primary_stage') or 'the current bottleneck stage'} and clear the longest-aging items first."
            )
        if duration_anomalies is not None and not duration_anomalies.empty:
            top_stage = duration_anomalies.iloc[0].get("STAGE_NAME", "the slowest stage")
            recommendations.append(
                f"- Review handoff rules, queue depth, and staffing around {top_stage} because it is the leading duration outlier."
            )
        if rejection_rate is not None and rejection_rate >= 0.20:
            recommendations.append(
                "- Audit input validation and source-file quality checks before processing because rejection volume is materially elevated."
            )
        if failed_count:
            recommendations.append(
                "- Triage the dominant failure statuses and re-run only the roster operations with actionable failure causes."
            )
        if market_df is not None and not market_df.empty:
            latest_percent = float(pd.to_numeric(market_df.iloc[-1].get("SCS_PERCENT"), errors="coerce") or 0)
            if latest_percent < 95:
                recommendations.append(
                    "- Escalate market-level remediation until SCS_PERCENT returns above the 95% operating threshold."
                )

        return recommendations or ["- Continue monitoring; no acute remediation signal was identified in the current scope."]

    @staticmethod
    def _aggregate_rejection_rate(pipeline_df):

        if not {"TOT_REC_CNT", "REJ_REC_CNT"}.issubset(pipeline_df.columns):
            return None
        total_records = pd.to_numeric(pipeline_df["TOT_REC_CNT"], errors="coerce").sum()
        rejected_records = pd.to_numeric(pipeline_df["REJ_REC_CNT"], errors="coerce").sum()
        if not total_records:
            return None
        return float(rejected_records / total_records)
