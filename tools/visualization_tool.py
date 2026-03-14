import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from tools.data_query_tool import DataQueryTool


class VisualizationTool:

    @staticmethod
    def _title_suffix(market=None, stages=None):

        parts = []
        if market:
            parts.append(str(market).upper())
        if stages:
            parts.append(", ".join(stages))
        return f" ({' | '.join(parts)})" if parts else ""

    def pipeline_stage_health_heatmap(self, pipeline_df, stages=None, health_flags=None, max_orgs=25, market=None):

        if pipeline_df is None or pipeline_df.empty:
            return None

        rows = []
        stage_names = stages or list(DataQueryTool.STAGE_METADATA.keys())
        for stage_name in stage_names:
            health_column = DataQueryTool.STAGE_METADATA.get(stage_name, {}).get("health_column")
            if health_column not in pipeline_df.columns:
                continue
            frame = pipeline_df[["ORG_NM", health_column]].copy()
            frame["STAGE"] = stage_name
            frame["HEALTH_FLAG"] = frame[health_column].fillna("UNKNOWN").astype(str).str.upper()
            rows.append(frame[["ORG_NM", "STAGE", "HEALTH_FLAG"]])

        if not rows:
            return None

        long_df = pd.concat(rows, ignore_index=True)
        if health_flags:
            long_df = long_df[long_df["HEALTH_FLAG"].isin(health_flags)]
        if long_df.empty:
            return None

        score_map = {"GREEN": 1, "YELLOW": 2, "RED": 3, "UNKNOWN": 0}
        long_df["HEALTH_SCORE"] = long_df["HEALTH_FLAG"].map(score_map).fillna(0)

        org_order = (
            long_df.groupby("ORG_NM")["HEALTH_SCORE"]
            .max()
            .sort_values(ascending=False)
            .head(max_orgs)
            .index
        )
        filtered = long_df[long_df["ORG_NM"].isin(org_order)]
        pivot = (
            filtered.pivot_table(
                index="ORG_NM",
                columns="STAGE",
                values="HEALTH_SCORE",
                aggfunc="max",
                fill_value=0,
            )
            .reindex(org_order)
        )
        if pivot.empty:
            return None

        figure = px.imshow(
            pivot,
            color_continuous_scale=[
                [0.0, "#cbd5e1"],
                [0.33, "#22c55e"],
                [0.66, "#facc15"],
                [1.0, "#ef4444"],
            ],
            aspect="auto",
            title=f"Pipeline Stage Health Heatmap{self._title_suffix(market=market, stages=stages)}",
            labels={"x": "Stage", "y": "Organization", "color": "Health"},
        )
        figure.update_coloraxes(
            colorbar=dict(
                tickvals=[0, 1, 2, 3],
                ticktext=["Unknown", "Green", "Yellow", "Red"],
            )
        )
        figure.update_traces(
            customdata=filtered.pivot_table(
                index="ORG_NM",
                columns="STAGE",
                values="HEALTH_FLAG",
                aggfunc="first",
                fill_value="UNKNOWN",
            ).reindex(org_order),
            hovertemplate="Organization=%{y}<br>Stage=%{x}<br>Health=%{z}<extra></extra>",
        )
        return figure

    def record_quality_breakdown(self, pipeline_df, ratio_columns=None, max_files=20, market=None):

        if pipeline_df is None or pipeline_df.empty:
            return None

        required = ["RO_ID", "TOT_REC_CNT", "SCS_REC_CNT", "FAIL_REC_CNT", "SKIP_REC_CNT", "REJ_REC_CNT"]
        if not all(column in pipeline_df.columns for column in required):
            return None

        frame = pipeline_df[required].copy()
        frame["TOT_REC_CNT"] = pd.to_numeric(frame["TOT_REC_CNT"], errors="coerce").replace(0, pd.NA)
        ratio_columns = ratio_columns or list(DataQueryTool.RATIO_METADATA.keys())
        ratio_columns = [column for column in ratio_columns if column in DataQueryTool.RATIO_METADATA]
        if not ratio_columns:
            return None

        percent_rows = []
        for source_column in ratio_columns:
            label = DataQueryTool.RATIO_METADATA[source_column]["label"]
            temp = frame[["RO_ID", "TOT_REC_CNT", source_column]].copy()
            temp["CATEGORY"] = label
            temp["PERCENT"] = (pd.to_numeric(temp[source_column], errors="coerce") / temp["TOT_REC_CNT"] * 100).fillna(0)
            percent_rows.append(temp[["RO_ID", "CATEGORY", "PERCENT"]])

        long_df = pd.concat(percent_rows, ignore_index=True)
        sort_label = DataQueryTool.RATIO_METADATA[ratio_columns[-1]]["label"]
        file_order = (
            long_df[long_df["CATEGORY"] == sort_label]
            .sort_values("PERCENT", ascending=False)["RO_ID"]
            .head(max_files)
            .tolist()
        )
        filtered = long_df[long_df["RO_ID"].isin(file_order)]
        if filtered.empty:
            return None

        figure = px.bar(
            filtered,
            x="RO_ID",
            y="PERCENT",
            color="CATEGORY",
            title=f"Record Quality Breakdown by File{self._title_suffix(market=market)}",
            labels={"RO_ID": "Roster Operation", "PERCENT": "Percent of Total Records"},
            category_orders={"RO_ID": file_order},
            color_discrete_map={
                "Success %": "#22c55e",
                "Fail %": "#f97316",
                "Skip %": "#94a3b8",
                "Reject %": "#ef4444",
            },
        )
        figure.update_layout(barmode="stack")
        return figure

    def duration_anomaly_chart(self, anomaly_df, stages=None, max_rows=25, market=None):

        if anomaly_df is None or anomaly_df.empty:
            return None

        frame = anomaly_df.copy()
        if stages and "STAGE_NAME" in frame.columns:
            frame = frame[frame["STAGE_NAME"].isin(stages)]
        if frame.empty:
            return None

        frame = frame.sort_values("ANOMALY_RATIO", ascending=False).head(max_rows).copy()
        frame["LABEL"] = frame["RO_ID"].astype(str) + " | " + frame["STAGE_NAME"].astype(str)
        actual_colors = frame["ANOMALY_RATIO"].apply(lambda ratio: "#ef4444" if ratio >= 3 else "#f59e0b")

        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=frame["LABEL"],
                y=frame["STAGE_DURATION_VALUE"],
                name="Actual Duration",
                marker_color=actual_colors,
                text=frame["ANOMALY_RATIO"].map(lambda ratio: f"{ratio:.1f}x"),
                textposition="outside",
                hovertemplate=(
                    "RO=%{x}<br>Actual=%{y:.2f} mins<br>"
                    "Avg=%{customdata[0]:.2f} mins<br>"
                    "Health=%{customdata[1]}<extra></extra>"
                ),
                customdata=frame[["HISTORICAL_AVG_DURATION", "STAGE_HEALTH_FLAG"]],
            )
        )
        figure.add_trace(
            go.Bar(
                x=frame["LABEL"],
                y=frame["HISTORICAL_AVG_DURATION"],
                name="Historical Average",
                marker_color="#94a3b8",
            )
        )
        figure.update_layout(
            title=f"Duration Anomaly Chart{self._title_suffix(market=market, stages=stages)}",
            xaxis_title="RO and Stage",
            yaxis_title="Minutes",
            barmode="group",
        )
        return figure

    def market_scs_percent_trend(self, market_df, market=None):

        if market_df is None or market_df.empty:
            return None

        frame = market_df.copy()
        if market and "MARKET" in frame.columns:
            frame = frame[frame["MARKET"].astype(str).str.upper() == market.upper()]
        if frame.empty:
            return None

        if "MONTH_DT" not in frame.columns and "MONTH" in frame.columns:
            frame["MONTH_DT"] = pd.to_datetime(frame["MONTH"], format="%m-%Y", errors="coerce")
        frame["SCS_PERCENT"] = pd.to_numeric(frame["SCS_PERCENT"], errors="coerce")
        frame = frame.sort_values(["MARKET", "MONTH_DT"])

        figure = px.line(
            frame,
            x="MONTH",
            y="SCS_PERCENT",
            color="MARKET",
            markers=True,
            title=f"Market SCS_PERCENT Trend{self._title_suffix(market=market)}",
            labels={"MONTH": "Month", "SCS_PERCENT": "Success Rate (%)", "MARKET": "Market"},
        )
        figure.add_hline(
            y=95,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="95% Threshold",
            annotation_position="top left",
        )
        return figure

    def retry_lift_chart(self, market_df, market=None):

        if market_df is None or market_df.empty:
            return None

        frame = market_df.copy()
        if market and "MARKET" in frame.columns:
            frame = frame[frame["MARKET"].astype(str).str.upper() == market.upper()]
        if frame.empty:
            return None

        plot_df = frame[["MARKET", "FIRST_ITER_SCS_CNT", "NEXT_ITER_SCS_CNT"]].copy()
        plot_df = plot_df.groupby("MARKET", dropna=False).sum().reset_index()
        plot_df["RECOVERED_BY_RETRY"] = plot_df["NEXT_ITER_SCS_CNT"]
        plot_df = plot_df.melt(
            id_vars="MARKET",
            value_vars=["FIRST_ITER_SCS_CNT", "NEXT_ITER_SCS_CNT"],
            var_name="SUCCESS_STAGE",
            value_name="SUCCESS_COUNT",
        )
        plot_df["SUCCESS_STAGE"] = plot_df["SUCCESS_STAGE"].replace(
            {
                "FIRST_ITER_SCS_CNT": "First Iteration Success",
                "NEXT_ITER_SCS_CNT": "Post-Retry Success",
            }
        )

        figure = px.bar(
            plot_df,
            x="MARKET",
            y="SUCCESS_COUNT",
            color="SUCCESS_STAGE",
            barmode="group",
            title=f"Retry Lift Chart{self._title_suffix(market=market)}",
            labels={"MARKET": "Market", "SUCCESS_COUNT": "Successful Transactions"},
            color_discrete_map={
                "First Iteration Success": "#22c55e",
                "Post-Retry Success": "#3b82f6",
            },
        )
        recovery_notes = frame.groupby("MARKET", dropna=False)["NEXT_ITER_SCS_CNT"].sum()
        for market_name, recovered_count in recovery_notes.items():
            figure.add_annotation(
                x=market_name,
                y=float(recovered_count),
                text=f"+{int(recovered_count)} recovered",
                showarrow=False,
                yshift=24,
                font=dict(color="#1d4ed8"),
            )
        return figure

    def stuck_ro_tracker(self, pipeline_df, stages=None, health_flags=None, max_rows=30, market=None):

        if pipeline_df is None or pipeline_df.empty:
            return None

        frame = pipeline_df.copy()
        if "IS_STUCK" not in frame.columns:
            return None

        stuck = frame[pd.to_numeric(frame["IS_STUCK"], errors="coerce").fillna(0).astype(int) == 1].copy()
        if stuck.empty:
            return None

        duration_columns = [
            column
            for column in stuck.columns
            if column.endswith("_DURATION") and not column.startswith("AVG_")
        ]
        if not duration_columns:
            return None

        stage_duration_map = {
            stage_name: metadata["duration_column"]
            for stage_name, metadata in DataQueryTool.STAGE_METADATA.items()
            if metadata["duration_column"] in stuck.columns
        }
        stage_health_map = {
            stage_name: metadata["health_column"]
            for stage_name, metadata in DataQueryTool.STAGE_METADATA.items()
            if metadata["health_column"] in stuck.columns
        }

        def duration_for_latest_stage(row):
            stage_name = str(row.get("LATEST_STAGE_NM") or "").upper()
            duration_column = stage_duration_map.get(stage_name)
            if not duration_column:
                return 0
            return pd.to_numeric(row.get(duration_column), errors="coerce")

        def health_for_latest_stage(row):
            stage_name = str(row.get("LATEST_STAGE_NM") or "").upper()
            health_column = stage_health_map.get(stage_name)
            if not health_column:
                return "UNKNOWN"
            return str(row.get(health_column) or "UNKNOWN").upper()

        stuck["TIME_IN_STAGE"] = stuck.apply(duration_for_latest_stage, axis=1).fillna(0)
        stuck["HEALTH_OVERLAY"] = stuck.apply(health_for_latest_stage, axis=1)
        if stages:
            stuck = stuck[stuck["LATEST_STAGE_NM"].isin(stages)]
        if health_flags:
            stuck = stuck[stuck["HEALTH_OVERLAY"].isin(health_flags)]
        if stuck.empty:
            return None
        stuck = stuck.sort_values("TIME_IN_STAGE", ascending=False).head(max_rows)

        return px.bar(
            stuck,
            x="RO_ID",
            y="TIME_IN_STAGE",
            color="HEALTH_OVERLAY",
            hover_data=["ORG_NM", "CNT_STATE", "LATEST_STAGE_NM"],
            title=f"Stuck RO Tracker{self._title_suffix(market=market, stages=stages)}",
            labels={"RO_ID": "Roster Operation", "TIME_IN_STAGE": "Time in Stage (mins)"},
            color_discrete_map={"GREEN": "#22c55e", "YELLOW": "#facc15", "RED": "#ef4444", "UNKNOWN": "#94a3b8"},
        )
