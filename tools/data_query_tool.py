import os
import re

import pandas as pd


class DataQueryTool:

    STATE_NAME_TO_CODE = {
        "alabama": "AL",
        "alaska": "AK",
        "arizona": "AZ",
        "arkansas": "AR",
        "california": "CA",
        "colorado": "CO",
        "connecticut": "CT",
        "delaware": "DE",
        "district of columbia": "DC",
        "washington dc": "DC",
        "florida": "FL",
        "georgia": "GA",
        "hawaii": "HI",
        "idaho": "ID",
        "illinois": "IL",
        "indiana": "IN",
        "iowa": "IA",
        "kansas": "KS",
        "kentucky": "KY",
        "louisiana": "LA",
        "maine": "ME",
        "maryland": "MD",
        "massachusetts": "MA",
        "michigan": "MI",
        "minnesota": "MN",
        "mississippi": "MS",
        "missouri": "MO",
        "montana": "MT",
        "nebraska": "NE",
        "nevada": "NV",
        "new hampshire": "NH",
        "new jersey": "NJ",
        "new mexico": "NM",
        "new york": "NY",
        "north carolina": "NC",
        "north dakota": "ND",
        "ohio": "OH",
        "oklahoma": "OK",
        "oregon": "OR",
        "pennsylvania": "PA",
        "rhode island": "RI",
        "south carolina": "SC",
        "south dakota": "SD",
        "tennessee": "TN",
        "texas": "TX",
        "utah": "UT",
        "vermont": "VT",
        "virginia": "VA",
        "washington": "WA",
        "west virginia": "WV",
        "wisconsin": "WI",
        "wyoming": "WY",
    }

    STAGE_METADATA = {
        "PRE_PROCESSING": {
            "aliases": ("pre processing", "pre-processing", "preprocessing"),
            "duration_column": "PRE_PROCESSING_DURATION",
            "health_column": "PRE_PROCESSING_HEALTH",
        },
        "MAPPING_APPROVAL": {
            "aliases": ("mapping approval", "mapping aproval"),
            "duration_column": "MAPPING_APROVAL_DURATION",
            "health_column": "MAPPING_APROVAL_HEALTH",
        },
        "ISF_GENERATION": {
            "aliases": ("isf generation", "isf gen", "isf"),
            "duration_column": "ISF_GEN_DURATION",
            "health_column": "ISF_GEN_HEALTH",
        },
        "DART_GENERATION": {
            "aliases": ("dart generation", "dart gen"),
            "duration_column": "DART_GEN_DURATION",
            "health_column": "DART_GEN_HEALTH",
        },
        "DART_REVIEW": {
            "aliases": ("dart review",),
            "duration_column": "DART_REVIEW_DURATION",
            "health_column": "DART_REVIEW_HEALTH",
        },
        "DART_UI_VALIDATION": {
            "aliases": ("dart ui validation", "ui validation"),
            "duration_column": "DART_UI_VALIDATION_DURATION",
            "health_column": "DART_UI_VALIDATION_HEALTH",
        },
        "SPS_LOAD": {
            "aliases": ("sps load",),
            "duration_column": "SPS_LOAD_DURATION",
            "health_column": "SPS_LOAD_HEALTH",
        },
    }

    RATIO_METADATA = {
        "SCS_REC_CNT": {
            "label": "Success %",
            "aliases": ("success ratio", "success rate", "success percent", "success percentage", "scs", "scs percent"),
        },
        "FAIL_REC_CNT": {
            "label": "Fail %",
            "aliases": ("fail ratio", "fail rate", "failure ratio", "failure rate", "fail percent", "failure percent"),
        },
        "SKIP_REC_CNT": {
            "label": "Skip %",
            "aliases": ("skip ratio", "skip rate", "skip percent", "skipped records"),
        },
        "REJ_REC_CNT": {
            "label": "Reject %",
            "aliases": ("reject ratio", "rejection ratio", "reject rate", "rejection rate", "reject percent", "rejection percent"),
        },
    }
    LOB_KEYWORDS = {
        "MEDICAID FFS": ("medicaid ffs",),
        "MEDICAID HMO": ("medicaid hmo",),
        "MEDICAID": ("medicaid",),
        "MEDICARE HMO": ("medicare hmo",),
        "MEDICARE PPO": ("medicare ppo",),
        "MEDICARE": ("medicare",),
        "COMMERCIAL HMO": ("commercial hmo",),
        "COMMERCIAL PPO/EPO": ("commercial ppo epo", "commercial ppo", "ppo epo"),
        "COMMERCIAL": ("commercial",),
        "HMO": ("hmo",),
        "PPO": ("ppo",),
    }

    def __init__(self):

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pipeline_path = os.getenv("ROSTERIQ_ROSTER_PATH") or os.path.join(
            base_dir, "data", "roster_processing_details.csv"
        )
        market_path = os.getenv("ROSTERIQ_AGGREGATED_PATH") or os.path.join(
            base_dir, "data", "aggregated_operational_metrics.csv"
        )

        self.pipeline = pd.read_csv(pipeline_path)
        self.market = pd.read_csv(market_path)
        self.pipeline.columns = self.pipeline.columns.str.strip()
        self.market.columns = self.market.columns.str.strip()
        self.market_codes = set(self.market["MARKET"].dropna().astype(str).str.upper().unique())
        org_names = (
            self.pipeline["ORG_NM"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()
            if "ORG_NM" in self.pipeline.columns
            else []
        )
        self.org_name_lookup = {
            self._normalize_text(org_name): org_name
            for org_name in org_names
            if self._normalize_text(org_name)
        }
        self.org_name_candidates = sorted(self.org_name_lookup.keys(), key=len, reverse=True)
        if {"ORG_NM", "CNT_STATE"}.issubset(self.pipeline.columns):
            org_market = (
                self.pipeline[["ORG_NM", "CNT_STATE"]]
                .dropna()
                .assign(ORG_NM=lambda frame: frame["ORG_NM"].astype(str).str.strip())
                .groupby("ORG_NM", dropna=False)["CNT_STATE"]
                .agg(lambda values: values.mode().iat[0] if not values.mode().empty else values.iloc[0])
            )
            self.org_market_lookup = org_market.to_dict()
        else:
            self.org_market_lookup = {}

    @staticmethod
    def _normalize_text(value):

        return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()

    @staticmethod
    def _dedupe(values):

        return list(dict.fromkeys(values))

    @classmethod
    def _contains_phrase(cls, normalized_text, phrase):

        normalized_phrase = cls._normalize_text(phrase)
        if not normalized_phrase:
            return False
        return re.search(rf"\b{re.escape(normalized_phrase)}\b", normalized_text) is not None

    def infer_market_from_text(self, query):

        raw_tokens = re.findall(r"[A-Za-z]{2,3}", str(query or ""))
        for token in raw_tokens:
            candidate = token.upper()
            if token.isupper() and candidate in self.market_codes:
                return candidate

        normalized = self._normalize_text(query)
        for state_name, state_code in sorted(self.STATE_NAME_TO_CODE.items(), key=lambda item: len(item[0]), reverse=True):
            if state_code in self.market_codes and self._contains_phrase(normalized, state_name):
                return state_code

        stopwords = {"IN", "ON", "BY", "OF", "TO", "FOR", "AT", "AND", "ME", "OR"}
        for token in normalized.split():
            candidate = token.upper()
            if candidate in stopwords:
                continue
            if candidate in self.market_codes:
                return candidate
        return None

    def infer_org_from_text(self, query):

        normalized = self._normalize_text(query)
        for normalized_org in self.org_name_candidates:
            if len(normalized_org) < 5:
                continue
            if self._contains_phrase(normalized, normalized_org):
                return self.org_name_lookup[normalized_org]

        stopwords = {
            "who",
            "is",
            "look",
            "up",
            "lookup",
            "research",
            "about",
            "this",
            "that",
            "the",
            "to",
            "add",
            "business",
            "context",
            "organization",
            "org",
            "pipeline",
            "anomaly",
            "anomalies",
            "market",
            "state",
            "general",
            "problem",
            "problems",
        }
        filtered_tokens = [token for token in normalized.split() if token not in stopwords]
        if any(
            phrase in normalized
            for phrase in ["all states", "across all states", "all markets", "across markets", "overall pipeline"]
        ):
            return None

        context_cues = {"who", "lookup", "look", "research", "background", "context"}
        org_name_markers = {
            "medical",
            "group",
            "foundation",
            "physicians",
            "physician",
            "hospital",
            "clinic",
            "system",
            "associates",
            "care",
        }
        has_context_cue = any(token in context_cues for token in normalized.split())
        org_marker_count = sum(1 for token in filtered_tokens if token in org_name_markers)
        if not has_context_cue and org_marker_count < 2:
            return None

        for window_size in range(min(5, len(filtered_tokens)), 1, -1):
            for start in range(0, len(filtered_tokens) - window_size + 1):
                phrase = " ".join(filtered_tokens[start : start + window_size]).strip()
                if len(phrase) < 8:
                    continue
                for normalized_org in self.org_name_candidates:
                    if phrase in normalized_org:
                        return self.org_name_lookup[normalized_org]
        return None

    @staticmethod
    def _month_window(year, month):

        start = pd.Timestamp(year=int(year), month=int(month), day=1)
        end = start + pd.offsets.MonthEnd(1)
        return start.normalize(), end.normalize()

    def _extract_time_window(self, query):

        text = str(query or "")
        normalized = self._normalize_text(text)
        month_map = {
            "jan": 1,
            "january": 1,
            "feb": 2,
            "february": 2,
            "mar": 3,
            "march": 3,
            "apr": 4,
            "april": 4,
            "may": 5,
            "jun": 6,
            "june": 6,
            "jul": 7,
            "july": 7,
            "aug": 8,
            "august": 8,
            "sep": 9,
            "sept": 9,
            "september": 9,
            "oct": 10,
            "october": 10,
            "nov": 11,
            "november": 11,
            "dec": 12,
            "december": 12,
        }

        iso_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
        if len(iso_dates) >= 2:
            start = pd.to_datetime(iso_dates[0], errors="coerce")
            end = pd.to_datetime(iso_dates[1], errors="coerce")
            if pd.notna(start) and pd.notna(end):
                start, end = sorted([start.normalize(), end.normalize()])
                return {"start": start, "end": end, "label": f"{start.date()} to {end.date()}"}
        if len(iso_dates) == 1:
            date_value = pd.to_datetime(iso_dates[0], errors="coerce")
            if pd.notna(date_value):
                date_value = date_value.normalize()
                return {"start": date_value, "end": date_value, "label": str(date_value.date())}

        month_matches = [
            (match.start(), int(match.group(2)), month_map[match.group(1).lower()])
            for match in re.finditer(
                r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{4})\b",
                text,
                flags=re.IGNORECASE,
            )
        ]
        month_matches.extend(
            (match.start(), int(match.group(2)), int(match.group(1)))
            for match in re.finditer(r"\b(0?[1-9]|1[0-2])[-/](\d{4})\b", text)
        )
        month_matches = sorted(month_matches, key=lambda item: item[0])
        if len(month_matches) >= 2 and any(
            self._contains_phrase(normalized, connector)
            for connector in ("from", "between", "through", "thru", "to")
        ):
            _, start_year, start_month = month_matches[0]
            _, end_year, end_month = month_matches[1]
            start, _ = self._month_window(start_year, start_month)
            _, end = self._month_window(end_year, end_month)
            start, end = sorted([start, end])
            return {
                "start": start,
                "end": end,
                "label": f"{start.strftime('%b %Y')} to {end.strftime('%b %Y')}",
            }
        if month_matches:
            _, year, month = month_matches[0]
            start, end = self._month_window(year, month)
            return {"start": start, "end": end, "label": start.strftime("%b %Y")}

        if self._contains_phrase(normalized, "last 30 days"):
            end = pd.Timestamp.today().normalize()
            start = end - pd.Timedelta(days=29)
            return {"start": start, "end": end, "label": "Last 30 days"}
        if self._contains_phrase(normalized, "last 7 days"):
            end = pd.Timestamp.today().normalize()
            start = end - pd.Timedelta(days=6)
            return {"start": start, "end": end, "label": "Last 7 days"}
        if self._contains_phrase(normalized, "last month"):
            today = pd.Timestamp.today().normalize()
            last_month = today - pd.offsets.MonthBegin(1)
            start, end = self._month_window(last_month.year, last_month.month)
            return {"start": start, "end": end, "label": start.strftime("%b %Y")}

        return None

    def extract_query_keywords(self, query):

        normalized = self._normalize_text(query)
        market = self.infer_market_from_text(query)
        org_name = self.infer_org_from_text(query)
        time_window = self._extract_time_window(query)

        lob_terms = []
        for canonical, aliases in self.LOB_KEYWORDS.items():
            if any(self._contains_phrase(normalized, alias) for alias in aliases):
                lob_terms.append(canonical)

        regulatory_terms = []
        for term in [
            "cms",
            "medicaid",
            "medicare",
            "regulation",
            "regulatory",
            "rule",
            "policy",
            "payer",
            "compliance",
            "provider directory",
            "provider roster",
            "roster compliance",
            "network adequacy",
            "submission requirement",
            "submission requirements",
            "data standard",
            "validation",
            "complete validation failure",
            "incompatible",
        ]:
            if self._contains_phrase(normalized, term):
                regulatory_terms.append(term)

        keywords = self._dedupe(
            [
                market,
                org_name,
                *(lob_terms or []),
                *(regulatory_terms or []),
            ]
        )

        return {
            "market": market,
            "org_name": org_name,
            "lob_terms": self._dedupe(lob_terms),
            "regulatory_terms": self._dedupe(regulatory_terms),
            "time_window": time_window,
            "keywords": keywords,
        }

    def apply_scope_filters(self, df, dataset_name, scope=None):

        frame = self._with_period_columns(df.copy(), dataset_name)
        scope = scope or {}
        market = scope.get("market")
        org_name = scope.get("org_name")
        lob_terms = scope.get("lob_terms") or []
        time_window = scope.get("time_window")

        if dataset_name == "pipeline":
            if market and "CNT_STATE" in frame.columns:
                frame = frame[frame["CNT_STATE"].astype(str).str.upper() == str(market).upper()]
            if org_name and "ORG_NM" in frame.columns:
                frame = frame[frame["ORG_NM"].astype(str).str.strip().str.upper() == str(org_name).strip().upper()]
            if lob_terms and "LOB" in frame.columns:
                lob_pattern = "|".join(re.escape(str(term).upper()) for term in lob_terms if term)
                if lob_pattern:
                    frame = frame[
                        frame["LOB"].astype(str).str.upper().str.contains(lob_pattern, na=False, regex=True)
                    ]
            if time_window and "FILE_RECEIVED_DT" in frame.columns:
                frame["FILE_RECEIVED_DT"] = pd.to_datetime(frame["FILE_RECEIVED_DT"], errors="coerce")
                frame = frame[
                    frame["FILE_RECEIVED_DT"].dt.normalize().between(time_window["start"], time_window["end"])
                ]
        elif dataset_name == "market":
            if market and "MARKET" in frame.columns:
                frame = frame[frame["MARKET"].astype(str).str.upper() == str(market).upper()]
            if time_window:
                if "MONTH_DT" not in frame.columns and "MONTH" in frame.columns:
                    frame["MONTH_DT"] = pd.to_datetime(frame["MONTH"], format="%m-%Y", errors="coerce")
                if "MONTH_DT" in frame.columns:
                    frame = frame[
                        frame["MONTH_DT"].dt.normalize().between(
                            time_window["start"].replace(day=1),
                            time_window["end"].replace(day=1),
                        )
                    ]
        return frame.reset_index(drop=True)

    def get_scoped_dataset(self, dataset_name, scope=None, sort_by=None, ascending=False):

        df = self._get_dataset(dataset_name)
        df = self.apply_scope_filters(df, dataset_name, scope=scope)
        if sort_by:
            sort_columns = [sort_by] if isinstance(sort_by, str) else list(sort_by)
            available = [column for column in sort_columns if column in df.columns]
            if available:
                df = df.sort_values(by=available, ascending=ascending)
        return df.reset_index(drop=True)

    def extract_visualization_scope(self, query, requested_charts=None, market=None):

        normalized = self._normalize_text(query)
        requested_charts = list(requested_charts or [])
        extracted_keywords = self.extract_query_keywords(query)
        org_name = extracted_keywords["org_name"]
        resolved_market = market or extracted_keywords["market"] or self.org_market_lookup.get(org_name)
        time_window = extracted_keywords["time_window"]
        explicit_chart_terms = []
        report_requested = any(
            self._contains_phrase(normalized, phrase)
            for phrase in (
                "full operational report",
                "operational report",
                "pipeline quality health report",
                "pipeline and quality health report",
                "full report",
            )
        )

        if any(self._contains_phrase(normalized, term) for term in ("heatmap", "health heatmap", "pipeline health", "stage health")):
            explicit_chart_terms.append("pipeline_stage_health_heatmap")
        if self._contains_phrase(normalized, "stuck"):
            explicit_chart_terms.append("stuck_ro_tracker")
        if any(self._contains_phrase(normalized, term) for term in ("duration", "outlier", "anomaly", "slow", "delay", "aging")):
            explicit_chart_terms.append("duration_anomaly_chart")
        if any(
            self._contains_phrase(normalized, term)
            for term in (
                "quality breakdown",
                "breakdown",
                "ratio",
                "percent of total",
                "record quality",
                "rejection",
                "reject",
                "skip",
            )
        ):
            explicit_chart_terms.append("record_quality_breakdown")
        if any(
            self._contains_phrase(normalized, term)
            for term in ("trend", "history", "historical", "monthly", "scs percent", "success rate", "threshold")
        ):
            explicit_chart_terms.append("market_scs_percent_trend")
        if any(self._contains_phrase(normalized, term) for term in ("retry", "rerun", "re-run", "iteration", "recover")):
            explicit_chart_terms.append("retry_lift_chart")

        if explicit_chart_terms:
            chart_preferences = explicit_chart_terms
        elif report_requested:
            chart_preferences = []
        elif requested_charts:
            chart_preferences = requested_charts
            if self._contains_phrase(normalized, "stuck"):
                chart_preferences = [chart for chart in chart_preferences if chart == "stuck_ro_tracker"] or ["stuck_ro_tracker"]
            elif any(self._contains_phrase(normalized, term) for term in ("duration", "outlier", "anomaly", "slow", "delay", "aging")):
                chart_preferences = [chart for chart in chart_preferences if chart == "duration_anomaly_chart"] or ["duration_anomaly_chart"]
            elif any(self._contains_phrase(normalized, term) for term in ("heatmap", "health heatmap", "pipeline health", "stage health")):
                chart_preferences = [chart for chart in chart_preferences if chart == "pipeline_stage_health_heatmap"] or ["pipeline_stage_health_heatmap"]
            elif any(self._contains_phrase(normalized, term) for term in ("quality breakdown", "breakdown", "ratio", "record quality", "reject", "skip")):
                chart_preferences = [chart for chart in chart_preferences if chart == "record_quality_breakdown"] or ["record_quality_breakdown"]
            elif any(
                self._contains_phrase(normalized, term)
                for term in ("trend", "history", "historical", "monthly", "scs percent", "success rate", "threshold")
            ):
                chart_preferences = [chart for chart in chart_preferences if chart == "market_scs_percent_trend"] or ["market_scs_percent_trend"]
            elif any(self._contains_phrase(normalized, term) for term in ("retry", "rerun", "re-run", "iteration", "recover")):
                chart_preferences = [chart for chart in chart_preferences if chart == "retry_lift_chart"] or ["retry_lift_chart"]
        else:
            chart_preferences = []

        stages = []
        for stage_name, metadata in self.STAGE_METADATA.items():
            if any(self._contains_phrase(normalized, alias) for alias in metadata["aliases"]):
                stages.append(stage_name)

        ratio_columns = []
        if any(self._contains_phrase(normalized, term) for term in ("ratio", "breakdown", "record quality", "quality")):
            for column, metadata in self.RATIO_METADATA.items():
                if any(self._contains_phrase(normalized, alias) for alias in metadata["aliases"]):
                    ratio_columns.append(column)
            if not ratio_columns and "record_quality_breakdown" in chart_preferences:
                ratio_columns = list(self.RATIO_METADATA.keys())

        health_flags = [flag for flag in ("GREEN", "YELLOW", "RED") if flag.lower() in normalized]

        labels = []
        if resolved_market:
            labels.append(str(resolved_market))
        if org_name:
            labels.append(str(org_name))
        for lob_term in extracted_keywords["lob_terms"]:
            labels.append(f"{lob_term} LOB")
        if time_window:
            labels.append(time_window.get("label"))

        max_items = None
        count_match = re.search(r"\btop\s+(\d{1,3})\b", normalized)
        if count_match:
            max_items = int(count_match.group(1))
        else:
            count_match = re.search(r"\b(\d{1,3})\s+(?:ros|rosters|files|organizations|orgs|markets)\b", normalized)
            if count_match:
                max_items = int(count_match.group(1))

        return {
            "market": resolved_market,
            "org_name": org_name,
            "lob_terms": extracted_keywords["lob_terms"],
            "regulatory_terms": extracted_keywords["regulatory_terms"],
            "query_keywords": self._dedupe(
                [
                    value
                    for value in [
                        resolved_market,
                        org_name,
                        *(extracted_keywords["lob_terms"] or []),
                        *(extracted_keywords["regulatory_terms"] or []),
                    ]
                    if value
                ]
            ),
            "labels": self._dedupe([label for label in labels if label]),
            "time_window": time_window,
            "is_full_operational_report": report_requested,
            "requested_charts": self._dedupe(chart_preferences),
            "stages": self._dedupe(stages),
            "ratio_columns": self._dedupe(ratio_columns),
            "health_flags": self._dedupe(health_flags),
            "max_items": max_items,
        }

    def _get_dataset(self, dataset_name):

        if dataset_name == "pipeline":
            return self.pipeline.copy()
        if dataset_name == "market":
            return self.market.copy()
        raise ValueError("dataset_name must be 'pipeline' or 'market'")

    def _with_period_columns(self, df, dataset_name):

        frame = df.copy()
        if dataset_name == "pipeline" and "FILE_RECEIVED_DT" in frame.columns:
            frame["FILE_RECEIVED_DT"] = pd.to_datetime(frame["FILE_RECEIVED_DT"], errors="coerce")
            frame["PERIOD"] = frame["FILE_RECEIVED_DT"].dt.strftime("%m-%Y")
        elif dataset_name == "market" and "MONTH" in frame.columns:
            frame["PERIOD"] = frame["MONTH"].astype(str)
            frame["MONTH_DT"] = pd.to_datetime(frame["MONTH"], format="%m-%Y", errors="coerce")
        return frame

    def aggregate_market_history(self, market_df):

        frame = self._with_period_columns(market_df.copy(), "market")
        if frame.empty:
            return frame

        grouped = (
            frame.groupby(["PERIOD", "MONTH_DT"], dropna=False)
            .agg(
                OVERALL_SCS_CNT=("OVERALL_SCS_CNT", lambda series: pd.to_numeric(series, errors="coerce").sum()),
                OVERALL_FAIL_CNT=("OVERALL_FAIL_CNT", lambda series: pd.to_numeric(series, errors="coerce").sum()),
                FIRST_ITER_SCS_CNT=("FIRST_ITER_SCS_CNT", lambda series: pd.to_numeric(series, errors="coerce").sum()),
                NEXT_ITER_SCS_CNT=("NEXT_ITER_SCS_CNT", lambda series: pd.to_numeric(series, errors="coerce").sum()),
            )
            .reset_index()
            .sort_values("MONTH_DT")
        )
        total_processed = grouped["OVERALL_SCS_CNT"] + grouped["OVERALL_FAIL_CNT"]
        grouped["SCS_PERCENT"] = (grouped["OVERALL_SCS_CNT"] / total_processed.replace(0, pd.NA) * 100).fillna(0)
        grouped["MARKET"] = "ALL"
        grouped["MONTH"] = grouped["PERIOD"]
        return grouped.reset_index(drop=True)

    def _file_rejection_metric(self, df):

        frame = df.copy()
        if {"REJ_REC_CNT", "TOT_REC_CNT"}.issubset(frame.columns):
            total = pd.to_numeric(frame["TOT_REC_CNT"], errors="coerce").replace(0, pd.NA)
            rejected = pd.to_numeric(frame["REJ_REC_CNT"], errors="coerce")
            frame["FILE_REJECTION_RATE"] = (rejected / total).fillna(0)
            frame["REJECTION_RATE_SOURCE"] = "record_counts"
        else:
            failed = pd.to_numeric(frame.get("IS_FAILED", 0), errors="coerce").fillna(0)
            frame["FILE_REJECTION_RATE"] = failed
            frame["REJECTION_RATE_SOURCE"] = "is_failed_proxy"
        return frame

    def filter_sort_aggregate(
        self,
        dataset_name,
        filters=None,
        sort_by=None,
        ascending=False,
        group_by=None,
        aggregations=None,
    ):

        df = self._get_dataset(dataset_name)
        df = self._with_period_columns(df, dataset_name)

        for column, value in (filters or {}).items():
            if column not in df.columns:
                continue
            if isinstance(value, (list, tuple, set)):
                df = df[df[column].isin(list(value))]
            else:
                df = df[df[column] == value]

        if group_by and aggregations:
            df = df.groupby(group_by, dropna=False).agg(aggregations).reset_index()

        if sort_by:
            sort_columns = [sort_by] if isinstance(sort_by, str) else list(sort_by)
            available = [column for column in sort_columns if column in df.columns]
            if available:
                df = df.sort_values(by=available, ascending=ascending)

        return df.reset_index(drop=True)

    def correlate_state_period_metrics(self):

        pipeline = self._with_period_columns(self.pipeline, "pipeline")
        pipeline = self._file_rejection_metric(pipeline)

        pipeline_state_period = (
            pipeline.groupby(["CNT_STATE", "PERIOD"], dropna=False)
            .agg(
                file_count=("RO_ID", "count"),
                total_records=("TOT_REC_CNT", lambda series: pd.to_numeric(series, errors="coerce").sum()),
                rejected_records=("REJ_REC_CNT", lambda series: pd.to_numeric(series, errors="coerce").sum()),
                average_file_rejection_rate=("FILE_REJECTION_RATE", "mean"),
                failed_file_count=("IS_FAILED", "sum"),
            )
            .reset_index()
            .rename(columns={"CNT_STATE": "STATE"})
        )
        pipeline_state_period["state_period_rejection_rate"] = (
            pipeline_state_period["rejected_records"]
            / pipeline_state_period["total_records"].replace(0, pd.NA)
        ).fillna(0)

        market = self._with_period_columns(self.market, "market")
        market_state_period = market.rename(columns={"MARKET": "STATE"})

        joined = pipeline_state_period.merge(
            market_state_period[
                ["STATE", "PERIOD", "SCS_PERCENT", "OVERALL_FAIL_CNT", "OVERALL_SCS_CNT"]
            ],
            on=["STATE", "PERIOD"],
            how="inner",
        )

        if not joined.empty:
            joined["failure_rate_from_market"] = 100 - pd.to_numeric(joined["SCS_PERCENT"], errors="coerce")

        return joined.sort_values(["STATE", "PERIOD"]).reset_index(drop=True)

    def detect_rejection_rate_anomalies(self, threshold=0.30, filters=None, scope=None):

        pipeline = self._file_rejection_metric(
            self.apply_scope_filters(self._get_dataset("pipeline"), "pipeline", scope=scope)
        )
        for column, value in (filters or {}).items():
            if column not in pipeline.columns:
                continue
            if isinstance(value, (list, tuple, set)):
                pipeline = pipeline[pipeline[column].isin(list(value))]
            else:
                pipeline = pipeline[pipeline[column] == value]

        anomalies = pipeline[pipeline["FILE_REJECTION_RATE"] > threshold].copy()
        anomalies["threshold"] = threshold

        columns = [
            column
            for column in [
                "RO_ID",
                "ORG_NM",
                "CNT_STATE",
                "RUN_NO",
                "FILE_REJECTION_RATE",
                "REJECTION_RATE_SOURCE",
                "threshold",
                "LATEST_STAGE_NM",
                "FAILURE_STATUS",
            ]
            if column in anomalies.columns
        ]
        return anomalies[columns].sort_values("FILE_REJECTION_RATE", ascending=False).reset_index(drop=True)

    def detect_stage_duration_anomalies(self, multiplier=2.0, filters=None, stages=None, scope=None):

        frame = self.apply_scope_filters(self._get_dataset("pipeline"), "pipeline", scope=scope)
        for column, value in (filters or {}).items():
            if column not in frame.columns:
                continue
            if isinstance(value, (list, tuple, set)):
                frame = frame[frame[column].isin(list(value))]
            else:
                frame = frame[frame[column] == value]

        duration_columns = [
            metadata["duration_column"]
            for stage_name, metadata in self.STAGE_METADATA.items()
            if metadata["duration_column"] in frame.columns and (not stages or stage_name in stages)
        ]
        if not duration_columns:
            return pd.DataFrame()

        numeric = frame[duration_columns].apply(pd.to_numeric, errors="coerce")
        historical_average = numeric.mean().dropna()

        anomaly_rows = []
        for duration_column, avg_value in historical_average.items():
            if avg_value <= 0:
                continue

            stage_name = next(
                (
                    name
                    for name, metadata in self.STAGE_METADATA.items()
                    if metadata["duration_column"] == duration_column
                ),
                duration_column.replace("_DURATION", ""),
            )
            health_column = self.STAGE_METADATA.get(stage_name, {}).get("health_column")
            flagged = frame[numeric[duration_column] > (multiplier * avg_value)].copy()
            if flagged.empty:
                continue

            if health_column and health_column in flagged.columns:
                flagged["STAGE_HEALTH_FLAG"] = flagged[health_column].fillna("UNKNOWN").astype(str).str.upper()
            else:
                flagged["STAGE_HEALTH_FLAG"] = "UNKNOWN"
            flagged["STAGE_NAME"] = stage_name
            flagged["STAGE_DURATION_COLUMN"] = duration_column
            flagged["STAGE_DURATION_VALUE"] = pd.to_numeric(flagged[duration_column], errors="coerce")
            flagged["HISTORICAL_AVG_DURATION"] = avg_value
            flagged["ANOMALY_RATIO"] = flagged["STAGE_DURATION_VALUE"] / avg_value
            flagged["OUTLIER_SEVERITY"] = flagged["ANOMALY_RATIO"].apply(
                lambda ratio: "Severe" if ratio >= 3 else "Moderate"
            )
            anomaly_rows.append(
                flagged[
                    [
                        "RO_ID",
                        "ORG_NM",
                        "CNT_STATE",
                        "RUN_NO",
                        "LATEST_STAGE_NM",
                        "STAGE_NAME",
                        "STAGE_DURATION_COLUMN",
                        "STAGE_DURATION_VALUE",
                        "HISTORICAL_AVG_DURATION",
                        "ANOMALY_RATIO",
                        "STAGE_HEALTH_FLAG",
                        "OUTLIER_SEVERITY",
                    ]
                ]
            )

        if not anomaly_rows:
            return pd.DataFrame()

        return (
            pd.concat(anomaly_rows, ignore_index=True)
            .sort_values("ANOMALY_RATIO", ascending=False)
            .reset_index(drop=True)
        )

    def analyze_retry_quality(self, scope=None):

        pipeline = self._file_rejection_metric(
            self.apply_scope_filters(self._get_dataset("pipeline"), "pipeline", scope=scope)
        )
        numeric_columns = [
            column
            for column in ["TOT_REC_CNT", "SCS_REC_CNT", "FAIL_REC_CNT", "SKIP_REC_CNT", "REJ_REC_CNT"]
            if column in pipeline.columns
        ]
        grouped = (
            pipeline.groupby(["RO_ID", "RUN_NO"], dropna=False)
            .agg(
                org_name=("ORG_NM", "first"),
                state=("CNT_STATE", "first"),
                file_rejection_rate=("FILE_REJECTION_RATE", "mean"),
                failed_file_count=("IS_FAILED", "sum"),
                stuck_file_count=("IS_STUCK", "sum"),
                latest_stage=("LATEST_STAGE_NM", "last"),
                **{
                    column.lower(): (column, lambda series, source_column=column: pd.to_numeric(series, errors="coerce").sum())
                    for column in numeric_columns
                },
            )
            .reset_index()
        )

        first_run = grouped[grouped["RUN_NO"] == 1].copy()
        retry_runs = grouped[grouped["RUN_NO"] > 1].copy()

        if first_run.empty or retry_runs.empty:
            return pd.DataFrame()

        retry_summary = (
            retry_runs.groupby("RO_ID", dropna=False)
            .agg(
                retry_run_count=("RUN_NO", "count"),
                retry_rejection_rate=("file_rejection_rate", "mean"),
                retry_failed_file_count=("failed_file_count", "sum"),
                retry_stuck_file_count=("stuck_file_count", "sum"),
                **{
                    f"retry_{column.lower()}": (column.lower(), "sum")
                    for column in numeric_columns
                },
            )
            .reset_index()
        )

        comparison = first_run.merge(retry_summary, on="RO_ID", how="inner")
        comparison = comparison.rename(
            columns={
                "file_rejection_rate": "run_1_rejection_rate",
                "failed_file_count": "run_1_failed_file_count",
                "stuck_file_count": "run_1_stuck_file_count",
                "latest_stage": "run_1_latest_stage",
                **{
                    column.lower(): f"run_1_{column.lower()}"
                    for column in numeric_columns
                },
            }
        )
        comparison["rejection_rate_change"] = (
            comparison["retry_rejection_rate"] - comparison["run_1_rejection_rate"]
        )
        if {"run_1_tot_rec_cnt", "run_1_scs_rec_cnt"}.issubset(comparison.columns):
            comparison["run_1_success_rate"] = (
                comparison["run_1_scs_rec_cnt"]
                / comparison["run_1_tot_rec_cnt"].replace(0, pd.NA)
            ).fillna(0)
        if {"retry_tot_rec_cnt", "retry_scs_rec_cnt"}.issubset(comparison.columns):
            comparison["retry_success_rate"] = (
                comparison["retry_scs_rec_cnt"]
                / comparison["retry_tot_rec_cnt"].replace(0, pd.NA)
            ).fillna(0)
        if {"run_1_rej_rec_cnt", "run_1_tot_rec_cnt"}.issubset(comparison.columns):
            comparison["run_1_record_rejection_rate"] = (
                comparison["run_1_rej_rec_cnt"]
                / comparison["run_1_tot_rec_cnt"].replace(0, pd.NA)
            ).fillna(0)
        if {"retry_rej_rec_cnt", "retry_tot_rec_cnt"}.issubset(comparison.columns):
            comparison["retry_record_rejection_rate"] = (
                comparison["retry_rej_rec_cnt"]
                / comparison["retry_tot_rec_cnt"].replace(0, pd.NA)
            ).fillna(0)
        if {"run_1_success_rate", "retry_success_rate"}.issubset(comparison.columns):
            comparison["success_rate_change"] = (
                comparison["retry_success_rate"] - comparison["run_1_success_rate"]
            )
        if {"run_1_record_rejection_rate", "retry_record_rejection_rate"}.issubset(comparison.columns):
            comparison["record_rejection_rate_change"] = (
                comparison["retry_record_rejection_rate"] - comparison["run_1_record_rejection_rate"]
            )

        sort_column = (
            "record_rejection_rate_change"
            if "record_rejection_rate_change" in comparison.columns
            else "rejection_rate_change"
        )
        return comparison.sort_values(sort_column, ascending=False).reset_index(drop=True)
