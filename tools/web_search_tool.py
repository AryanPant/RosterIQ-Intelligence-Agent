import os
import re

from dotenv import load_dotenv

load_dotenv()


class WebSearchTool:

    MAX_SNIPPET_LENGTH = 280
    REGULATORY_DOMAINS = ["cms.gov", "medicaid.gov"]
    COMPLIANCE_DOMAINS = ["cms.gov", "medicaid.gov", "ecfr.gov"]
    STATE_DOMAIN_HINTS = {
        "VA": ["dmmas.virginia.gov", "virginia.gov", "law.lis.virginia.gov"],
        "KS": ["kancare.ks.gov", "kdhe.ks.gov", "kmap-state-ks.us", "kslegislature.gov"],
    }
    ORG_DOMAIN_HINTS = {
        "Cedars-Sinai Medical Care Foundation": ["cedars-sinai.org"],
        "MercyOne Medical Group": ["mercyone.org"],
    }
    REGULATORY_TERMS = (
        "cms",
        "medicaid",
        "medicare",
        "regulation",
        "regulatory",
        "rule",
        "policy",
        "payer",
        "provider directory",
        "provider roster",
        "network adequacy",
        "directory accuracy",
        "roster compliance",
        "submission requirement",
        "submission requirements",
    )
    VALIDATION_TERMS = (
        "validation",
        "complete validation failure",
        "data standard",
        "schema",
        "incompatible",
        "directory",
        "provider directory",
        "roster compliance",
    )
    LOB_TERMS = (
        "lob",
        "line of business",
        "medicaid",
        "medicare",
        "commercial",
        "ffs",
        "hmo",
        "ppo",
        "submission",
        "requirement",
        "requirements",
    )
    ORG_CONTEXT_TERMS = (
        "business context",
        "background",
        "who is",
        "organization",
        "medical group",
        "foundation",
        "health system",
        "provider org",
        "provider organization",
    )
    ANOMALY_TERMS = (
        "anomaly",
        "anomalies",
        "spike",
        "surge",
        "jump",
        "drop",
        "decline",
        "rejection",
        "reject",
        "failure",
        "pipeline",
        "stuck",
    )

    def __init__(self):

        self.api_key = os.getenv("TAVILY_API_KEY")
        self.provider = (os.getenv("ROSTERIQ_WEB_PROVIDER") or "tavily").strip().lower()
        self.disabled = os.getenv("ROSTERIQ_DISABLE_WEB_SEARCH", "").strip().lower() in {"1", "true", "yes"}

    @staticmethod
    def _contains_any(text, phrases):

        normalized = str(text or "").lower()
        return any(phrase in normalized for phrase in phrases)

    def _truncate_snippet(self, text):

        cleaned = re.sub(r"\s+", " ", (text or "")).strip()
        if len(cleaned) <= self.MAX_SNIPPET_LENGTH:
            return cleaned

        truncated = cleaned[: self.MAX_SNIPPET_LENGTH + 1]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]

        return truncated.rstrip(" .,;:") + "..."

    @staticmethod
    def _dedupe_text(values):

        return [value for value in dict.fromkeys(value for value in values if value)]

    @staticmethod
    def _safe_int(value, default=0):

        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(value, default=0.0):

        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _state_domains(self, market):

        return self._dedupe_text(
            [
                *self.STATE_DOMAIN_HINTS.get(str(market or "").upper(), []),
                *self.REGULATORY_DOMAINS,
            ]
        )

    def _offline_result(self, category, title, url, snippet, purpose, query):

        return {
            "category": category,
            "title": title,
            "url": url,
            "snippet": self._truncate_snippet(snippet),
            "purpose": purpose,
            "query": query,
            "source": "offline-fallback",
        }

    def _offline_fallback(self, search_plan):

        results = []
        for item in search_plan:
            query = item["query"]
            purpose = item["purpose"]
            category = item["category"]
            market = item.get("market")
            org_name = item.get("org_name")
            failure_status = item.get("failure_status")
            lob_name = item.get("lob")

            if category == "regulatory_change":
                results.append(
                    self._offline_result(
                        category,
                        f"{market or 'Market'} regulatory context",
                        f"https://{self._state_domains(market)[0]}" if self._state_domains(market) else "https://www.cms.gov/",
                        "CMS and state Medicaid policy changes are the first place to check when a market shows a rejection spike or a success-rate drop tied to roster validation and submission rules.",
                        purpose,
                        query,
                    )
                )
            elif category == "compliance_standard":
                label = failure_status or "Complete Validation Failure"
                results.append(
                    self._offline_result(
                        category,
                        f"{label} compliance context",
                        "https://www.cms.gov/",
                        "Validation-heavy failure labels usually point to provider-directory, enrollment, or file-schema checks rather than a pure queueing or retry problem.",
                        purpose,
                        query,
                    )
                )
            elif category == "lob_policy":
                results.append(
                    self._offline_result(
                        category,
                        f"{market or 'Market'} {lob_name or 'LOB'} submission context",
                        f"https://{self._state_domains(market)[0]}" if self._state_domains(market) else "https://www.medicaid.gov/",
                        "LOB-specific provider roster rules often differ for Medicaid FFS, Medicaid managed care, Medicare, and commercial submission streams, so payer and state guidance should be checked separately.",
                        purpose,
                        query,
                    )
                )
            elif category == "org_context":
                org_domains = self.ORG_DOMAIN_HINTS.get(org_name or "", [])
                org_url = f"https://{org_domains[0]}" if org_domains else "local://rosteriq/org-context"
                results.append(
                    self._offline_result(
                        category,
                        org_name or "Provider organization context",
                        org_url,
                        "Provider-organization context can help determine whether an anomaly is tied to a large physician foundation, regional medical group, or multi-market health-system network.",
                        purpose,
                        query,
                    )
                )
        return results

    def _search_live(self, search_item, max_results):

        if self.provider not in {"", "tavily"}:
            return []
        if not self.api_key:
            return []

        try:
            response = None
            try:
                from tavily import TavilyClient
            except ImportError:
                TavilyClient = None

            if TavilyClient is not None:
                client = TavilyClient(api_key=self.api_key)
                response = client.search(
                    query=search_item["query"],
                    max_results=max_results,
                    include_domains=search_item.get("domains") or None,
                    search_depth=search_item.get("search_depth", "advanced"),
                    topic=search_item.get("topic", "general"),
                    time_range=search_item.get("time_range"),
                    include_answer="basic",
                    auto_parameters=False,
                    timeout=45,
                )
            else:
                import requests

                http_response = requests.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.api_key,
                        "query": search_item["query"],
                        "max_results": max_results,
                        "include_domains": search_item.get("domains") or None,
                        "search_depth": search_item.get("search_depth", "advanced"),
                        "topic": search_item.get("topic", "general"),
                        "time_range": search_item.get("time_range"),
                        "include_answer": "basic",
                        "auto_parameters": False,
                    },
                    timeout=45,
                )
                http_response.raise_for_status()
                response = http_response.json()

            answer = self._truncate_snippet(response.get("answer", ""))
            rows = []
            for item in response.get("results", []):
                rows.append(
                    {
                        "title": item.get("title", "Untitled"),
                        "url": item.get("url", ""),
                        "snippet": self._truncate_snippet(item.get("content", "") or answer),
                        "search_answer": answer,
                        "source": "tavily",
                    }
                )
            if not rows and answer:
                rows.append(
                    {
                        "title": search_item.get("purpose", "External context"),
                        "url": "",
                        "snippet": answer,
                        "search_answer": answer,
                        "source": "tavily",
                    }
                )
            return rows
        except Exception:
            return []

    def _extract_lob_terms(self, state):

        scope = state.get("investigation_brief", {}).get("query_scope", {})
        if scope.get("lob_terms"):
            return self._dedupe_text(scope.get("lob_terms", []))

        pipeline_df = state.get("record_quality", {}).get("pipeline_df")
        if pipeline_df is None or pipeline_df.empty or "LOB" not in pipeline_df.columns:
            return []

        lob_counts = {}
        for raw_value in pipeline_df["LOB"].dropna().astype(str):
            for item in [piece.strip() for piece in raw_value.split(",") if piece.strip()]:
                normalized = item.upper()
                lob_counts[normalized] = lob_counts.get(normalized, 0) + 1

        ordered = sorted(lob_counts.items(), key=lambda item: item[1], reverse=True)
        return [item for item, _ in ordered[:3]]

    def _pipeline_rejection_rate(self, state):

        pipeline_df = state.get("record_quality", {}).get("pipeline_df")
        if pipeline_df is None or pipeline_df.empty:
            return 0.0
        if not {"TOT_REC_CNT", "REJ_REC_CNT"}.issubset(pipeline_df.columns):
            return 0.0

        total_records = 0.0
        rejected_records = 0.0
        for _, row in pipeline_df.iterrows():
            total_records += self._safe_float(row.get("TOT_REC_CNT"))
            rejected_records += self._safe_float(row.get("REJ_REC_CNT"))
        if total_records <= 0:
            return 0.0
        return rejected_records / total_records

    def _failure_status_signals(self, state):

        root_cause = state.get("root_cause", {})
        top_failure = root_cause.get("top_failure_status") or {}
        failure_statuses = state.get("record_quality", {}).get("failure_statuses", [])
        total = sum(self._safe_int(item.get("count")) for item in failure_statuses)
        top_count = self._safe_int(top_failure.get("count"))
        ratio = (top_count / total) if total and top_count else 0.0
        return {
            "status": top_failure.get("status") or "",
            "count": top_count,
            "ratio": ratio,
        }

    def _collect_signals(self, state):

        query = state.get("query") or ""
        normalized_query = query.lower()
        brief = state.get("investigation_brief", {})
        scope = brief.get("query_scope", {})
        metrics = state.get("record_quality", {}).get("market_metrics", {})
        failure_signal = self._failure_status_signals(state)
        org_candidates = self._dedupe_text(
            [
                scope.get("org_name"),
                (state.get("root_cause", {}).get("top_impacted_org") or {}).get("org"),
            ]
        )
        lob_terms = self._extract_lob_terms(state)
        success_delta = self._safe_float(metrics.get("success_rate_delta"))
        rejection_rate = self._pipeline_rejection_rate(state)
        external_intent = "external_context" in set(brief.get("intents", []))
        explicit_regulatory_request = self._contains_any(normalized_query, self.REGULATORY_TERMS)
        explicit_validation_request = self._contains_any(normalized_query, self.VALIDATION_TERMS)
        explicit_lob_request = self._contains_any(normalized_query, self.LOB_TERMS) and self._contains_any(
            normalized_query,
            ("submission", "requirement", "requirements", "policy", "payer", "regulatory", "medicaid", "medicare"),
        )
        explicit_org_request = self._contains_any(normalized_query, self.ORG_CONTEXT_TERMS) or (
            bool(scope.get("org_name")) and self._contains_any(normalized_query, self.ANOMALY_TERMS)
        )
        rejection_spike_request = (
            bool(scope.get("market") or state.get("market"))
            and self._contains_any(normalized_query, ("spike", "surge", "jump", "drop", "decline"))
            and self._contains_any(normalized_query, ("rejection", "reject", "failure", "validation"))
        )
        validation_issue = self._contains_any(failure_signal["status"].lower(), ("validation", "incompatible")) and (
            failure_signal["count"] > 0
        )
        market_pressure = (
            bool(scope.get("market") or state.get("market"))
            and (success_delta <= -0.5 or rejection_rate >= 0.10 or rejection_spike_request)
        )
        medicaid_lobs = [lob for lob in lob_terms if "MEDICAID" in str(lob).upper()]
        return {
            "query": query,
            "normalized_query": normalized_query,
            "market": scope.get("market") or state.get("market"),
            "query_keywords": scope.get("query_keywords", []),
            "regulatory_terms": scope.get("regulatory_terms", []),
            "lob_terms": lob_terms,
            "medicaid_lobs": medicaid_lobs,
            "org_candidates": org_candidates,
            "failure_status": failure_signal["status"],
            "failure_ratio": failure_signal["ratio"],
            "success_delta": success_delta,
            "rejection_rate": rejection_rate,
            "external_intent": external_intent,
            "explicit_regulatory_request": explicit_regulatory_request,
            "explicit_validation_request": explicit_validation_request,
            "explicit_lob_request": explicit_lob_request,
            "explicit_org_request": explicit_org_request,
            "rejection_spike_request": rejection_spike_request,
            "validation_issue": validation_issue,
            "market_pressure": market_pressure,
        }

    def build_search_plan(self, state, max_queries=4):

        signals = self._collect_signals(state)
        market = signals["market"]
        failure_status = signals["failure_status"]
        lob_terms = signals["lob_terms"]
        named_org = bool(state.get("investigation_brief", {}).get("query_scope", {}).get("org_name"))
        query_keywords = self._dedupe_text(signals["query_keywords"] + signals["regulatory_terms"])
        plan = []

        regulatory_needed = market and (
            signals["explicit_regulatory_request"]
            or signals["rejection_spike_request"]
            or (signals["external_intent"] and signals["market_pressure"])
        )
        if regulatory_needed:
            plan.append(
                {
                    "category": "regulatory_change",
                    "purpose": f"Check CMS or {market} Medicaid rule changes that could explain rejection spikes or success-rate movement.",
                    "query": " ".join(
                        self._dedupe_text(
                            [
                                market,
                                "Medicaid",
                                "provider roster",
                                "provider directory",
                                "rule change",
                                "validation",
                                "submission requirements",
                                *query_keywords[:4],
                            ]
                        )
                    ),
                    "domains": self._state_domains(market),
                    "topic": "general",
                    "time_range": "year",
                    "search_depth": "advanced",
                    "market": market,
                }
            )

        validation_needed = signals["explicit_validation_request"] or signals["validation_issue"]
        if validation_needed:
            failure_phrase = failure_status or "Complete Validation Failure"
            plan.append(
                {
                    "category": "compliance_standard",
                    "purpose": "Explain what the leading validation failure means in provider roster compliance terms.",
                    "query": " ".join(
                        self._dedupe_text(
                            [
                                f'"{failure_phrase}"',
                                "provider roster",
                                "provider directory",
                                "validation",
                                "compliance standard",
                                "CMS",
                                "Medicaid",
                            ]
                        )
                    ),
                    "domains": self.COMPLIANCE_DOMAINS,
                    "topic": "general",
                    "time_range": "year",
                    "search_depth": "advanced",
                    "failure_status": failure_phrase,
                }
            )

        lob_needed = market and lob_terms and (
            signals["explicit_lob_request"]
            or (signals["external_intent"] and bool(signals["medicaid_lobs"]))
            or (signals["rejection_spike_request"] and bool(signals["medicaid_lobs"]))
        )
        if lob_needed:
            primary_lob = signals["medicaid_lobs"][0] if signals["medicaid_lobs"] else lob_terms[0]
            plan.append(
                {
                    "category": "lob_policy",
                    "purpose": f"Fetch {primary_lob} submission and roster-policy context for {market}.",
                    "query": " ".join(
                        self._dedupe_text(
                            [
                                market,
                                primary_lob,
                                "provider roster",
                                "submission requirements",
                                "provider directory",
                                "Medicaid",
                                *query_keywords[:3],
                            ]
                        )
                    ),
                    "domains": self._state_domains(market),
                    "topic": "general",
                    "time_range": "year",
                    "search_depth": "advanced",
                    "market": market,
                    "lob": primary_lob,
                }
            )

        org_needed = signals["org_candidates"] and (signals["explicit_org_request"] or named_org)
        for org_name in signals["org_candidates"][:2]:
            if not org_needed:
                break
            plan.append(
                {
                    "category": "org_context",
                    "purpose": f"Look up {org_name} to add business context to the anomaly.",
                    "query": " ".join(
                        self._dedupe_text(
                            [
                                f'"{org_name}"',
                                "medical group",
                                "foundation",
                                "provider organization",
                                market,
                            ]
                        )
                    ).strip(),
                    "domains": self.ORG_DOMAIN_HINTS.get(org_name, []),
                    "topic": "general",
                    "search_depth": "basic",
                    "org_name": org_name,
                    "market": market,
                }
            )

        return plan[:max_queries]

    def search(self, query, max_results=3):

        if self.disabled:
            return []

        generic_plan = [
            {
                "category": "regulatory_change",
                "purpose": "General external context lookup.",
                "query": query,
                "domains": self.REGULATORY_DOMAINS,
                "topic": "general",
                "search_depth": "basic",
            }
        ]
        rows = self._search_live(generic_plan[0], max_results=max_results)
        if rows:
            return rows
        return self._offline_fallback(generic_plan)[:max_results]

    def search_external_context(self, state, max_results_per_query=2):

        if self.disabled:
            return []

        plan = self.build_search_plan(state)
        if not plan:
            return []

        collected = []
        for item in plan:
            rows = self._search_live(item, max_results=max_results_per_query)
            if not rows:
                rows = self._offline_fallback([item])[:max_results_per_query]
            for row in rows:
                enriched = dict(row)
                enriched["category"] = item["category"]
                enriched["purpose"] = item["purpose"]
                enriched["query"] = item["query"]
                collected.append(enriched)

        deduped = []
        seen = set()
        for row in collected:
            key = (
                row.get("category"),
                row.get("url") or row.get("title"),
                row.get("purpose"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)

        return deduped
