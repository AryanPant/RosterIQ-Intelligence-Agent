import os
import re


class WebSearchTool:

    MAX_SNIPPET_LENGTH = 280
    STATE_DOMAIN_HINTS = {
        "VA": "site:dmmas.virginia.gov OR site:virginia.gov",
        "KS": "site:kancare.ks.gov OR site:kdhe.ks.gov OR site:kmap-state-ks.us",
    }
    ORG_DOMAIN_HINTS = {
        "Cedars-Sinai Medical Care Foundation": "site:cedars-sinai.org",
        "MercyOne Medical Group": "site:mercyone.org",
    }

    def __init__(self):

        self.api_key = os.getenv("TAVILY_API_KEY")
        self.provider = (os.getenv("ROSTERIQ_WEB_PROVIDER") or "tavily").strip().lower()
        self.disabled = os.getenv("ROSTERIQ_DISABLE_WEB_SEARCH", "").strip().lower() in {"1", "true", "yes"}

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
            if item["category"] == "regulatory_change":
                results.append(
                    self._offline_result(
                        item["category"],
                        "CMS Provider Directory and Access Rules",
                        "https://www.cms.gov/",
                        "CMS and state Medicaid access and directory requirements are relevant when rejection spikes suggest roster accuracy, validation, or submission-rule changes.",
                        item["purpose"],
                        item["query"],
                    )
                )
            elif item["category"] == "compliance_standard":
                results.append(
                    self._offline_result(
                        item["category"],
                        "Provider Data Validation Compliance Context",
                        "https://www.cms.gov/",
                        "Validation-heavy failure labels often point to directory-data quality rules, schema checks, or provider enrollment data standard mismatches rather than pure pipeline latency.",
                        item["purpose"],
                        item["query"],
                    )
                )
            elif item["category"] == "lob_policy":
                results.append(
                    self._offline_result(
                        item["category"],
                        "Medicaid and LOB Submission Guidance",
                        "https://www.medicaid.gov/",
                        "LOB-specific roster or provider data requirements can differ across Medicaid FFS, Medicaid managed care, Medicare, and commercial submissions.",
                        item["purpose"],
                        item["query"],
                    )
                )
            elif item["category"] == "org_context":
                results.append(
                    self._offline_result(
                        item["category"],
                        "Provider Organization Context",
                        "local://rosteriq/org-context",
                        "Provider-organization context can help explain whether anomalies are tied to a large integrated delivery network, foundation model, or multi-market medical group.",
                        item["purpose"],
                        item["query"],
                    )
                )
        return results

    def _search_live(self, query, max_results):

        if self.provider not in {"", "tavily"}:
            return []
        if not self.api_key:
            return []

        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=self.api_key)
            response = client.search(query=query, max_results=max_results)
            rows = []
            for item in response.get("results", []):
                rows.append(
                    {
                        "title": item.get("title", "Untitled"),
                        "url": item.get("url", ""),
                        "snippet": self._truncate_snippet(item.get("content", "")),
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
                lob_counts[item] = lob_counts.get(item, 0) + 1

        ordered = sorted(lob_counts.items(), key=lambda item: item[1], reverse=True)
        return [item[0].upper() for item, _ in ordered[:2]]

    def build_search_plan(self, state, max_queries=4):

        query = state.get("query") or ""
        query_lower = query.lower()
        scope = state.get("investigation_brief", {}).get("query_scope", {})
        market = scope.get("market") or state.get("market")
        query_keywords = scope.get("query_keywords", [])
        regulatory_terms = scope.get("regulatory_terms", [])
        root_cause = state.get("root_cause", {})
        failure_status = (root_cause.get("top_failure_status") or {}).get("status", "")
        org_candidates = self._dedupe_text(
            [
                scope.get("org_name"),
                (root_cause.get("top_impacted_org") or {}).get("org"),
            ]
        )
        lob_terms = self._extract_lob_terms(state)

        plan = []
        regulatory_trigger = any(
            token in query_lower
            for token in ["regulation", "regulatory", "rule", "policy", "payer", "medicaid", "cms", "compliance", "spike", "drop"]
        ) or regulatory_terms or scope.get("is_full_operational_report")
        if market and regulatory_trigger:
            domain_hint = self.STATE_DOMAIN_HINTS.get(str(market).upper(), "site:cms.gov OR site:medicaid.gov")
            keyword_hint = " ".join(query_keywords[:4])
            plan.append(
                {
                    "category": "regulatory_change",
                    "purpose": f"Check CMS or state Medicaid rule changes that could explain rejection or success-rate movement in {market}.",
                    "query": f'{domain_hint} {market} {keyword_hint} provider directory roster data rule change Medicaid policy validation',
                }
            )

        validation_trigger = any(
            token in query_lower
            for token in ["validation", "complete validation failure", "provider directory", "data standard", "schema", "incompatible"]
        ) or "validation" in failure_status.lower() or "incompatible" in failure_status.lower()
        if validation_trigger:
            keyword_hint = " ".join(self._dedupe_text([failure_status, *regulatory_terms])[:4])
            plan.append(
                {
                    "category": "compliance_standard",
                    "purpose": "Explain failure modes like Complete Validation Failure or incompatible provider data against provider-directory compliance standards.",
                    "query": f'site:cms.gov provider directory validation requirements "{failure_status or "Complete Validation Failure"}" {keyword_hint} provider data compliance',
                }
            )

        if market and lob_terms:
            primary_lob = lob_terms[0]
            domain_hint = self.STATE_DOMAIN_HINTS.get(str(market).upper(), "site:medicaid.gov OR site:cms.gov")
            keyword_hint = " ".join(self._dedupe_text([primary_lob, *query_keywords])[:5])
            plan.append(
                {
                    "category": "lob_policy",
                    "purpose": f"Fetch {primary_lob} submission or payer-policy context for {market}.",
                    "query": f'{domain_hint} {market} {keyword_hint} provider data submission requirements roster',
                }
            )

        for org_name in org_candidates[:2]:
            domain_hint = self.ORG_DOMAIN_HINTS.get(org_name, "")
            keyword_hint = " ".join(self._dedupe_text([org_name, *query_keywords])[:5])
            search_query = f'{domain_hint} "{org_name}" {keyword_hint} medical group foundation provider organization'.strip()
            plan.append(
                {
                    "category": "org_context",
                    "purpose": f"Look up {org_name} to add business context to roster anomalies.",
                    "query": search_query,
                }
            )

        return plan[:max_queries]

    def search(self, query, max_results=3):

        if self.disabled:
            return []

        rows = self._search_live(query, max_results=max_results)
        if rows:
            return rows

        return self._offline_fallback(
            [{"category": "regulatory_change", "purpose": "General external context lookup.", "query": query}]
        )[:max_results]

    def search_external_context(self, state, max_results_per_query=2):

        if self.disabled:
            return []

        plan = self.build_search_plan(state)
        if not plan:
            return []

        collected = []
        for item in plan:
            rows = self._search_live(item["query"], max_results=max_results_per_query)
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
            key = (row.get("category"), row.get("url"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)

        return deduped
