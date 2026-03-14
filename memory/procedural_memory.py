import copy
import os
import re

import yaml

from utils.openrouter_client import OpenRouterClient


class ProceduralMemory:

    DEFAULT_PROCEDURES = {
        "triage_stuck_ros": {
            "description": "Finds all IS_STUCK=1 ROs, ranks by time stuck and Red health flags, recommends escalation priority.",
            "function": "escalation_score = duration_ratio + red_health_flag_bonus + stuck_indicator",
            "logic": [
                "Filter pipeline records where IS_STUCK = 1.",
                "Rank by time stuck using duration anomalies and current stage duration.",
                "Prioritize rows with Red health flags and highest anomaly ratios.",
            ],
        },
        "record_quality_audit": {
            "description": "For a given state or org, computes FAIL_REC_CNT + REJ_REC_CNT + SKIP_REC_CNT as a percentage of TOT_REC_CNT and flags files below SCS_PCT threshold.",
            "function": "quality_issue_rate = (FAIL_REC_CNT + REJ_REC_CNT + SKIP_REC_CNT) / TOT_REC_CNT",
            "logic": [
                "Filter records to the requested market or organization.",
                "Compute quality loss as (FAIL_REC_CNT + REJ_REC_CNT + SKIP_REC_CNT) / TOT_REC_CNT.",
                "Flag files with low SCS_PCT or elevated combined failure, rejection, and skip proportions.",
            ],
        },
        "market_health_report": {
            "description": "Correlates a market's SCS_PERCENT with pipeline-level file rejection rates for the same state and period.",
            "function": "state_period_join = merge(file_rejection_rate_by_state_period, market_scs_percent_by_state_period)",
            "logic": [
                "Aggregate file-level rejection metrics by state and period.",
                "Join state-period pipeline metrics to market-level SCS_PERCENT.",
                "Identify whether file-level rejection pressure aligns with market decline.",
            ],
        },
        "retry_effectiveness_analysis": {
            "description": "For RUN_NO > 1, compares first-pass vs. retry SCS_PCT to determine if reprocessing is actually improving outcomes.",
            "function": "retry_lift = retry_scs_pct - first_pass_scs_pct",
            "logic": [
                "Compare RUN_NO = 1 results to retry runs for the same RO_ID.",
                "Measure whether retries improve success percentage or reduce rejection rates.",
                "Flag retry patterns that fail to recover the original issue.",
            ],
        },
        "lob_rejection_breakdown": {
            "description": "Groups rejection rates by Line of Business to analyze the distribution of rejections across different business lines.",
            "function": "rejection_rate_by_lob = SUM(REJ_REC_CNT) / SUM(TOT_REC_CNT) GROUP BY LOB",
            "logic": [
                "Filter records to the desired time period and market or organization if needed.",
                "Group the data by Line of Business (LOB).",
                "For each LOB, compute total REJ_REC_CNT and total TOT_REC_CNT.",
                "Calculate rejection rate as SUM(REJ_REC_CNT) divided by SUM(TOT_REC_CNT).",
                "Return the rejection rate per LOB for further analysis or reporting.",
            ],
        },
    }

    def __init__(self, path=None):

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.path = path or os.path.join(base_dir, "procedures", "procedures.yaml")
        self.llm = OpenRouterClient()
        self.procedures = self._load_procedures()
        self._ensure_defaults()

    def _load_procedures(self):

        if not os.path.exists(self.path):
            return {}

        with open(self.path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def reload(self):

        self.procedures = self._load_procedures()
        self._ensure_defaults()
        return self.procedures

    def _save(self):

        with open(self.path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.procedures, handle, sort_keys=False)

    def _ensure_defaults(self):

        changed = False
        for name, payload in self.DEFAULT_PROCEDURES.items():
            existing = self.procedures.setdefault(name, {})
            for key, value in payload.items():
                if key not in existing or not existing.get(key):
                    existing[key] = copy.deepcopy(value)
                    changed = True

        if changed:
            self._save()

    @staticmethod
    def _slugify(text):

        slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
        return slug or "custom_procedure"

    @staticmethod
    def _tokenize(text):

        return re.findall(r"[a-z0-9_]+", (text or "").lower())

    def _procedure_match_score(self, query, procedure_name, procedure_payload):

        query_text = (query or "").lower()
        query_tokens = set(self._tokenize(query_text))
        if not query_tokens:
            return 0

        logic = procedure_payload.get("logic", [])
        if isinstance(logic, str):
            logic_text = logic
        elif isinstance(logic, list):
            logic_text = " ".join(str(item) for item in logic)
        else:
            logic_text = str(logic or "")

        name_tokens = set(self._tokenize(procedure_name.replace("_", " ")))
        description_tokens = set(self._tokenize(procedure_payload.get("description", "")))
        logic_tokens = set(self._tokenize(logic_text))

        score = 0
        normalized_name_phrase = procedure_name.replace("_", " ").lower()
        if procedure_name.lower() in query_text:
            score += 100
        if normalized_name_phrase and normalized_name_phrase in query_text:
            score += 80

        score += len(query_tokens.intersection(name_tokens)) * 10
        score += len(query_tokens.intersection(description_tokens)) * 3
        score += len(query_tokens.intersection(logic_tokens)) * 1
        return score

    def _infer_target_procedure(self, user_query):

        normalized = (user_query or "").lower()
        explicit_name_patterns = [
            r"\bprocedure\s+(?:called|named)\s+([A-Za-z0-9_]+)",
            r"\bcreate\s+(?:a\s+new\s+)?procedure\s+(?:called|named)?\s*([A-Za-z0-9_]+)",
            r"\bdefine\s+(?:a\s+new\s+)?procedure\s+(?:called|named)?\s*([A-Za-z0-9_]+)",
            r"\badd\s+(?:a\s+new\s+)?procedure\s+(?:called|named)?\s*([A-Za-z0-9_]+)",
            r"\b(?:update|modify|change|refine|improve)\s+(?:the\s+)?(?:procedure\s+)?([A-Za-z0-9_ ]+?)(?:\s+(?:to|with|so|by|for)\b|$)",
        ]
        original_query = user_query or ""

        for pattern in explicit_name_patterns:
            match = re.search(pattern, original_query, flags=re.IGNORECASE)
            if match:
                return self._slugify(match.group(1))

        best_match = None
        best_score = 0
        for name, payload in self.procedures.items():
            score = self._procedure_match_score(user_query, name, payload or {})
            if score > best_score:
                best_score = score
                best_match = name

        if best_match and best_score > 0:
            return best_match

        if "procedure" in normalized:
            tail = normalized.split("procedure", 1)[1].strip()
            if tail:
                return self._slugify(tail[:40])

        return None

    def infer_target_procedure(self, user_query):

        self.reload()
        return self._infer_target_procedure(user_query)

    def is_update_query(self, user_query):

        self.reload()
        normalized = (user_query or "").lower()
        inferred_target = self._infer_target_procedure(user_query)
        explicit_procedure_markers = [
            "create procedure",
            "create a new procedure",
            "new procedure",
            "define procedure",
            "define a new procedure",
            "add procedure",
            "store procedure",
            "update procedure",
            "modify procedure",
            "change procedure",
            "refine procedure",
            "improve procedure",
        ]
        if any(marker in normalized for marker in explicit_procedure_markers):
            return True

        direct_update_verbs = ("update ", "modify ", "change ", "refine ", "improve ")
        if normalized.startswith(direct_update_verbs):
            return bool(inferred_target)

        procedure_edit_markers = [
            "also include",
            "include",
            "also add",
            "exclude",
            "remove",
            "replace",
            "instead of",
            "should include",
            "should use",
            "should calculate",
            "should compute",
        ]
        has_edit_action = re.search(r"\b(add|include|exclude|remove|replace)\b", normalized) is not None
        normalized_target = inferred_target.replace("_", " ").lower() if inferred_target else ""
        explicit_target_reference = bool(normalized_target and normalized_target in normalized)
        if inferred_target and explicit_target_reference and (
            has_edit_action or any(marker in normalized for marker in procedure_edit_markers)
        ):
            return True

        if "procedure" in normalized and any(marker in normalized for marker in procedure_edit_markers):
            return True

        return False

    @staticmethod
    def _normalize_logic(logic, fallback_text):

        if isinstance(logic, str):
            logic_items = [logic]
        elif isinstance(logic, list):
            logic_items = logic
        elif logic is None:
            logic_items = []
        else:
            logic_items = [str(logic)]

        normalized = []
        for item in logic_items:
            text = str(item).strip()
            if text and text not in normalized:
                normalized.append(text)

        if not normalized and fallback_text:
            normalized.append(fallback_text.strip())

        return normalized

    def _fallback_interpret_query(self, user_query, inferred_target, existing_context):

        normalized = (user_query or "").lower()
        target = inferred_target or "custom_procedure"

        if target == "record_quality_audit" and any(
            token in normalized for token in ["skip_rec_cnt", "skip rec cnt", "skipped records", "skip count", "include skip"]
        ):
            return {
                "action": "update",
                "procedure": "record_quality_audit",
                "description": self.DEFAULT_PROCEDURES["record_quality_audit"]["description"],
                "function": self.DEFAULT_PROCEDURES["record_quality_audit"]["function"],
                "logic": copy.deepcopy(self.DEFAULT_PROCEDURES["record_quality_audit"]["logic"]),
                "change_summary": "Updated record_quality_audit to include SKIP_REC_CNT in the quality issue rate.",
            }

        if "line of business" in normalized or "lob" in normalized:
            if "reject" in normalized or "rejection" in normalized:
                return {
                    "action": "create" if target not in self.procedures else "update",
                    "procedure": inferred_target or "lob_rejection_breakdown",
                    "description": self.DEFAULT_PROCEDURES["lob_rejection_breakdown"]["description"],
                    "function": self.DEFAULT_PROCEDURES["lob_rejection_breakdown"]["function"],
                    "logic": copy.deepcopy(self.DEFAULT_PROCEDURES["lob_rejection_breakdown"]["logic"]),
                    "change_summary": "Stored a Line of Business rejection breakdown procedure from the user instruction.",
                }

        if inferred_target:
            before = existing_context or {}
            return {
                "action": "update" if before else "create",
                "procedure": inferred_target,
                "description": str(before.get("description") or f"User-defined procedure derived from: {user_query}").strip(),
                "function": str(before.get("function") or "manual_review_required = 1").strip(),
                "logic": self._normalize_logic(before.get("logic"), user_query),
                "change_summary": f"Stored structured procedural memory for {inferred_target}.",
            }

        return None

    def _interpret_query(self, user_query):

        self.reload()
        inferred_target = self._infer_target_procedure(user_query)
        existing_context = self.procedures.get(inferred_target or "", {})

        prompt = f"""
Convert the user's instruction into a procedure definition.
Return JSON only with keys: action, procedure, description, function, logic, change_summary.

Rules:
- procedure must be snake_case
- function must be a concise mathematical or computational expression
- logic must be a list of implementation steps
- if the user is refining an existing procedure, preserve the procedure name and update the function or logic accordingly
- if the user is modifying an existing procedure, rewrite the stored function and logic rather than appending conversational text
- do not return markdown, explanations, or prose outside the JSON object

Example:
{{
  "action": "update",
  "procedure": "record_quality_audit",
  "description": "For a given state or org, computes FAIL_REC_CNT + REJ_REC_CNT + SKIP_REC_CNT as a percentage of TOT_REC_CNT and flags files below SCS_PCT threshold.",
  "function": "quality_issue_rate = (FAIL_REC_CNT + REJ_REC_CNT + SKIP_REC_CNT) / TOT_REC_CNT",
  "logic": [
    "Filter records to the requested market or organization.",
    "Compute quality loss as (FAIL_REC_CNT + REJ_REC_CNT + SKIP_REC_CNT) / TOT_REC_CNT.",
    "Flag files with low SCS_PCT or elevated combined failure, rejection, and skip proportions."
  ],
  "change_summary": "Updated record_quality_audit to include SKIP_REC_CNT in the quality issue rate."
}}

Known procedures:
{yaml.safe_dump(self.procedures, sort_keys=False)}

Likely target procedure:
{inferred_target or "unknown"}

Existing target procedure:
{yaml.safe_dump(existing_context, sort_keys=False)}

User instruction:
{user_query}
""".strip()

        retry_prompt = f"""
Return a JSON object only.

Task:
Update or create a diagnostic procedure definition from this user instruction.

Required keys:
action, procedure, description, function, logic, change_summary

Target procedure:
{inferred_target or "unknown"}

Existing procedure:
{yaml.safe_dump(existing_context, sort_keys=False)}

User instruction:
{user_query}
""".strip()

        for raw in [
            self.llm.generate(prompt, max_tokens=900, temperature=0.1),
            self.llm.generate(retry_prompt, max_tokens=500, temperature=0.0),
        ]:
            if not raw:
                continue

            try:
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                parsed = yaml.safe_load(cleaned)
                if not isinstance(parsed, dict):
                    continue

                procedure_name = parsed.get("procedure") or inferred_target
                description = str(parsed.get("description", "")).strip()
                function = str(parsed.get("function", "")).strip()
                logic = self._normalize_logic(parsed.get("logic", []), "")
                change_summary = str(parsed.get("change_summary", "")).strip()

                if not procedure_name or not description or not function or not logic:
                    continue

                return {
                    "action": str(parsed.get("action", "update")).strip() or "update",
                    "procedure": self._slugify(procedure_name),
                    "description": description,
                    "function": function,
                    "logic": logic,
                    "change_summary": change_summary or "Updated procedure logic from user instruction.",
                }
            except Exception:
                continue

        return self._fallback_interpret_query(user_query, inferred_target, existing_context)

    def list(self):

        self.reload()
        return list(self.procedures.keys())

    def get(self, name):

        self.reload()
        return self.procedures.get(name, {})

    def update(self, name, procedure_payload):

        self.reload()
        self.procedures.setdefault(name, {})
        self.procedures[name].update(procedure_payload)
        self._save()

    def upsert_from_query(self, user_query):

        self.reload()
        interpreted = self._interpret_query(user_query)
        target = self._infer_target_procedure(user_query) or "custom_procedure"
        before = copy.deepcopy(self.procedures.get(target, {}))

        if not interpreted:
            return {
                "procedure": target,
                "before": before,
                "after": copy.deepcopy(before),
                "confirmation": "Procedural memory was not updated because the model did not return a valid structured procedure definition.",
                "updated": False,
            }

        name = interpreted["procedure"]
        before = copy.deepcopy(self.procedures.get(name, {}))

        self.procedures.setdefault(name, {})
        self.procedures[name]["description"] = interpreted["description"]
        self.procedures[name]["function"] = interpreted["function"]
        self.procedures[name]["logic"] = interpreted["logic"]
        self._save()

        return {
            "procedure": name,
            "before": before,
            "after": copy.deepcopy(self.procedures[name]),
            "confirmation": interpreted["change_summary"],
            "updated": True,
        }

    def improve(self, name, user_correction):

        prefixed_query = f"Update procedure {name}: {user_correction}"
        result = self.upsert_from_query(prefixed_query)
        if result.get("updated"):
            result["confirmation"] = f"{result['confirmation']} Stored under {name}."
        return result
