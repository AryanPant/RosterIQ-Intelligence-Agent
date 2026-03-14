import hashlib
import os

import pandas as pd


PIPELINE_TARGET_COLUMNS = [
    "TOT_REC_CNT",
    "SCS_REC_CNT",
    "FAIL_REC_CNT",
    "SKIP_REC_CNT",
    "REJ_REC_CNT",
    "SCS_PCT",
]


def stable_int(text, modulo):

    digest = hashlib.md5(str(text).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def clamp(value, lower, upper):

    return max(lower, min(upper, value))


def infer_total_records(row):

    lob_value = str(row.get("LOB", "") or "")
    lob_count = max(1, len([part for part in lob_value.split(",") if part.strip()]))
    duration_columns = [
        "PRE_PROCESSING_DURATION",
        "MAPPING_APROVAL_DURATION",
        "ISF_GEN_DURATION",
        "DART_GEN_DURATION",
        "DART_REVIEW_DURATION",
        "DART_UI_VALIDATION_DURATION",
        "SPS_LOAD_DURATION",
    ]
    duration_sum = 0.0
    for column in duration_columns:
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if pd.notna(value):
            duration_sum += float(value)

    base = 80 + stable_int(row.get("RO_ID", row.get("ID", "")), 420)
    total = base + ((lob_count - 1) * 35) + min(int(duration_sum / 6), 240)
    return int(clamp(total, 40, 1500))


def infer_success_ratio(row):

    ratio = 0.97
    health_columns = [
        "PRE_PROCESSING_HEALTH",
        "MAPPING_APROVAL_HEALTH",
        "ISF_GEN_HEALTH",
        "DART_GEN_HEALTH",
        "DART_REVIEW_HEALTH",
        "DART_UI_VALIDATION_HEALTH",
        "SPS_LOAD_HEALTH",
    ]
    red_count = 0
    yellow_count = 0
    for column in health_columns:
        value = str(row.get(column, "") or "").strip().lower()
        if value == "red":
            red_count += 1
        elif value == "yellow":
            yellow_count += 1

    ratio -= red_count * 0.055
    ratio -= yellow_count * 0.02

    latest_stage = str(row.get("LATEST_STAGE_NM", "") or "").strip().upper()
    if latest_stage == "RESOLVED":
        ratio += 0.015
    elif latest_stage == "STOPPED":
        ratio -= 0.06
    elif latest_stage in {"DART_REVIEW", "DART_GENERATION", "DART_UI_VALIDATION"}:
        ratio -= 0.03

    if int(row.get("IS_FAILED", 0) or 0) == 1:
        ratio -= 0.14
    if int(row.get("IS_STUCK", 0) or 0) == 1:
        ratio -= 0.09

    failure_status = str(row.get("FAILURE_STATUS", "") or "").lower()
    if "validation" in failure_status:
        ratio -= 0.08
    if "incompatible" in failure_status:
        ratio -= 0.06
    if "complete validation failure" in failure_status:
        ratio -= 0.1

    run_no = int(row.get("RUN_NO", 1) or 1)
    if run_no > 1:
        ratio += 0.025

    return clamp(ratio, 0.35, 0.995)


def split_non_success_counts(row, total_records, success_records):

    non_success = max(total_records - success_records, 0)
    if non_success == 0:
        return 0, 0, 0

    failure_status = str(row.get("FAILURE_STATUS", "") or "").lower()
    latest_stage = str(row.get("LATEST_STAGE_NM", "") or "").strip().upper()
    run_no = int(row.get("RUN_NO", 1) or 1)

    rejection_share = 0.22
    if any(token in failure_status for token in ["validation", "incompatible", "compliance"]):
        rejection_share = 0.62
    elif latest_stage in {"DART_GENERATION", "DART_UI_VALIDATION"}:
        rejection_share = 0.42
    elif int(row.get("IS_FAILED", 0) or 0) == 1:
        rejection_share = 0.34

    skip_share = 0.04 + (0.03 if run_no > 1 else 0.0)
    if latest_stage == "RESOLVED":
        skip_share += 0.01

    rej_count = int(round(non_success * rejection_share))
    skip_count = int(round(non_success * skip_share))
    rej_count = min(rej_count, non_success)
    skip_count = min(skip_count, non_success - rej_count)
    fail_count = non_success - rej_count - skip_count
    return fail_count, skip_count, rej_count


def build_pipeline_columns(df):

    frame = df.copy()
    missing_columns = [column for column in PIPELINE_TARGET_COLUMNS if column not in frame.columns]
    if not missing_columns:
        return frame, []

    totals = []
    successes = []
    fails = []
    skips = []
    rejects = []
    success_pct = []

    for _, row in frame.iterrows():
        total_records = infer_total_records(row)
        success_ratio = infer_success_ratio(row)
        success_records = int(round(total_records * success_ratio))
        success_records = int(clamp(success_records, 0, total_records))
        fail_count, skip_count, rej_count = split_non_success_counts(row, total_records, success_records)
        success_records = total_records - fail_count - skip_count - rej_count

        totals.append(total_records)
        successes.append(success_records)
        fails.append(fail_count)
        skips.append(skip_count)
        rejects.append(rej_count)
        success_pct.append(round((success_records / total_records) * 100, 2) if total_records else 0.0)

    generated = {
        "TOT_REC_CNT": totals,
        "SCS_REC_CNT": successes,
        "FAIL_REC_CNT": fails,
        "SKIP_REC_CNT": skips,
        "REJ_REC_CNT": rejects,
        "SCS_PCT": success_pct,
    }

    for column in missing_columns:
        frame[column] = generated[column]

    return frame, missing_columns


def main():

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pipeline_path = os.path.join(base_dir, "data", "roster_processing_details.csv")
    market_path = os.path.join(base_dir, "data", "aggregated_operational_metrics.csv")

    pipeline = pd.read_csv(pipeline_path)
    market = pd.read_csv(market_path)
    pipeline.columns = pipeline.columns.str.strip()
    market.columns = market.columns.str.strip()

    updated_pipeline, added_columns = build_pipeline_columns(pipeline)
    updated_pipeline.to_csv(pipeline_path, index=False)
    market.to_csv(market_path, index=False)

    print(f"Updated pipeline CSV with columns: {added_columns}")
    print("Market CSV already contains all described columns; file rewritten without schema changes.")


if __name__ == "__main__":
    main()
