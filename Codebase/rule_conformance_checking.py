import pandas as pd
from pathlib import Path


INPUT_CSV = "Gemma/TS5.csv"
OUTPUT_DIR = "rule_based_conformance_results"

CASE_COL = "Case ID"
ACTIVITY_COL = "Activity"
TIMESTAMP_COL = "Timestamp"
GROUP_COL = "Group"
LIFECYCLE_COL = "Lifecycle"

SEP = "/"

TRIAGE = "ER Sepsis Triage"
IV_ANTIBIOTICS = "IV Antibiotics"
LACTIC_ACID = "LacticAcid"
ADMISSION_NC = "Admission NC"
ADMISSION_IC = "Admission IC"
RETURN_ER = "Return ER"

RELEASE_PREFIX = "Release"

STRICT_LESS_THAN = False

ANTIBIOTICS_LIMIT_MIN = 60
LACTIC_LIMIT_MIN = 180

EXTREME_DELAY_HOURS = 24



def split_cell(value):
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(SEP)]


def parse_timestamp(value):
    return pd.to_datetime(value, errors="coerce")


def explode_one_row_per_case(df):

    events = []

    for _, row in df.iterrows():
        case_id = row[CASE_COL]

        activities = split_cell(row[ACTIVITY_COL])
        timestamps = split_cell(row[TIMESTAMP_COL])
        groups = split_cell(row[GROUP_COL]) if GROUP_COL in df.columns else []
        lifecycles = split_cell(row[LIFECYCLE_COL]) if LIFECYCLE_COL in df.columns else []

        max_len = max(len(activities), len(timestamps), len(groups), len(lifecycles))

        for i in range(max_len):
            events.append({
                CASE_COL: case_id,
                "event_index": i,
                ACTIVITY_COL: activities[i] if i < len(activities) else None,
                TIMESTAMP_COL: parse_timestamp(timestamps[i]) if i < len(timestamps) else pd.NaT,
                GROUP_COL: groups[i] if i < len(groups) else None,
                LIFECYCLE_COL: lifecycles[i] if i < len(lifecycles) else None,
            })

    event_df = pd.DataFrame(events)
    event_df = event_df.sort_values([CASE_COL, TIMESTAMP_COL, "event_index"], na_position="last")
    return event_df


def pct(numerator, denominator):
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def within_limit(delay_minutes, limit_minutes):
    if pd.isna(delay_minutes):
        return False
    if STRICT_LESS_THAN:
        return delay_minutes < limit_minutes
    return delay_minutes <= limit_minutes


def first_event_time(case_events, activity):
    matches = case_events[case_events[ACTIVITY_COL] == activity]
    if matches.empty:
        return pd.NaT
    return matches[TIMESTAMP_COL].min()


def first_event_time_after(case_events, activity, start_time):
    matches = case_events[
        (case_events[ACTIVITY_COL] == activity) &
        (case_events[TIMESTAMP_COL] >= start_time)
    ]

    if matches.empty:
        return pd.NaT

    return matches[TIMESTAMP_COL].min()


def first_event_time_anywhere(case_events, activity):
    matches = case_events[case_events[ACTIVITY_COL] == activity]
    if matches.empty:
        return pd.NaT
    return matches[TIMESTAMP_COL].min()


def evaluate_time_rule_for_case(case_id, case_events, target_activity, limit_minutes):
    triage_time = first_event_time(case_events, TRIAGE)

    target_any_time = first_event_time_anywhere(case_events, target_activity)

    if pd.isna(triage_time):
        return {
            CASE_COL: case_id,
            "rule_target": target_activity,
            "has_triage": False,
            "has_target": not pd.isna(target_any_time),
            "triage_time": triage_time,
            "target_time": target_any_time,
            "delay_minutes": pd.NA,
            "status": "not_applicable_no_triage",
            "compliant": False,
            "violated": False,
            "missing_target": False,
            "target_before_triage": False,
            "extreme_delay_flag": False,
        }

    target_time = first_event_time_after(case_events, target_activity, triage_time)

    # If no target after triage exists, check whether one exists before triage.
    target_before_triage = False
    if pd.isna(target_time) and not pd.isna(target_any_time):
        target_before_triage = target_any_time < triage_time
        target_time = target_any_time

    has_target = not pd.isna(target_time)

    if not has_target:
        return {
            CASE_COL: case_id,
            "rule_target": target_activity,
            "has_triage": True,
            "has_target": False,
            "triage_time": triage_time,
            "target_time": pd.NaT,
            "delay_minutes": pd.NA,
            "status": "missing_target",
            "compliant": False,
            "violated": True,
            "missing_target": True,
            "target_before_triage": False,
            "extreme_delay_flag": False,
        }

    delay_minutes = (target_time - triage_time).total_seconds() / 60

    if target_before_triage or delay_minutes < 0:
        status = "data_quality_target_before_triage"
        compliant = False
        violated = True
    elif within_limit(delay_minutes, limit_minutes):
        status = "compliant"
        compliant = True
        violated = False
    else:
        status = "violated_late"
        compliant = False
        violated = True

    extreme_delay_flag = delay_minutes >= EXTREME_DELAY_HOURS * 60

    return {
        CASE_COL: case_id,
        "rule_target": target_activity,
        "has_triage": True,
        "has_target": True,
        "triage_time": triage_time,
        "target_time": target_time,
        "delay_minutes": delay_minutes,
        "status": status,
        "compliant": compliant,
        "violated": violated,
        "missing_target": False,
        "target_before_triage": target_before_triage or delay_minutes < 0,
        "extreme_delay_flag": extreme_delay_flag,
    }


def evaluate_question_1(event_df):
    rows = []

    for case_id, case_events in event_df.groupby(CASE_COL):
        rows.append(
            evaluate_time_rule_for_case(
                case_id=case_id,
                case_events=case_events,
                target_activity=IV_ANTIBIOTICS,
                limit_minutes=ANTIBIOTICS_LIMIT_MIN,
            )
        )

        rows.append(
            evaluate_time_rule_for_case(
                case_id=case_id,
                case_events=case_events,
                target_activity=LACTIC_ACID,
                limit_minutes=LACTIC_LIMIT_MIN,
            )
        )

    q1_details = pd.DataFrame(rows)

    summary_rows = []

    for target_activity, group in q1_details.groupby("rule_target"):
        eligible = group[group["has_triage"] == True]
        n_eligible = len(eligible)

        n_compliant = int(eligible["compliant"].sum())
        n_violated = int(eligible["violated"].sum())
        n_missing = int(eligible["missing_target"].sum())
        n_before = int(eligible["target_before_triage"].sum())
        n_extreme = int(eligible["extreme_delay_flag"].sum())

        valid_delay = eligible[
            (eligible["has_target"] == True) &
            (eligible["target_before_triage"] == False)
        ]

        summary_rows.append({
            "question": "Q1 medical guidelines",
            "rule": f"{TRIAGE} -> {target_activity}",
            "limit_minutes": ANTIBIOTICS_LIMIT_MIN if target_activity == IV_ANTIBIOTICS else LACTIC_LIMIT_MIN,
            "eligible_cases_with_triage": n_eligible,
            "compliant_cases": n_compliant,
            "compliant_pct": pct(n_compliant, n_eligible),
            "violated_cases": n_violated,
            "violated_pct": pct(n_violated, n_eligible),
            "missing_target_cases": n_missing,
            "missing_target_pct": pct(n_missing, n_eligible),
            "target_before_triage_cases": n_before,
            "target_before_triage_pct": pct(n_before, n_eligible),
            "extreme_delay_cases": n_extreme,
            "extreme_delay_pct": pct(n_extreme, n_eligible),
            "average_delay_minutes": valid_delay["delay_minutes"].mean(),
            "median_delay_minutes": valid_delay["delay_minutes"].median(),
            "average_delay_hours": valid_delay["delay_minutes"].mean() / 60,
        })

    return q1_details, pd.DataFrame(summary_rows)


def classify_question_2_trajectory(case_events):
    activities = list(case_events[ACTIVITY_COL].dropna())

    has_nc = ADMISSION_NC in activities
    has_ic = ADMISSION_IC in activities
    has_release = any(str(a).startswith(RELEASE_PREFIX) for a in activities)

    first_nc_idx = activities.index(ADMISSION_NC) if has_nc else None
    first_ic_idx = activities.index(ADMISSION_IC) if has_ic else None

    if has_nc and has_ic:
        if first_nc_idx < first_ic_idx:
            return "admission_nc_then_ic"
        else:
            return "admission_ic_then_nc"

    if has_nc and not has_ic:
        return "admission_nc_only"

    if has_ic and not has_nc:
        return "admission_ic_only"

    if has_release and not has_nc and not has_ic:
        return "discharge_without_admission"

    return "other_or_unclassified"


def evaluate_question_2(event_df):
    rows = []

    for case_id, case_events in event_df.groupby(CASE_COL):
        rows.append({
            CASE_COL: case_id,
            "trajectory_q2": classify_question_2_trajectory(case_events),
        })

    details = pd.DataFrame(rows)

    total_cases = details[CASE_COL].nunique()

    summary = (
        details["trajectory_q2"]
        .value_counts()
        .rename_axis("trajectory_q2")
        .reset_index(name="cases")
    )

    summary["pct_of_all_cases"] = summary["cases"].apply(lambda x: pct(x, total_cases))
    summary.insert(0, "question", "Q2 patient trajectories")

    return details, summary


def evaluate_question_3(event_df):
    rows = []

    for case_id, case_events in event_df.groupby(CASE_COL):
        case_events = case_events.sort_values([TIMESTAMP_COL, "event_index"])

        first_case_time = case_events[TIMESTAMP_COL].min()

        return_events = case_events[case_events[ACTIVITY_COL] == RETURN_ER]
        has_return = not return_events.empty

        if has_return:
            first_return_time = return_events[TIMESTAMP_COL].min()
            days_to_return = (first_return_time - first_case_time).total_seconds() / (60 * 60 * 24)
            returned_within_28_days = days_to_return <= 28
        else:
            first_return_time = pd.NaT
            days_to_return = pd.NA
            returned_within_28_days = False

        rows.append({
            CASE_COL: case_id,
            "has_return": has_return,
            "first_case_time": first_case_time,
            "first_return_time": first_return_time,
            "days_to_return": days_to_return,
            "returned_within_28_days": returned_within_28_days,
        })

    details = pd.DataFrame(rows)

    total_cases = len(details)
    return_cases = int(details["has_return"].sum())
    return_28_cases = int(details["returned_within_28_days"].sum())

    returned = details[details["has_return"] == True]

    summary = pd.DataFrame([
        {
            "question": "Q3 returning patients",
            "metric": "patients_returning_any_time",
            "cases": return_cases,
            "pct_of_all_cases": pct(return_cases, total_cases),
            "average_days_to_return": returned["days_to_return"].mean(),
            "median_days_to_return": returned["days_to_return"].median(),
        },
        {
            "question": "Q3 returning patients",
            "metric": "patients_returning_within_28_days",
            "cases": return_28_cases,
            "pct_of_all_cases": pct(return_28_cases, total_cases),
            "average_days_to_return": returned[returned["returned_within_28_days"] == True]["days_to_return"].mean(),
            "median_days_to_return": returned[returned["returned_within_28_days"] == True]["days_to_return"].median(),
        }
    ])

    return details, summary


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    required_cols = {CASE_COL, ACTIVITY_COL, TIMESTAMP_COL}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    event_df = explode_one_row_per_case(df)

    q1_details, q1_summary = evaluate_question_1(event_df)
    q2_details, q2_summary = evaluate_question_2(event_df)
    q3_details, q3_summary = evaluate_question_3(event_df)

    all_summary = pd.concat(
        [
            q1_summary,
            q2_summary,
            q3_summary,
        ],
        ignore_index=True,
        sort=False,
    )

    event_df.to_csv(output_dir / "exploded_event_log.csv", index=False)

    q1_details.to_csv(output_dir / "q1_guideline_rule_details.csv", index=False)
    q1_summary.to_csv(output_dir / "q1_guideline_rule_summary.csv", index=False)

    q2_details.to_csv(output_dir / "q2_trajectory_details.csv", index=False)
    q2_summary.to_csv(output_dir / "q2_trajectory_summary.csv", index=False)

    q3_details.to_csv(output_dir / "q3_returning_patient_details.csv", index=False)
    q3_summary.to_csv(output_dir / "q3_returning_patient_summary.csv", index=False)

    all_summary.to_csv(output_dir / "all_questions_summary.csv", index=False)
    print("\n  Q1 guideline summary  ")
    print(q1_summary.to_string(index=False))
    print("\n  Q2 trajectory summary  ")
    print(q2_summary.to_string(index=False))
    print("\n  Q3 returning patients summary  ")
    print(q3_summary.to_string(index=False))
    print(f"\nSaved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()