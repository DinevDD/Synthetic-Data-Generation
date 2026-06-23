import os
import pandas as pd
import numpy as np
from collections import Counter

try:
    from scipy.stats import wasserstein_distance
except ImportError:
    wasserstein_distance = None

REAL_LOG_PATH = "Data/filtered_log.xes.gz"
SYNTHETIC_LOG_PATH = "Gemma/generated_ts5_output(1).csv"
OUTPUT_DIR = "windowed_workload_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CASE_COL = "Case ID"
ACTIVITY_COL = "Activity"
TIMESTAMP_COL = "Timestamp"
RESOURCE_COL = "Group"
LIFECYCLE_COL = "Lifecycle"
NGRAM_N = 3

WINDOW_STEP_FRACTION = 0.5
MIN_WINDOW_CASES = 10

PROFILE_POINTS = 50

def load_event_log(path):
    path_lower = path.lower()

    if path_lower.endswith(".xes") or path_lower.endswith(".xes.gz"):
        return load_xes_event_log(path)

    return load_csv_event_log(path)


def load_xes_event_log(path):
    try:
        import pm4py
    except ImportError:
        raise ImportError(
            "PM4Py is required to load XES files. Install it with:\n"
            "pip install pm4py"
        )

    log = pm4py.read_xes(path)
    df = pm4py.convert_to_dataframe(log)

    print("\nLoaded XES columns:")
    print(df.columns.tolist())

    column_map = {}

    if "case:concept:name" in df.columns:
        column_map["case:concept:name"] = CASE_COL

    if "concept:name" in df.columns:
        column_map["concept:name"] = ACTIVITY_COL

    if "time:timestamp" in df.columns:
        column_map["time:timestamp"] = TIMESTAMP_COL

    if "org:resource" in df.columns:
        column_map["org:resource"] = RESOURCE_COL
    elif "org:group" in df.columns:
        column_map["org:group"] = RESOURCE_COL

    if "lifecycle:transition" in df.columns:
        column_map["lifecycle:transition"] = LIFECYCLE_COL

    df = df.rename(columns=column_map)

    required = [CASE_COL, ACTIVITY_COL, TIMESTAMP_COL]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{path} is missing required normalized columns: {missing}\n"
            f"Available columns after loading are:\n{df.columns.tolist()}"
        )

    keep_cols = [CASE_COL, ACTIVITY_COL, TIMESTAMP_COL]

    if RESOURCE_COL in df.columns:
        keep_cols.append(RESOURCE_COL)

    if LIFECYCLE_COL in df.columns:
        keep_cols.append(LIFECYCLE_COL)
    df = df[keep_cols].copy()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce", utc=True)
    df = df.dropna(subset=[CASE_COL, ACTIVITY_COL, TIMESTAMP_COL])
    df[TIMESTAMP_COL] = df[TIMESTAMP_COL].dt.tz_localize(None)
    df[CASE_COL] = df[CASE_COL].astype(str)
    df[ACTIVITY_COL] = df[ACTIVITY_COL].astype(str)

    df = df.sort_values([CASE_COL, TIMESTAMP_COL]).reset_index(drop=True)

    return df


def load_csv_event_log(path):
    df = pd.read_csv(path)

    required = [CASE_COL, ACTIVITY_COL, TIMESTAMP_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}\n"
            f"Available columns are:\n{df.columns.tolist()}"
        )

    compressed = (
        df[ACTIVITY_COL].astype(str).str.contains("/", regex=False).any()
        or df[TIMESTAMP_COL].astype(str).str.contains("/", regex=False).any()
    )
    if not compressed:
        df = df.copy()
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
        df = df.dropna(subset=[CASE_COL, ACTIVITY_COL, TIMESTAMP_COL])
        df[CASE_COL] = df[CASE_COL].astype(str)
        df[ACTIVITY_COL] = df[ACTIVITY_COL].astype(str)
        df = df.sort_values([CASE_COL, TIMESTAMP_COL]).reset_index(drop=True)
        return df

    rows = []
    for _, row in df.iterrows():
        case_id = row[CASE_COL]

        activities = str(row[ACTIVITY_COL]).split("/")
        timestamps = str(row[TIMESTAMP_COL]).split("/")

        if len(activities) != len(timestamps):
            print(
                f"Skipping case {case_id}: "
                f"{len(activities)} activities but {len(timestamps)} timestamps"
            )
            continue

        groups = None
        lifecycles = None

        if RESOURCE_COL in df.columns and pd.notna(row.get(RESOURCE_COL)):
            groups = str(row[RESOURCE_COL]).split("/")

        if LIFECYCLE_COL in df.columns and pd.notna(row.get(LIFECYCLE_COL)):
            lifecycles = str(row[LIFECYCLE_COL]).split("/")

        for i, (activity, timestamp) in enumerate(zip(activities, timestamps)):
            new_row = {
                CASE_COL: case_id,
                ACTIVITY_COL: activity.strip(),
                TIMESTAMP_COL: timestamp.strip(),
            }

            if groups is not None and i < len(groups):
                new_row[RESOURCE_COL] = groups[i].strip()

            if lifecycles is not None and i < len(lifecycles):
                new_row[LIFECYCLE_COL] = lifecycles[i].strip()

            rows.append(new_row)

    expanded = pd.DataFrame(rows)

    expanded[TIMESTAMP_COL] = pd.to_datetime(expanded[TIMESTAMP_COL], errors="coerce")
    expanded = expanded.dropna(subset=[CASE_COL, ACTIVITY_COL, TIMESTAMP_COL])

    expanded[CASE_COL] = expanded[CASE_COL].astype(str)
    expanded[ACTIVITY_COL] = expanded[ACTIVITY_COL].astype(str)

    expanded = expanded.sort_values([CASE_COL, TIMESTAMP_COL]).reset_index(drop=True)

    return expanded

def get_case_bounds(df):
    return (
        df.groupby(CASE_COL)[TIMESTAMP_COL]
        .agg(case_start="min", case_end="max")
        .reset_index()
        .sort_values("case_start")
        .reset_index(drop=True)
    )


def add_case_bounds(df):
    bounds = get_case_bounds(df)
    return df.merge(bounds, on=CASE_COL, how="left")


def case_duration_seconds(df):
    bounds = get_case_bounds(df)
    durations = (bounds["case_end"] - bounds["case_start"]).dt.total_seconds()
    return durations[durations >= 0]


def create_real_case_windows(real_df, target_cases, step_fraction=0.5):

    bounds = get_case_bounds(real_df)

    if target_cases < MIN_WINDOW_CASES:
        raise ValueError(
            f"Synthetic log has only {target_cases} cases. "
            f"This is too small for stable workload evaluation."
        )

    step = max(1, int(target_cases * step_fraction))
    windows = []

    for start_idx in range(0, len(bounds) - target_cases + 1, step):
        window_cases = bounds.iloc[start_idx:start_idx + target_cases][CASE_COL]
        window_df = real_df[real_df[CASE_COL].isin(window_cases)].copy()

        window_start = bounds.iloc[start_idx]["case_start"]
        window_end = bounds.iloc[start_idx + target_cases - 1]["case_start"]

        windows.append({
            "window_id": len(windows) + 1,
            "window_start": window_start,
            "window_end": window_end,
            "case_count": window_df[CASE_COL].nunique(),
            "event_count": len(window_df),
            "df": window_df.sort_values([CASE_COL, TIMESTAMP_COL]).reset_index(drop=True)
        })

    return windows
def get_wip_profile(df, points=50):

    bounds = get_case_bounds(df)
    total_cases = len(bounds)

    start = bounds["case_start"].min()
    end = bounds["case_end"].max()

    if start == end:
        return np.zeros(points)

    timestamps = pd.date_range(start=start, end=end, periods=points)

    wip_values = []

    for t in timestamps:
        active = (
            (bounds["case_start"] <= t)
            & (bounds["case_end"] >= t)
        ).sum()

        wip_pct = (active / total_cases) * 100 if total_cases > 0 else 0
        wip_values.append(wip_pct)

    return np.array(wip_values)


def label_workload_windows(windows):
    workload_scores = []

    for w in windows:
        profile = get_wip_profile(w["df"], points=PROFILE_POINTS)
        score = np.mean(profile)
        w["mean_wip_pct"] = score
        w["max_wip_pct"] = np.max(profile)
        workload_scores.append(score)

    q33 = np.quantile(workload_scores, 0.33)
    q66 = np.quantile(workload_scores, 0.66)

    for w in windows:
        score = w["mean_wip_pct"]

        if score <= q33:
            w["workload_level"] = "Low workload"
        elif score <= q66:
            w["workload_level"] = "Medium workload"
        else:
            w["workload_level"] = "High workload"

    return windows


def compute_wip_distribution_distance_pct(real_window_df, synthetic_df):

    real_profile = get_wip_profile(real_window_df, points=PROFILE_POINTS)
    syn_profile = get_wip_profile(synthetic_df, points=PROFILE_POINTS)

    if wasserstein_distance is not None:
        distance = wasserstein_distance(real_profile, syn_profile)
    else:
        distance = np.mean(np.abs(np.sort(real_profile) - np.sort(syn_profile)))

    return distance

def ngrams(sequence, n):
    if len(sequence) < n:
        return []
    return [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]


def get_ngram_distribution(df, n=3):
    counter = Counter()

    for _, case_df in df.groupby(CASE_COL):
        seq = (
            case_df.sort_values(TIMESTAMP_COL)[ACTIVITY_COL]
            .astype(str)
            .tolist()
        )

        counter.update(ngrams(seq, n))

    total = sum(counter.values())

    if total == 0:
        return {}

    return {k: v / total for k, v in counter.items()}


def compute_ngd_percentage(real_df, synthetic_df, n=3):
    real_dist = get_ngram_distribution(real_df, n=n)
    syn_dist = get_ngram_distribution(synthetic_df, n=n)

    all_ngrams = set(real_dist.keys()) | set(syn_dist.keys())

    if not all_ngrams:
        return np.nan

    tvd = 0.5 * sum(
        abs(real_dist.get(g, 0) - syn_dist.get(g, 0))
        for g in all_ngrams
    )

    return tvd * 100

def remaining_cycle_times_at_relative_point(df, relative_position):
    bounds = get_case_bounds(df)

    start = bounds["case_start"].min()
    end = bounds["case_end"].max()

    if start == end:
        return np.array([])

    t = start + (end - start) * relative_position

    active = bounds[
        (bounds["case_start"] <= t)
        & (bounds["case_end"] >= t)
    ].copy()

    if active.empty:
        return np.array([])

    rct = (active["case_end"] - t).dt.total_seconds()
    rct = rct[rct >= 0]

    return rct.to_numpy()


def fallback_emd_1d(x, y):
    if len(x) == 0 or len(y) == 0:
        return np.nan

    q = np.linspace(0, 1, 101)
    xq = np.quantile(x, q)
    yq = np.quantile(y, q)

    return np.mean(np.abs(xq - yq))


def compute_profile_rctd_percentage(real_df, synthetic_df, points=50):

    relative_points = np.linspace(0.05, 0.95, points)

    rctd_values = []
    raw_seconds_values = []

    for p in relative_points:
        real_rct = remaining_cycle_times_at_relative_point(real_df, p)
        syn_rct = remaining_cycle_times_at_relative_point(synthetic_df, p)

        if len(real_rct) == 0 or len(syn_rct) == 0:
            continue

        if wasserstein_distance is not None:
            distance = wasserstein_distance(real_rct, syn_rct)
        else:
            distance = fallback_emd_1d(real_rct, syn_rct)

        real_mean = np.mean(real_rct)

        if real_mean <= 0:
            continue

        rctd_pct = (distance / real_mean) * 100

        rctd_values.append(rctd_pct)
        raw_seconds_values.append(distance)

    if not rctd_values:
        return {
            "R_CTD_pct": np.nan,
            "R_CTD_seconds": np.nan,
            "valid_rctd_points": 0
        }

    return {
        "R_CTD_pct": np.mean(rctd_values),
        "R_CTD_seconds": np.mean(raw_seconds_values),
        "valid_rctd_points": len(rctd_values)
    }

def compute_case_duration_distance_pct(real_df, synthetic_df):
    real_durations = case_duration_seconds(real_df)
    syn_durations = case_duration_seconds(synthetic_df)

    if len(real_durations) == 0 or len(syn_durations) == 0:
        return {
            "case_duration_distance_pct": np.nan,
            "case_duration_distance_seconds": np.nan,
            "real_mean_case_duration_seconds": np.nan,
            "synthetic_mean_case_duration_seconds": np.nan
        }

    if wasserstein_distance is not None:
        distance = wasserstein_distance(real_durations, syn_durations)
    else:
        distance = fallback_emd_1d(real_durations, syn_durations)

    real_mean = real_durations.mean()

    return {
        "case_duration_distance_pct": (distance / real_mean) * 100 if real_mean > 0 else np.nan,
        "case_duration_distance_seconds": distance,
        "real_mean_case_duration_seconds": real_mean,
        "synthetic_mean_case_duration_seconds": syn_durations.mean()
    }


def compare_global(real_df, synthetic_df):
    duration_result = compute_case_duration_distance_pct(real_df, synthetic_df)

    row = {
        "Comparison": "Full real log vs full synthetic log",
        "Real Cases": real_df[CASE_COL].nunique(),
        "Synthetic Cases": synthetic_df[CASE_COL].nunique(),
        "Real Events": len(real_df),
        "Synthetic Events": len(synthetic_df),
        "NGD %": compute_ngd_percentage(real_df, synthetic_df, n=NGRAM_N),
        "WiP Distribution Distance %": compute_wip_distribution_distance_pct(real_df, synthetic_df),
        "Case Duration Distance %": duration_result["case_duration_distance_pct"],
        "Case Duration Distance Seconds": duration_result["case_duration_distance_seconds"],
        "Real Mean Case Duration Seconds": duration_result["real_mean_case_duration_seconds"],
        "Synthetic Mean Case Duration Seconds": duration_result["synthetic_mean_case_duration_seconds"]
    }

    return pd.DataFrame([row])

def compare_synthetic_against_real_windows(real_df, synthetic_df):
    synthetic_case_count = synthetic_df[CASE_COL].nunique()

    windows = create_real_case_windows(
        real_df=real_df,
        target_cases=synthetic_case_count,
        step_fraction=WINDOW_STEP_FRACTION
    )

    windows = label_workload_windows(windows)

    rows = []

    for w in windows:
        real_window_df = w["df"]

        rctd = compute_profile_rctd_percentage(
            real_df=real_window_df,
            synthetic_df=synthetic_df,
            points=PROFILE_POINTS
        )

        duration_result = compute_case_duration_distance_pct(
            real_window_df,
            synthetic_df
        )

        row = {
            "Window ID": w["window_id"],
            "Workload Level": w["workload_level"],
            "Window Start": w["window_start"],
            "Window End": w["window_end"],

            "Real Window Cases": real_window_df[CASE_COL].nunique(),
            "Synthetic Cases": synthetic_df[CASE_COL].nunique(),
            "Real Window Events": len(real_window_df),
            "Synthetic Events": len(synthetic_df),

            "Real Mean WiP %": w["mean_wip_pct"],
            "Real Max WiP %": w["max_wip_pct"],
            "Synthetic Mean WiP %": np.mean(get_wip_profile(synthetic_df, points=PROFILE_POINTS)),
            "Synthetic Max WiP %": np.max(get_wip_profile(synthetic_df, points=PROFILE_POINTS)),

            "WiP Distribution Distance %": compute_wip_distribution_distance_pct(
                real_window_df,
                synthetic_df
            ),

            "NGD %": compute_ngd_percentage(
                real_window_df,
                synthetic_df,
                n=NGRAM_N
            ),

            "R-CTD %": rctd["R_CTD_pct"],
            "R-CTD Seconds": rctd["R_CTD_seconds"],
            "Valid R-CTD Points": rctd["valid_rctd_points"],

            "Case Duration Distance %": duration_result["case_duration_distance_pct"],
            "Case Duration Distance Seconds": duration_result["case_duration_distance_seconds"],
            "Real Mean Case Duration Seconds": duration_result["real_mean_case_duration_seconds"],
            "Synthetic Mean Case Duration Seconds": duration_result["synthetic_mean_case_duration_seconds"],
        }

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_by_workload(window_results):
    summary = (
        window_results
        .groupby("Workload Level")
        .agg(
            Windows=("Window ID", "count"),
            Mean_NGD_pct=("NGD %", "mean"),
            Std_NGD_pct=("NGD %", "std"),
            Mean_R_CTD_pct=("R-CTD %", "mean"),
            Std_R_CTD_pct=("R-CTD %", "std"),
            Mean_WiP_Distance_pct=("WiP Distribution Distance %", "mean"),
            Std_WiP_Distance_pct=("WiP Distribution Distance %", "std"),
            Mean_Case_Duration_Distance_pct=("Case Duration Distance %", "mean"),
            Std_Case_Duration_Distance_pct=("Case Duration Distance %", "std"),
        )
        .reset_index()
    )

    return summary


def get_best_matching_windows(window_results, top_n=10):

    df = window_results.copy()

    metric_cols = [
        "NGD %",
        "R-CTD %",
        "WiP Distribution Distance %"
    ]

    df["Overall Distance Score"] = df[metric_cols].mean(axis=1, skipna=True)

    return df.sort_values("Overall Distance Score").head(top_n)

real_log = load_event_log(REAL_LOG_PATH)
synthetic_log = load_event_log(SYNTHETIC_LOG_PATH)

real_log = add_case_bounds(real_log)
synthetic_log = add_case_bounds(synthetic_log)

print("\n  Dataset sizes  ")
print(f"Real cases: {real_log[CASE_COL].nunique()}")
print(f"Synthetic cases: {synthetic_log[CASE_COL].nunique()}")
print(f"Real events: {len(real_log)}")
print(f"Synthetic events: {len(synthetic_log)}")

global_results = compare_global(real_log, synthetic_log)
window_results = compare_synthetic_against_real_windows(real_log, synthetic_log)
workload_summary = summarize_by_workload(window_results)
best_windows = get_best_matching_windows(window_results, top_n=10)

print("\n  Global comparison  ")
print(global_results)

print("\n  Workload summary  ")
print(workload_summary)

print("\n  Best matching real windows  ")
print(best_windows[
    [
        "Window ID",
        "Workload Level",
        "Overall Distance Score",
        "NGD %",
        "R-CTD %",
        "WiP Distribution Distance %",
        "Case Duration Distance %"
    ]
])

global_results.to_csv(
    os.path.join(OUTPUT_DIR, "global_comparison.csv"),
    index=False
)

window_results.to_csv(
    os.path.join(OUTPUT_DIR, "windowed_real_comparison.csv"),
    index=False
)

workload_summary.to_csv(
    os.path.join(OUTPUT_DIR, "workload_summary.csv"),
    index=False
)

best_windows.to_csv(
    os.path.join(OUTPUT_DIR, "best_matching_real_windows.csv"),
    index=False
)

print(f"\nSaved results to: {OUTPUT_DIR}")