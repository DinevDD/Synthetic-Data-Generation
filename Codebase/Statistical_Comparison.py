import os
import pm4py
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity



original_path = "Data/Sepsis Cases - Event Log.xes.gz"
synthetic_path = "Data/data.csv"

output_folder = "comparison_results"
os.makedirs(output_folder, exist_ok=True)

TOP_N = 20



def load_original_xes(path):
    log = pm4py.read_xes(path)
    df = pm4py.convert_to_dataframe(log)

    df["time:timestamp"] = pd.to_datetime(
        df["time:timestamp"],
        utc=True,
        errors="coerce"
    )

    df = df.sort_values(["case:concept:name", "time:timestamp"])

    return df


def load_csv_event_log_dataframe(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)

    print(f"    Raw rows  : {len(df)}")
    print(f"    Columns   : {list(df.columns)}")

    required_cols = ["Case ID", "Activity", "Timestamp"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise KeyError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    if "Group" not in df.columns:
        df["Group"] = None

    if "Lifecycle" not in df.columns:
        df["Lifecycle"] = "complete"

    event_rows = []

    for row_idx, row in df.iterrows():
        case_id = str(row["Case ID"]).strip()

        activities = [
            part.strip()
            for part in str(row["Activity"]).split("/")
            if part.strip()
        ]

        timestamp_parts = [
            part.strip()
            for part in str(row["Timestamp"]).split("/")
            if part.strip()
        ]

        group_parts = [
            part.strip()
            for part in str(row.get("Group", "")).split("/")
            if part.strip()
        ]

        lifecycle_parts = [
            part.strip()
            for part in str(row.get("Lifecycle", "complete")).split("/")
            if part.strip()
        ]

        if not activities:
            raise ValueError(f"Row {row_idx} has no activity: {row.to_dict()}")

        if not timestamp_parts:
            raise ValueError(f"Row {row_idx} has no timestamp: {row.to_dict()}")

        for idx, activity in enumerate(activities):
            if len(timestamp_parts) == len(activities):
                timestamp_value = timestamp_parts[idx]

            elif len(timestamp_parts) == 1:
                timestamp_value = (
                    pd.to_datetime(
                        timestamp_parts[0],
                        utc=True,
                        errors="coerce"
                    )
                    + pd.Timedelta(seconds=idx)
                )
            else:
                raise ValueError(
                    f"Row {row_idx} has {len(activities)} activities but "
                    f"{len(timestamp_parts)} timestamps. Use either one timestamp "
                    "or one timestamp per activity."
                )

            if len(group_parts) == len(activities):
                group_value = group_parts[idx]
            elif len(group_parts) == 1:
                group_value = group_parts[0]
            else:
                group_value = None

            if len(lifecycle_parts) == len(activities):
                lifecycle_value = lifecycle_parts[idx]
            elif len(lifecycle_parts) == 1:
                lifecycle_value = lifecycle_parts[0]
            else:
                lifecycle_value = "complete"

            event_rows.append({
                "case:concept:name": case_id,
                "concept:name": activity,
                "time:timestamp": timestamp_value,
                "org:group": group_value,
                "lifecycle:transition": lifecycle_value,
            })

    event_df = pd.DataFrame(event_rows)

    event_df["time:timestamp"] = pd.to_datetime(
        event_df["time:timestamp"],
        utc=True,
        errors="coerce"
    )

    if event_df["time:timestamp"].isna().any():
        bad_rows = event_df[event_df["time:timestamp"].isna()]
        raise ValueError(f"Some timestamps could not be parsed:\n{bad_rows}")

    event_df = event_df.sort_values(
        ["case:concept:name", "time:timestamp"]
    ).reset_index(drop=True)

    print(f"    Event rows : {len(event_df)}")
    print(f"    Traces     : {event_df['case:concept:name'].nunique()}")
    print(f"    Events     : {len(event_df)}")
    print(
        f"    Activities ({event_df['concept:name'].nunique()}): "
        f"{sorted(event_df['concept:name'].dropna().unique())}"
    )
    return event_df


def load_synthetic_csv(path):
    print(f"\nLoading synthetic CSV: {path}")
    return load_csv_event_log_dataframe(path)
def jaccard_similarity(set_a, set_b):
    union = set_a | set_b

    if len(union) == 0:
        return 1.0

    return len(set_a & set_b) / len(union)

def safe_cosine(vec_a, vec_b):
    if np.sum(vec_a) == 0 or np.sum(vec_b) == 0:
        return 0.0

    return cosine_similarity([vec_a], [vec_b])[0][0]

def safe_js_divergence(vec_a, vec_b):
    if np.sum(vec_a) == 0 or np.sum(vec_b) == 0:
        return 0.0

    js_distance = jensenshannon(vec_a, vec_b, base=2)

    return js_distance ** 2

def distribution_vectors(dist_a, dist_b):
    all_keys = sorted(set(dist_a.index) | set(dist_b.index))

    vec_a = dist_a.reindex(all_keys, fill_value=0).values
    vec_b = dist_b.reindex(all_keys, fill_value=0).values

    return all_keys, vec_a, vec_b


def distribution_similarity_metrics(dist_a, dist_b):
    all_keys, vec_a, vec_b = distribution_vectors(dist_a, dist_b)

    support_a = set(dist_a[dist_a > 0].index)
    support_b = set(dist_b[dist_b > 0].index)

    return {
        "keys": all_keys,
        "vec_a": vec_a,
        "vec_b": vec_b,
        "cosine": safe_cosine(vec_a, vec_b),
        "js_divergence": safe_js_divergence(vec_a, vec_b),
        "jaccard": jaccard_similarity(support_a, support_b),
    }


def save_distribution_comparison(
    keys,
    vec_a,
    vec_b,
    key_column,
    output_filename
):
    comparison = pd.DataFrame({
        key_column: keys,
        "original_percentage": vec_a * 100,
        "synthetic_percentage": vec_b * 100,
        "absolute_difference": np.abs(vec_a - vec_b) * 100
    })

    comparison = comparison.sort_values(
        "absolute_difference",
        ascending=False
    )

    comparison.to_csv(
        os.path.join(output_folder, output_filename),
        index=False
    )

    return comparison


def summarize_numeric_series(series, name):
    return {
        f"{name}_mean": series.mean(),
        f"{name}_median": series.median(),
        f"{name}_std": series.std(),
        f"{name}_min": series.min(),
        f"{name}_max": series.max(),
        f"{name}_q25": series.quantile(0.25),
        f"{name}_q75": series.quantile(0.75),
    }


def get_activity_distribution(df):
    return df["concept:name"].value_counts(normalize=True)


def get_final_event_distribution(df):
    final_events = (
        df.sort_values(["case:concept:name", "time:timestamp"])
        .groupby("case:concept:name")
        .tail(1)
    )

    return final_events["concept:name"].value_counts(normalize=True)

def get_case_variants(df):
    variants = (
        df.sort_values(["case:concept:name", "time:timestamp"])
        .groupby("case:concept:name")["concept:name"]
        .apply(lambda events: tuple(events))
    )

    return variants
def get_case_variant_distribution(df):
    variants = get_case_variants(df)

    variant_dist = variants.value_counts(normalize=True)

    variant_dist.index = [
        " -> ".join(variant)
        for variant in variant_dist.index
    ]

    return variant_dist

def get_case_durations_hours(df):
    case_stats = (
        df
        .groupby("case:concept:name")
        .agg(
            start_time=("time:timestamp", "min"),
            end_time=("time:timestamp", "max")
        )
    )

    case_durations = (
        case_stats["end_time"] - case_stats["start_time"]
    ).dt.total_seconds() / 3600

    return case_durations

def get_events_per_case(df):
    return df.groupby("case:concept:name").size()

def get_events_per_case_distribution(df):
    events_per_case = get_events_per_case(df)

    return events_per_case.value_counts(normalize=True).sort_index()


print("\nLoading original XES log:")
original_df = load_original_xes(original_path)

synthetic_df = load_synthetic_csv(synthetic_path)


original_activity_dist = get_activity_distribution(original_df)
synthetic_activity_dist = get_activity_distribution(synthetic_df)

activity_metrics = distribution_similarity_metrics(
    original_activity_dist,
    synthetic_activity_dist
)

activity_comparison = save_distribution_comparison(
    activity_metrics["keys"],
    activity_metrics["vec_a"],
    activity_metrics["vec_b"],
    "activity",
    "activity_distribution_comparison.csv"
)

activity_comparison.head(TOP_N).to_csv(
    os.path.join(output_folder, "top_activity_distribution_comparison.csv"),
    index=False
)


original_final_event_dist = get_final_event_distribution(original_df)
synthetic_final_event_dist = get_final_event_distribution(synthetic_df)

final_event_metrics = distribution_similarity_metrics(
    original_final_event_dist,
    synthetic_final_event_dist
)

final_event_comparison = save_distribution_comparison(
    final_event_metrics["keys"],
    final_event_metrics["vec_a"],
    final_event_metrics["vec_b"],
    "final_event",
    "final_event_distribution_comparison.csv"
)

final_event_comparison.head(TOP_N).to_csv(
    os.path.join(output_folder, "top_final_event_distribution_comparison.csv"),
    index=False
)

original_variant_dist = get_case_variant_distribution(original_df)
synthetic_variant_dist = get_case_variant_distribution(synthetic_df)

variant_metrics = distribution_similarity_metrics(
    original_variant_dist,
    synthetic_variant_dist
)

variant_comparison = save_distribution_comparison(
    variant_metrics["keys"],
    variant_metrics["vec_a"],
    variant_metrics["vec_b"],
    "case_variant",
    "case_variant_distribution_comparison.csv"
)

variant_comparison.head(TOP_N).to_csv(
    os.path.join(output_folder, "top_case_variant_distribution_comparison.csv"),
    index=False
)

original_case_durations = get_case_durations_hours(original_df)
synthetic_case_durations = get_case_durations_hours(synthetic_df)

case_duration_wasserstein = wasserstein_distance(
    original_case_durations,
    synthetic_case_durations
)

duration_comparison = pd.DataFrame({
    "original_case_duration_hours": pd.Series(original_case_durations.values),
    "synthetic_case_duration_hours": pd.Series(synthetic_case_durations.values)
})

duration_comparison.to_csv(
    os.path.join(output_folder, "case_duration_comparison.csv"),
    index=False
)

original_events_per_case = get_events_per_case(original_df)
synthetic_events_per_case = get_events_per_case(synthetic_df)

original_events_per_case_dist = get_events_per_case_distribution(original_df)
synthetic_events_per_case_dist = get_events_per_case_distribution(synthetic_df)

events_per_case_metrics = distribution_similarity_metrics(
    original_events_per_case_dist,
    synthetic_events_per_case_dist
)

events_per_case_wasserstein = wasserstein_distance(
    original_events_per_case,
    synthetic_events_per_case
)

events_per_case_comparison = save_distribution_comparison(
    events_per_case_metrics["keys"],
    events_per_case_metrics["vec_a"],
    events_per_case_metrics["vec_b"],
    "events_per_case",
    "events_per_case_distribution_comparison.csv"
)


case_duration_summary = {
    **summarize_numeric_series(
        original_case_durations,
        "original_case_duration_hours"
    ),
    **summarize_numeric_series(
        synthetic_case_durations,
        "synthetic_case_duration_hours"
    ),
}

events_per_case_summary = {
    **summarize_numeric_series(
        original_events_per_case,
        "original_events_per_case"
    ),
    **summarize_numeric_series(
        synthetic_events_per_case,
        "synthetic_events_per_case"
    ),
}

summary_statistics = pd.DataFrame(
    list(case_duration_summary.items())
    + list(events_per_case_summary.items()),
    columns=["Statistic", "Value"]
)

summary_statistics.to_csv(
    os.path.join(output_folder, "case_duration_and_length_summary.csv"),
    index=False
)


results = pd.DataFrame({
    "Metric": [
        "Cosine similarity of event distribution",
        "Jaccard similarity of event distribution",
        "Jensen-Shannon divergence of event distribution",

        "Final event cosine similarity",
        "Final event Jaccard similarity",
        "Final event Jensen-Shannon divergence",

        "Case variant cosine similarity",
        "Case variant Jaccard similarity",
        "Case variant Jensen-Shannon divergence",

        "Wasserstein distance of case durations in hours",

        "Events-per-case cosine similarity",
        "Events-per-case Jaccard similarity",
        "Events-per-case Jensen-Shannon divergence",
        "Events-per-case Wasserstein distance",
    ],
    "Value": [
        activity_metrics["cosine"],
        activity_metrics["jaccard"],
        activity_metrics["js_divergence"],

        final_event_metrics["cosine"],
        final_event_metrics["jaccard"],
        final_event_metrics["js_divergence"],

        variant_metrics["cosine"],
        variant_metrics["jaccard"],
        variant_metrics["js_divergence"],

        case_duration_wasserstein,

        events_per_case_metrics["cosine"],
        events_per_case_metrics["jaccard"],
        events_per_case_metrics["js_divergence"],
        events_per_case_wasserstein,
    ]
})

results.to_csv(
    os.path.join(output_folder, "comparison_metrics.csv"),
    index=False
)

print("\nComparison metrics:")
print(results.to_string(index=False))

print(f"\nAll comparison results saved in folder: {output_folder}")