import os
import pm4py
import pandas as pd
import matplotlib.pyplot as plt


log = pm4py.read_xes("Data/Sepsis Cases - Event Log.xes.gz")
df = pm4py.convert_to_dataframe(log)
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
df = df.sort_values(["case:concept:name", "time:timestamp"])


case_lengths = df.groupby("case:concept:name").size()
threshold = case_lengths.quantile(0.95)
valid_cases = case_lengths[case_lengths <= threshold].index
df_filtered = df[df["case:concept:name"].isin(valid_cases)].copy()
filtered_log = pm4py.convert_to_event_log(df_filtered)
pm4py.write_xes(filtered_log, "Data/filtered_log.xes.gz")



output_folder = "visualizations"
os.makedirs(output_folder, exist_ok=True)

case_stats = (
    df_filtered
    .groupby("case:concept:name")
    .agg(
        start_time=("time:timestamp", "min"),
        end_time=("time:timestamp", "max"),
        events_per_case=("concept:name", "count")
    )
)

case_stats["case_duration"] = case_stats["end_time"] - case_stats["start_time"]
case_stats["case_duration_hours"] = case_stats["case_duration"].dt.total_seconds() / 3600
case_stats["case_duration_days"] = case_stats["case_duration_hours"] / 24

summary_stats = pd.DataFrame({
    "Metric": [
        "Number of events",
        "Number of cases",
        "Number of unique activities",
        "Average events per case",
        "Median events per case",
        "Min events per case",
        "Max events per case",
        "Average case duration in hours",
        "Median case duration in hours",
        "Min case duration in hours",
        "Max case duration in hours"
    ],
    "Value": [
        len(df_filtered),
        df_filtered["case:concept:name"].nunique(),
        df_filtered["concept:name"].nunique(),
        case_stats["events_per_case"].mean(),
        case_stats["events_per_case"].median(),
        case_stats["events_per_case"].min(),
        case_stats["events_per_case"].max(),
        case_stats["case_duration_hours"].mean(),
        case_stats["case_duration_hours"].median(),
        case_stats["case_duration_hours"].min(),
        case_stats["case_duration_hours"].max()
    ]
})

summary_stats.to_csv(
    os.path.join(output_folder, "summary_statistics.csv"),
    index=False
)

print("\nSummary statistics after preprocessing:")
print(summary_stats.to_string(index=False))

print("\nPreprocessing summary:")
print(f"Original number of cases: {df['case:concept:name'].nunique()}")
print(f"Filtered number of cases: {df_filtered['case:concept:name'].nunique()}")
print(f"Removed cases: {df['case:concept:name'].nunique() - df_filtered['case:concept:name'].nunique()}")
print(f"95th percentile event-count threshold: {threshold}")



def save_plot(filename):
    path = os.path.join(output_folder, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")



activity_counts = df_filtered["concept:name"].value_counts().head(20)
plt.figure(figsize=(10, 6))
activity_counts.sort_values().plot(kind="barh")
plt.xlabel("Frequency")
plt.ylabel("Activity")
save_plot("top_15_most_common_events.png")



plt.figure(figsize=(8, 5))
plt.boxplot(case_stats["case_duration_hours"], vert=False)
plt.xlabel("Case Duration in Hours")
save_plot("boxplot_case_durations.png")


plt.figure(figsize=(8, 5))
plt.boxplot(case_stats["events_per_case"], vert=False)
plt.xlabel("Number of Events per Case")
save_plot("boxplot_events_per_case.png")


plt.figure(figsize=(8, 5))
plt.hist(case_stats["case_duration_hours"], bins=30)
plt.xlabel("Case Duration in Hours")
plt.ylabel("Number of Cases")
save_plot("histogram_case_durations.png")



plt.figure(figsize=(8, 5))
plt.hist(case_stats["events_per_case"], bins=30)
plt.xlabel("Number of Events per Case")
plt.ylabel("Number of Cases")
save_plot("histogram_events_per_case.png")



start_activities = (
    df_filtered
    .groupby("case:concept:name")
    .first()["concept:name"]
    .value_counts()
)

plt.figure(figsize=(10, 6))
start_activities.sort_values().plot(kind="barh")
plt.xlabel("Number of Cases")
plt.ylabel("Start Activity")
save_plot("start_activity_distribution.png")



end_activities = (
    df_filtered
    .groupby("case:concept:name")
    .last()["concept:name"]
    .value_counts()
)

plt.figure(figsize=(10, 6))
end_activities.sort_values().plot(kind="barh")
plt.xlabel("Number of Cases")
plt.ylabel("End Activity")
save_plot("end_activity_distribution.png")



variants = (
    df_filtered
    .groupby("case:concept:name")["concept:name"]
    .apply(lambda x: " → ".join(x))
)

variant_counts = variants.value_counts().head(10)

plt.figure(figsize=(12, 6))
variant_counts.sort_values().plot(kind="barh")
plt.xlabel("Number of Cases")
plt.ylabel("Variant")
save_plot("top_10_most_common_variants.png")



duration_percentiles = case_stats["case_duration_hours"].quantile([
    0.00, 0.05, 0.25, 0.50, 0.75, 0.95, 1.00
])

duration_percentiles.to_csv(
    os.path.join(output_folder, "case_duration_percentiles.csv"),
    header=["case_duration_hours"]
)



event_count_percentiles = case_stats["events_per_case"].quantile([
    0.00, 0.05, 0.25, 0.50, 0.75, 0.95, 1.00
])

event_count_percentiles.to_csv(
    os.path.join(output_folder, "events_per_case_percentiles.csv"),
    header=["events_per_case"]
)


activity_percentage = (
    df_filtered["concept:name"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

activity_percentage.to_csv(
    os.path.join(output_folder, "activity_frequency_percentage.csv"),
    header=["percentage"]
)

print(f"\nAll diagrams and statistics saved in folder: {output_folder}")