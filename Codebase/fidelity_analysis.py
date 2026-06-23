import os
import gzip
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import ks_2samp, wasserstein_distance


ORIGINAL_PATH = "Data/filtered_log.xes.gz"
SYNTHETIC_PATH = "DeepSeek/TS5.csv"

OUTPUT_PATH = "fidelity_analysis/results.csv"

CASE_COL = "Case ID"
ACTIVITY_COL = "Activity"
TIMESTAMP_COL = "Timestamp"
SEPARATOR = "/"



def split_cell(value):
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(SEPARATOR)]


def expand_truncated_log(df):
    rows = []

    for _, row in df.iterrows():
        case_id = row[CASE_COL]
        activities = split_cell(row[ACTIVITY_COL])

        if len(activities) == 0:
            continue

        for i, activity in enumerate(activities):
            event = {
                CASE_COL: case_id,
                ACTIVITY_COL: activity
            }

            for col in df.columns:
                if col in [CASE_COL, ACTIVITY_COL]:
                    continue

                values = split_cell(row[col])

                if len(values) == len(activities):
                    event[col] = values[i]
                else:
                    event[col] = row[col]

            rows.append(event)

    expanded = pd.DataFrame(rows)

    if TIMESTAMP_COL in expanded.columns:
        expanded[TIMESTAMP_COL] = pd.to_datetime(
            expanded[TIMESTAMP_COL],
            errors="coerce",
            utc=True
        ).dt.tz_localize(None)

    return expanded


def local_name(tag):
    return tag.split("}")[-1]


def iter_xes_children(parent, wanted_name):
    for child in parent:
        if local_name(child.tag) == wanted_name:
            yield child


def read_xes_or_xes_gz(path):
    if path.lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            tree = ET.parse(f)
    else:
        tree = ET.parse(path)

    root = tree.getroot()
    rows = []

    for trace_index, trace in enumerate(iter_xes_children(root, "trace")):
        case_id = None
        case_attrs = {}

        for child in trace:
            tag = local_name(child.tag)
            key = child.attrib.get("key")
            value = child.attrib.get("value")

            if tag in {"string", "date", "int", "float", "boolean"} and key:
                if key == "concept:name":
                    case_id = value
                else:
                    case_attrs[f"case:{key}"] = value

        if case_id is None:
            case_id = f"case_{trace_index}"

        for event in iter_xes_children(trace, "event"):
            row = {CASE_COL: case_id}
            row.update(case_attrs)

            for attr in event:
                tag = local_name(attr.tag)
                key = attr.attrib.get("key")
                value = attr.attrib.get("value")

                if not key:
                    continue

                if key == "concept:name":
                    row[ACTIVITY_COL] = value
                elif key == "time:timestamp":
                    row[TIMESTAMP_COL] = value
                elif key == "org:group":
                    row["Group"] = value
                elif key == "lifecycle:transition":
                    row["Lifecycle"] = value
                else:
                    row[key] = value

            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No events were read from the XES file.")

    if ACTIVITY_COL not in df.columns:
        raise ValueError(
            f"XES file does not contain event activity key 'concept:name'. "
            f"Available columns: {list(df.columns)}"
        )

    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = pd.to_datetime(
            df[TIMESTAMP_COL],
            errors="coerce",
            utc=True
        ).dt.tz_localize(None)

    return df


def load_event_log(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist: {path}")

    path_lower = path.lower()

    if path_lower.endswith(".xes") or path_lower.endswith(".xes.gz"):
        return read_xes_or_xes_gz(path)

    if path_lower.endswith(".gz"):
        try:
            return read_xes_or_xes_gz(path)
        except Exception:
            return pd.read_csv(path, dtype=str, compression="gzip")

    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()

    rename_map = {
        "case:concept:name": CASE_COL,
        "case": CASE_COL,
        "case_id": CASE_COL,
        "case id": CASE_COL,
        "CaseID": CASE_COL,
        "Case ID": CASE_COL,

        "concept:name": ACTIVITY_COL,
        "activity": ACTIVITY_COL,
        "Activity": ACTIVITY_COL,

        "time:timestamp": TIMESTAMP_COL,
        "timestamp": TIMESTAMP_COL,
        "Timestamp": TIMESTAMP_COL,

        "org:group": "Group",
        "lifecycle:transition": "Lifecycle"
    }

    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

    if CASE_COL not in df.columns:
        raise ValueError(
            f"Missing column: {CASE_COL}. Available columns: {list(df.columns)}"
        )

    if ACTIVITY_COL not in df.columns:
        raise ValueError(
            f"Missing column: {ACTIVITY_COL}. Available columns: {list(df.columns)}"
        )

    is_truncated = df[ACTIVITY_COL].astype(str).str.contains(SEPARATOR, regex=False).any()

    if is_truncated:
        df = expand_truncated_log(df)
    else:
        if TIMESTAMP_COL in df.columns:
            df[TIMESTAMP_COL] = pd.to_datetime(
                df[TIMESTAMP_COL],
                errors="coerce",
                utc=True
            ).dt.tz_localize(None)

    return df

def get_traces(df):
    temp = df.copy()

    if TIMESTAMP_COL in temp.columns:
        temp = temp.sort_values([CASE_COL, TIMESTAMP_COL])
    else:
        temp = temp.sort_values([CASE_COL])

    return (
        temp.groupby(CASE_COL)[ACTIVITY_COL]
        .apply(lambda x: tuple(x.astype(str)))
        .to_dict()
    )


def get_trace_variants(df):
    return Counter(get_traces(df).values())


def get_trace_lengths(df):
    return np.array([len(trace) for trace in get_traces(df).values()], dtype=float)


def get_activity_distribution(df):
    return Counter(df[ACTIVITY_COL].astype(str))


def get_throughput_times(df):
    if TIMESTAMP_COL not in df.columns:
        return np.array([])

    temp = df.dropna(subset=[TIMESTAMP_COL]).copy()

    if temp.empty:
        return np.array([])

    grouped = temp.groupby(CASE_COL)[TIMESTAMP_COL]
    throughput = grouped.max() - grouped.min()

    return throughput.dt.total_seconds().to_numpy(dtype=float)

def aligned_probabilities(counter_a, counter_b):
    keys = sorted(set(counter_a.keys()) | set(counter_b.keys()))

    a = np.array([counter_a.get(k, 0) for k in keys], dtype=float)
    b = np.array([counter_b.get(k, 0) for k in keys], dtype=float)

    if a.sum() > 0:
        a /= a.sum()

    if b.sum() > 0:
        b /= b.sum()

    return a, b


def hellinger_similarity(counter_a, counter_b):
    p, q = aligned_probabilities(counter_a, counter_b)

    if len(p) == 0:
        return np.nan

    distance = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
    return 1 - distance


def ks_similarity(values_a, values_b):
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)

    values_a = values_a[~np.isnan(values_a)]
    values_b = values_b[~np.isnan(values_b)]

    if len(values_a) == 0 or len(values_b) == 0:
        return np.nan

    return 1 - ks_2samp(values_a, values_b).statistic


def normalized_emd_similarity(values_a, values_b):
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)

    values_a = values_a[~np.isnan(values_a)]
    values_b = values_b[~np.isnan(values_b)]

    if len(values_a) == 0 or len(values_b) == 0:
        return np.nan

    emd = wasserstein_distance(values_a, values_b)

    combined = np.concatenate([values_a, values_b])
    value_range = combined.max() - combined.min()

    if value_range == 0:
        return 1.0 if emd == 0 else 0.0

    return max(0.0, 1 - (emd / value_range))


def levenshtein_distance(seq1, seq2):
    seq1 = list(seq1)
    seq2 = list(seq2)

    m = len(seq1)
    n = len(seq2)

    dp = np.zeros((m + 1, n + 1), dtype=int)

    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            substitution_cost = 0 if seq1[i - 1] == seq2[j - 1] else 1

            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + substitution_cost
            )

    return dp[m, n]


def normalized_trace_distance(trace_a, trace_b):
    max_len = max(len(trace_a), len(trace_b))

    if max_len == 0:
        return 0.0

    return levenshtein_distance(trace_a, trace_b) / max_len


def trace_variant_emd_similarity(original_df, synthetic_df, max_variants=1000):
    original_variants = get_trace_variants(original_df)
    synthetic_variants = get_trace_variants(synthetic_df)

    if len(original_variants) == 0 or len(synthetic_variants) == 0:
        return np.nan

    original_items = original_variants.most_common(max_variants)
    synthetic_items = synthetic_variants.most_common(max_variants)

    original_traces = [x[0] for x in original_items]
    synthetic_traces = [x[0] for x in synthetic_items]

    original_weights = np.array([x[1] for x in original_items], dtype=float)
    synthetic_weights = np.array([x[1] for x in synthetic_items], dtype=float)

    original_weights /= original_weights.sum()
    synthetic_weights /= synthetic_weights.sum()

    cost_matrix = np.zeros((len(original_traces), len(synthetic_traces)))

    for i, original_trace in enumerate(original_traces):
        for j, synthetic_trace in enumerate(synthetic_traces):
            cost_matrix[i, j] = normalized_trace_distance(original_trace, synthetic_trace)

    c = cost_matrix.flatten()

    A_eq = []
    b_eq = []

    for i in range(len(original_traces)):
        row = np.zeros(len(original_traces) * len(synthetic_traces))
        row[i * len(synthetic_traces):(i + 1) * len(synthetic_traces)] = 1
        A_eq.append(row)
        b_eq.append(original_weights[i])

    for j in range(len(synthetic_traces)):
        row = np.zeros(len(original_traces) * len(synthetic_traces))
        row[j::len(synthetic_traces)] = 1
        A_eq.append(row)
        b_eq.append(synthetic_weights[j])

    result = linprog(
        c,
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        bounds=(0, None),
        method="highs"
    )

    if not result.success:
        return np.nan

    return max(0.0, 1 - result.fun)

def trace_variant_hellinger_similarity(original_df, synthetic_df):


    original_variants = get_trace_variants(original_df)
    synthetic_variants = get_trace_variants(synthetic_df)

    return hellinger_similarity(original_variants, synthetic_variants)


def infer_attribute_columns(df):
    excluded = {CASE_COL, ACTIVITY_COL, TIMESTAMP_COL}
    return [col for col in df.columns if col not in excluded]


def is_numeric_like(series):
    converted = pd.to_numeric(series, errors="coerce")
    return converted.notna().mean() >= 0.8


def evaluate_attributes(original_df, synthetic_df):
    rows = []

    common_columns = sorted(
        set(infer_attribute_columns(original_df))
        & set(infer_attribute_columns(synthetic_df))
    )

    for col in common_columns:
        original_col = original_df[col]
        synthetic_col = synthetic_df[col]

        if is_numeric_like(original_col) and is_numeric_like(synthetic_col):
            original_values = pd.to_numeric(original_col, errors="coerce").to_numpy()
            synthetic_values = pd.to_numeric(synthetic_col, errors="coerce").to_numpy()

            ks_score = ks_similarity(original_values, synthetic_values)

            rows.append({
                "metric": f"attribute_resemblance::{col}",
                "value": ks_score,
                "interpretation": "Numerical attribute resemblance based on 1 - KS statistic",
                "included_in_paper_resemblance": True
            })

            rows.append({
                "metric": f"attribute_emd_similarity::{col}",
                "value": normalized_emd_similarity(original_values, synthetic_values),
                "interpretation": "Additional numerical attribute Earth Mover similarity",
                "included_in_paper_resemblance": False
            })

        else:
            original_counter = Counter(original_col.fillna("MISSING").astype(str))
            synthetic_counter = Counter(synthetic_col.fillna("MISSING").astype(str))

            hellinger_score = hellinger_similarity(original_counter, synthetic_counter)

            rows.append({
                "metric": f"attribute_resemblance::{col}",
                "value": hellinger_score,
                "interpretation": "Categorical/boolean attribute resemblance based on 1 - Hellinger distance",
                "included_in_paper_resemblance": True
            })

    return rows


def evaluate_fidelity(original_df, synthetic_df):
    rows = []

    original_traces = get_traces(original_df)
    synthetic_traces = get_traces(synthetic_df)

    original_trace_lengths = get_trace_lengths(original_df)
    synthetic_trace_lengths = get_trace_lengths(synthetic_df)

    original_throughput = get_throughput_times(original_df)
    synthetic_throughput = get_throughput_times(synthetic_df)

    rows.append({
        "metric": "num_traces_original",
        "value": len(original_traces),
        "interpretation": "Absolute number of original cases"
    })

    rows.append({
        "metric": "num_traces_synthetic",
        "value": len(synthetic_traces),
        "interpretation": "Absolute number of synthetic cases"
    })

    rows.append({
        "metric": "num_trace_variants_original",
        "value": len(get_trace_variants(original_df)),
        "interpretation": "Absolute number of original trace variants"
    })

    rows.append({
        "metric": "num_trace_variants_synthetic",
        "value": len(get_trace_variants(synthetic_df)),
        "interpretation": "Absolute number of synthetic trace variants"
    })

    rows.append({
        "metric": "activity_occurrence_hellinger_similarity",
        "value": hellinger_similarity(
            get_activity_distribution(original_df),
            get_activity_distribution(synthetic_df)
        ),
        "interpretation": "Similarity of activity frequency distributions"
    })

    rows.append({
        "metric": "trace_length_hellinger_similarity",
        "value": hellinger_similarity(
            Counter(original_trace_lengths),
            Counter(synthetic_trace_lengths)
        ),
        "interpretation": "Similarity of trace length distributions"
    })

    rows.append({
        "metric": "trace_length_emd_similarity",
        "value": normalized_emd_similarity(
            original_trace_lengths,
            synthetic_trace_lengths
        ),
        "interpretation": "Earth Mover similarity of trace length distributions"
    })

    rows.append({
        "metric": "trace_variant_emd_similarity",
        "value": trace_variant_emd_similarity(
            original_df,
            synthetic_df,
            max_variants=300
        ),
        "interpretation": "Earth Mover similarity of trace variant distributions"
    })

    rows.append({
        "metric": "throughput_time_ks_similarity",
        "value": ks_similarity(
            original_throughput,
            synthetic_throughput
        ),
        "interpretation": "Similarity of case duration distributions"
    })

    rows.append({
        "metric": "throughput_time_emd_similarity",
        "value": normalized_emd_similarity(
            original_throughput,
            synthetic_throughput
        ),
        "interpretation": "Earth Mover similarity of case duration distributions"
    })

    rows.append({
        "metric": "trace_variant_hellinger_similarity",
        "value": trace_variant_hellinger_similarity(
            original_df,
            synthetic_df
        ),
        "interpretation": "Similarity of exact case/trace variant frequency distributions"
    })
    rows.extend(evaluate_attributes(original_df, synthetic_df))

    result = pd.DataFrame(rows)

    paper_mask = result.get("included_in_paper_resemblance", False) == True

    paper_resemblance = result.loc[paper_mask, "value"].mean()

    broad_mask = (
            result["metric"].str.contains("similarity", regex=False)
            | result["metric"].str.contains("resemblance", regex=False)
    )

    broad_fidelity = result.loc[broad_mask, "value"].mean()

    result = pd.concat(
        [
            result,
            pd.DataFrame([
                {
                    "metric": "paper_attribute_resemblance_R",
                    "value": paper_resemblance,
                    "interpretation": "Paper-style R: average resemblance over event/trace attributes only",
                    "included_in_paper_resemblance": False
                },
                {
                    "metric": "average_fidelity_similarity",
                    "value": broad_fidelity,
                    "interpretation": "Broader custom average over all normalized fidelity metrics",
                    "included_in_paper_resemblance": False
                }
            ])
        ],
        ignore_index=True
    )

    return result



if __name__ == "__main__":
    original_df = load_event_log(ORIGINAL_PATH)
    synthetic_df = load_event_log(SYNTHETIC_PATH)

    print("Original columns:", list(original_df.columns))
    print("Synthetic columns:", list(synthetic_df.columns))
    print("Original rows:", len(original_df))
    print("Synthetic rows:", len(synthetic_df))

    results = evaluate_fidelity(original_df, synthetic_df)

    print(results.to_string(index=False))

    results.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved fidelity analysis to: {OUTPUT_PATH}")