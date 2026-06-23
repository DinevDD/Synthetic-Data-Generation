import json
import math
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


DFG = Dict[Tuple[str, str], float]
CountDistribution = Dict[str, float]


def load_json_graph(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = ["activities", "start_activities", "end_activities", "edges"]
    missing = [key for key in required_keys if key not in data]

    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")

    return data


def load_pm4py_dfg(path: str) -> dict:
    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    i = 0

    activity_count = int(lines[i])
    i += 1

    activities_list = lines[i:i + activity_count]
    i += activity_count

    activities = {activity: 0.0 for activity in activities_list}

    start_count = int(lines[i])
    i += 1

    start_activities = {}

    for _ in range(start_count):
        idx, count = lines[i].split("x", 1)
        activity = activities_list[int(idx)]
        start_activities[activity] = float(count)
        i += 1

    end_count = int(lines[i])
    i += 1

    end_activities = {}

    for _ in range(end_count):
        idx, count = lines[i].split("x", 1)
        activity = activities_list[int(idx)]
        end_activities[activity] = float(count)
        i += 1

    edges = {}

    for line in lines[i:]:
        left, count = line.rsplit("x", 1)
        source_idx, target_idx = left.split(">", 1)

        source = activities_list[int(source_idx)]
        target = activities_list[int(target_idx)]
        count = float(count)

        edges[f"{source}||{target}"] = count

        activities[source] += count
        activities[target] += count

    return {
        "activities": activities,
        "start_activities": start_activities,
        "end_activities": end_activities,
        "edges": edges,
    }


def load_graph(path: str) -> dict:
    suffix = Path(path).suffix.lower()

    if suffix == ".json":
        return load_json_graph(path)

    if suffix == ".dfg":
        return load_pm4py_dfg(path)

    raise ValueError(f"Unsupported file type: {suffix}")




def extract_dfg_edges(graph: dict) -> DFG:
    dfg = {}

    for edge, count in graph["edges"].items():
        if "||" not in edge:
            raise ValueError(f"Invalid edge format: {edge}")

        source, target = edge.split("||", 1)
        dfg[(source, target)] = float(count)

    return dfg


def extract_counts(graph: dict, key: str) -> CountDistribution:
    return {
        item: float(count)
        for item, count in graph.get(key, {}).items()
    }


def safe_divide(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def total_variation_similarity(original: Dict, synthetic: Dict) -> float:

    all_keys = set(original) | set(synthetic)

    if not all_keys:
        return 1.0

    original_total = sum(original.values())
    synthetic_total = sum(synthetic.values())

    if original_total == 0 and synthetic_total == 0:
        return 1.0

    if original_total == 0 or synthetic_total == 0:
        return 0.0

    total_variation_distance = 0.5 * sum(
        abs(
            original.get(key, 0.0) / original_total
            - synthetic.get(key, 0.0) / synthetic_total
        )
        for key in all_keys
    )

    return 1 - total_variation_distance


def weighted_jaccard(original: Dict, synthetic: Dict) -> float:
    all_keys = set(original) | set(synthetic)

    if not all_keys:
        return 1.0

    numerator = sum(
        min(original.get(key, 0.0), synthetic.get(key, 0.0))
        for key in all_keys
    )

    denominator = sum(
        max(original.get(key, 0.0), synthetic.get(key, 0.0))
        for key in all_keys
    )

    return safe_divide(numerator, denominator)


def cosine_similarity(original: Dict, synthetic: Dict) -> float:
    all_keys = set(original) | set(synthetic)

    if not all_keys:
        return 1.0

    dot_product = sum(
        original.get(key, 0.0) * synthetic.get(key, 0.0)
        for key in all_keys
    )

    original_norm = math.sqrt(
        sum(original.get(key, 0.0) ** 2 for key in all_keys)
    )

    synthetic_norm = math.sqrt(
        sum(synthetic.get(key, 0.0) ** 2 for key in all_keys)
    )

    if original_norm == 0 and synthetic_norm == 0:
        return 1.0

    if original_norm == 0 or synthetic_norm == 0:
        return 0.0

    return dot_product / (original_norm * synthetic_norm)



def edge_precision_recall_f1(original_dfg: DFG, synthetic_dfg: DFG) -> dict:
    original_edges = set(original_dfg)
    synthetic_edges = set(synthetic_dfg)

    shared_edges = original_edges & synthetic_edges

    recall = safe_divide(len(shared_edges), len(original_edges))
    precision = safe_divide(len(shared_edges), len(synthetic_edges))

    return {
        "edge_recall": recall,
        "edge_precision": precision,
        "edge_f1": f1_score(precision, recall),
        "original_edges": len(original_edges),
        "synthetic_edges": len(synthetic_edges),
        "shared_edges": len(shared_edges),
        "missing_original_edges": len(original_edges - synthetic_edges),
        "extra_synthetic_edges": len(synthetic_edges - original_edges),
    }


def count_distribution_metrics(
    original_counts: CountDistribution,
    synthetic_counts: CountDistribution,
    prefix: str,
) -> dict:
    original_items = set(original_counts)
    synthetic_items = set(synthetic_counts)
    shared_items = original_items & synthetic_items

    recall = safe_divide(len(shared_items), len(original_items))
    precision = safe_divide(len(shared_items), len(synthetic_items))

    return {
        f"{prefix}_recall": recall,
        f"{prefix}_precision": precision,
        f"{prefix}_f1": f1_score(precision, recall),
        f"{prefix}_distribution_similarity": total_variation_similarity(
            original_counts,
            synthetic_counts,
        ),
        f"{prefix}_weighted_jaccard": weighted_jaccard(
            original_counts,
            synthetic_counts,
        ),
        f"{prefix}_cosine_similarity": cosine_similarity(
            original_counts,
            synthetic_counts,
        ),
        f"{prefix}_original_count": len(original_counts),
        f"{prefix}_synthetic_count": len(synthetic_counts),
        f"{prefix}_shared_count": len(shared_items),
    }


def compare_dfg_graphs(original_path: str, synthetic_path: str) -> dict:
    original_graph = load_graph(original_path)
    synthetic_graph = load_graph(synthetic_path)

    original_dfg = extract_dfg_edges(original_graph)
    synthetic_dfg = extract_dfg_edges(synthetic_graph)

    original_activities = extract_counts(original_graph, "activities")
    synthetic_activities = extract_counts(synthetic_graph, "activities")

    original_start_activities = extract_counts(original_graph, "start_activities")
    synthetic_start_activities = extract_counts(synthetic_graph, "start_activities")

    original_end_activities = extract_counts(original_graph, "end_activities")
    synthetic_end_activities = extract_counts(synthetic_graph, "end_activities")

    results = {}

    # DFG structure
    results.update(edge_precision_recall_f1(original_dfg, synthetic_dfg))

    # DFG frequency distribution
    results["dfg_distribution_similarity"] = total_variation_similarity(
        original_dfg,
        synthetic_dfg,
    )
    results["dfg_weighted_jaccard"] = weighted_jaccard(
        original_dfg,
        synthetic_dfg,
    )
    results["dfg_cosine_similarity"] = cosine_similarity(
        original_dfg,
        synthetic_dfg,
    )

    # Activity frequencies
    results.update(count_distribution_metrics(
        original_activities,
        synthetic_activities,
        "activity",
    ))

    # Start activity frequencies
    results.update(count_distribution_metrics(
        original_start_activities,
        synthetic_start_activities,
        "start_activity",
    ))

    # End activity frequencies
    results.update(count_distribution_metrics(
        original_end_activities,
        synthetic_end_activities,
        "end_activity",
    ))

    return results



if __name__ == "__main__":


    original_path = "pm4py_outputs_inductive/discovery/dfg.json"
    synthetic_path = "DeepSeek/Discovery/dfg.json"

    output_path = "model_comparison/dfg_metrics_results.csv"

    results = compare_dfg_graphs(original_path, synthetic_path)

    results_df = pd.DataFrame([results])
    results_df.to_csv(output_path, index=False)

    print("\n  DFG metrics results  ")
    print(results_df.T)

    print(f"\nSaved metrics to: {output_path}")