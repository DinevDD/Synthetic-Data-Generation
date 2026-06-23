


#Direction:Original data against synthetic models


import os
import json
import pm4py
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments

from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator

import pandas as pd


ORIGINAL_LOG_PATH = "Data/filtered_log.xes.gz"

SYNTHETIC_MODELS_DIR = "pm4py_outputs_inductive/discovery_synthetic"

OUTPUT_DIR = "pm4py_outputs_inductive/conformance_synhtetic"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def import_original_log(path: str):
    print(f"\n{'=' * 60}")
    print(f"[1] Importing original XES log: {path}")
    print('=' * 60)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Original log not found: {path}")

    log_obj = pm4py.read_xes(path)

    if isinstance(log_obj, pd.DataFrame):
        df = log_obj.copy()

        df["time:timestamp"] = pd.to_datetime(
            df["time:timestamp"],
            utc=True,
            errors="coerce"
        )

        df = df.sort_values(["case:concept:name", "time:timestamp"])

        log = pm4py.convert_to_event_log(df)

        traces = df["case:concept:name"].nunique()
        events = len(df)
        activities = sorted(df["concept:name"].dropna().unique())

    else:
        log = log_obj

        traces = len(log)
        events = sum(len(t) for t in log)
        activities = sorted(
            set(
                e["concept:name"]
                for t in log
                for e in t
                if "concept:name" in e
            )
        )

    print(f"    Traces    : {traces}")
    print(f"    Events    : {events}")
    print(f"    Activities: {activities}")

    return log


def find_synthetic_pnml_models(models_dir: str):
    print(f"\n{'=' * 60}")
    print(f"[2] Searching for synthetic PNML models in: {models_dir}")
    print('=' * 60)

    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Synthetic models directory not found: {models_dir}")

    pnml_files = []

    for root, _, files in os.walk(models_dir):
        for file in files:
            if file.lower().endswith(".pnml"):
                pnml_files.append(os.path.join(root, file))

    if not pnml_files:
        raise FileNotFoundError(
            f"No PNML models found in: {models_dir}\n"
            "Make sure the synthetic Petri nets were exported first."
        )

    pnml_files = sorted(pnml_files)

    print(f"    Found {len(pnml_files)} PNML model(s):")
    for path in pnml_files:
        print(f"      - {path}")

    return pnml_files


def load_petri_net(path: str):
    print(f"\n{'=' * 60}")
    print(f"Loading synthetic Petri Net: {path}")
    print('=' * 60)

    net, initial_marking, final_marking = pnml_importer.apply(path)

    print(f"    Places      : {len(net.places)}")
    print(f"    Transitions : {len(net.transitions)}")
    print(f"    Arcs        : {len(net.arcs)}")

    return net, initial_marking, final_marking


def run_token_replay(log, net, initial_marking, final_marking):
    print(f"\n{'=' * 60}")
    print("[3] Token-Based Replay")
    print('=' * 60)

    replayed = token_replay.apply(log, net, initial_marking, final_marking)

    print("\n    Per-trace summary, first 10 traces:")
    print(
        f"    {'Trace':>6}  {'Fit':>6}  {'Consumed':>9}  "
        f"{'Produced':>9}  {'Missing':>8}  {'Remaining':>10}"
    )
    print("    " + "-" * 58)

    for i, r in enumerate(replayed[:10]):
        print(
            f"    {i:>6}  {r['trace_fitness']:>6.3f}  "
            f"{r['consumed_tokens']:>9}  {r['produced_tokens']:>9}  "
            f"{r['missing_tokens']:>8}  {r['remaining_tokens']:>10}"
        )

    fitness_tbr = pm4py.fitness_token_based_replay(
        log,
        net,
        initial_marking,
        final_marking
    )

    print("\n    [Token-Based Fitness]")
    for k, v in fitness_tbr.items():
        print(f"      {k:<35}: {v:.4f}" if isinstance(v, float) else f"      {k:<35}: {v}")

    return replayed, fitness_tbr


def run_alignments(log, net, initial_marking, final_marking):
    print(f"\n{'=' * 60}")
    print("[4] Alignment-Based Conformance")
    print('=' * 60)

    aligned_traces = alignments.apply_log(
        log,
        net,
        initial_marking,
        final_marking
    )

    print("\n    Per-trace summary, first 10 traces:")
    print(
        f"    {'Trace':>6}  {'Cost':>8}  {'BWC':>8}  {'Fitness':>8}  "
        "Alignment"
    )
    print("    " + "-" * 80)

    for i, a in enumerate(aligned_traces[:10]):
        moves = []

        for log_mv, model_mv in a["alignment"]:
            if log_mv == model_mv:
                moves.append(str(log_mv))
            elif model_mv == ">>":
                moves.append(f"LOG:{log_mv}")
            else:
                moves.append(f"MDL:{model_mv}")

        alignment_str = " → ".join(moves)

        if len(alignment_str) > 50:
            alignment_str = alignment_str[:50] + "..."

        bwc = a.get("bwc", "?")
        fitness = a.get("fitness", "?")
        fit_str = f"{fitness:.4f}" if isinstance(fitness, float) else str(fitness)

        print(
            f"    {i:>6}  {a['cost']:>8}  {str(bwc):>8}  "
            f"{fit_str:>8}  {alignment_str}"
        )

    fitness_align = pm4py.fitness_alignments(
        log,
        net,
        initial_marking,
        final_marking
    )

    print("\n    [Alignment-Based Fitness]")
    for k, v in fitness_align.items():
        print(f"      {k:<35}: {v:.4f}" if isinstance(v, float) else f"      {k:<35}: {v}")

    return aligned_traces, fitness_align


def compute_all_metrics(
    log,
    net,
    initial_marking,
    final_marking,
    replayed_traces,
    aligned_traces,
    fitness_tbr,
    fitness_align
):
    print(f"\n{'=' * 60}")
    print("[5] Full Conformance Metrics Report")
    print('=' * 60)

    precision = precision_evaluator.apply(
        log,
        net,
        initial_marking,
        final_marking,
        variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
    )

    generalization = generalization_evaluator.apply(
        log,
        net,
        initial_marking,
        final_marking
    )

    simplicity = simplicity_evaluator.apply(net)

    n_traces = len(replayed_traces)

    total_missing = sum(r["missing_tokens"] for r in replayed_traces)
    total_remaining = sum(r["remaining_tokens"] for r in replayed_traces)
    total_consumed = sum(r["consumed_tokens"] for r in replayed_traces)
    total_produced = sum(r["produced_tokens"] for r in replayed_traces)

    n_fit_traces = sum(
        1 for r in replayed_traces
        if r.get("missing_tokens", 1) == 0 and r.get("remaining_tokens", 1) == 0
    )

    avg_align_cost = (
        sum(a["cost"] for a in aligned_traces) / len(aligned_traces)
        if aligned_traces
        else None
    )

    f_tbr = fitness_tbr.get(
        "log_fitness",
        fitness_tbr.get("average_trace_fitness", 0.0)
    )

    f_align = fitness_align.get(
        "log_fitness",
        fitness_align.get(
            "averageFitness",
            fitness_align.get("average_trace_fitness", 0.0)
        )
    )

    f1_tbr = (
        (2 * f_tbr * precision) / (f_tbr + precision)
        if (f_tbr + precision) > 0
        else 0.0
    )

    f1_align = (
        (2 * f_align * precision) / (f_align + precision)
        if (f_align + precision) > 0
        else 0.0
    )

    report = {
        **{f"tbr_{k}": v for k, v in fitness_tbr.items()},
        **{f"align_{k}": v for k, v in fitness_align.items()},

        "tbr_total_traces": n_traces,
        "tbr_fitting_traces": n_fit_traces,
        "tbr_pct_fitting_traces": round(n_fit_traces / n_traces * 100, 2)
        if n_traces > 0
        else 0.0,

        "tbr_total_missing_tokens": total_missing,
        "tbr_total_remaining_tokens": total_remaining,
        "tbr_total_consumed_tokens": total_consumed,
        "tbr_total_produced_tokens": total_produced,

        "align_avg_cost_per_trace": avg_align_cost,

        "precision": precision,
        "generalization": generalization,
        "simplicity": simplicity,

        "f1_tbr_fitness_x_precision": f1_tbr,
        "f1_align_fitness_x_precision": f1_align,
    }

    max_key = max(len(k) for k in report)

    print(f"\n    {'METRIC':<{max_key}}  VALUE")
    print("    " + "-" * (max_key + 12))

    for k, v in report.items():
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"    {k:<{max_key}}  {val_str}")

    return report

def save_report(report, model_path):
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    report_path = os.path.join(
        OUTPUT_DIR,
        f"{model_name}_original_data_vs_synthetic_model_report.json"
    )

    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n     Report saved -> {report_path}")

    return report_path


def save_summary(all_reports):
    summary_path = os.path.join(
        OUTPUT_DIR,
        "summary_original_data_vs_synthetic_models.json"
    )

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(all_reports, fh, indent=2)

    print(f"\n Summary saved -> {summary_path}")



def main():

    print("  Conformance Checking Pipeline")
    print("  Direction: Original data against synthetic models")


    original_log = import_original_log(ORIGINAL_LOG_PATH)

    synthetic_models = find_synthetic_pnml_models(SYNTHETIC_MODELS_DIR)

    all_reports = {}

    for model_path in synthetic_models:

        print(f"  Evaluating synthetic model: {model_path}")

        net, im, fm = load_petri_net(model_path)

        replayed_traces, fitness_tbr = run_token_replay(
            original_log,
            net,
            im,
            fm
        )

        aligned_traces, fitness_align = run_alignments(
            original_log,
            net,
            im,
            fm
        )

        report = compute_all_metrics(
            original_log,
            net,
            im,
            fm,
            replayed_traces,
            aligned_traces,
            fitness_tbr,
            fitness_align
        )

        report["direction"] = "original_data_against_synthetic_model"
        report["log_path"] = ORIGINAL_LOG_PATH
        report["model_path"] = model_path

        report_path = save_report(report, model_path)

        model_name = os.path.splitext(os.path.basename(model_path))[0]

        all_reports[model_name] = {
            "model_path": model_path,
            "report_path": report_path,
            "direction": "original_data_against_synthetic_model",
            "metrics": report
        }

    save_summary(all_reports)


    print(f"  Conformance complete. Reports → {OUTPUT_DIR}")



if __name__ == "__main__":
    main()