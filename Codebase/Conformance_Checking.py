
import os
import json
import pandas as pd
import pm4py
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator


CSV_LOG_PATH = "Data/data.csv"

OUTPUT_DIR   = "pm4py_outputs_inductive/conformance"   # inductive miner
PNML_PATH = os.path.join("pm4py_outputs_inductive/discovery", "petri_net.pnml") # inductive miner


CSV_CASE_ID_COL   = "case:concept:name"
CSV_ACTIVITY_COL  = "concept:name"
CSV_TIMESTAMP_COL = "time:timestamp"
def load_petri_net(path: str):
    print(f"\n{'='*60}")
    print(f"[1] Loading Petri Net from: {path}")
    print('='*60)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PNML file not found: {path}\n"
            "  → Please run discovery.py first."
        )

    net, initial_marking, final_marking = pnml_importer.apply(path)
    print(f"    Places      : {len(net.places)}")
    print(f"    Transitions : {len(net.transitions)}")
    print(f"    Arcs        : {len(net.arcs)}")
    return net, initial_marking, final_marking

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
                timestamp_value = pd.to_datetime(
                    timestamp_parts[0],
                    utc=True,
                    errors="coerce"
                ) + pd.Timedelta(seconds=idx)
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
    print(f"    Activities ({event_df['concept:name'].nunique()}): "
          f"{sorted(event_df['concept:name'].dropna().unique())}")

    return event_df

def import_csv_log(path: str):
    print(f"\n{'='*60}")
    print(f"[2] Importing CSV log: {path}")
    print('='*60)

    event_df = load_csv_event_log_dataframe(path)

    log = log_converter.apply(
        event_df,
        parameters={
            log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY:
            "case:concept:name"
        },
    )

    return log

def run_token_replay(log, net, initial_marking, final_marking):
    print(f"\n{'='*60}")
    print("[3] Token-Based Replay …")
    print('='*60)

    replayed = token_replay.apply(log, net, initial_marking, final_marking)

    print("\n    Per-trace summary (first 10 traces):")
    print(f"    {'Trace':>6}  {'Fit':>6}  {'Consumed':>9}  "
          f"{'Produced':>9}  {'Missing':>8}  {'Remaining':>10}")
    print("    " + "-" * 58)
    for i, r in enumerate(replayed[:10]):
        print(f"    {i:>6}  {r['trace_fitness']:>6.3f}  "
              f"{r['consumed_tokens']:>9}  {r['produced_tokens']:>9}  "
              f"{r['missing_tokens']:>8}  {r['remaining_tokens']:>10}")

    fitness_tbr = pm4py.fitness_token_based_replay(
        log, net, initial_marking, final_marking
    )
    print(f"\n    [Token-Based Fitness]")
    for k, v in fitness_tbr.items():
        print(f"      {k:<35}: {v:.4f}" if isinstance(v, float)
              else f"      {k:<35}: {v}")

    return replayed, fitness_tbr

def run_alignments(log, net, initial_marking, final_marking):
    print(f"\n{'='*60}")
    print("[4] Alignment-Based Conformance …")
    print('='*60)

    aligned_traces = alignments.apply_log(log, net, initial_marking, final_marking)

    print("\n    Per-trace summary (first 10 traces):")
    print(f"    {'Trace':>6}  {'Cost':>8}  {'BWC':>8}  {'Fitness':>8}  "
          "Alignment (sync moves shown as activity name)")
    print("    " + "-" * 80)
    for i, a in enumerate(aligned_traces[:10]):
        moves = []
        for (log_mv, model_mv) in a["alignment"]:
            if log_mv == model_mv:
                moves.append(str(log_mv))
            elif model_mv == ">>":
                moves.append(f"LOG:{log_mv}")
            else:
                moves.append(f"MDL:{model_mv}")
        alignment_str = " → ".join(moves)
        if len(alignment_str) > 50:
            alignment_str = alignment_str[:50] + "…"
        bwc     = a.get("bwc", "?")
        fitness = a.get("fitness", "?")
        fit_str = f"{fitness:.4f}" if isinstance(fitness, float) else str(fitness)
        print(f"    {i:>6}  {a['cost']:>8}  {str(bwc):>8}  {fit_str:>8}  {alignment_str}")

    fitness_align = pm4py.fitness_alignments(
        log, net, initial_marking, final_marking
    )
    print(f"\n    [Alignment-Based Fitness]")
    for k, v in fitness_align.items():
        print(f"      {k:<35}: {v:.4f}" if isinstance(v, float)
              else f"      {k:<35}: {v}")

    return aligned_traces, fitness_align


def compute_all_metrics(log, net, initial_marking, final_marking,
                        replayed_traces, aligned_traces,
                        fitness_tbr, fitness_align):
    print(f"\n{'='*60}")
    print("[5] Full Conformance Metrics Report")
    print('='*60)

    precision     = precision_evaluator.apply(
        log, net, initial_marking, final_marking,
        variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
    )
    generalization = generalization_evaluator.apply(
        log, net, initial_marking, final_marking
    )
    simplicity = simplicity_evaluator.apply(net)

    n_traces        = len(replayed_traces)
    total_missing   = sum(r["missing_tokens"]   for r in replayed_traces)
    total_remaining = sum(r["remaining_tokens"] for r in replayed_traces)
    total_consumed  = sum(r["consumed_tokens"]  for r in replayed_traces)
    total_produced  = sum(r["produced_tokens"]  for r in replayed_traces)
    n_fit_traces    = sum(
        1 for r in replayed_traces
        if r.get("missing_tokens", 1) == 0 and r.get("remaining_tokens", 1) == 0
    )
    avg_align_cost = sum(a["cost"] for a in aligned_traces) / len(aligned_traces)

    f_tbr   = fitness_tbr.get("log_fitness",
              fitness_tbr.get("average_trace_fitness", 0.0))
    f_align = fitness_align.get("log_fitness",
              fitness_align.get("averageFitness",
              fitness_align.get("average_trace_fitness", 0.0)))

    f1_tbr   = (2*f_tbr  *precision)/(f_tbr  +precision) if (f_tbr  +precision)>0 else 0.0
    f1_align = (2*f_align*precision)/(f_align+precision) if (f_align+precision)>0 else 0.0

    report = {
        **{f"tbr_{k}":   v for k, v in fitness_tbr.items()},
        **{f"align_{k}": v for k, v in fitness_align.items()},
        "tbr_total_traces"           : n_traces,
        "tbr_fitting_traces"         : n_fit_traces,
        "tbr_pct_fitting_traces"     : round(n_fit_traces / n_traces * 100, 2),
        "tbr_total_missing_tokens"   : total_missing,
        "tbr_total_remaining_tokens" : total_remaining,
        "tbr_total_consumed_tokens"  : total_consumed,
        "tbr_total_produced_tokens"  : total_produced,
        "align_avg_cost_per_trace"   : avg_align_cost,
        "precision"                  : precision,
        "generalization"             : generalization,
        "simplicity"                 : simplicity,
        "f1_tbr_fitness_x_precision" : f1_tbr,
        "f1_align_fitness_x_precision": f1_align,
    }

    max_key = max(len(k) for k in report)
    print(f"\n    {'METRIC':<{max_key}}  VALUE")
    print("    " + "-" * (max_key + 12))
    for k, v in report.items():
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"    {k:<{max_key}}  {val_str}")

    report_path = os.path.join(OUTPUT_DIR, "conformance_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n    Full report saved -> {report_path}")

    return report


def main():

    print("  Conformance Checking Pipeline")


    net, im, fm = load_petri_net(PNML_PATH)


    log = import_csv_log(CSV_LOG_PATH)


    replayed_traces, fitness_tbr = run_token_replay(log, net, im, fm)


    aligned_traces, fitness_align = run_alignments(log, net, im, fm)


    compute_all_metrics(log, net, im, fm,
                        replayed_traces, aligned_traces,
                        fitness_tbr, fitness_align)


    print(f"  Conformance complete.  Report → {OUTPUT_DIR}/conformance_report.json")



if __name__ == "__main__":
    main()
