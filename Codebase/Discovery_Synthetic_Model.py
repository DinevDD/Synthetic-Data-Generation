import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pandas as pd
import pm4py

from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.objects.bpmn.exporter import exporter as bpmn_exporter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.statistics.start_activities.log import get as start_act_get
from pm4py.statistics.end_activities.log import get as end_act_get
from pm4py.objects.process_tree.exporter import exporter as pt_exporter
from pm4py.objects.dfg.exporter import exporter as dfg_exporter

CSV_LOG_PATH = "Data/data.csv"
OUTPUT_DIR = "pm4py_outputs_inductive/discovery_synthetic"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


# Import CSV log
def import_csv_log(path: str):
    print(f"\n{'=' * 60}")
    print(f"[1] Importing CSV log: {path}")
    print('=' * 60)

    event_df = load_csv_event_log_dataframe(path)
    log = pm4py.convert_to_event_log(event_df)

    return log



def discover_model(log):
    print(f"\n{'=' * 60}")
    print("[2] Running Inductive Miner …")
    print('=' * 60)

    process_tree = pm4py.discover_process_tree_inductive(log)  # , noise_threshold=0.2
    print(f"    Process tree: {process_tree}")

    net, initial_marking, final_marking = pt_converter.apply(process_tree)
    print(f"    Petri Net — places      : {len(net.places)}")
    print(f"    Petri Net — transitions : {len(net.transitions)}")
    print(f"    Petri Net — arcs        : {len(net.arcs)}")
    return process_tree, net, initial_marking, final_marking


def export_models(log, process_tree, net, initial_marking, final_marking):
    print(f"\n{'=' * 60}")
    print("[3] Exporting models & visualisations …")
    print('=' * 60)

    # Process tree
    pt_path = os.path.join(OUTPUT_DIR, "process_tree.ptml")
    pt_exporter.apply(process_tree, pt_path)
    print(f"      Process Tree (PTML) saved  -> {pt_path}")

    pt_txt_path = os.path.join(OUTPUT_DIR, "process_tree.txt")
    with open(pt_txt_path, "w", encoding="utf-8") as fh:
        fh.write(str(process_tree))
    print(f"      Process Tree (TXT)  saved  -> {pt_txt_path}")
    pt_img_path = os.path.join(OUTPUT_DIR, "process_tree.png")
    pm4py.save_vis_process_tree(process_tree, pt_img_path)
    print(f"      Process Tree (PNG)  saved  -> {pt_img_path}")

    # Petri net
    pnml_path = os.path.join(OUTPUT_DIR, "petri_net.pnml")
    pnml_exporter.apply(net, initial_marking, pnml_path,
                        final_marking=final_marking)
    print(f"      Petri Net (PNML) saved  -> {pnml_path}")

    pn_img_path = os.path.join(OUTPUT_DIR, "petri_net.png")
    pm4py.save_vis_petri_net(net, initial_marking, final_marking, pn_img_path)
    print(f"      Petri Net (PNG)  saved  -> {pn_img_path}")

    # BPMN
    bpmn_path = os.path.join(OUTPUT_DIR, "process_model.bpmn")
    bpmn_graph = pm4py.convert_to_bpmn(net, initial_marking, final_marking)
    bpmn_exporter.apply(bpmn_graph, bpmn_path)
    print(f"      BPMN (XML)       saved  -> {bpmn_path}")

    bpmn_img_path = os.path.join(OUTPUT_DIR, "process_model.png")
    pm4py.save_vis_bpmn(bpmn_graph, bpmn_img_path)
    print(f"      BPMN (PNG)       saved  -> {bpmn_img_path}")

    #DFG
    dfg = dfg_discovery.apply(log)
    activities = pm4py.get_event_attribute_values(log, "concept:name")
    start_acts = start_act_get.get_start_activities(log)
    end_acts = end_act_get.get_end_activities(log)

    dfg_img_path = os.path.join(OUTPUT_DIR, "dfg.png")
    pm4py.save_vis_dfg(dfg, start_acts, end_acts, dfg_img_path)
    print(f"      DFG (PNG)        saved  -> {dfg_img_path}")

    #DFG file for later comparison
    dfg_path = os.path.join(OUTPUT_DIR, "dfg.dfg")
    dfg_exporter.apply(
        dfg,
        dfg_path,
        parameters={
            dfg_exporter.Variants.CLASSIC.value.Parameters.START_ACTIVITIES: start_acts,
            dfg_exporter.Variants.CLASSIC.value.Parameters.END_ACTIVITIES: end_acts
        }
    )
    print(f"      DFG (.dfg)       saved  -> {dfg_path}")

    #JSON
    dfg_json_path = os.path.join(OUTPUT_DIR, "dfg.json")
    dfg_payload = {
        "activities": {act: cnt for act, cnt in activities.items()},
        "start_activities": start_acts,
        "end_activities": end_acts,
        "edges": {
            f"{src}||{tgt}": cnt
            for (src, tgt), cnt in dfg.items()
        },
    }

    with open(dfg_json_path, "w", encoding="utf-8") as fh:
        json.dump(dfg_payload, fh, indent=2, ensure_ascii=False)

    print(f"      DFG (JSON)       saved  -> {dfg_json_path}")


def main():
    print(" Process Discovery Pipeline")

    # Calling the newly defined CSV importer
    log = import_csv_log(CSV_LOG_PATH)
    process_tree, net, im, fm = discover_model(log)
    export_models(log, process_tree, net, im, fm)

    print(f"  Discovery complete.  All outputs  -> {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()