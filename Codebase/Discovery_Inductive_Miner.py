import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.objects.bpmn.exporter import exporter as bpmn_exporter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.statistics.start_activities.log import get as start_act_get
from pm4py.statistics.end_activities.log import get as end_act_get
from pm4py.objects.process_tree.exporter import exporter as pt_exporter
from pm4py.objects.dfg.exporter import exporter as dfg_exporter
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.visualization.dfg import visualizer as dfg_visualizer

XES_LOG_PATH = "Data/filtered_log.xes.gz"
OUTPUT_DIR   = "pm4py_outputs_inductive/discovery"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save_pdf(gviz, pdf_path: str) -> None:
    stem = pdf_path.removesuffix(".pdf")
    gviz.render(filename=stem, format="pdf", cleanup=True)
    print(f"   PDF saved → {pdf_path}")


def import_xes_log(path: str):
    print(f"\n{'='*60}")
    print(f"[1] Importing XES log: {path}")
    print('='*60)
    log = xes_importer.apply(path)
    print(f"    Traces     : {len(log)}")
    print(f"    Events     : {sum(len(t) for t in log)}")
    activities = set(e["concept:name"] for t in log for e in t)
    print(f"    Activities ({len(activities)}): {sorted(activities)}")
    return log

def discover_model(log):
    print(f"\n{'='*60}")
    print("[2] Running Inductive Miner …")
    print('='*60)

    process_tree = pm4py.discover_process_tree_inductive(log)  # , noise_threshold=0.2
    print(f"    Process tree: {process_tree}")

    net, initial_marking, final_marking = pt_converter.apply(process_tree)
    print(f"    Petri Net - places      : {len(net.places)}")
    print(f"    Petri Net - transitions : {len(net.transitions)}")
    print(f"    Petri Net - arcs        : {len(net.arcs)}")
    return process_tree, net, initial_marking, final_marking


def export_models(log, process_tree, net, initial_marking, final_marking):
    print(f"\n{'='*60}")
    print("[3] Exporting models & visualisations …")
    print('='*60)

    pt_path = os.path.join(OUTPUT_DIR, "process_tree.ptml")
    pt_exporter.apply(process_tree, pt_path)
    print(f"    Process Tree (PTML) saved -> {pt_path}")

    pt_txt_path = os.path.join(OUTPUT_DIR, "process_tree.txt")
    with open(pt_txt_path, "w", encoding="utf-8") as fh:
        fh.write(str(process_tree))
    print(f"  Process Tree (TXT)  saved -> {pt_txt_path}")

    pt_img_path = os.path.join(OUTPUT_DIR, "process_tree.png")
    pt_gviz = pt_visualizer.apply(process_tree)
    pt_visualizer.save(pt_gviz, pt_img_path)
    print(f"  Process Tree (PNG)  saved ->{pt_img_path}")
    _save_pdf(pt_gviz, os.path.join(OUTPUT_DIR, "process_tree.pdf"))

    pnml_path = os.path.join(OUTPUT_DIR, "petri_net.pnml")
    pnml_exporter.apply(net, initial_marking, pnml_path,
                        final_marking=final_marking)
    print(f"    Petri Net (PNML) saved -> {pnml_path}")

    pn_img_path = os.path.join(OUTPUT_DIR, "petri_net.png")
    pn_gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    pn_visualizer.save(pn_gviz, pn_img_path)
    print(f"   Petri Net (PNG)  saved -> {pn_img_path}")
    _save_pdf(pn_gviz, os.path.join(OUTPUT_DIR, "petri_net.pdf"))

    bpmn_path = os.path.join(OUTPUT_DIR, "process_model.bpmn")
    bpmn_graph = pm4py.convert_to_bpmn(net, initial_marking, final_marking)
    bpmn_exporter.apply(bpmn_graph, bpmn_path)
    print(f"   BPMN (XML)       saved -> {bpmn_path}")

    bpmn_img_path = os.path.join(OUTPUT_DIR, "process_model.png")
    bpmn_gviz = bpmn_visualizer.apply(bpmn_graph)
    bpmn_visualizer.save(bpmn_gviz, bpmn_img_path)
    print(f"   BPMN (PNG)       saved -> {bpmn_img_path}")
    _save_pdf(bpmn_gviz, os.path.join(OUTPUT_DIR, "process_model.pdf"))


    dfg = dfg_discovery.apply(log)
    activities = pm4py.get_event_attribute_values(log, "concept:name")
    start_acts = start_act_get.get_start_activities(log)
    end_acts = end_act_get.get_end_activities(log)

    dfg_img_path = os.path.join(OUTPUT_DIR, "dfg.png")
    dfg_gviz = dfg_visualizer.apply(dfg, parameters={
        dfg_visualizer.Variants.FREQUENCY.value.Parameters.START_ACTIVITIES: start_acts,
        dfg_visualizer.Variants.FREQUENCY.value.Parameters.END_ACTIVITIES: end_acts,
    })
    dfg_visualizer.save(dfg_gviz, dfg_img_path)
    print(f"     DFG (PNG)        saved -> {dfg_img_path}")
    _save_pdf(dfg_gviz, os.path.join(OUTPUT_DIR, "dfg.pdf"))

    dfg_path = os.path.join(OUTPUT_DIR, "dfg.dfg")
    dfg_exporter.apply(
        dfg,
        dfg_path,
        parameters={
            dfg_exporter.Variants.CLASSIC.value.Parameters.START_ACTIVITIES: start_acts,
            dfg_exporter.Variants.CLASSIC.value.Parameters.END_ACTIVITIES: end_acts,
        },
    )
    print(f"  DFG (.dfg)       saved -> {dfg_path}")

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
    print(f"   DFG (JSON)       saved -> {dfg_json_path}")


def main():
    print("  Process Discovery Pipeline")

    log = import_xes_log(XES_LOG_PATH)
    process_tree, net, im, fm = discover_model(log)
    export_models(log, process_tree, net, im, fm)

    print(f"  Discovery complete.  All outputs → {OUTPUT_DIR}/")
    print("  Run conformance.py next.")


if __name__ == "__main__":
    main()