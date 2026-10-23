"""Phase 6b: Detector -> Explainer -> Recommender as a LangGraph relay."""
import os
import re
import joblib
import pandas as pd


from pathlib import Path
from typing import TypedDict


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from src.data.loader import load_config
from src.genai.explain_chain import get_llm, retrieve, sensor_evidence
from src.genai.rag import DASHES, ManualIndex
#from src.models.detector import Detector, pick_engines

EXPLAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the Explainer agent in a maintenance team. A Detector agent has already measured the machine.
Use ONLY the facts in DETECTOR REPORT and SENSOR EVIDENCE. Quote numbers exactly as written, add no units, never invent values.
Say 'consistent with', never claim a confirmed diagnosis. Plain English; explain technical terms in a few words.
Write 3 to 5 sentences, no headings, no lists. State the failure risk percentage and the estimated cycles left,
and say what the two anomaly checks found."""),
    ("human", "DETECTOR REPORT\n{report}\n\nSENSOR EVIDENCE\n{evidence}\n\nWrite the explanation.")])

RECOMMEND_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the Recommender agent in a maintenance team.
The urgency level was decided by fixed rules; do not change it or argue with it.
Use ONLY the MANUAL EXCERPTS for actions. End every action with its procedure ID in square brackets, e.g. [HPC-01].
Never invent procedure IDs. If the manuals do not cover something, say so.
Write ONLY a numbered list of at most 4 actions in priority order; the first should address the strongest sensor evidence.
If you recommend any physical inspection, make the last action the safety step from POL-02."""),
    ("human", "URGENCY (decided by rules): {urgency}\n\nEXPLANATION\n{explanation}\n\n"
              "MANUAL EXCERPTS\n{manual}\n\nWrite the numbered actions.")])


class State(TypedDict, total=False):
    history: pd.DataFrame
    detection: dict
    evidence: list
    chunks: list
    explanation: str
    actions: str
    unknown_citations: list
    trace: list


class MaintenanceCrew:
    def __init__(self, config_path: str = "configs/config.yaml", detector=None):
        cfg = load_config(config_path)
        art, proc = Path(cfg["paths"]["artifacts_dir"]), Path(cfg["paths"]["processed_dir"])
        #self.detector = detector or Detector(config_path)
        #self.index = ManualIndex(config_path)
        lite = os.getenv("PM_LITE", "").strip() == "1"
        if detector is None:
            if lite:
                from src.models.detector_lite import DetectorLite
                detector = DetectorLite(config_path)
            else:
                from src.models.detector import Detector
                detector = Detector(config_path)
        self.detector = detector
        if lite:
            from src.genai.rag_lite import ManualIndexLite
            self.index = ManualIndexLite(config_path)
        else:
            self.index = ManualIndex(config_path)
        self.sensors = self.detector.sensors
        self.scaler = joblib.load(art / "scaler.joblib")
        train = pd.read_csv(proc / "train.csv", usecols=self.sensors + ["rul"])
        self.healthy = train.loc[train["rul"] >= 100, self.sensors].mean()
        llm, parser = get_llm(), StrOutputParser()
        self.explain_chain = EXPLAIN_PROMPT | llm | parser
        self.recommend_chain = RECOMMEND_PROMPT | llm | parser
        self.app = self._build_graph()

    # ---------- the three runners (+ a shortcut) ----------
    def detector_node(self, state: State) -> dict:
        det = self.detector.analyze(state["history"])
        return {"detection": det, "trace": state.get("trace", []) + ["detector"]}

    def route(self, state: State) -> str:
        d = state["detection"]
        quiet = d["level"] == 1 and not (d["alarm"] or d["iforest_flag"] or d["ae_flag"])
        return "clear" if quiet else "explain"

    def explainer_node(self, state: State) -> dict:
        d = state["detection"]
        evidence = [sensor_evidence(x, self.scaler, self.sensors, self.healthy)
                    for x in d["drivers"]]
        chunks = retrieve(self.index, evidence)
        by_id = {c["id"]: c for c in self.index.chunks}
        if "POL-02" in by_id and "POL-02" not in {c["id"] for c in chunks}:
            chunks.append(by_id["POL-02"])                    # safety step is always available

        if d.get("lite"):
            ae_line = "Autoencoder anomaly check: not run (lightweight deployment)."
        else:
            ae_line = (f"Autoencoder anomaly check: {'FLAGGED' if d['ae_flag'] else 'normal'} "
                       f"(score {d['ae_score']:.3f}, limit {d['ae_limit']:.3f}).")

        report = (
            f"Machine unit {d['unit']}, cycle {d['cycle']}.\n"
            f"Failure risk {d['risk']:.1%} (alarm line {d['threshold']:.0%}, "
            f"alarm {'ON' if d['alarm'] else 'off'}).\n"
            f"Estimated cycles left: {d['rul_text']}.\n"
            f"Isolation Forest anomaly check: {'FLAGGED' if d['iforest_flag'] else 'normal'} "
            f"(score {d['iforest_score']:.3f}, limit {d['iforest_limit']:.3f}).\n"
            f"Autoencoder anomaly check: {'FLAGGED' if d['ae_flag'] else 'normal'} "
            f"(score {d['ae_score']:.3f}, limit {d['ae_limit']:.3f}).\n"
            f"Sensors drifting together: {d['n_drift']}.")
        text = self.explain_chain.invoke({
            "report": report, "evidence": "\n".join("- " + e["text"] for e in evidence)})
        return {"evidence": evidence, "chunks": chunks, "explanation": text,
                "trace": state["trace"] + ["explainer"]}

    def recommender_node(self, state: State) -> dict:
        d = state["detection"]
        manual = "\n\n".join(f"[{c['id']}] ({c['source']})\n{c['text']}" for c in state["chunks"])
        actions = self.recommend_chain.invoke({
            "urgency": f"Level {d['level']} {d['level_name']}",
            "explanation": state["explanation"], "manual": manual})
        cited = set(re.findall(r"\[([A-Z]{2,4}-\d{2})\]", DASHES.sub("-", actions)))
        unknown = sorted(cited - {c["id"] for c in state["chunks"]})
        return {"actions": actions, "unknown_citations": unknown,
                "trace": state["trace"] + ["recommender"]}

    def all_clear_node(self, state: State) -> dict:
        d = state["detection"]
        return {"explanation": (f"Failure risk is {d['risk']:.1%}, below the alarm line of "
                                f"{d['threshold']:.0%}. Estimated cycles left: {d['rul_text']}. "
                                f"Both anomaly checks are normal."),
                "actions": "1. No action needed. Keep routine checks. [POL-01]",
                "unknown_citations": [], "chunks": [], "evidence": [],
                "trace": state["trace"] + ["all_clear (LLM skipped)"]}

    # ---------- wiring ----------
    def _build_graph(self):
        g = StateGraph(State)
        g.add_node("detector", self.detector_node)
        g.add_node("explainer", self.explainer_node)
        g.add_node("recommender", self.recommender_node)
        g.add_node("all_clear", self.all_clear_node)
        g.add_edge(START, "detector")
        g.add_conditional_edges("detector", self.route,
                                {"explain": "explainer", "clear": "all_clear"})
        g.add_edge("explainer", "recommender")
        g.add_edge("recommender", END)
        g.add_edge("all_clear", END)
        return g.compile()

    def run(self, history: pd.DataFrame) -> dict:
        return self.app.invoke({"history": history, "trace": []})


def format_report(s: dict) -> str:
    d = s["detection"]
    return (f"Agents that ran: {' -> '.join(s['trace'])}\n\n"
            f"WHY\n{s['explanation']}\n\nRECOMMENDED ACTIONS\n{s['actions']}\n\n"
            f"URGENCY (rule-based): Level {d['level']} {d['level_name']}\n"
            + "\n".join(f"  - {r}" for r in d["reasons"]))


if __name__ == "__main__":
    cfg = load_config()
    test = pd.read_csv(Path(cfg["paths"]["processed_dir"]) / "test.csv")
    crew = MaintenanceCrew()
    for tag, unit in pick_engines(crew.detector, test).items():
        out = crew.run(test[test["unit"] == unit])
        print("=" * 72)
        print(f"{tag}: unit {unit}")
        print(format_report(out))
        bad = out["unknown_citations"]
        print("\nCitation check:", "all cited IDs were retrieved" if not bad
              else f"INVENTED IDs: {bad}")
