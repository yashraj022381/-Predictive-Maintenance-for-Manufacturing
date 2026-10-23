"""Phase 6a: SHAP evidence + manual excerpts -> grounded explanation (LangChain + Groq)."""
import os
import re
import json
import joblib

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.data.loader import load_config
from src.explain.names import SENSOR_NAMES
#from src.explain.shap_explainer import RiskExplainer
#from src.genai.rag import ManualIndex, DASHES
from src.genai.rag import DASHES

SYSTEM = """You are a maintenance assistant for a plant technician.
Use ONLY the facts in SENSOR EVIDENCE and MANUAL EXCERPTS. Never invent sensor values, procedures or procedure IDs.
Cite manual procedures by their ID in square brackets, for example [HPC-01].
If the manuals do not cover something, say it is not covered by the manuals.
The evidence comes from a statistical model, so say 'consistent with', never claim a confirmed diagnosis.
Write in plain English. Explain any technical term in a few words.
Quote numbers exactly as written. Do not add units that are not in the evidence.
Always state the failure risk percentage in the first section."""

HUMAN = """MACHINE FACTS
{facts}

SENSOR EVIDENCE (from the ML model; compared with healthy typical levels)
{evidence}

MANUAL EXCERPTS
{manual}

Write exactly these three sections:

## Why this machine is at risk
2 to 4 sentences. Use the evidence numbers.

## Recommended actions
At most 4 numbered actions in priority order. Each action must end with the procedure ID it comes from.

## Urgency
One line: the urgency level and the reason, ending with the policy ID, for example [POL-01]."""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])


def get_llm() -> ChatGroq:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not found. Put it in a .env file in the project root.")
    return ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                    temperature=0.2, max_retries=2)


def sensor_evidence(driver: dict, scaler, sensors: list[str], healthy: pd.Series) -> dict:
    """Turn a scaled 0..1 SHAP driver into real units compared with the healthy level."""
    base, _, kind = driver["feature"].partition("_")
    i = sensors.index(base)
    rng, mn = scaler.data_range_[i], scaler.data_min_[i]
    v, name = driver["value"], SENSOR_NAMES.get(base, base)
    win = re.sub(r"\D", "", kind) or "10"
    if kind.startswith("diff"):
        change = v * rng
        text = f"{name}: changed by {change:+.3f} over the last {win} cycles"
        direction = "rising" if change > 0 else "falling"
    elif kind.startswith("std"):
        text, direction = f"{name}: readings fluctuating by about {v * rng:.3f}", None
    else:                                    # raw reading or rolling mean
        real = v * rng + mn
        typical = healthy[base] * rng + mn
        pct = (real - typical) / abs(typical) * 100
        text = f"{name}: recent level {real:.2f} vs healthy typical {typical:.2f} ({pct:+.1f}%)"
        direction = "higher" if pct > 0 else "lower"
    return {"sensor": name, "direction": direction,
            "text": f"{text}; model effect: {driver['effect']}"}


#def retrieve(index: ManualIndex, evidence: list[dict]) -> list[dict]:
def retrieve(index, evidence: list[dict]) -> list[dict]:
    """One search per distinct sensor symptom + one search for the policy."""
    seen, out = set(), []
    queries = list(dict.fromkeys(f"{e['sensor']} {e['direction']}"
                                 for e in evidence if e["direction"]))[:5]
    queries.append("maintenance urgency level, remaining useful life, several sensors drifting")
    for q in queries:
        for c in index.search(q, k=2 if q.startswith("maintenance") else 1):
            if c["id"] not in seen:
                seen.add(c["id"])
                out.append(c)
    return out


class GenAIExplainer:
    def __init__(self, config_path: str = "configs/config.yaml"):
        from src.explain.shap_explainer import RiskExplainer
        from src.genai.rag import ManualIndex
        cfg = load_config(config_path)
        art, proc = Path(cfg["paths"]["artifacts_dir"]), Path(cfg["paths"]["processed_dir"])
        self.sensors = json.loads((art / "feature_cols.json").read_text())["sensors"]
        self.scaler = joblib.load(art / "scaler.joblib")
        train = pd.read_csv(proc / "train.csv", usecols=self.sensors + ["rul"])
        self.healthy = train.loc[train["rul"] >= 100, self.sensors].mean()   # scaled units
        self.risk = RiskExplainer(config_path)
        self.index = ManualIndex(config_path)
        self.chain = PROMPT | get_llm() | StrOutputParser()
        self.rul_cap = cfg["data"]["rul_cap"]

    def explain(self, row: pd.DataFrame, rul_estimate: float | None = None) -> dict:
        r = self.risk.explain(row, top_k=5)
        evidence = [sensor_evidence(d, self.scaler, self.sensors, self.healthy)
                    for d in r["drivers"]]
        chunks = retrieve(self.index, evidence)

        if rul_estimate is None:
            rul_text = "not available"
        elif rul_estimate >= self.rul_cap - 1:
            rul_text = f"{self.rul_cap}+ cycles (healthy range)"
        else:
            rul_text = f"about {rul_estimate:.0f} cycles (typical error about 12 cycles)"

        facts = (f"Machine unit {int(row['unit'].iloc[0])}, cycle {int(row['cycle'].iloc[0])}. "
                 f"Failure risk {r['risk_probability']:.1%} (alarm line {r['threshold']:.0%}, "
                 f"alarm {'ON' if r['alarm'] else 'off'}). Estimated cycles left: {rul_text}.")
        manual = "\n\n".join(f"[{c['id']}] ({c['source']})\n{c['text']}" for c in chunks)

        text = self.chain.invoke({"facts": facts,
                                  "evidence": "\n".join("- " + e["text"] for e in evidence),
                                  "manual": manual})

        #cited = set(re.findall(r"\[([A-Z]{2,4}-\d{2})\]", text))
        cited = set(re.findall(r"\[([A-Z]{2,4}-\d{2})\]", DASHES.sub("-", text)))
        unknown = sorted(cited - {c["id"] for c in chunks})
        return {"facts": facts, "evidence": evidence, "sources": [c["id"] for c in chunks],
                "text": text, "unknown_citations": unknown}


if __name__ == "__main__":
    cfg = load_config()
    art = Path(cfg["paths"]["artifacts_dir"])
    test = pd.read_csv(Path(cfg["paths"]["processed_dir"]) / "test.csv")
    cols = json.loads((art / "feature_cols.json").read_text())["features"]
    last = test.sort_values(["unit", "cycle"]).groupby("unit").tail(1).reset_index(drop=True)

    gen = GenAIExplainer()
    probs = gen.risk.model.predict_proba(last[cols])[:, 1]
    rul_model = joblib.load(art / "rul_xgb.joblib")        # fast RUL estimate for now (LSTM in 6b)

    picks = [("HIGHEST RISK", int(probs.argmax())),
             ("BORDERLINE", int(np.abs(probs - gen.risk.threshold).argmin()))]
    for tag, i in picks:
        row = last.iloc[[i]]
        est = float(np.clip(rul_model.predict(row[cols])[0], 0, gen.rul_cap))
        out = gen.explain(row, rul_estimate=est)
        print("=" * 70)
        print(f"{tag} | true cycles left: {int(row['rul'].iloc[0])}")
        print(out["facts"])
        print("Evidence given to the AI:")
        for e in out["evidence"]:
            print("  -", e["text"])
        print("Manual sections retrieved:", out["sources"])
        print("\n" + out["text"])
        print("\nCitation check:", "all cited IDs were retrieved"
              if not out["unknown_citations"] else f"INVENTED IDs: {out['unknown_citations']}")
