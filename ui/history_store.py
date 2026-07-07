"""Session state defaults, chat history persistence, and artifact export."""

from __future__ import annotations

import csv as _csv
import json
from pathlib import Path

import streamlit as st

from paths import ARTIFACTS_DIR, HISTORY_FILE


def _load_history() -> dict:
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_history():
    data = {
        "messages":         st.session_state["messages"],
        "conv_history":     st.session_state["conv_history"],
        "active_synthesis": st.session_state["active_synthesis"],
    }
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    except Exception:
        pass  # never crash the UI over a failed write


def _init_state():
    if "history_loaded" not in st.session_state:
        saved = _load_history()
        st.session_state["messages"]         = saved.get("messages", [])
        st.session_state["conv_history"]     = saved.get("conv_history", [])
        st.session_state["active_synthesis"] = saved.get("active_synthesis", None)
        st.session_state["viewing_history"]  = False
        st.session_state["history_snapshot"] = []
        st.session_state["history_loaded"]   = True
        st.session_state["pending_save"]     = None  # msg index waiting for artifact name
    else:
        st.session_state.setdefault("viewing_history",  False)
        st.session_state.setdefault("history_snapshot", [])
        st.session_state.setdefault("pending_save",     None)


def _save_artifacts(result: dict, artifact_name: str) -> Path:
    base          = ARTIFACTS_DIR / artifact_name
    responses_dir = base / "responses"
    kg_dir        = base / "kg_triples"
    responses_dir.mkdir(parents=True, exist_ok=True)
    kg_dir.mkdir(parents=True, exist_ok=True)

    mode = result["mode"]

    # Normalise all modes to a flat list of history entries
    if mode in ("neukrag", "neukrag-inter"):
        agent_label = "neukrag_inter" if mode == "neukrag-inter" else "neukrag"
        entries = [{
            "agent":      agent_label,
            "round":      1,
            "statement":  result["final_hypothesis"],
            "references": result.get("references", ""),
            "triples":    result.get("triples", []),
            "agreed":     None,
        }]
    elif mode == "synthesis":
        entries = [
            {
                "agent":      name,
                "round":      1,
                "statement":  data["statement"],
                "references": data["references"],
                "triples":    data.get("triples", []),
                "agreed":     None,
            }
            for name, data in result["agent_hypotheses"].items()
        ]
    else:
        entries = result.get("debate_history", [])

    for entry in entries:
        agent     = entry["agent"]
        round_num = entry["round"]
        stem      = f"round_{round_num:02d}_{agent}"

        # Response JSON (everything except raw triples)
        resp = {k: v for k, v in entry.items() if k != "triples"}
        with open(responses_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(resp, f, indent=2, ensure_ascii=False, default=str)

        # KG triples CSV (only when the agent queried the KG this round)
        triples = entry.get("triples") or []
        if triples:
            with open(kg_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
                writer = _csv.DictWriter(f, fieldnames=["h", "r", "t", "document_id"])
                writer.writeheader()
                for t in triples:
                    writer.writerow({
                        "h":           t.get("h", ""),
                        "r":           t.get("r", ""),
                        "t":           t.get("t", ""),
                        "document_id": t.get("document_id", ""),
                    })

    # Final synthesis
    with open(responses_dir / "final_synthesis.json", "w", encoding="utf-8") as f:
        json.dump(
            {"mode": mode, "query": result.get("query", ""), "final_hypothesis": result["final_hypothesis"]},
            f, indent=2, ensure_ascii=False,
        )

    return base
