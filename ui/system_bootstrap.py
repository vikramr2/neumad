"""Cached DSPy/KG system bootstrap — one per (k_hops, max_triples) combination."""

from __future__ import annotations

from pathlib import Path

import dspy
import streamlit as st

from orchestration import (
    CONFIG_PATH,
    LLM_MODEL,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
    HypothesisGenerator,
    Mediator,
    SpecialistAgent,
    load_graph,
    load_metadata,
    load_toml,
)
from run_neukrag import EntityExtractor


def _configure_dspy_lm() -> None:
    """Configure the shared DSPy LM, tolerating the case where another cached
    bootstrap (build_system / build_neukrag_system) already did it first.

    dspy.settings is a single global config; only the first thread to call
    configure() may write to it, but the LM it sets is globally readable by every
    other thread regardless of who configured it (see
    dspy.dsp.utils.settings.Settings — main_thread_config is shared, not
    per-thread). Each @st.cache_resource-wrapped bootstrap only runs its body once
    (on cache miss), and Streamlit's rerun/session threading means that miss can
    land on a different OS thread per bootstrap — so whichever one runs second
    hits dspy's single-owner-thread guard. Both bootstraps construct the same
    LLM_MODEL/OLLAMA_BASE_URL/OLLAMA_API_KEY, so skipping a redundant call here is
    safe rather than crashing.
    """
    lm = dspy.LM(LLM_MODEL, api_base=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY, cache=False)
    try:
        dspy.configure(lm=lm)
    except RuntimeError:
        pass


@st.cache_resource(show_spinner="Loading knowledge graphs…")
def build_system(k_hops: int, max_triples: int):
    _configure_dspy_lm()

    cfg      = load_toml(CONFIG_PATH)
    kg_cfg   = cfg.get("kg_paths", {})
    meta_cfg = cfg.get("metadata_paths", {})

    metadata = {
        name: load_metadata(Path(meta_cfg[f"{name}_metadata"]).expanduser())
        for name in ("neuroscience", "aiml", "neuromorphic")
        if meta_cfg.get(f"{name}_metadata")
    }

    agents = [
        SpecialistAgent("neuroscience", Path(kg_cfg["neuroscience_kg"]).expanduser(),
                        k_hops, max_triples, metadata=metadata.get("neuroscience")),
        SpecialistAgent("aiml",         Path(kg_cfg["aiml_kg"]).expanduser(),
                        k_hops, max_triples, metadata=metadata.get("aiml")),
        SpecialistAgent("neuromorphic", Path(kg_cfg["neuromorphic_kg"]).expanduser(),
                        k_hops, max_triples, metadata=metadata.get("neuromorphic")),
    ]
    return agents, Mediator()


@st.cache_resource(show_spinner="Loading knowledge graph…")
def build_neukrag_system(kg_name: str, k_hops: int, max_triples: int):
    _configure_dspy_lm()

    cfg      = load_toml(CONFIG_PATH)
    kg_cfg   = cfg.get("kg_paths", {})
    meta_cfg = cfg.get("metadata_paths", {})

    kg_key  = "all_kg" if kg_name == "all" else f"{kg_name}_kg"
    graph   = load_graph(Path(kg_cfg[kg_key]).expanduser())

    if kg_name == "all":
        unified_meta: dict = {}
        for name in ("neuroscience", "aiml", "neuromorphic"):
            if meta_cfg.get(f"{name}_metadata"):
                unified_meta.update(
                    load_metadata(Path(meta_cfg[f"{name}_metadata"]).expanduser())
                )
        metadata = unified_meta
    else:
        metadata = (
            load_metadata(Path(meta_cfg[f"{kg_name}_metadata"]).expanduser())
            if meta_cfg.get(f"{kg_name}_metadata") else {}
        )

    return graph, metadata, EntityExtractor(), HypothesisGenerator()
