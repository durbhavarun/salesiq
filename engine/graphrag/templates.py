"""
engine/graphrag/templates.py
────────────────────────────
Three GraphRAG query templates.

CRITICAL attribute facts (from graph/populate.py — do NOT change):
  Node type attribute : "type"         (Account, Deal, Signal)
  Edge rel  attribute : "rel"          (HAS_DEAL, HAS_SIGNAL, REFERRED_BY, SHARES_GUARANTOR)

Score card key facts (from engine/scoring_engine.py — do NOT change):
  score key           : "score"        (not total_score)
  confidence key      : "confidence"   (not confidence_band)
  proximity key       : "network_proximity"  → dict with "hops" and "display"
"""
from __future__ import annotations
import os, logging, pickle
from pathlib import Path
from typing import Any
import networkx as nx

logger = logging.getLogger(__name__)
_GRAPH_CACHE: nx.DiGraph | None = None


def _get_graph() -> nx.DiGraph:
    global _GRAPH_CACHE
    if _GRAPH_CACHE is not None:
        return _GRAPH_CACHE
    pkl = Path(__file__).parent.parent.parent / "data" / "demo" / "graph.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            _GRAPH_CACHE = pickle.load(f)
            return _GRAPH_CACHE
    _GRAPH_CACHE = nx.DiGraph()
    return _GRAPH_CACHE


def _node_str(G: nx.DiGraph, nid: str) -> str:
    a = G.nodes.get(nid, {})
    t = a.get("type", "node")          # real attr: "type"
    parts = [f"{k}={v}" for k, v in list(a.items())[:6] if k != "type"]
    return f"[{t}:{nid}] " + ", ".join(parts)


def _no_llm(template: str, params: dict, paths: list, nodes: list) -> dict[str, Any]:
    return {
        "answer": (f"LLM disabled (SALESIQ_LLM_PROVIDER=none). "
                   f"Template '{template}' matched. {len(paths)} graph paths retrieved. "
                   f"Enable Groq or Ollama for natural-language answers."),
        "paths_used": paths, "nodes_cited": nodes,
        "template": template, "provider": "none",
        "valid": False, "error": "LLM provider disabled",
    }


def run_explain(lead_id: str, scores_state: dict, question: str = "") -> dict[str, Any]:
    if not lead_id:
        return _no_llm("explain", {"lead_id": lead_id}, [], [])
    G = _get_graph()
    paths: list[str] = []
    nodes: list[str] = []
    lines: list[str] = []

    if lead_id in G.nodes:
        lines.append("=== Lead Account ===")
        lines.append(_node_str(G, lead_id))
        nodes.append(lead_id)
    else:
        lines.append(f"Lead {lead_id} not in graph.")

    sc = scores_state.get(lead_id)
    if sc:
        lines.append("\n=== Score Breakdown ===")
        for k in ("score", "icp_fit", "intent_score", "confidence", "recommended_action"):
            if k in sc:
                lines.append(f"  {k}: {sc[k]}")
        prox = sc.get("network_proximity", {})
        if isinstance(prox, dict):
            lines.append(f"  network_proximity: {prox.get('display','unknown')}")
        for s in sc.get("positive_signals", [])[:5]:
            lines.append(f"  signal: {s.get('factor')} — {s.get('evidence')} [{s.get('source','')}]")
        for g in sc.get("gaps", []):
            lines.append(f"  gap: {g}")

    lines.append("\n=== Graph Signals ===")
    cnt = 0
    for _, dst, data in G.edges(lead_id, data=True):
        if data.get("rel") == "HAS_SIGNAL":          # real attr: "rel"
            lines.append("  " + _node_str(G, dst))
            paths.append(f"{lead_id} --[HAS_SIGNAL]--> {dst}")
            nodes.append(dst)
            cnt += 1
            if cnt >= 8:
                break

    lines.append("\n=== Network Connections ===")
    found = False
    for _, dst, data in G.edges(lead_id, data=True):
        rel = data.get("rel", "")
        if rel in ("REFERRED_BY", "SHARES_GUARANTOR"):
            lines.append(f"  {rel}: {_node_str(G, dst)}")
            paths.append(f"{lead_id} --[{rel}]--> {dst}")
            nodes.append(dst)
            found = True
            for _, ddst, dd in G.edges(dst, data=True):
                if dd.get("rel") == "HAS_DEAL" and G.nodes.get(ddst, {}).get("outcome") == "won":
                    lines.append(f"    won deal: {_node_str(G, ddst)}")
                    paths.append(f"{dst} --[HAS_DEAL]--> {ddst} (won)")
                    nodes.append(ddst)
    if not found:
        lines.append("  None detected within 2 hops")

    context = "\n".join(lines)
    nodes = list(set(nodes))
    if os.getenv("SALESIQ_LLM_PROVIDER", "groq").lower() == "none":
        return _no_llm("explain", {"lead_id": lead_id}, paths, nodes)
    from engine.llm_gateway import generate_graphrag_answer
    llm = generate_graphrag_answer(question or f"Why did {lead_id} receive its score?", context)
    return {"answer": llm["answer"], "paths_used": paths, "nodes_cited": nodes,
            "template": "explain", "provider": llm["provider"],
            "valid": llm["valid"], "error": llm.get("error")}


def run_pattern(lead_id: str, scores_state: dict, question: str = "") -> dict[str, Any]:
    G = _get_graph()
    paths: list[str] = []
    nodes: list[str] = []
    lines: list[str] = []
    match_fields = ["industry", "employee_band", "business_model",
                    "loan_type", "dscr_band", "collateral_type", "geography"]
    la = G.nodes.get(lead_id, {})
    lead_vals = {f: la.get(f) for f in match_fields if la.get(f)}
    lines.append(f"=== Lead Profile: {lead_id} ===")
    lines.append(", ".join(f"{k}={v}" for k, v in lead_vals.items()))
    nodes.append(lead_id)
    lines.append("\n=== Similar Won Accounts (2+ matching params) ===")
    found = 0
    for nid, na in G.nodes(data=True):
        if na.get("type") != "Account" or nid == lead_id:   # real attr: "type"
            continue
        matches = [(f, v) for f, v in lead_vals.items() if na.get(f) == v]
        if len(matches) < 2:
            continue
        won = [ddst for _, ddst, dd in G.edges(nid, data=True)
               if dd.get("rel") == "HAS_DEAL"                # real attr: "rel"
               and G.nodes.get(ddst, {}).get("outcome") == "won"]
        if not won:
            continue
        lines.append(f"\n  {nid}: {', '.join(f+chr(61)+str(v) for f,v in matches)}")
        for did in won[:2]:
            lines.append(f"  Won deal: {_node_str(G, did)}")
            paths.append(f"{nid} --[HAS_DEAL]--> {did} (won)")
            nodes.append(did)
        nodes.append(nid)
        found += 1
        if found >= 5:
            lines.append("  ... (truncated)")
            break
    if found == 0:
        lines.append("  No similar won accounts found with 2+ matching parameters.")
    context = "\n".join(lines)
    nodes = list(set(nodes))
    if os.getenv("SALESIQ_LLM_PROVIDER", "groq").lower() == "none":
        return _no_llm("pattern", {"lead_id": lead_id}, paths, nodes)
    from engine.llm_gateway import generate_graphrag_answer
    llm = generate_graphrag_answer(
        question or f"Have we won accounts similar to {lead_id}?", context)
    return {"answer": llm["answer"], "paths_used": paths, "nodes_cited": nodes,
            "template": "pattern", "provider": llm["provider"],
            "valid": llm["valid"], "error": llm.get("error")}


def run_compare(lead_id_a: str, lead_id_b: str, scores_state: dict,
                question: str = "") -> dict[str, Any]:
    G = _get_graph()
    paths: list[str] = []
    nodes: list[str] = []
    lines: list[str] = []
    for lid, label in [(lead_id_a, "Lead A"), (lead_id_b, "Lead B")]:
        lines.append(f"\n=== {label}: {lid} ===")
        if lid in G.nodes:
            lines.append(_node_str(G, lid))
            nodes.append(lid)
        sc = scores_state.get(lid)
        if sc:
            for k in ("score", "icp_fit", "intent_score", "confidence"):
                if k in sc:
                    lines.append(f"  {k}: {sc[k]}")
            prox = sc.get("network_proximity", {})
            if isinstance(prox, dict):
                lines.append(f"  network_proximity: {prox.get('display','unknown')}")
        cnt = 0
        for _, dst, data in G.edges(lid, data=True):
            if data.get("rel") == "HAS_SIGNAL":
                lines.append(f"  signal: {_node_str(G, dst)}")
                paths.append(f"{lid} --[HAS_SIGNAL]--> {dst}")
                nodes.append(dst)
                cnt += 1
                if cnt >= 4:
                    break
    context = "\n".join(lines)
    nodes = list(set(nodes))
    if os.getenv("SALESIQ_LLM_PROVIDER", "groq").lower() == "none":
        return _no_llm("compare", {"lead_id_a": lead_id_a, "lead_id_b": lead_id_b}, paths, nodes)
    from engine.llm_gateway import generate_graphrag_answer
    llm = generate_graphrag_answer(
        question or f"Why is {lead_id_a} scored differently from {lead_id_b}?", context)
    return {"answer": llm["answer"], "paths_used": paths, "nodes_cited": nodes,
            "template": "compare", "provider": llm["provider"],
            "valid": llm["valid"], "error": llm.get("error")}
