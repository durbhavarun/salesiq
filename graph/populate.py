"""
graph/populate.py
=================
Loads all data into the knowledge graph.
Supports two backends:
  - Neo4j (via Docker):     SALESIQ_GRAPH_BACKEND=neo4j
  - NetworkX (in-memory):   SALESIQ_GRAPH_BACKEND=memory  (default)

Why this abstraction exists:
  Neo4j requires Docker. Not everyone has Docker running.
  NetworkX runs in pure Python with zero infrastructure.
  Both implement the same graph structure and support the same queries.
  The env var switches between them — no code changes needed.
"""

import os, json, time, sys
import pandas as pd
import networkx as nx
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BACKEND  = os.getenv("SALESIQ_GRAPH_BACKEND", "memory")
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "demo"

# ── paths ──────────────────────────────────────────────────────────────
DEALS_CSV   = DATA_DIR / "historical_deals.csv"
LEADS_CSV   = DATA_DIR / "new_leads.csv"
SIGNALS_JSON = DATA_DIR / "signals.json"

# ═══════════════════════════════════════════════════════════════════════
# NETWORKX BACKEND (in-memory, no Docker)
# ═══════════════════════════════════════════════════════════════════════

def build_networkx_graph() -> nx.DiGraph:
    """
    Build the complete knowledge graph in NetworkX.
    Returns a directed graph where:
      - Nodes have a 'type' attribute ('Account','Deal','Signal',etc.)
      - Edges have a 'rel' attribute matching Neo4j relationship names
    """
    G = nx.DiGraph()

    deals_df   = pd.read_csv(DEALS_CSV)
    leads_df   = pd.read_csv(LEADS_CSV)
    signals    = json.loads(SIGNALS_JSON.read_text())

    print(f"  Loading {len(deals_df)} historical deals...")
    print(f"  Loading {len(leads_df)} new leads...")
    print(f"  Loading {len(signals)} signals...")

    # ── Add Account nodes from historical deals ──
    for _, row in deals_df.iterrows():
        acc_id = row["account_id"]
        if acc_id not in G:
            G.add_node(acc_id, type="Account", **{
                k: v for k, v in row.items()
                if k not in ["deal_id","outcome","deal_created_date","deal_closed_date"]
            })

        # Add Deal node
        deal_id = row["deal_id"]
        G.add_node(deal_id, type="Deal",
                   deal_id=deal_id,
                   account_id=acc_id,
                   outcome=row["outcome"],
                   deal_created_date=row["deal_created_date"],
                   deal_closed_date=row.get("deal_closed_date",""))

        # HAS_DEAL edge
        G.add_edge(acc_id, deal_id, rel="HAS_DEAL")

    # ── Add Account nodes from new leads ──
    for _, row in leads_df.iterrows():
        acc_id = row["account_id"]
        if acc_id not in G:
            G.add_node(acc_id, type="Account", **{
                k: v for k, v in row.items()
            })

    # ── Add Signal nodes ──
    for sig in signals:
        sig_id = sig["signal_id"]
        acc_id = sig["account_id"]
        G.add_node(sig_id, type="Signal", **sig)
        if acc_id in G:
            G.add_edge(acc_id, sig_id, rel="HAS_SIGNAL")

    # ── Add SHARES_GUARANTOR edges (Network Proximity — Pattern 3) ──
    # Accounts referred by portfolio are connected to a reference won account
    referred_accounts = deals_df[deals_df["referred_by_portfolio"] == "yes"]["account_id"].tolist()
    lead_referrals = leads_df[leads_df["referred_by_portfolio"] == "yes"]["account_id"].tolist()
    anchor_won = deals_df[deals_df["outcome"] == "won"]["account_id"].iloc[0]

    for acc_id in referred_accounts + lead_referrals:
        if acc_id in G and anchor_won in G and acc_id != anchor_won:
            G.add_edge(acc_id, anchor_won, rel="REFERRED_BY")

    # ── Add SHARES_GUARANTOR edges for Manufacturing pattern ──
    mfg_won = deals_df[
        (deals_df["industry"] == "Manufacturing") &
        (deals_df["outcome"] == "won") &
        (deals_df["loan_purpose"] == "equipment_financing")
    ]["account_id"].tolist()

    mfg_leads = leads_df[
        (leads_df["industry"] == "Manufacturing") &
        (leads_df["loan_purpose"] == "equipment_financing")
    ]["account_id"].tolist()

    if mfg_won:
        anchor_mfg = mfg_won[0]
        for acc_id in mfg_leads:
            if acc_id in G and anchor_mfg in G and acc_id != anchor_mfg:
                G.add_edge(acc_id, anchor_mfg, rel="SHARES_GUARANTOR")

    node_counts = {}
    for n, d in G.nodes(data=True):
        t = d.get("type","unknown")
        node_counts[t] = node_counts.get(t, 0) + 1

    print(f"  Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    for t, c in sorted(node_counts.items()):
        print(f"    {t}: {c}")

    return G


# ═══════════════════════════════════════════════════════════════════════
# NEO4J BACKEND
# ═══════════════════════════════════════════════════════════════════════

def populate_neo4j():
    """
    Load all data into Neo4j.
    Includes a retry loop because Neo4j in Docker takes ~30s to start.
    Without the retry, populate.py would crash immediately on docker-compose up.
    """
    from neo4j import GraphDatabase

    uri      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user     = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "salesiq_password_2024")

    # ── Retry loop: wait up to 60s for Neo4j to boot ──
    driver = None
    for attempt in range(12):
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()
            print(f"  Connected to Neo4j after {attempt+1} attempt(s)")
            break
        except Exception as e:
            print(f"  Neo4j not ready (attempt {attempt+1}/12) — waiting 5s...")
            time.sleep(5)

    if driver is None:
        print("ERROR: Could not connect to Neo4j after 60s. Is Docker running?")
        print("TIP: Run 'docker-compose up -d' first, wait 30s, then retry.")
        sys.exit(1)

    deals_df  = pd.read_csv(DEALS_CSV)
    leads_df  = pd.read_csv(LEADS_CSV)
    signals   = json.loads(SIGNALS_JSON.read_text())

    with driver.session() as session:

        # Clear existing data
        session.run("MATCH (n) DETACH DELETE n")
        print("  Cleared existing graph data")

        # Load schema constraints
        schema_path = Path(__file__).parent / "schema.cypher"
        if schema_path.exists():
            for stmt in schema_path.read_text().split(";"):
                stmt = stmt.strip()
                if stmt and not stmt.startswith("//"):
                    try:
                        session.run(stmt)
                    except Exception:
                        pass  # Constraints may already exist

        # ── Accounts from historical deals ──
        for _, row in deals_df.iterrows():
            session.run("""
                MERGE (a:Account {account_id: $account_id})
                SET a.industry = $industry,
                    a.employee_band = $employee_band,
                    a.company_stage = $company_stage,
                    a.geography = $geography,
                    a.referred_by_portfolio = $referred_by_portfolio
            """, account_id=row["account_id"],
                 industry=row["industry"],
                 employee_band=row["employee_band"],
                 company_stage=row["company_stage"],
                 geography=row["geography"],
                 referred_by_portfolio=row.get("referred_by_portfolio","no"))

            session.run("""
                MERGE (d:Deal {deal_id: $deal_id})
                SET d.outcome = $outcome,
                    d.deal_created_date = $deal_created_date,
                    d.account_id = $account_id
                WITH d
                MATCH (a:Account {account_id: $account_id})
                MERGE (a)-[:HAS_DEAL]->(d)
            """, deal_id=row["deal_id"],
                 outcome=row["outcome"],
                 deal_created_date=row["deal_created_date"],
                 account_id=row["account_id"])

        # ── Accounts from new leads ──
        for _, row in leads_df.iterrows():
            session.run("""
                MERGE (a:Account {account_id: $account_id})
                SET a.industry = $industry,
                    a.company_stage = $company_stage,
                    a.geography = $geography,
                    a.referred_by_portfolio = $referred_by_portfolio
            """, account_id=row["account_id"],
                 industry=row["industry"],
                 company_stage=row["company_stage"],
                 geography=row["geography"],
                 referred_by_portfolio=row.get("referred_by_portfolio","no"))

        # ── Signals ──
        for sig in signals:
            session.run("""
                MERGE (s:Signal {signal_id: $signal_id})
                SET s.signal_type = $signal_type,
                    s.signal_value = $signal_value,
                    s.extraction_confidence = $extraction_confidence,
                    s.score_eligibility_date = $score_eligibility_date,
                    s.evidence_snippet = $evidence_snippet
                WITH s
                MATCH (a:Account {account_id: $account_id})
                MERGE (a)-[:HAS_SIGNAL]->(s)
            """, **sig)

        # ── Network Proximity edges ──
        referred = deals_df[deals_df["referred_by_portfolio"]=="yes"]["account_id"].tolist()
        lead_ref = leads_df[leads_df["referred_by_portfolio"]=="yes"]["account_id"].tolist()
        anchor   = deals_df[deals_df["outcome"]=="won"]["account_id"].iloc[0]

        for acc in referred + lead_ref:
            if acc != anchor:
                session.run("""
                    MATCH (a:Account {account_id: $acc})
                    MATCH (b:Account {account_id: $anchor})
                    MERGE (a)-[:REFERRED_BY]->(b)
                """, acc=acc, anchor=anchor)

        result = session.run("MATCH (n) RETURN labels(n)[0] AS lbl, count(*) AS cnt")
        print("  Neo4j node counts:")
        for record in result:
            print(f"    {record['lbl']}: {record['cnt']}")

    driver.close()
    print("  Neo4j population complete")


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Backend: {BACKEND}")
    print()

    if BACKEND == "neo4j":
        print("Populating Neo4j graph...")
        populate_neo4j()
    else:
        print("Building NetworkX in-memory graph...")
        G = build_networkx_graph()
        # Save a pickle for the API to load at startup
        import pickle
        graph_path = DATA_DIR / "graph.pkl"
        with open(graph_path, "wb") as f:
            pickle.dump(G, f)
        print(f"  Graph saved to {graph_path}")

    print()
    print("PASS: Graph population complete")
