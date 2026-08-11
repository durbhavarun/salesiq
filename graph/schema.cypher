// ============================================================
// SalesIQ — Neo4j Graph Schema
// Run with: neo4j-admin or via Python driver on startup
// ============================================================

// --- Uniqueness constraints (also create indexes automatically) ---

// Every Account has a unique account_id
CREATE CONSTRAINT account_id_unique IF NOT EXISTS
FOR (a:Account) REQUIRE a.account_id IS UNIQUE;

// Every Deal has a unique deal_id
CREATE CONSTRAINT deal_id_unique IF NOT EXISTS
FOR (d:Deal) REQUIRE d.deal_id IS UNIQUE;

// Every Signal has a unique signal_id
CREATE CONSTRAINT signal_id_unique IF NOT EXISTS
FOR (s:Signal) REQUIRE s.signal_id IS UNIQUE;

// Every ICPVersion has a unique version_id
CREATE CONSTRAINT icp_version_id_unique IF NOT EXISTS
FOR (v:ICPVersion) REQUIRE v.version_id IS UNIQUE;

// Every ScoreResult has a unique score_id
CREATE CONSTRAINT score_id_unique IF NOT EXISTS
FOR (sr:ScoreResult) REQUIRE sr.score_id IS UNIQUE;

// Every Override has a unique override_id
CREATE CONSTRAINT override_id_unique IF NOT EXISTS
FOR (o:Override) REQUIRE o.override_id IS UNIQUE;

// --- Relationship types used in this schema ---
// Account  -[HAS_DEAL]->         Deal
// Account  -[HAS_SIGNAL]->       Signal
// Signal   -[EVIDENCE_OF]->      Deal
// ScoreResult -[SCORED_BY]->     Account
// ScoreResult -[USED_ICP]->      ICPVersion
// Override -[OVERRIDES]->        ScoreResult
// Account  -[SHARES_GUARANTOR]-> Account   (Network Proximity)
// Account  -[REFERRED_BY]->      Account   (Network Proximity)
