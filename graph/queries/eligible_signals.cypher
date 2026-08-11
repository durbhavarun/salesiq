// ============================================================
// Eligible Signals Query — Temporal Leakage Prevention
// Only returns signals where score_eligibility_date
// is on or before the deal_created_date.
// This prevents future information from leaking into scores.
// ============================================================

// Usage: pass $account_id and $as_of_date as parameters
MATCH (a:Account {account_id: $account_id})-[:HAS_SIGNAL]->(s:Signal)
WHERE s.score_eligibility_date <= $as_of_date
RETURN
    s.signal_id            AS signal_id,
    s.signal_type          AS signal_type,
    s.signal_value         AS signal_value,
    s.extraction_confidence AS confidence,
    s.evidence_snippet     AS evidence,
    s.score_eligibility_date AS eligible_from
ORDER BY s.score_eligibility_date DESC
