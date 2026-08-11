// ============================================================
// Network Proximity Query
// Find the shortest path from a lead to a won deal
// within 2 hops via SHARES_GUARANTOR or REFERRED_BY edges
// ============================================================

// Usage: pass $account_id as parameter
MATCH (start:Account {account_id: $account_id})
MATCH (target:Account)-[:HAS_DEAL]->(d:Deal {outcome: 'won'})
WHERE start <> target
MATCH path = shortestPath(
    (start)-[:SHARES_GUARANTOR|REFERRED_BY*1..2]-(target)
)
RETURN
    target.account_id AS connected_account,
    d.deal_id         AS won_deal_id,
    length(path)      AS hops,
    [r IN relationships(path) | type(r)] AS relationship_types
ORDER BY hops ASC
LIMIT 1
