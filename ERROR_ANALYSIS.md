# Error analysis: phase one

EXP-001 is a no-training diagnostic, not a competitive classifier. It makes 20,075 correct matches and zero false-positive links on development fold 0, but misses 98.69% of positive links. Most non-singletons remain empty. Therefore blocking/recall is the immediate bottleneck; tuning a precision threshold cannot repair missing candidates.

Full positive diagnostics (7,638,365 links) and examples are under outputs/audit/PAIR-002/. Difficult types: cross-script Indic names; field abbreviation/reordering; missing addresses; inserted leading numbers; short aliases and domain-style names; strong address with weak name. The 10,000-positive transliteration sample improves some name comparisons but has no negative-control estimate yet.

A deterministic 4,220-S1 probe against all targets found 38,477 same-normalized-name nonmatches. Common-name businesses at different addresses are routine. One hard case shares the street/city and business name but differs by house number 531 vs 532, with address JW above 0.94. Numeric contradiction and informative-location tokens need separate features. A blanket first-number veto is also unsafe because positives contain inserted leading numbers.

Next error report must distinguish: candidate miss, candidate retrieved but scored low, singleton false merge, competing S1 ownership, missing address, cross-script retrieval, common-name collision, misleading high token-set score, source-specific address noise, and country-transfer degradation. Report counts and weighted entity score loss, not only selected anecdotes.
