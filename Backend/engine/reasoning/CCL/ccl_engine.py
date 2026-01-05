# ccl_engine.py
from .domain_db import (
    load_domain_keywords,
    load_domain_normalization,
    load_metaphor_rules
)
DOMAIN_KEYWORDS = load_domain_keywords()
DOMAIN_NORMALIZATION_MAP = load_domain_normalization()
METAPHOR_RULES = load_metaphor_rules()
ABSTRACT_ENTITIES = {a for a, _ in METAPHOR_RULES}
PHYSICAL_ACTIONS = {p for _, p in METAPHOR_RULES}

def minimal_ccl_decision(claim_text: str) -> dict:
    """
    Minimal CCL (Observe-only)
    Determines likely knowledge domain(s) and metaphor risk.
    No scoring impact. Debug-only output.
    """
    text = claim_text.lower()
    domains = []

    # --- Domain detection ---
    for raw_domain, keywords in DOMAIN_KEYWORDS.items():
        if any(k in text for k in keywords):
            domains.append(raw_domain)

    if not domains:
        domains.append("general_knowledge")

    print("CCL raw domains:", domains)

    # --- Domain normalization ---
    raw_domains = domains
    primary_raw = raw_domains[0]
    secondary_raw = raw_domains[1:]

    primary_domain = DOMAIN_NORMALIZATION_MAP.get(
        primary_raw, "Opinion / Interpretation"
    )

    secondary_domains = [
        DOMAIN_NORMALIZATION_MAP.get(d, "Opinion / Interpretation")
        for d in secondary_raw
    ]

    # --- Option B: Metaphor suspicion ---
    possible_metaphor = (
        any(a in text for a in ABSTRACT_ENTITIES)
        and any(p in text for p in PHYSICAL_ACTIONS)
    )

    return {
        "primary_domain": primary_domain,
        "secondary_domains": secondary_domains,
        "confidence": 0.8 if domains else 0.5,
        "reason": "Matched minimal CCL keyword rules",
        "notes": {
            "ambiguous": len(domains) > 1,
            "possible_metaphor": possible_metaphor,
            "mixed_claim": len(domains) > 1
        }
    }
