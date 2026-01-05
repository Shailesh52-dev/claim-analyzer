# nli/level2/implicit_conflict.py

from typing import Dict, List

# -------------------------------------------------
# Fundamental priors (VERY SMALL, VERY STRONG)
# -------------------------------------------------

FUNDAMENTAL_PRIORS = [
    {
        "concept": "sun_direction",
        "trigger_terms": ["sun", "sunrise", "sunset"],
        "invalid_phrases": ["sets in the east", "rises in the west"],
        "confidence": 0.85,
        "reason": "Violates basic astronomical directionality"
    },
    {
        "concept": "human_respiration",
        "trigger_terms": ["human", "people", "man", "woman"],
        "invalid_phrases": ["breathe underwater", "live underwater without oxygen"],
        "confidence": 0.80,
        "reason": "Violates biological respiration constraints"
    },
    {
        "concept": "fire_properties",
        "trigger_terms": ["fire", "flame"],
        "invalid_phrases": ["freezes", "makes things cold"],
        "confidence": 0.75,
        "reason": "Violates thermodynamic behavior of fire"
    }
]

# -------------------------------------------------
# Core detector
# -------------------------------------------------

def detect_implicit_conflict(text: str) -> Dict:
    text_lower = text.lower()

    detected_conflicts: List[Dict] = []

    for prior in FUNDAMENTAL_PRIORS:
        # Check if the concept is even relevant
        if not any(term in text_lower for term in prior["trigger_terms"]):
            continue

        # Check for implicit violation phrases
        for phrase in prior["invalid_phrases"]:
            if phrase in text_lower:
                detected_conflicts.append({
                    "concept": prior["concept"],
                    "confidence": prior["confidence"],
                    "reason": prior["reason"],
                    "matched_phrase": phrase
                })

    if not detected_conflicts:
        return {
            "implicit_conflict": False,
            "confidence": 0.0,
            "details": []
        }

    # Use the strongest detected conflict
    strongest = max(detected_conflicts, key=lambda x: x["confidence"])

    return {
        "implicit_conflict": True,
        "confidence": strongest["confidence"],
        "details": detected_conflicts
    }
