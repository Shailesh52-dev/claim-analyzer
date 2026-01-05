import re

def extract_main_verb(text: str) -> str | None:
    """
    Very lightweight verb extractor.
    Intended for rhetorical analysis, not full parsing.
    """

    if not text:
        return None

    text = text.lower()

    # Common metaphor / action verbs (extend later)
    VERB_CANDIDATES = [
        "buckling", "collapsing", "bleeding", "surging",
        "stalling", "navigating", "blocking", "crashing",
        "eroding", "accelerating", "slowing"
    ]

    for v in VERB_CANDIDATES:
        if re.search(rf"\b{v}\b", text):
            return v

    return None
