# backend/cdvm/core/validator.py

from typing import Tuple

# ---- Schema lock ----

REQUIRED_KEYS = {
    "sentence",
    "subject_domain",
    "object_domain",
    "verb",
    "verb_native_domain",
    "cross_domain",
    "rhetorical_intensity",
    "notes",
}

ALLOWED_DOMAINS = {
    "politics",
    "economics",
    "society",
    "technology",
    "environment",
    "healthcare",
    "law",
}

ALLOWED_NATIVE_DOMAINS = {
    "physical",
    "biological",
    "mechanical",
    "moral",
}

ALLOWED_INTENSITY = {
    "low",
    "medium",
    "high",
}

MAX_SENTENCE_WORDS = 30
MAX_VERB_TOKENS = 4


def validate_cdvm_entry(entry: dict) -> Tuple[bool, str]:
    # 1. Exact schema check (no missing / no extra)
    if set(entry.keys()) != REQUIRED_KEYS:
        return False, "Schema mismatch (missing or extra keys)"

    # 2. Cross-domain must be true
    if entry["cross_domain"] is not True:
        return False, "cross_domain must be true"

    # 3. Enum validation
    if entry["subject_domain"] not in ALLOWED_DOMAINS:
        return False, "Invalid subject_domain"

    if entry["object_domain"] not in ALLOWED_DOMAINS:
        return False, "Invalid object_domain"

    if entry["verb_native_domain"] not in ALLOWED_NATIVE_DOMAINS:
        return False, "Invalid verb_native_domain"

    if entry["rhetorical_intensity"] not in ALLOWED_INTENSITY:
        return False, "Invalid rhetorical_intensity"

    # 4. Linguistic hard limits
    sentence = entry["sentence"].strip()
    verb = entry["verb"].strip()

    if not sentence or not verb:
        return False, "Empty sentence or verb"

    if len(sentence.split()) > MAX_SENTENCE_WORDS:
        return False, "Sentence too long"

    if len(verb.split()) > MAX_VERB_TOKENS:
        return False, "Verb phrase too long"

    return True, "Valid"
