# Backend/cdvm/core/cdvm_analyzer.py

from typing import Optional, Dict

from engine.reasoning.cdvm.core.priors import(
    get_verb_prior,
    get_verb_intensity_prior,
    is_common_transfer,
)


def analyze_cdvm(
    verb: str,
    verb_native_domain: str,
    subject_domain: str,
    object_domain: str,
) -> Optional[Dict]:
    """
    Analyze Cross-Domain Verb Metaphor (CDVM).

    Returns a CDVM signal dict if cross-domain conditions are met,
    otherwise returns None.

    This function is OBSERVE-ONLY.
    """

    # ---- Basic cross-domain check ----
    if verb_native_domain == subject_domain:
        return None

    if verb_native_domain == object_domain:
        return None

    # ---- Query priors ----
    verb_prior = get_verb_prior(verb)
    intensity_prior = get_verb_intensity_prior(verb)
    common_transfer = is_common_transfer(
        verb_native_domain, object_domain
    )

    # ---- Build signal ----
    signal = {
        "type": "cdvm",
        "present": True,
        "verb": verb,
        "verb_native_domain": verb_native_domain,
        "subject_domain": subject_domain,
        "object_domain": object_domain,
        "domain_transfer": f"{verb_native_domain}->{object_domain}",
        "verb_intensity_prior": intensity_prior,
        "is_common_transfer": common_transfer,
        "confidence": round(
            0.6 * intensity_prior + 0.4 * (1.0 if common_transfer else 0.3),
            3
        ),
        "mode": "observe_only",
    }

    # ---- Optional metadata ----
    if verb_prior:
        signal["verb_seen_count"] = verb_prior.get("count", 0)
    else:
        signal["verb_seen_count"] = 0
        signal["note"] = "Verb not seen in calibration data"

    return signal
