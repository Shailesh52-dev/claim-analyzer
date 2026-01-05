# Backend/option3/option3_analyzer.py

from typing import Optional, Dict, List

# ---- Strategy enums (locked) ----

CAUSAL_DRAMATIZATION = "causal_dramatization"
STRUCTURAL_FAILURE = "structural_failure_framing"
URGENCY_FRAMING = "urgency_framing"
AGENT_INTENSIFICATION = "agent_intensification"


# ---- Risk note templates (neutral, reviewer-safe) ----

RISK_NOTES = {
    CAUSAL_DRAMATIZATION:
        "Frames cause–effect relationships in a heightened manner.",
    STRUCTURAL_FAILURE:
        "Uses breakdown metaphors to emphasize systemic stress.",
    URGENCY_FRAMING:
        "Employs uncommon metaphors that increase perceived urgency.",
    AGENT_INTENSIFICATION:
        "Attributes strong causal agency to an actor.",
}


def _add_unique(lst: List[str], item: str):
    if item not in lst:
        lst.append(item)


def _derive_strategies_from_cdvm(cdvm: Dict) -> List[str]:
    strategies: List[str] = []

    intensity = cdvm.get("verb_intensity_prior", 0.5)
    native_domain = cdvm.get("verb_native_domain")
    is_common = cdvm.get("is_common_transfer", False)
    subject_domain = cdvm.get("subject_domain")

    # Rule A — causal dramatization
    if intensity >= 0.4:
        _add_unique(strategies, CAUSAL_DRAMATIZATION)

    # Rule B — structural failure framing
    if intensity >= 0.8 and native_domain in {"physical", "mechanical"}:
        _add_unique(strategies, STRUCTURAL_FAILURE)

    # Rule C — urgency framing
    if intensity >= 0.6 and not is_common:
        _add_unique(strategies, URGENCY_FRAMING)

    # Rule D — agent intensification
    if subject_domain == "politics" and intensity >= 0.6:
        _add_unique(strategies, AGENT_INTENSIFICATION)

    return strategies


def _overall_intensity(strategies: List[str], cdvm_confidence: float) -> str:
    if not strategies:
        return "low"

    if cdvm_confidence < 0.5:
        return "low"

    if len(strategies) == 1 and strategies[0] == CAUSAL_DRAMATIZATION:
        return "medium"

    return "high"


def _build_explainability(strategies: List[str], confidence: float) -> Dict:
    base = "The claim uses metaphorical language to emphasize causality"

    if STRUCTURAL_FAILURE in strategies:
        base += " by portraying the issue as structurally strained or failing"
    elif URGENCY_FRAMING in strategies:
        base += " by conveying heightened urgency"

    base += "."

    if confidence >= 0.75:
        level = "high"
    elif confidence >= 0.5:
        level = "moderate"
    else:
        level = "low"

    return {
        "summary": base,
        "confidence_level": level,
    }


def analyze_option3(cdvm_signal: Optional[Dict]) -> Dict:
    """
    Maps CDVM signals to a rhetorical profile.
    Observe-only. No truth judgment.
    """

    if not cdvm_signal or not cdvm_signal.get("present"):
        return {
            "present": False
        }

    strategies = _derive_strategies_from_cdvm(cdvm_signal)
    cdvm_conf = cdvm_signal.get("confidence", 0.5)

    overall = _overall_intensity(strategies, cdvm_conf)

    risk_notes = [RISK_NOTES[s] for s in strategies]

    profile = {
        "present": True,
        "overall_intensity": overall,
        "strategies": strategies,
        "signals": {
            "cdvm": {
                "verb": cdvm_signal.get("verb"),
                "verb_native_domain": cdvm_signal.get("verb_native_domain"),
                "domain_transfer": cdvm_signal.get("domain_transfer"),
                "intensity_prior": cdvm_signal.get("verb_intensity_prior"),
                "is_common_transfer": cdvm_signal.get("is_common_transfer"),
            }
        },
        "risk_notes": risk_notes,
        "explainability": _build_explainability(strategies, cdvm_conf),
    }

    return profile
