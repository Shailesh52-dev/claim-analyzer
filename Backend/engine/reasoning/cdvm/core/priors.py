# Backend/cdvm/core/priors.py

import json
from pathlib import Path
from typing import Optional, Dict

# ---- Paths (derived data only) ----

BASE_DIR = Path(__file__).resolve().parents[2]  # Backend/
DERIVED_DIR = BASE_DIR / "cdvm" / "data" / "derived"

VERB_PRIORS_PATH = DERIVED_DIR / "verb_priors.json"
DOMAIN_TRANSFERS_PATH = DERIVED_DIR / "domain_transfers.json"


# ---- Internal caches (loaded once) ----

_VERB_PRIORS: Optional[Dict] = None
_DOMAIN_TRANSFERS: Optional[Dict] = None


# ---- Loaders ----

def _load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"CDVM derived file missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_loaded():
    global _VERB_PRIORS, _DOMAIN_TRANSFERS

    if _VERB_PRIORS is None:
        _VERB_PRIORS = _load_json(VERB_PRIORS_PATH)

    if _DOMAIN_TRANSFERS is None:
        _DOMAIN_TRANSFERS = _load_json(DOMAIN_TRANSFERS_PATH)


# ---- Public accessors (read-only) ----

def get_verb_prior(verb: str) -> Optional[Dict]:
    """
    Returns:
        {
          "native_domain": str,
          "intensity_prior": float,
          "count": int
        }
        or None if verb is unseen
    """
    _ensure_loaded()
    return _VERB_PRIORS.get(verb)


def get_verb_intensity_prior(verb: str, default: float = 0.5) -> float:
    """
    Returns intensity prior for verb.
    Falls back to a neutral default if unseen.
    """
    _ensure_loaded()
    entry = _VERB_PRIORS.get(verb)
    if entry is None:
        return default
    return entry.get("intensity_prior", default)


def get_domain_transfer_prior(
    native_domain: str,
    object_domain: str
) -> Optional[Dict]:
    """
    Returns:
        {
          "count": int,
          "frequency": float
        }
        or None if unseen
    """
    _ensure_loaded()
    key = f"{native_domain}->{object_domain}"
    return _DOMAIN_TRANSFERS.get(key)


def is_common_transfer(
    native_domain: str,
    object_domain: str,
    threshold: float = 0.05
) -> bool:
    """
    Returns True if the domain transfer frequency
    exceeds the given threshold.
    """
    _ensure_loaded()
    key = f"{native_domain}->{object_domain}"
    entry = _DOMAIN_TRANSFERS.get(key)
    if entry is None:
        return False
    return entry.get("frequency", 0.0) >= threshold


# ---- Optional: debug helpers ----

def list_known_verbs():
    _ensure_loaded()
    return sorted(_VERB_PRIORS.keys())


def list_domain_transfers():
    _ensure_loaded()
    return sorted(_DOMAIN_TRANSFERS.keys())
