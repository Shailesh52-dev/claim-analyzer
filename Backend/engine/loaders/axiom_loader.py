import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AXIOM_DIR = os.path.join(BASE_DIR, "..", "axioms")


def _load_jsonl(path: str):
    """
    Generic JSONL loader.
    Each line must be a valid JSON object.
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_fkc_axioms():
    """
    Loads FKC axioms from JSONL.

    Returns:
    [
      {
        "id": str,
        "text": str,
        "severity": "absolute" | "physical" | "soft",
        "domain": str
      },
      ...
    ]
    """
    path = os.path.join(AXIOM_DIR, "fkc_axioms.jsonl")
    return _load_jsonl(path)


def load_fkc_nouns():
    """
    Loads FKC symbolic nouns / keywords from JSONL.

    Returns:
    [
      { "type": str, "value": str | list },
      ...
    ]
    """
    path = os.path.join(AXIOM_DIR, "fkc_nouns.jsonl")
    return _load_jsonl(path)
