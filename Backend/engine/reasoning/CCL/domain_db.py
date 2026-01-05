import sqlite3
import os

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "data", "ccl_rules.db")

def load_domain_keywords():
    """
    Returns:
        dict[str, list[str]]  → raw_domain -> [keywords]
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT raw_domain, keyword
        FROM domains
    """)

    rows = cursor.fetchall()
    conn.close()

    domain_keywords = {}
    for raw_domain, keyword in rows:
        domain_keywords.setdefault(raw_domain, []).append(keyword.lower())

    return domain_keywords


def load_domain_normalization():
    """
    Returns:
        dict[str, str] → raw_domain -> normalized_domain
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT raw_domain, normalized_domain
        FROM domains
    """)

    rows = cursor.fetchall()
    conn.close()

    return {raw: norm for raw, norm in rows}


def load_metaphor_rules():
    """
    Returns:
        list[tuple[str, str]] → (abstract_entity, physical_action)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT abstract_entity, physical_action
        FROM metaphors
    """)

    rows = cursor.fetchall()
    conn.close()

    return [(a.lower(), p.lower()) for a, p in rows]
