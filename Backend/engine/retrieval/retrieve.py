# Backend/evidence_retrieval.py

import os
import requests
from urllib.parse import urlparse

def extract_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""

DOMAIN_TIER_MAP = {
    # Tier 1 – Authoritative / Primary
    "who.int": "T1",
    "nih.gov": "T1",
    "ncbi.nlm.nih.gov": "T1",   # PubMed
    "cdc.gov": "T1",
    "nature.com": "T1",
    "thelancet.com": "T1",

    # Tier 2 – High-quality secondary
    "britannica.com": "T2",
    "sciencedirect.com": "T2",
    "mayoclinic.org": "T2",
    "healthline.com": "T2",

    # Tier 3 – Everything else (default)
}


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")


def google_cse_search(query: str, max_results: int = 5):
    """
    Primary evidence retrieval using Google Custom Search JSON API.
    Returns list of evidence dicts or None on failure.
    """
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return None

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": max_results,
    }

    try:
        resp = requests.get(url, params=params, timeout=4)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("items", []):
            url = item.get("link", "")
            domain = extract_domain(url)
            
            tier = "T3"  # default (conservative)
            for known_domain, mapped_tier in DOMAIN_TIER_MAP.items():
                if domain.endswith(known_domain):
                    tier = mapped_tier
                    break
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": url,
                "source": "google",
                "domain": domain,
                "tier": tier
            })

        return results if results else None

    except Exception as e:
        print("[Google CSE failed]", e)
        return None
    
def retrieve_evidence(query: str):
    """
    Google-first evidence retrieval with DuckDuckGo fallback.
    Returns: (evidence_list, evidence_source)
    """

    # 1️⃣ Try Google first (primary path)
    google_results = google_cse_search(query)
    if google_results:
        return google_results, "google"

    # 3️⃣ Nothing found
    return [], "none"