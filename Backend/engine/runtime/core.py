
 # FIX: Corrected CORSMiddleware import name (lowercase 'w')
import math
import requests 
import json
import os
import urllib.parse
from urllib.parse import urlparse
from bs4 import BeautifulSoup 
from typing import List, Dict, Any, Union, Optional
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import time 
import torch
import re
from io import BytesIO
from PIL import Image, ImageEnhance
from engine.loaders.axiom_loader import load_fkc_axioms, load_fkc_nouns
from engine.nli.level2.implicit_conflict import detect_implicit_conflict
from engine.reasoning.CCL.ccl_engine import minimal_ccl_decision
from engine.reasoning.cdvm.core.cdvm_analyzer import analyze_cdvm
from engine.reasoning.option3.option3_analyzer import analyze_option3
from engine.utils.verb_extractor import extract_main_verb
from engine.retrieval.retrieve import retrieve_evidence
from dotenv import load_dotenv
load_dotenv() 
# --- transformers availability guard ---
TRANSFORMERS_AVAILABLE = False

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
    )
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    # Do NOT crash at import time (Cloud Run safe)
    print(f"[WARN] transformers not available: {e}")

NN_LOAD_STARTED = False
NN_MODEL_LOADING = False
FKC = None
def load_local_nn_model():
    global LOCAL_NN_MODEL, LOCAL_NN_TOKENIZER, NN_MODEL_AVAILABLE
    global VLM_MODEL, TROCR_PROCESSOR
    global NN_MODEL_LOADING, FKC

    # --- guard: prevent double load ---
    if NN_MODEL_LOADING:
        return

    NN_MODEL_LOADING = True

    try:
        # --- availability check (NO crash) ---
        if not TRANSFORMERS_AVAILABLE:
            print("[WARN] transformers not available, skipping NN load")
            return

        # ===============================
        # 1. NN MODEL LOADING
        # ===============================
        # if os.path.exists(LOCAL_NN_MODEL_PATH):
        #     LOCAL_NN_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        #     LOCAL_NN_MODEL = AutoModelForSequenceClassification.from_pretrained(
        #         LOCAL_NN_MODEL_PATH,
        #         num_labels=NUM_OUTPUT_HEADS
        #     )
        #     LOCAL_NN_MODEL.eval()
        #     NN_MODEL_AVAILABLE = True
        # else:
        #     NN_MODEL_AVAILABLE = False

        # ===============================
        # 1.5 LOAD FKC MODEL (SAFE)
        # ===============================
        if FKC is not None:
            FKC.load_model()

        # ===============================
        # 2. OCR / TrOCR (OPTIONAL)
        # ===============================
        # Only load if you already had this logic before
        # (leave commented if not needed immediately)
        #
        # TROCR_PROCESSOR = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        # VLM_MODEL = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    except Exception as e:
        print(f"[NN LOAD FAILED] {e}")

    finally:
        NN_MODEL_LOADING = False
def start_nn_model_loader():
    global NN_LOAD_STARTED
    if NN_LOAD_STARTED:
        return
    NN_LOAD_STARTED = True

    import threading

    def _safe_load():
        try:
            load_local_nn_model()
        except Exception as e:
            print("❌ NN model load failed:", e)

    threading.Thread(
        target=_safe_load,
        daemon=True
).start()
    # --- EXECUTION MODES ---
SAFE_MODE = "SAFE"        # Submission / production-safe
OBSERVE_MODE = "OBSERVE"  # Analysis only, no effects
FULL_ACT_MODE = "FULL_ACT"  # Experimental, aggressive

ALLOWED_MODES = {SAFE_MODE, OBSERVE_MODE, FULL_ACT_MODE}



# --- PADDLEOCR CONFIGURATION ---
# Fix for connectivity check hang/error
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

# --- PDF Dependency Check ---
# To enable PDF processing, install PyMuPDF: pip install PyMuPDF
PDF_SUPPORT_AVAILABLE = False
try:
    # PyMuPDF is often used for fast PDF handling
    import fitz 
    PDF_SUPPORT_AVAILABLE = True
    print("PDF conversion dependency (PyMuPDF/fitz) loaded.")
except ImportError:
    # Do not print warning here, as the failure is handled gracefully later if a PDF is uploaded.
    pass
# -----------------------------

# --- PADDLEOCR/TRANSFORMERS Imports (Restructured for stability) ---

# Always import common Transformers components

# Attempt to import PaddleOCR
PADDLEOCR_AVAILABLE = False
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    print("WARNING: PaddleOCR not found. Will attempt TrOCR fallback.")

# --- CRITICAL CONFIGURATION ---
API_KEY = os.getenv("API_KEY") 
MAX_RETRIES = 5
# --- CRITICAL FIX: Define MODEL_NAME for NN Tokenizer/Model loading ---
MODEL_NAME = 'distilbert-base-uncased' 
FKC_MODEL_NAME = 'facebook/bart-large-mnli' # New NLI Model for Fundamental Checker

# NEW: Configurable thresholds for NN signal mapping (0.0 to 1.0)
SIGNAL_THRESHOLDS = {
"high": 0.8,
"medium": 0.6,
"low": 0.4
}

# --- LOCAL MODEL INTEGRATION ---
NN_MODEL_AVAILABLE = False
NUM_OUTPUT_HEADS = 4
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_NN_MODEL_PATH = os.path.join(BASE_DIR, "local_nn_model")


# Initialize placeholders for VLM models to check their existence later
VLM_MODEL = None
TROCR_PROCESSOR = None

# --- NEW COMPONENT: KNOWLEDGE SANITY ASSISTANT (KSA) [DISABLED] ---
class KnowledgeSanityAssistant:
    def __init__(self):
        self.active = False # STRICTLY DISABLED
        self.known_false_patterns = [
            "cure all diseases",
            "doctors don't want you to know",
            "secret insiders reveal",
            "overnight cure",
            "hidden global control",
            "shut down the internet worldwide",
            "all financial assets will be frozen",
            "stock market will crash to zero",
        ]
        print("Knowledge Sanity Assistant (KSA) initialized in DORMANT mode.")

    def check(self, text: str) -> dict:
        """
        Input: Text content
        Output: Flag and reason (but functionality is disabled via self.active)
        """
        if not self.active:
            return {"flag": False, "reason": "KSA Disabled"}
            
        text_lower = text.lower()
        for pattern in self.known_false_patterns:
            if pattern in text_lower:
                return {
                    "flag": True, 
                    "reason": f"KSA Match: '{pattern}'"
                }
        return {"flag": False, "reason": None}

KSA = KnowledgeSanityAssistant()

# --- NEW COMPONENT: FUNDAMENTAL KNOWLEDGE CHECKER (FKC) [INVARIANT GUARDRAIL] ---
class FundamentalKnowledgeChecker:
    def __init__(self):
        self.enabled = True
        self.model_ready = False
        self.tokenizer = None
        self.model = None

        # Load axioms
        try:
            self.axioms = load_fkc_axioms()
            self.keyword_sets = load_fkc_nouns()
            print(f"[FKC] Loaded {len(self.axioms)} axioms and {len(self.keyword_sets)} keywords.")
        except Exception as e:
            print(f"[FKC] Failed to load axioms, using fallback: {e}")
            self.axioms = [
                {"text": "Tuesday comes before Wednesday.", "severity": "absolute"},
                {"text": "Water is chemically composed of hydrogen and oxygen.", "severity": "absolute"},
                {"text": "A triangle has three sides.", "severity": "absolute"},
                {"text": "A square has four sides.", "severity": "absolute"},
            ]
            self.keyword_sets = {}

    def load_model(self):
        if not TRANSFORMERS_AVAILABLE:
            print("[FKC] Transformers not available.")
            return

        try:
            print(f"[FKC] Initializing NLI model: {FKC_MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(FKC_MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(FKC_MODEL_NAME)
            self.model_ready = True
            print("[FKC] Model ready.")
        except Exception as e:
            print(f"[FKC] Model init failed: {e}")
            self.model_ready = False
            self.enabled = False

    def check_claim_simple(self, claim_text: str) -> dict:
        # --- DEBUG STATE (MUST BE FIRST) ---
        nli_debug = {
        "checked_axioms": [],
        "best_match": None,
        "decision": None
        }
        
        # Gate
        if not self.enabled or not self.model_ready:
            return {
                "skipped": True,
                "violation": False,
                "reason": "FKC disabled or model not ready",
                "debug": {
                    "nli": nli_debug
                }
        }

        normalized = claim_text.lower().strip()



        # Absolute axiom fast-path
        for axiom in self.axioms:
            axiom_text = axiom["text"]
            if axiom["severity"] != "absolute":
                continue

            ax_text = axiom_text.lower()
            # 1. Numeric / definition contradictions
            numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
            for n in numbers:
                if n in normalized and n in axiom_text and normalized != axiom_text:
                    return {
                        "skipped": False,
                        "violation": True,
                        "is_hard": True,
                        "reason": "Absolute axiom contradiction (numeric)",
                        "debug": {
                            "axiom": axiom["text"],
                            "claim": claim_text
                        }
                    }
        # 2. Composition contradictions
        if ("made of" in normalized or "composed of" in normalized):
            if "water" in normalized and "fire" in normalized:
                return {
                    "skipped": False,
                    "violation": True,
                    "is_hard": True,
                    "reason": "Invalid material composition",
                    "debug": {
                        "axiom": axiom["text"],
                        "claim": claim_text
                    }
                }
        # 3. Calendar / identity contradictions
        if "tuesday" in normalized and "wednesday" in normalized:
                return {
                    "skipped": False,
                    "violation": True,
                    "is_hard": True,
                    "reason": "Calendar invariant violation",
                    "debug": {
                        "axiom": axiom["text"],
                        "claim": claim_text
                    }
                }

                        

        # Length / context gate
        if len(normalized.split()) < 4:
            return {
                "skipped": False,
                "violation": False,
                "debug": {"skipped_length": True}
            }
        # NLI reasoning
        strongest = 0.0
        violated_axiom = None
        severity = "soft"

        for axiom in self.axioms:
            if axiom["severity"] == "soft":
                continue

            nli = self.run_nli_inference(axiom["text"], claim_text)
            contradiction = nli.get("contradiction", 0.0)

            if contradiction > strongest:
                strongest = contradiction
                violated_axiom = axiom["text"]
                severity = axiom["severity"]

        threshold = 0.97 if severity == "absolute" else 0.75

        if strongest >= threshold and violated_axiom:
            return {
                "violation": True,
                "is_hard": severity == "absolute",
                "confidence": strongest,
                "axiom_text": violated_axiom,
                "debug": {"source": "nli"}
            }

        return {
            "skipped": False,
            "violation": False
            }
    def calibrate_contradiction(self, p: float) -> float:
        """
        Maps raw NLI contradiction probability to a calibrated score.
        Suppresses low-confidence noise.
        """
        if p < 0.60:
            return 0.0
        return min(1.0, (p - 0.60) / 0.40)


    # --- GENERIC NLI INFERENCE HELPER (Used by CRCS) ---
    def run_nli_inference(self, premise: str, hypothesis: str) -> dict:
        """
        Runs NLI on arbitrary premise/hypothesis pairs.
        Returns raw probability map: {'entailment': float, 'contradiction': float, 'neutral': float}
        """
        if not self.enabled:
            return {'entailment': 0.0, 'contradiction': 0.0, 'neutral': 1.0}

        try:
            inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = logits.softmax(dim=1)
            
            label_map = self.model.config.id2label
            result = {}
            for idx, label in label_map.items():
                result[label.lower()] = probs[0][idx].item()
            
            return result
        except Exception as e:
            return {'entailment': 0.0, 'contradiction': 0.0, 'neutral': 1.0}

    def check_claim(self, claim_text: str) -> dict:
        """
            Checks a claim against axioms using NLI. 
            Returns structured data about violations.
            """
        nli_debug = {
        "checked_axioms": [],
        "best_match": None,
        "decision": None
    }
        default_result = {'violation': False, 'is_hard': False, 'warning': None, 'confidence': 0.0, 'debug': {}}

            # --- FIX 2: CONTEXT & BELIEF SHIELDING ---
        PROTECTED_CONTEXT_TERMS = [
                "believed", "thought", "according to", "historically",
                "in the past", "medieval", "ancient", "once believed",
                "claims that", "hypothetically", "if ", "suppose"
            ]
        if any(term in claim_text.lower() for term in PROTECTED_CONTEXT_TERMS):
                default_result['debug']['shielded'] = True
                return default_result

            # FIX 1 gating handled in analyze_content, keeping length safety
        if not self.enabled or not claim_text or len(claim_text.split()) < 6:
                default_result['debug']['skipped_length'] = True
                return default_result

        lower_claim = claim_text.lower()

            # --- FIX 1 & 2: GLOBAL ABSOLUTE INVARIANT PRECHECKS (SYMBOLIC) ---
        COMPOSITION_INDICATORS = ["made of", "composed of", "consists of", "made from", "created from"]
        IMPOSSIBLE_MATERIALS = ["fire", "stone", "magic", "nothing", "pure energy"]
            
        if any(ind in lower_claim for ind in COMPOSITION_INDICATORS):
            if any(mat in lower_claim for mat in IMPOSSIBLE_MATERIALS):
                    comp_axiom = next((a["text"] for a in self.axioms if "Water" in a["text"] and a["severity"] == "absolute"), None)
            if not comp_axiom:
                        comp_axiom = next((a["text"] for a in self.axioms if a["severity"] == "absolute"), "Material Composition Laws")
                    
            return {
                        'violation': True, 
                        'is_hard': True, 
                        'confidence': 1.0, 
                        'warning': f"Scientific Conflict (ABSOLUTE): Invalid composition claim detected against '{comp_axiom}'",
                        'axiom_text': comp_axiom,
                        'debug': {'source': 'symbolic_composition_precheck'}
                    }

        IDENTITY_PHRASES = ["is the same as", "equals", "is identical to", "is the same day as"]
        if any(p in lower_claim for p in IDENTITY_PHRASES):
                if "tuesday" in lower_claim and "wednesday" in lower_claim:
                    tues_axiom = next((a["text"] for a in self.axioms if "Tuesday" in a["text"]), "Calendar Invariants")
                    return {
                        'violation': True, 'is_hard': True, 'confidence': 1.0,
                        'warning': f"Logical Contradiction (ABSOLUTE): Identity error detected against '{tues_axiom}'",
                        'axiom_text': tues_axiom,
                        'debug': {'source': 'symbolic_identity_precheck'}
                    }

        PHYSICAL_PHRASES = ["fall upward", "gravity repels", "pushes objects away", "earth is flat"]
        if any(p in lower_claim for p in PHYSICAL_PHRASES):
                phys_axiom = next((a["text"] for a in self.axioms if a["severity"] == "physical"), "Physical Laws")
                return {
                    'violation': True, 'is_hard': False, 'confidence': 1.0,
                    'warning': f"Scientific Conflict (PHYSICAL): Claim contains anti-physical phrasing.",
                    'axiom_text': phys_axiom,
                    'debug': {'source': 'symbolic_physical_precheck'}
                }

        # --- NLI CHECK (GLOBAL / CONTEXT-INDEPENDENT) ---
        strongest_contradiction = 0.0
        violated_axiom_text = ""
        violated_axiom_severity = "soft"

        try:
                for axiom in self.axioms:
                    # --- BIDIRECTIONAL NLI ---
                    nli_forward = self.run_nli_inference(axiom["text"], claim_text)
                    nli_reverse = self.run_nli_inference(claim_text, axiom["text"])
                    raw_contradiction = max(
                        nli_forward.get("contradiction", 0.0),
                        nli_reverse.get("contradiction", 0.0),
                    )
                    raw_entailment = max(
                        nli_forward.get("entailment", 0.0),
                        nli_reverse.get("entailment", 0.0)
                    )
                    # --- MARGIN-BASED CONTRADICTION SCORING ---
                    CONTRADICTION_MARGIN = 0.20
                    
                    margin_score = raw_contradiction - raw_entailment
                    
                    # ✅ ADD DEBUG COLLECTION HERE
                    
                    nli_debug["checked_axioms"].append({
                        "axiom": axiom["text"],
                        "severity": axiom["severity"],
                        "forward": nli_forward,
                        "reverse": nli_reverse,
                        "raw_contradiction": raw_contradiction,
                        "raw_entailment": raw_entailment,
                        "margin_score": margin_score
                    })   
                    if margin_score <= CONTRADICTION_MARGIN:
                        
                    # NLI is confused or ambiguous → ignore
                        contradiction_score = 0.0
                        continue
                    else:
                        contradiction_score = self.calibrate_contradiction(raw_contradiction)
                    # --- FIX: SOFT axioms must not dominate axiom selection ---
                    if axiom["severity"] == "soft":
                        continue
                    if contradiction_score > strongest_contradiction:
                        strongest_contradiction = contradiction_score
                        violated_axiom_text = axiom["text"]
                        violated_axiom_severity = axiom["severity"]
                
                # --- STRICT THRESHOLD CHECK WITH SOFT AXIOM FIX ---
                    # --- SEVERITY-AWARE THRESHOLDING ---
                    if violated_axiom_severity == "absolute":
                            threshold = 0.97
                    elif violated_axiom_severity == "physical":
                            threshold = 0.75
                    else:
                            threshold = 1.1  # soft axioms can never trigger
                    # --- FIX: SOFT AXIOMS MUST NEVER TRIGGER VIOLATION ---
                    if violated_axiom_severity == "soft":
                        return {
                            "violation": False,
                            "is_hard": False,
                            "warning": "Informational (SOFT): Potential contextual mismatch detected",
                            "confidence": strongest_contradiction,
                            "debug": {
                                "severity": "soft",
                                "note": "SOFT axioms are non-penalizing",
                                "source": "nli_model",
                                "axiom": violated_axiom_text
                            }
                        }

                    is_hard_violation = (violated_axiom_severity == "absolute")
                    # --- FINAL NLI DECISION DEBUG ---
                    nli_debug["best_match"] = violated_axiom_text
                    nli_debug["decision"] = {
                        "severity": violated_axiom_severity,
                        "threshold_used": threshold,
                        "final_score": strongest_contradiction
                        }
                    
                    return {
                        'violation': True,
                        'is_hard': is_hard_violation,
                        'confidence': strongest_contradiction,
                        'warning': f"Scientific Conflict ({violated_axiom_severity.upper()}): Claim contradicts '{violated_axiom_text}'",
                        'axiom_text': violated_axiom_text,
                        'debug': {'source': "nli_level_1_bidirectional_margin","decision": {
                            "selected_axiom": violated_axiom_text,
                            "severity": violated_axiom_severity,
                            "threshold_used": threshold,
                            "final_score": strongest_contradiction 
                            },
                            "nli": nli_debug
                        } 
                    }
                
        except Exception as e:
                # print(f"FKC Inference Error: {e}")
                pass
                
        return default_result

FKC = FundamentalKnowledgeChecker()

# --- MODULE: REBUTTAL DETECTION v1.1 (Targeted Heuristic) ---
class RebuttalDetectionEngine:
    def __init__(self):
        self.rebuttal_cues = [
            "experts reject",
            "strongly reject",
            "warn against",
            "no scientific evidence",
            "debunked",
            "false claim",
            "misinformation",
            "has been disproven",
            "authorities say this is false",
            "must never be consumed",
            "dangerous",
            "toxic",
            "warn", 
            "reject"
            "this claim is false"
        ]
        
        # New indicators for ORIGINATING claims (the false ones)
        self.originating_indicators = [
            "claims that", "suggests that", "alleges", "posts circulating", 
            "shared online", "according to social media", "viral post", "circulating online"
        ]
        
        # New indicators for CORRECTIVE statements (the true ones, to ignore)
        self.corrective_indicators = [
            "experts say", "researchers note", "according to", "scientists state", 
            "studies show", "data indicates", "scientific agencies", "health authorities"
        ]

    def is_rebuttal_sentence(self, text: str) -> bool:
        """Checks if the claim ITSELF contains rebuttal language."""
        text_lower = text.lower()
        return any(cue in text_lower for cue in self.rebuttal_cues)
        
    def is_originating_claim(self, text: str, index: int) -> bool:
        """
        Determines if a claim is an originating assertion (target) rather than a correction.
        """
        text_lower = text.lower()
        
        # 1. Check for specific originating markers
        if any(ind in text_lower for ind in self.originating_indicators):
            return True
            
        # 2. Check for corrective markers (Exclusion)
        if any(ind in text_lower for ind in self.corrective_indicators):
            return False
            
        # 3. Fallback: First claim is often the target
        if index == 0:
            return True
            
        return False

    def scan(self, text: str, claims: List[str]) -> List[dict]:
        """
        Scans text for rebuttal cues near claims.
        Returns a list of debug objects for each detected rebuttal.
        """
        findings = []
        text_lower = text.lower()
        
        # Simple proximity check (intra-article scope)
        detected_cues = [cue for cue in self.rebuttal_cues if cue in text_lower]
        
        if detected_cues:
            for i, claim in enumerate(claims):
                # FIX v1.1: STRICT TARGETING RULE
                # Mark REFUTED only if:
                # 1. It is an ORIGINATING claim (viral post, allegation)
                # 2. It is NOT a corrective statement itself
                # 3. Rebuttal cues exist in context
                
                is_origin = self.is_originating_claim(claim, i)
                is_corrective = self.is_rebuttal_sentence(claim)
                
                if is_origin and not is_corrective:
                    findings.append({
                        "claim": claim,
                        "status": "REFUTED_BY_CONTEXT",
                        "trigger_phrases": detected_cues,
                        "confidence": "high",
                        "scope": "intra-article"
                    })
                
        return findings

REBUTTAL_DETECTOR = RebuttalDetectionEngine()

# --- MODULE: COMMON-SENSE PRIOR (CSP) ---
class CommonSensePrior:
    def __init__(self):
        self.exclusions = [
            "suggests", "may", "might", "could", "predicts", "will", "opinion",
            "study", "research", "scientists", "experts", "believe", "think",
            "reportedly", "allegedly", "politics", "election", "policy", "social",
            "should", "better", "worse", "good", "bad", "new", "recent"
        ]
        self.positive_priors = [
            "earth orbits", "water freezes", "humans need oxygen", 
            "triangle has three sides", "gravity pulls", "fire is hot",
            "sun rises", "ice is cold", "plants need water"
        ]
        self.negative_priors = [
            "cures cancer instantly", "survive without oxygen", 
            "flat earth", "magic spell", "gravity is fake", "moon is made of cheese",
            "perpetual motion", "thinking cures"
        ]

    def compute_score(self, text: str, has_conflict: bool) -> dict:
        """
        Returns { "CSP_score": float } based on common-sense heuristics.
        Range: [-1.0, 1.0]. Neutral: 0.0.
        """
        if has_conflict: 
            return {"CSP_score": 0.0}
        
        text_lower = text.lower()
        
        # Check Exclusions (ambiguity, future, opinion)
        if any(ex in text_lower for ex in self.exclusions):
            return {"CSP_score": 0.0}
            
        # Positive Match (Universal Truths)
        if any(p in text_lower for p in self.positive_priors):
            return {"CSP_score": 0.4}
            
        # Negative Match (Causal Nonsense)
        if any(n in text_lower for n in self.negative_priors):
            return {"CSP_score": -0.4}
            
        return {"CSP_score": 0.0}

CSP_MODULE = CommonSensePrior()

# --- MODULE: CARS v2 (Claim-Aligned Retrieval System) ---
# Purpose: Retrieve authoritative contextual information. 
# Does NOT verify or judge. Returns "Context Packets" only.
class CARS_v2_Engine:
    def __init__(self):
        # 1️⃣ CONTROLLED SOURCE POOL (FIXED)
        self.sources_db = {
            "T1": { "weight": 1.0, "sources": ["NASA", "WHO", "CDC", "ESA", "UIDAI", "RBI"] },
            "T2": { "weight": 0.8, "sources": ["Britannica", "PubMed"] },
            "T3": { "weight": 0.6, "sources": ["Reuters", "AP News", "BBC"] }
        }
        
        # General Fallback Contexts (Mock Data for Tier-0 Claims)
        self.fallback_contexts = {
            "general_science": [
                {
                    "source": "Britannica",
                    "tier": "T2",
                    "text": "Scientific consensus relies on reproducible evidence and peer-reviewed studies to establish facts about the natural world, including physics, biology, and chemistry."
                },
                {
                    "source": "ScienceDaily",
                    "tier": "T2",
                    "text": "The scientific method involves observation, hypothesis testing, and validation of natural phenomena to build reliable knowledge."
                }
            ],
            "health": [
                {
                    "source": "WHO",
                    "tier": "T1", 
                    "text": "Human health and survival depend on biological processes such as cellular respiration, which requires the continuous intake of oxygen."
                },
                {
                    "source": "PubMed",
                    "tier": "T2",
                    "text": "Oxygen is a critical element for human survival, essential for metabolic processes and cellular function."
                },
                {
                    "source": "Britannica",
                    "tier": "T2",
                    "text": "Respiration is the process by which organisms exchange gases, specifically taking in oxygen and expelling carbon dioxide, vital for life."
                }
            ],
            "physics": [
                {
                    "source": "NASA",
                    "tier": "T1",
                    "text": "The laws of physics, including gravity and thermodynamics, govern the behavior of matter and energy in the universe."
                },
                {
                    "source": "Britannica",
                    "tier": "T2",
                    "text": "Gravity is a fundamental interaction which causes mutual attraction between all things with mass or energy."
                }
            ]
        }

    def identify_domain(self, text: str) -> str:
        text_lower = text.lower()
        if any(x in text_lower for x in ["planet", "star", "orbit", "galaxy", "physics", "gravity", "earth", "space", "sun", "moon"]):
            return "astronomy"
        if any(x in text_lower for x in ["health", "virus", "vaccine", "medicine", "cancer", "disease", "smoking", "human", "body", "oxygen", "breathe"]):
            return "medicine"
        return "general"

    def retrieve_context(self, claim_text: str, domain: str) -> dict:
        # 3️⃣ CARS v2 INTERNAL BEHAVIOR (Mock Retrieval)
        # Mirrors human fact-checking: "Let me see what reliable sources say."
        # Returns context packets based on domain and keywords.
        
        contexts = []
        claim_lower = claim_text.lower()
        
        # --- MOCK: Authoritative Context Retrieval (Domain-Constrained) ---
        if domain == "medicine" and "smoking" in claim_lower and "cancer" in claim_lower:
            contexts.append({"source": "WHO", "tier": "T1", "text": "Tobacco smoking is the primary cause of lung cancer, responsible for over two-thirds of lung cancer deaths globally."})
            contexts.append({"source": "CDC", "tier": "T1", "text": "Cigarette smoking is the number one risk factor for lung cancer. In the United States, cigarette smoking is linked to about 80% to 90% of lung cancer deaths."})
            contexts.append({"source": "PubMed", "tier": "T2", "text": "Epidemiological studies have consistently demonstrated a strong causal association between tobacco smoking and lung carcinoma."})
            
        elif domain == "astronomy" and ("flat" in claim_lower or "round" in claim_lower or "sphere" in claim_lower or "revolves" in claim_lower):
            contexts.append({"source": "NASA", "tier": "T1", "text": "Earth is a planet. It is round like a ball. It orbits the Sun."})
            contexts.append({"source": "ESA", "tier": "T1", "text": "The Earth is an oblate spheroid, meaning it is a sphere slightly flattened at the poles."})
            contexts.append({"source": "Britannica", "tier": "T2", "text": "Earth, third planet from the Sun and the fifth largest planet in the solar system... its shape is nearly spherical."})

        # --- FIX 1 & 2: GENERAL FALLBACK RETRIEVAL PATH (MULTI-SOURCE) ---
        # If no specific context found, provide 2-3 authoritative fallback contexts.
        if not contexts:
            if domain == "medicine":
                contexts.extend(self.fallback_contexts["health"])
            elif domain == "astronomy":
                contexts.extend(self.fallback_contexts["physics"]) # Physics often covers astronomy basics
            else:
                contexts.extend(self.fallback_contexts["general_science"])
                
            # Additional keyword-based check for "Water" (Chemistry) - Treated as explicit fallback
            if "water" in claim_lower and "h2o" not in claim_lower: # avoid duplicate if covered
                contexts.append({"source": "Britannica", "tier": "T2", "text": "Water is a chemical substance composed of hydrogen and oxygen, vital for all known forms of life."})
                contexts.append({"source": "PubMed", "tier": "T2", "text": "Water (H2O) is the most abundant compound on Earth's surface and is essential for survival."})


        return {
            "claim_text": claim_text,
            "domain": domain,
            "retrieved": len(contexts) > 0,
            "contexts": contexts
        }

CARS = CARS_v2_Engine()

# --- MODULE: CONTROLLED RETRIEVAL & CONSENSUS SIGNALS (CRCS) ---
# Purpose: Aggregates NLI outputs (from CARS contexts). Computes Consensus Strength (ACS).
class CRCS_Engine:
    def __init__(self, cars_engine):
        self.cars = cars_engine

    def compute_acs(self, nli_results_list: List[dict]) -> tuple:
        # 4️⃣ CONSENSUS SIGNAL COMPUTATION (FIXED)
        # CS_i = source_weight * agreement * clarity
        # Agreement: Support (+1), Contradiction (-1), Neutral (0)
        # Clarity: 1.0 (assuming explicit contexts from CARS)
        
        if not nli_results_list:
            return 0.0, 0, []

        total_score = 0.0
        total_weight = 0.0
        sources_used = []

        for item in nli_results_list:
            # Determine agreement from NLI scores
            # Simple heuristic: if entailment > 0.7 -> +1, if contradiction > 0.7 -> -1, else 0
            agreement = 0.0
            if item['scores']['entailment'] > 0.5: agreement = 1.0
            elif item['scores']['contradiction'] > 0.5: agreement = -1.0
            
            weight = self.cars.sources_db[item['tier']]['weight']
            cs_i = weight * agreement * 1.0 # Clarity assumed 1.0 for manual context
            
            total_score += cs_i
            total_weight += weight
            sources_used.append(item['source'])

        if total_weight == 0: return 0.0, 0, []
        
        acs = total_score / total_weight
        return max(-1.0, min(1.0, acs)), len(nli_results_list), list(set(sources_used))

    def evaluate_consensus(self, claim_text: str, cars_output: dict, nli_provider,metaphor_flag: bool, primary_domain: str) -> dict:
        # 6️⃣ HOW CRCS USES NLI OUTPUTS
        # 1. Take CARS output (contexts)
        # 2. Run NLI for each context
        # 3. Compute ACS
        E = 0.0
        C = 0.0
        N = 1.0
        
        nli_results_list = []
        
        if cars_output.get("retrieved"):
            for ctx in cars_output['contexts']:
                # --- CONTEXTUAL BUILD: tier → context_strength (ONCE) ---
                tier_weight = {
                    "T1": 1.0,
                    "T2": 0.7,
                    "T3": 0.4
                }.get(ctx.get("tier", "T3"), 0.4)
                # retrieval_confidence assumed 1.0 for now
                ctx["context_strength"] = tier_weight * 1.0
                # FKC has run_nli_inference via check_claim internals, but not exposed directly.
                # We need access to the model. Using a temporary helper method on FKC if available, 
                # or we rely on the main loop. Since we can't edit FKC to add methods easily without full refactor,
                # we will simulate or assume access.
                # Actually, in the previous turn I added `run_nli_inference` to FKC. I will reuse that.
                
                # Check if FKC has the method, if not, skip (safety)
                if (
    hasattr(nli_provider, "model")
    and hasattr(nli_provider, "tokenizer")
    and nli_provider.model
    and nli_provider.tokenizer
):
                    inputs = nli_provider.tokenizer(ctx['text'], claim_text, return_tensors="pt", truncation=True)
                    with torch.no_grad():
                        logits = nli_provider.model(**inputs).logits
                    probs = logits.softmax(dim=1)
                    
                    label_map = nli_provider.model.config.id2label
                    scores = {}
                    for idx, label in label_map.items():
                        scores[label.lower()] = probs[0][idx].item()
                    
                    nli_results_list.append({
                        "source": ctx['source'],
                        "tier": ctx['tier'],
                        "scores": scores
                    })
        
        acs, ev_count, sources = self.compute_acs(nli_results_list)
        context_strength = self._compute_context_strength(cars_output)
        
        # --- DAY 6: Aggregate NLI proportions ---
        if nli_results_list:
            entailments = []
            contradictions = []
            neutrals = []
            
            for r in nli_results_list:
                scores = r.get("scores", {})
                entailments.append(scores.get("entailment", 0.0))
                contradictions.append(scores.get("contradiction", 0.0))
                neutrals.append(scores.get("neutral", 0.0))
            
            E = sum(entailments) / len(entailments)
            C = sum(contradictions) / len(contradictions)
            N = sum(neutrals) / len(neutrals)
            E, C, N = self._apply_uncertainty_decay(E, C, N, context_strength)
            # --- DAY 6: Truth / Fake / Unverified (Base) ---
        true_raw = E
        fake_raw = C
        unverified_raw = N
        if metaphor_flag:
            true_raw = 0.0
            fake_raw = 0.0
            unverified_raw = 1.0
            
        # --- DAY 6: Factual Prior (Distribution-Level) ---
        
        if (
            not metaphor_flag
            and primary_domain in {
                "General Knowledge",
                "Physical Reality",
                "Science",
                "Demographics",
                "Geography",
            }
            and fake_raw < 0.10
            and ev_count >= 1
        ):
            shift = min(0.30, unverified_raw)
            true_raw += shift
            unverified_raw -= shift
        # --- DAY 6: Normalize Truth / Fake / Unverified ---
        total = true_raw + fake_raw + unverified_raw
        if total > 0:
            true_pct = round(100 * true_raw / total)
            fake_pct = round(100 * fake_raw / total)
            unverified_pct = 100 - true_pct - fake_pct
        else:
            true_pct = fake_pct = 0
            unverified_pct = 100

        # 5️⃣ CONSENSUS INTERPRETATION & 9️⃣ EXPLAINABILITY OBJECT
        # --- PASS B: FACTUAL PRIOR (Day 5) ---
        consensus_label = "No consensus"
        if (
            metaphor_flag is False
            and primary_domain in {
                "General Knowledge",
                "Physical Reality",
                "Science",
                "Demographics",
                "Geography",
            }
            and ev_count > 0
            and acs >= -0.10
            and E>C
        ):
            consensus_label = "Weakly Supported (Factual)"
        if acs >= 0.75: consensus_label = "Strong TRUE consensus"
        elif acs >= 0.30: consensus_label = "Moderate TRUE"
        elif acs <= -0.75: consensus_label = "Strong FALSE consensus"
        elif acs <= -0.30: consensus_label = "Moderate FALSE"
        
        print("[CRCS RETURN HIT] Truth_Breakdown about to return")
        
        confidence = 50
        if ev_count >= 2:
            confidence += 10
        elif ev_count == 1:
            confidence += 5
        if primary_domain in {
            "General Knowledge",
            "Physical Reality",
            "Science",
            "Demographics",
            "Geography",
        }:
            confidence += 10
        if metaphor_flag:
            confidence -= 30
        entropy = 0.0
        E = max(0.0, min(1.0, E))
        C = max(0.0, min(1.0, C))
        N = max(0.0, min(1.0, N))
        for v in (E, C, N):
            if v > 0:
                entropy -= v * math.log(v)
        if entropy < 0.6:
            confidence += 10
        elif entropy > 0.9:
            confidence -= 10
        if unverified_pct >= 70:
            confidence -= 15
        confidence = max(5, min(95, confidence))

        return {
            "ACS": round(acs, 2),
            "Consensus": consensus_label,
            "Sources": sources,
            "E": round(E, 4),
            "C": round(C, 4),
            "N": round(N, 4),
            "Truth_Breakdown": {"true": true_pct, "fake": fake_pct, "unverified": unverified_pct},
            "Confidence": confidence,
            "Evidence_Count": ev_count,
            "Effective_Domain": primary_domain,
            "NLI_Raw": nli_results_list # For internal use if needed
        }
        # --- CRCS HELPERS (Day 7) ---

    # --- DAY 7: CONTEXT STRENGTH HELPERS ---
    
    def _compute_context_strength(self, cars_context):
        """
    Aggregates precomputed context_strength values.
    Does NOT interpret tier or source quality.
        """
        if not cars_context or not cars_context.get("contexts"):
            return 0.0
        
        strength = 0.0
        for ctx in cars_context["contexts"]:
            strength += ctx.get("context_strength", 0.0)
        return strength

    def _apply_uncertainty_decay(self, E, C, N, context_strength):
        """
        Reduces uncertainty as quality context accumulates.
        Never invents direction.
        """
        if context_strength <= 0 or N <= 0:
            return E, C, N

        decay_cap = 0.6
        decay = min(decay_cap, 0.25 * math.log1p(context_strength))

        N_new = N * (1 - decay)
        released = N - N_new

        directional = E + C
        if directional > 0:
            E += released * (E / directional)
            C += released * (C / directional)
            N = N_new

        return E, C, N
                
    

    def get_adjustment_signal(self, consensus_data: dict, current_label: str, base_confidence: float) -> dict:
        # 6️⃣ OVER-CAUTIOUSNESS & 7️⃣ CONTRADICTION Logic
        acs = consensus_data["ACS"]
        
        signal = {
            "OverCautious": False,
            "ContradictionPenalty": False,
            "Confidence_Adjustment": 0.0,
            "Notes": ""
        }

        # OCC Detection
        is_unverified = current_label in ["Unsure", "Unverified"]
        if (is_unverified and 
            base_confidence < 0.65 and 
            acs >= 0.75 and 
            consensus_data["Evidence_Count"] >= 3):
            
            signal["OverCautious"] = True
            signal["Confidence_Adjustment"] = 0.25
            signal["Notes"] = "Over-cautious neutrality detected. Applying consensus boost."
            return signal

        # Contradiction Penalty
        if acs <= -0.75:
            signal["ContradictionPenalty"] = True
            signal["Confidence_Adjustment"] = -0.30
            signal["Notes"] = "Strong authoritative contradiction detected."
            return signal
            
        # General Consensus Boost (Moderate)
        if acs >= 0.75:
            signal["Confidence_Adjustment"] = 0.15
            signal["Notes"] = "High authoritative agreement detected."
            
        return signal

CRCS = CRCS_Engine(CARS)
# --- MODULE: CLAIM CONTEXT LOCATOR (CCL) [OBSERVE MODE ONLY] ---
# Purpose: Decide *where* a claim should be checked (not fetch, not judge)
# --- CCL DOMAIN NORMALIZATION MAP (Observe-only) ---
# DOMAIN_NORMALIZATION_MAP = {
#     # Physical world
#     "chemistry": "Physical Reality",
#     "physics": "Physical Reality",
#     "astronomy_physics": "Physical Reality",
#     "materials": "Physical Reality",

#     # Living systems
#     "biology_medicine": "Biological / Medical Reality",
#     "biology": "Biological / Medical Reality",
#     "medicine": "Biological / Medical Reality",
#     "health": "Biological / Medical Reality",

#     # Time & history
#     "history_society": "Temporal / Historical Reality",
#     "history": "Temporal / Historical Reality",
#     "temporal": "Temporal / Historical Reality",

#     # Society & institutions
#     "government": "Social / Institutional Reality",
#     "policy": "Social / Institutional Reality",
#     "law": "Social / Institutional Reality",
    
#     # Economy & systems
#     "economics": "Socioeconomic Systems",
#     "economy": "Socioeconomic Systems",
#     "finance": "Socioeconomic Systems",
#     "market": "Socioeconomic Systems",
#     "inflation": "Socioeconomic Systems",


#     # Fallback
#     "general_knowledge": "Opinion / Interpretation"
# }
# def minimal_ccl_decision(claim_text: str) -> dict:
#         """
#         Minimal CCL (Observe-only)
#         Determines the most likely knowledge domain(s) a claim belongs to.
#         No scoring impact. Debug-only output.
#         """
#         text = claim_text.lower()
#         domains = []
#         # Astronomy / Physics
#         if any(k in text for k in [
#          "sun", "moon", "earth", "planet", "orbit", "gravity", "space", "universe"
#         ]):
#          domains.append("astronomy_physics")
#         # Biology / Medicine
#         if any(k in text for k in [
#         "human", "body", "oxygen", "breathe", "disease", "virus", "health", "medicine"
#         ]):
#             domains.append("biology_medicine")
#         # Chemistry / Materials
#         if any(k in text for k in [
#             "water", "fire", "chemical", "compound", "element", "reaction"
#         ]):
#             domains.append("chemistry")
#         # History / Society
#         if any(k in text for k in [
#             "history", "ancient", "medieval", "government", "policy", "law"
#         ]):
#             domains.append("history_society")
#         if any(k in text for k in [
#             "economy", "economic", "inflation", "market", "finance"
#         ]):
#             domains.append("economics")
#         if not domains:
#             domains.append("general_knowledge")
        
#         print("CCL raw domains:", domains)
            
#         raw_domains = domains if domains else ["general_knowledge"]
#         primary_raw = raw_domains[0]
#         secondary_raw = raw_domains[1:] if len(raw_domains) > 1 else []
#         primary_domain = DOMAIN_NORMALIZATION_MAP.get(
#             primary_raw,
#             "Opinion / Interpretation"
#         )
        
#         secondary_domains = [
#             DOMAIN_NORMALIZATION_MAP.get(d, "Opinion / Interpretation")
#             for d in secondary_raw
#         ]
        
#         # --- Option B: Metaphor Suspicion (observe-only) ---
#         abstract_entities = [
#             "economy", "market", "society", "democracy",
#             "system", "nation", "culture", "government"
#         ]
#         physical_actions = [
#             "bleeding", "dying", "dead", "eating",
#             "choking", "burning", "freezing", "collapsing"
#         ]
#         possible_metaphor = (
#             any(a in text for a in abstract_entities)
#             and any(p in text for p in physical_actions)
#         )
        
#         return {
#             "primary_domain": primary_domain,
#             "secondary_domains": secondary_domains,
#             "confidence": 0.8 if domains else 0.5,
#             "reason": "Matched minimal CCL keyword rules",
#             "notes": {
#                 "ambiguous": len(domains) > 1,
#                 "possible_metaphor": possible_metaphor,
#                 "mixed_claim": len(domains) > 1
#             }
#         }

    # 1. NN Model Loading
if os.path.exists(LOCAL_NN_MODEL_PATH):
    LOCAL_NN_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    LOCAL_NN_MODEL = AutoModelForSequenceClassification.from_pretrained(
        LOCAL_NN_MODEL_PATH,
        num_labels=NUM_OUTPUT_HEADS
    )
    LOCAL_NN_MODEL.eval()
    NN_MODEL_AVAILABLE = True
else:
    NN_MODEL_AVAILABLE = False



    # 1.5 Load FKC
    FKC.load_model()

    # 2. VLM/OCR Model Loading (PaddleOCR or TrOCR)
    
    # --- PADDLEOCR RE-ENABLED ---
    if PADDLEOCR_AVAILABLE:
        # Initialize PaddleOCR with optimized settings for documents/layout complexity
        # Removed unsupported arguments (rec_algorithm, rec_batch_num)
        VLM_MODEL = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            det_db_thresh=0.25,
            det_db_box_thresh=0.4,
            det_db_unclip_ratio=2.0
        )
        print("Local PaddleOCR VLM Model loaded with optimized document settings.")
    else:
        # Fallback to TrOCR if PaddleOCR is not available (due to missing dependency)
        TROCR_MODEL_NAME = "microsoft/trocr-base-printed"
        TROCR_PROCESSOR = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
        VLM_MODEL = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)
        print(f"Local TrOCR VLM Model loaded ({TROCR_MODEL_NAME}).")
# ---------------------------------------------


# class TextRequest(BaseModel):
#     text: str
#     debug: Optional[bool] = False # Add optional debug flag

# class UrlRequest(BaseModel):
#     url: str
#     debug: Optional[bool] = False # Add optional debug flag
    
TRUSTED_DOMAINS = {
    'reuters.com', 'apnews.com', 'bbc.com', 'cnn.com', 'nytimes.com', 
    'nasa.gov', 'who.int', 'cdc.gov', 'gov.in', 'pib.gov.in', 'washingtonpost.com', 
    'time.com', 'hindu.com', 'indianexpress.com', 'ndtv.com', 'sciencemag.org'
}

def get_domain_from_url(url: str):
    try:
        netloc = urlparse(url).netloc
        return netloc.replace('www.', '')
    except Exception:
        return None

# --- LINE IMPORTANCE SCORING LOGIC (Used for filtering text output) ---

ACTION_VERBS = [
    "announced", "confirmed", "reported", "approved",
    "said", "returned", "launched", "banned"
]

def score_line(line: str) -> int:
    """Scores a line based on typical article-like characteristics vs. junk."""
    score = 0
    words = line.split()

    # Positive Scoring
    if len(words) >= 6:
        score += 2
    if len(words) >= 10:
        score += 2

    # Score highly for claims containing action verbs
    if any(v in line.lower() for v in ACTION_VERBS):
        score += 3

    # Score for specificity (digits imply numbers/time/money)
    if any(char.isdigit() for char in line):
        score += 1

    # Negative Scoring (Junk)
    if line.isupper() and len(words) > 1 and len(words) < 15:
        score -= 3

    if re.search(r"http|www|\.\w{2,4}", line.lower()):
        score -= 3

    return score


# --- CRITICAL FIX 1, 2, 3: MANUAL FEATURE DEFINITIONS ---

# FIX 1: Authority Score Definition (Expanded)
AUTHORITY_SOURCES = [
    "nasa",
    "who",
    "world health organization", # Added full name
    "united nations",
    "cdc",
    "fda",
    "esa",
    "ipcc",
    "government",
    "official press release"
]
def authority_score(text: str) -> float:
    text_l = text.lower()
    hits = sum(1 for src in AUTHORITY_SOURCES if src in text_l)
    return min(hits * 30, 100) # Max 100

# FIX 2: Manipulation Score Definition (Expanded)
MANIPULATIVE_TERMS = [
    "you must",
    "panic",
    "secret",
    "hidden",
    "they don't want you to know",
    "mysterious forces",
    "immediately",
    "guaranteed"
]
def manipulation_score(text: str) -> float:
    text_l = text.lower()
    hits = sum(1 for term in MANIPULATIVE_TERMS if term in text_l)
    return min(hits * 25, 100) # Max 100

# FIX 3: Reporting Style Bonus Definition
REPORTING_PHRASES = [
    "confirmed",
    "according to",
    "press release",
    "reported by",
    "announced"
]
def reporting_style_bonus(text: str) -> float:
    return min(sum(1 for p in REPORTING_PHRASES if p in text.lower()) * 15, 60) # Max 60

# --- REMOVED RAW KNOWN_FALSE_PATTERNS (Moved to KSA Class) ---

# FIX 5: Lexical Uncertainty
UNCERTAINTY_TERMS = [
    "may", "might", "could", "potentially",
    "suggests", "preliminary", "early-stage",
    "further testing required", "not yet confirmed",
    "unknown amount"
]

# --- END MANUAL FEATURE DEFINITIONS ---

# --- NEW: CONSENSUS INDICATOR CHECKER ---
CONSENSUS_INDICATORS = [
    "decades of research",
    "epidemiological research",
    "well-established",
    "strong consensus",
    "widely accepted",
    "meta-analysis",
    "systematic review",
    "scientific consensus"
]

def consensus_score(text: str) -> float:
    """
    Detects linguistic markers of mature scientific consensus.
    Returns a score 0-100 based on presence of indicators.
    """
    text_lower = text.lower()
    hits = sum(1 for ind in CONSENSUS_INDICATORS if ind in text_lower)
    # 2 hits is enough for max score (very specific phrases)
    return min(hits * 50, 100)


# --- LOCAL IMAGE PROCESSING FUNCTION (RE-ENABLED FOR DOCUMENT/IMAGE ANALYSIS) ---
# Modified return type to Dict[str, Any]
def extract_text_from_image_local(contents: bytes, file_type: str) -> Dict[str, Any]:
    """
    Handles OCR using the initialized model (PaddleOCR or TrOCR).
    Focuses on document/general image analysis with layout-aware sorting.
    Returns: {'text': str, 'document_type': str, 'text_extraction_method': str}
    """
    
    # NOTE: Assuming TROCR_PROCESSOR and VLM_MODEL are defined in the global try/except
    global TROCR_PROCESSOR, VLM_MODEL 

    # --- CRITICAL DEBUG FIX: Check if VLM_MODEL was successfully instantiated. ---
    if VLM_MODEL is None:
        print(f"DEBUG: VLM Check failed. VLM_MODEL is None. Cannot proceed with OCR.")
        return {
            'text': "Local VLM unavailable. Cannot extract claim from document.",
            'document_type': "unknown",
            'text_extraction_method': "failed"
        }
    # ----------------------------

    raw_extracted_text = ""
    document_type = "image"
    text_extraction_method = "ocr"
    
    try:
        is_pdf = file_type == 'application/pdf'
        
        # --- CRITICAL FIX 1 & 2: PDF Text Extraction Priority ---
        if is_pdf:
            document_type = "digital_pdf"
            if not PDF_SUPPORT_AVAILABLE:
                return {
                    'text': "PDF Handling Error: Document OCR requires a PDF-to-Image library (like PyMuPDF) to be installed in the backend environment. Please upload an image file or install the dependency.",
                    'document_type': "digital_pdf",
                    'text_extraction_method': "failed"
                }
            
            # 1a. Attempt raw text extraction (fastest, most accurate for digital PDFs)
            doc = fitz.open(stream=contents, filetype="pdf")
            raw_pdf_text = "".join([page.get_text() for page in doc])
            doc.close()
            
            # 1b. Check if extracted text is substantial (e.g., > 100 characters)
            if len(raw_pdf_text) > 100:
                print(f"DEBUG: Using direct PDF text extraction (Length: {len(raw_pdf_text)}). Skipping OCR.")
                text_extraction_method = "direct"
                return {
                    'text': raw_pdf_text, 
                    'document_type': document_type,
                    'text_extraction_method': text_extraction_method
                }
            
            # 1c. FALLBACK to high-resolution OCR (Scanned PDF)
            print("DEBUG: Raw PDF text too short. Falling back to high-DPI image rendering for OCR.")
            document_type = "scanned_pdf"
            text_extraction_method = "ocr_scan"
            
            doc = fitz.open(stream=contents, filetype="pdf")
            page = doc.load_page(0)
            
            # CRITICAL FIX #1: Render PDF at 300 DPI for high quality OCR
            zoom = 300 / 72
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            
            # CRITICAL FIX #3 (Debug): Log PDF size
            image_data = pix.tobytes("ppm")
            image = Image.open(BytesIO(image_data))
            print(f"PDF Image Size: {image.width} x {image.height}") # Log size
            
            doc.close()
        else:
            # Handle standard image types (always OCR)
            image = Image.open(BytesIO(contents)).convert("RGB")
            document_type = "image"
            text_extraction_method = "ocr_scan"
            
        # CRITICAL FIX #2: Wrap resizing so it ONLY applies to non-PDFs (which were NOT resized above)
        if not is_pdf:
            # 1. Resize small text (Upscale by factor 2 for better OCR) 
            image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
        
        
        if PADDLEOCR_AVAILABLE:
            # --- PADDLEOCR PATH: Layout-Aware Extraction and Sorting ---
            
            img_path = BytesIO()
            image.save(img_path, format="PNG")
            
            result = VLM_MODEL.ocr(img_path.getvalue())
            
            # --- DEBUG CHECK START ---
            # REMOVED: Raw debug block as requested
            # --- DEBUG CHECK END ---

            extracted_blocks = []
            if result and result[0]: # Check if any results were found (result[0] is the list of blocks)
                for line_block in result[0]:
                    # line_block structure: [ [[box_coords]], [text, confidence] ]
                    box = line_block[0] # Bounding box coordinates
                    text = line_block[1][0].strip() # Recognized text string, stripped
                    
                    # CRITICAL FIX: Skip empty lines explicitly
                    if not text:
                        continue
                        
                    # Approximate top-left corner for sorting
                    min_x = min(p[0] for p in box)
                    min_y = min(p[1] for p in box)
                    
                    extracted_blocks.append({'text': text, 'x': min_x, 'y': min_y})
            
            # CRITICAL FIX: Sort blocks by vertical position (Y), then horizontal position (X) 
            # to reconstruct the correct reading order for multi-column documents.
            extracted_blocks.sort(key=lambda b: (b['y'], b['x']))
            
            # Recombine the text in reading order
            raw_extracted_text = " ".join([b['text'] for b in extracted_blocks])
            
            
        else:
            # --- TROCR FALLBACK PATH: Standard Text Extraction ---
            text_extraction_method = "trocr_fallback"
            
            # 2. Convert to high-contrast grayscale (Luminance)
            grayscale_image = image.convert("L")
            
            # 3. Increase contrast (factor 2.0)
            enhancer = ImageEnhance.Contrast(grayscale_image)
            grayscale_image = enhancer.enhance(2.0)
            
            # 4. CRITICAL FIX: Convert back to 3-channel RGB for TrOCRProcessor input
            image_for_processor = grayscale_image.convert("RGB")
            
            # TrOCR inference 
            pixel_values = TROCR_PROCESSOR(images=image_for_processor, return_tensors="pt").pixel_values
            generated_ids = VLM_MODEL.generate(pixel_values, max_length=512) 
            raw_extracted_text = TROCR_PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        
        # --- Stage 3: Semantic Segmentation and Filtering (Common to both models) ---
        # The raw_extracted_text is now in the correct reading order if PaddleOCR was used.
        raw_lines_segmented = re.split(r'(?<=[.!?])\s*', raw_extracted_text.strip())
        
        position_filtered_lines = [line.strip() for line in raw_lines_segmented if line.strip()]
        
        important_lines = []
        discarded_semantic = []
        
        for line in position_filtered_lines:
            score = score_line(line)

            if score >= 3: # Keep lines scoring 3 or higher
                important_lines.append(line)
            else:
                discarded_semantic.append(line)

        # Join the important lines back together (the full context block)
        filtered_text = " ".join(important_lines)
        
        # --- CRITICAL FIX: Minimum Semantic Threshold ---
        word_count = len(filtered_text.split())
        sentence_count = filtered_text.count(".") + filtered_text.count("!") + filtered_text.count("?")
        
        final_output = ""
        
        if word_count < 10 or sentence_count == 0:
            # Fallback if text is too short or incoherent
            final_output = "Document processed, but no substantial, coherently structured text was extracted or deemed relevant for analysis."
        else:
            final_output = filtered_text
            final_output = final_output[0].upper() + final_output[1:]


        # --- Debug Logging (UPDATED) ---
        print("\n--- OCR Filter Debug Info ---")
        print(f"OCR Model: {'PaddleOCR (Layout-Sorted)' if PADDLEOCR_AVAILABLE else 'TrOCR (Fallback)'}")
        # UPDATED: Log up to 500 characters of raw OCR text
        print(f"RAW OCR Text (Sorted): {raw_extracted_text[:500]}{'...' if len(raw_extracted_text) > 500 else ''}")
        print(f"Total Lines Detected (Pre-Filter): {len(raw_lines_segmented)}")
        # NOTE: Total lines detected now accurately reflects the number of non-empty segments
        print(f"Lines Kept (Scored >= 3): {len(important_lines)}")
        
        print(f"FINAL INPUT Word Count: {word_count}")
        print(f"FINAL INPUT Text: {final_output[:80]}...")
        print("-----------------------------\n")
        
        return {
            'text': final_output, 
            'document_type': document_type,
            'text_extraction_method': text_extraction_method
        }

    except Exception as e:
        print(f"OCR processing failed due to exception: {e}") 
        # If the failure happened outside the PDF check (e.g. image processing failed)
        return {
            'text': f"OCR processing failed: {e.__class__.__name__} error. If uploading an image, ensure the file is not corrupted.",
            'document_type': document_type,
            'text_extraction_method': "failed"
        }


# --- Neural Network Inference (Signal-Only) ---
def run_nn_prediction(text: str) -> List[float]:
    """
    Runs local Multi-Head NN model.
    """
    print("NN_MODEL_AVAILABLE =", NN_MODEL_AVAILABLE, "| module =", __name__)

    if not NN_MODEL_AVAILABLE:
        return [0.5, 0.5, 0.5, 0.5] 
    
    # CRITICAL FIX: Handle empty/whitespace input gracefully
    if not text or not text.strip():
        print("NN Prediction skipped: Input text is empty.")
        return [0.5, 0.5, 0.5, 0.5]

    try:
        inputs = LOCAL_NN_TOKENIZER(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        # Ensure we run inference without computing gradients
        with torch.no_grad():
            outputs = LOCAL_NN_MODEL(**inputs)
        
        probabilities = torch.sigmoid(outputs.logits).cpu().numpy().flatten().tolist()
        
        return probabilities[:NUM_OUTPUT_HEADS] 

    except Exception as e:
        print(f"NN Model Prediction Error: {e}")
        return [0.5] * NUM_OUTPUT_HEADS 
# --- Neural Network Training Forward (TRAINING ONLY) ---
def run_nn_training_forward(text: str):
    global NN_MODEL_AVAILABLE

    # ✅ SAFETY: ensure model is loaded for training
    if not NN_MODEL_AVAILABLE:
        load_local_nn_model()

    if not NN_MODEL_AVAILABLE:
        raise RuntimeError(
            "NN model not available for training forward pass"
        )

    if not text or not text.strip():
        raise ValueError("Empty input text for NN training forward")

    inputs = LOCAL_NN_TOKENIZER(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    outputs = LOCAL_NN_MODEL(
        **inputs,
        output_hidden_states=True
    )

    logits = outputs.logits.squeeze(0)
    embeddings = outputs.hidden_states[-1].mean(dim=1).squeeze(0)

    return {
        "logits": logits,
        "embeddings": embeddings
    }

# --- HELPER FUNCTION: Confidence Label Mapping ---
def get_confidence_label(score: int) -> str:
    """Maps 0-100 score to a descriptive credibility label."""
    # Updated thresholds for clearer classification
    if score >= 85: 
        return "Highly Credible"
    if score >= 70: 
        return "Likely Credible"
    if score >= 50: 
        return "Moderately Credible"
    if score >= 30:
        return "Unverified"
    return "Highly Unlikely"

# =========================
# CDVM Domain Mapping
# =========================
def map_ccl_to_cdvm_domain(ccl_domain: str | None) -> str | None:
    if not ccl_domain:
        return None

    d = ccl_domain.lower()

    if "economic" in d:
        return "economics"
    if "politic" in d:
        return "politics"
    if "societ" in d:
        return "society"
    if "law" in d:
        return "law"
    if "environment" in d:
        return "environment"

    return None
    
# --- CORE LOGIC: SIGNAL AGGREGATION AND FINAL JUDGMENT (NN + HCDM PURE) ---
def aggregate_signals(
    claims_data: dict,
    nn_signals_01: List[float],
    source_type: str,
    url: str,
    crcs_signal: dict = None,
    cars_context: dict = None,
    debug: bool = False
) -> tuple:
    factors = []
    # --- SAFE DEFAULTS FOR OPTIONAL DEBUG SIGNALS ---
    rebuttal_findings = None
    
    # 1. Extract Neural Signals
    nn_plau, nn_evid, nn_bias, nn_uncert = nn_signals_01 
    
    # --- PART 1: DYNAMIC BASELINE ---
    # Instead of a static 60, we calculate a baseline from the raw neural quality.
    # Quality = Plausibility + Evidence + Certainty (1 - Uncertainty)
    # Range: 45 (Weak) to 85 (Strong)
    
    certainty_score = 1.0 - nn_uncert
    
    # Weighted Signal Balance (0.0 to 1.0)
    # Plausibility (35%) + Evidence (45%) + Certainty (20%)
    signal_quality = (nn_plau * 0.35) + (nn_evid * 0.45) + (certainty_score * 0.20)
    
    # FIX 8: EXPAND DYNAMIC BASELINE RANGE
    # Map to 45-85 range (width 40 instead of 30) to better reward strong signals
    base_score = 45.0 + (signal_quality * 40.0)
    
    # --- NEW: High Confidence Bonus ---
    # Rewards obvious truths (High Plausibility AND High Evidence)
    if nn_plau > 0.85 and nn_evid > 0.85:
        high_conf_bonus = 15.0
        base_score += high_conf_bonus
        factors.append(f"✅ Dynamic Baseline: {int(base_score)} (Strong Neural Signals + High Confidence Bonus).")
    else:
        factors.append(f"✅ Dynamic Baseline: {int(base_score)} (Derived from Neural Signals).")


    # --- Step 2: Claim-Level Analysis & Metrics ---
    # We need to calculate ratios for contradictions and support
    supported_count = 0
    contradicted_count = 0
    
    # Maps for claim scoring (unchanged logic, just used for counts)
    PLAU_MAP = {"high": 100, "medium": 50, "low": 10}
    EVID_MAP = {"high": 100, "medium": 60, "low": 30} 
    
    # Track FKC Violations
    fkc_warnings = []
    hard_violation_count = 0
    any_fkc_violation = False # Track if any FKC flag exists (physical OR absolute)

    total_claims = len(claims_data.get('claims', []))
    
    # FIX 1 (Step 2a): DISABLE FKC EFFECTS WHEN NO VALID CLAIMS EXIST
    # Check if there is at least one extracted claim that is NOT a placeholder
    valid_claims_exist = any(
        "No substantial claim found" not in c.get('claim_text', "") 
        for c in claims_data.get('claims', [])
    )

    for claim in claims_data.get('claims', []):
        # 1. Collect FKC Data (ONLY IF VALID CLAIMS EXIST)
        fkc_result = claim.get('fkc_result') or {}
        # --- FKC PENALTY (SEVERITY-AWARE, NON-DECISIVE) ---
        if fkc_result.get("violation") and valid_claims_exist:
            any_fkc_violation = True
            debug_info = fkc_result.get("debug", {})
            severity = debug_info.get("severity", "physical")
            fkc_confidence = debug_info.get("confidence", 1.0)
            # Severity-aware penalty
            if severity == "absolute":
                penalty = 40.0
            elif severity == "physical":
                penalty = 20.0
            else:
                penalty = 0.0
            # Confidence-scaled penalty
            applied_penalty = penalty * float(fkc_confidence)
            if applied_penalty > 0:
                base_score -= applied_penalty
                factors.append(f"⚠️ Fundamental Knowledge Violation ({severity.upper()}): "f"-{int(applied_penalty)}")

        # Only process violations if extraction was successful/valid
        if valid_claims_exist and fkc_result.get('violation'):
            any_fkc_violation = True # Flag presence of ANY violation
            fkc_warnings.append(fkc_result.get('warning'))
            
            # FIX 2 (Current Request): ABSOLUTE VIOLATIONS MUST PROPAGATE
            # Check for HARD violation (Severity=Absolute).
            # This relies on the FKC result logic, which sets is_hard=True for absolute prechecks OR high-conf NLI absolute.
            if fkc_result.get('is_hard'):
                hard_violation_count += 1

        # 2. Determine Claim Verdict (Supported/Contradicted/Speculative)
        signals = claim.get('signals', {})
        plausibility = signals.get('plausibility', 'medium').lower()
        evidence = signals.get('evidence_specificity', 'medium').lower()

        claim_score = (PLAU_MAP.get(plausibility, 50) * 0.20) + \
                    (EVID_MAP.get(evidence, 60) * 0.45) + 50 
        
        if claim_score >= 80 and EVID_MAP.get(evidence, 60) > 60:
            supported_count += 1
            claim['claim_verdict'] = 'Supported'
        elif claim_score <= 30 and PLAU_MAP.get(plausibility, 50) < 30:
            contradicted_count += 1
            claim['claim_verdict'] = 'Contradicted'
        else:
            claim['claim_verdict'] = 'Speculative'

    # Calculate Ratios
    if total_claims > 0:
        supported_ratio = supported_count / total_claims
        contradicted_ratio = contradicted_count / total_claims
    else:
        supported_ratio = 0
        contradicted_ratio = 0


    # --- Step 3: Calculate Content Features ---
    # Using the first claim text for feature extraction (Authority, Manipulation)
    text_content = claims_data.get('claims', [{}])[0].get('claim_text', "")
    
    avg_authority_score = authority_score(text_content)       # 0-100
    avg_manipulation_score = manipulation_score(text_content) # 0-100
    avg_reporting_bonus = reporting_style_bonus(text_content) # 0-60
    
    # NEW: Calculate Consensus Score (0-100)
    avg_consensus_score = consensus_score(text_content)


    # --- PART 4: SCALED POSITIVE BOOSTS ---
    # Authority: Scaled boost (Max 25)
    # e.g., Score 60 -> +24, Score 30 -> +12
    auth_boost = min(avg_authority_score * 0.4, 25.0)
    if auth_boost > 3:
        base_score += auth_boost
        factors.append(f"✅ Authority Boost: +{int(auth_boost)} (Scaled by source strength).")

    # Reporting Style: Scaled boost (Max 15)
    report_boost = min(avg_reporting_bonus * 0.4, 15.0)
    if report_boost > 3:
        base_score += report_boost
        factors.append(f"✅ Confirmation Language: +{int(report_boost)} (Scaled by journalistic phrasing).")

    # Claim Consistency: Scaled by ratio (Max 15)
    if supported_ratio > 0:
        consistency_boost = supported_ratio * 15.0
        base_score += consistency_boost
        factors.append(f"✅ Claim Consistency: +{int(consistency_boost)} (Supported claims ratio).")
    
    # NEW: CONSENSUS CONFIDENCE BOOST (Bounded & Guarded)
    # Only applies if NO FKC violation, Low Manipulation, Low Bias, AND High NN Support
    if (avg_consensus_score > 0 and 
        not any_fkc_violation and 
        avg_manipulation_score < 30 and 
        (nn_bias * 100) < 30 and
        nn_plau > 0.6): # Must be reasonably plausible
        
        # Boost formula: Base 5 + portion of consensus score (Max 15)
        consensus_bonus = 5.0 + (avg_consensus_score * 0.1)
        base_score += consensus_bonus
        factors.append(f"✅ Scientific Consensus: +{int(consensus_bonus)} (Mature evidence indicators detected).")


    # --- PART 2: SCALED PENALTIES ---
    # --- NLI LEVEL-2: IMPLICIT CONTRADICTION PENALTY (SOFT) ---
        implicit_penalty_total = 0.0
        for claim in claims_data.get('claims', []):
            implicit_result = claim.get("implicit_result", {})
        
    
    # 1. Manipulation Penalty (Scaled)
    # Score 100 -> -45, Score 50 -> -22.5
    manip_penalty = min(avg_manipulation_score * 0.5, 45.0)
    if manip_penalty > 5:
        base_score -= manip_penalty
        factors.append(f"🚨 Manipulation Penalty: -{int(manip_penalty)} (Proportional to alarmist language).")

    # 2. Bias Penalty (Scaled, Threshold > 50)
    nn_bias_score = nn_bias * 100
    if nn_bias_score > 50:
        # Scale: Bias 50 -> 0 penalty. Bias 100 -> 25 penalty.
        bias_penalty = (nn_bias_score - 50) * 0.5
        base_score -= bias_penalty
        factors.append(f"⚖️ Bias Penalty: -{int(bias_penalty)} (Proportional to bias level).")

    # 3. Contradiction Penalty (Scaled by ratio)
    # 100% contradicted -> -50 penalty
    if contradicted_ratio > 0:
        contradict_penalty = contradicted_ratio * 50.0
        base_score -= contradict_penalty
        factors.append(f"🚩 Internal Contradiction: -{int(contradict_penalty)} (Ratio of contradicted claims).")


    # --- PART 3: FKC / NLI INTEGRATION (GUARDRAIL ONLY) ---
    # FIX 4 (Ordering): Apply Absolute Penalty BEFORE NN penalties
    # FIX 7: HCDM ABSOLUTE INVARIANT PENALTY (BOUNDED)
    # Formula: 25 + (10 * count), Cap at 55
    if hard_violation_count > 0:
        inv_penalty = min(25.0 + (10.0 * hard_violation_count), 55.0)
        base_score -= inv_penalty
        factors.append(f"🚨 Invariant Violation: -{int(inv_penalty)} ({hard_violation_count} absolute contradiction{'s' if hard_violation_count > 1 else ''}).")

    # Append FKC Warnings (Soft or Hard) as informative factors (No score impact for soft)
    if fkc_warnings:
        unique_warnings = list(set(fkc_warnings))
        for warning in unique_warnings:
            factors.append(f"⚠️ {warning}")


    # FIX 9 & FIX 3: CLEAN CLAIM BONUS (CONTROLLED & GUARDED)
    # Applied LAST.
    # Conditions: No FKC warnings (any kind), Low Manipulation, Low Bias
    if not any_fkc_violation and avg_manipulation_score < 30 and (nn_bias * 100) < 30:
        base_score += 8
        factors.append("✅ Clean Claim Bonus: +8 (No logical contradictions or manipulation detected).")


    # --- CRCS INTEGRATION ---
    if crcs_signal:
        # 6️⃣ OVER-CAUTIOUSNESS DETECTION
        if crcs_signal.get("OverCautious"):
            adj = crcs_signal.get("Confidence_Adjustment", 0.0) * 100
            if adj > 0:
                base_score += adj
                factors.append(f"✅ CRCS Boost: +{int(adj)} (Correcting over-cautious neutrality).")
        
        # 7️⃣ CONTRADICTION PENALTY
        elif crcs_signal.get("ContradictionPenalty"):
            adj = crcs_signal.get("Confidence_Adjustment", 0.0) * 100
            if adj < 0:
                base_score += adj # Adding negative
                factors.append(f"🚨 CRCS Penalty: {int(adj)} (Strong authoritative contradiction).")
        
        # General Consensus Boost
        elif crcs_signal.get("Confidence_Adjustment", 0.0) > 0:
            adj = crcs_signal.get("Confidence_Adjustment", 0.0) * 100
            base_score += adj
            factors.append(f"✅ CRCS Consensus: +{int(adj)} ({crcs_signal.get('Consensus')}).")

    # --- NEW: COMMON-SENSE PRIOR (CSP) ADVISORY ---
    # Trigger: No FKC violation, No Internal Contradiction, No External Contradiction
    has_conflict = any_fkc_violation or (contradicted_ratio > 0) or (crcs_signal and crcs_signal.get("ContradictionPenalty"))
    
    csp_output = CSP_MODULE.compute_score(text_content, has_conflict)
    csp_val = csp_output["CSP_score"]
    
    if csp_val != 0.0:
        # Weight: 25 points * score (Result: +/- 10 points max)
        csp_adj = csp_val * 25.0
        base_score += csp_adj
        factors.append(f"{'✅' if csp_adj > 0 else '⚠️'} Common-Sense Prior: {'+' if csp_adj > 0 else ''}{int(csp_adj)} (Heuristic certainty adjustment).")

    # --- PART 5: UNCERTAINTY HANDLING (SOFT CENTERING) ---
    # Keep uncertainty as a soft centering mechanism, not a punishment.
    # If uncertainty is high (>0.5), we confine the score to a mid-range band (35-65).
    # This prevents "Uncertain" claims from being marked as "Fake" (Verified False) or "Real" (Verified True).
    
    if nn_uncert > 0.5:
        current_score = base_score
        
        # If score is too high (Real), pull down to Unverified
        if current_score > 65:
            base_score = 65
            factors.append("⚠️ High Uncertainty: Score capped at Unverified level.")
            
        # If score is too low (Fake), pull up to Unverified (Speculative is better than Proven Fake)
        elif current_score < 35:
            base_score = 35
            factors.append("⚠️ High Uncertainty: Score raised to Unverified level (Speculative).")
            
        # If score is already 35-65, we leave it alone (it is effectively centered).


    # --- Final Clamping & Domain Bonus ---
    
    # Trusted Domain Boost (Flat +5)
    is_trusted_domain = url and (get_domain_from_url(url) in TRUSTED_DOMAINS)
    if is_trusted_domain:
        base_score += 5
        factors.append(f"✅ Domain Override: +5 (Trusted Source).")

    final_score = max(0, min(100, base_score))
    
    factors.insert(0, "✅ Decision Method: Dynamic HCDM Aggregation.")

    # ----------------------------------------------------
    
    score_int = int(final_score)
    
    if final_score >= 80:
        final_classification = 'Real'
        explanation = f"Assessed as Real ({get_confidence_label(final_score)}). "
        "The claim aligns with established facts and shows no signs of contradiction "
        "or manipulation based on available evidence."
    elif final_score <= 30: 
        final_classification = 'Fake'
        explanation = f"Assessed as Fake ({get_confidence_label(final_score)}). "
        "The claim conflicts with verified knowledge or exhibits strong indicators "
        "of misinformation." 
    else:
        final_classification = 'Insufficient Evidence'
        explanation = f"Assessed as Insufficient Evidence ({get_confidence_label(final_score)}). "
        "The claim is plausible but cannot be conclusively verified or falsified "
        "with the currently available information."
    conf_real = final_score 
    conf_fake = 100 - final_score
    
    # --- BUG FIX: DEFINE VARIABLES BEFORE USE IN FINAL_RESPONSE ---
    suggestion = "To improve credibility, ensure claims reference specific, verifiable entities (journals, official bodies) and use neutral language."

    related_news = []
    # If CRCS found sources, use them in related news
    if crcs_signal and 'Sources' in crcs_signal and crcs_signal['Sources']:
        related_news = [{"title": f"Reference from {src}", "url": "#"} for src in crcs_signal["Sources"]]
    elif cars_context and cars_context.get("retrieved"): # Fallback to raw CARS sources if available
        related_news = [{"title": f"Reference from {ctx['source']}", "url": "#"} for ctx in cars_context["contexts"]]

    verification_tools = [
        {"source": "Google Fact Check", "url": "https://toolbox.google.com/factcheck/explorer"},
        {"source": "Snopes", "url": "https://www.snopes.com/"}
    ]
    # ----------------------------------------------------------------

    # Check if ANY claim triggered the FKC (Hard or Soft) for UI Flag
    fundamental_contradiction_detected = any(c.get('fkc_flag') for c in claims_data.get('claims', []))
    # --- DEBUG BOX DATA ---
    debug_box = {}
    if debug:
        debug_box = {
            "ccl_analysis": [
                {"claim": c.get("claim_text"),
                "primary_domain": c.get("ccl_result", {}).get("primary_domain"),
                "secondary_domains": c.get("ccl_result", {}).get("secondary_domains", []),
                "confidence": c.get("ccl_result", {}).get("confidence"),
                "reason": c.get("ccl_result", {}).get("reason"),
                "possible_metaphor": c.get("ccl_result", {}).get("notes", {}).get("possible_metaphor"),
                "notes": c.get("ccl_result", {}).get("notes"),
                }
                for c in claims_data.get("claims", [])
            ],
            "nli_level2_implicit": [
                {"claim": c["claim_text"],
                "triggered": c.get("implicit_result", {}).get("implicit_conflict", False),
                "confidence": c.get("implicit_result", {}).get("confidence", 0.0),
                "details": c.get("implicit_result", {}).get("details", [])\
                } for c in claims_data.get('claims', [])
            ],
            "cars_retrieval": {
                "retrieved": cars_context["retrieved"] if cars_context else False,
                "context_count": len(cars_context["contexts"]) if cars_context else 0,
                "sources": [c["source"] for c in cars_context["contexts"]] if cars_context else []
            },
            "crcs_consensus": {
                "acs": crcs_signal["ACS"] if crcs_signal else "N/A",
                "consensus_label": crcs_signal["Consensus"] if crcs_signal else "N/A",
                "evidence_count":crcs_signal["Evidence_Count"] if crcs_signal else 0,
                # --- Day 6 additions ---
                "truth_breakdown": crcs_signal.get("Truth_Breakdown") if crcs_signal else None,
                "confidence": crcs_signal.get("Confidence") if crcs_signal else None,
                "E": crcs_signal.get("E") if crcs_signal else None,
                "C": crcs_signal.get("C") if crcs_signal else None,
                "N": crcs_signal.get("N") if crcs_signal else None,
            },
            "fkc_checks": [
                {
                    "claim": c["claim_text"],
                    "violation": c["fkc_result"]["violation"],
                    "warning": c.get("fkc_result", {}).get("warning", None),
                    "debug_info": c["fkc_result"].get("debug", {}),
                    "admission": c.get("admission_info") # New Admission Info
                } for c in claims_data.get('claims', [])
            ],
            "nli_raw_samples": crcs_signal.get("NLI_Raw", []) if crcs_signal else [],
            "rebuttal_detection": rebuttal_findings # NEW: Rebuttal Detection Debug Info
        }
        # --- CDVM (observe-only, per-claim) ---
    cdvm_signals = [
        analyze_cdvm(
        verb=c.get("verb") or c.get("main_verb"),
        verb_native_domain="physical",
        subject_domain=map_ccl_to_cdvm_domain(
            c.get("ccl_result", {}).get("primary_domain")
        ),
        object_domain="society"
    )
        for c in claims_data.get("claims", [])


    ]

        
        # Add print statement to log debug info to the backend terminal
    print(f"\n--- DEBUG BOX ---\n{json.dumps(debug_box, indent=2)}\n-----------------\n")
        
    # Prepare final metadata for the document case
    confidence_label = final_classification
    final_response = {
        "classification": final_classification,
        "confidenceReal": round(conf_real / 100.0, 4), 
        "confidenceFake": round(conf_fake / 100.0, 4),
        "credibility_score": score_int,
        "confidence_label": confidence_label, # ADDED FEATURE 2
        "factors": factors,
        "explanation": explanation, # FEATURE 1: Now dynamic
        "suggestion": suggestion,
        "claims_data": claims_data, 
        "related_news": related_news,
        "verification_tools": verification_tools,
        "fundamental_contradiction": fundamental_contradiction_detected, # UI Flag
        "crcs_data": crcs_signal, # Expose for debug/UI if needed
        "debug_box": debug_box # Pre-HCDM Observability Data (Empty if debug=False)
}
    # --- DAY-7: UI VERDICT ADAPTER (NO LOGIC CHANGE) ---
    try:
        if crcs_signal:
            final_response["verdict"] = {
                "label": final_classification,  # Real / Fake / Insufficient Evidence
                "truth_breakdown": crcs_signal.get("Truth_Breakdown"),
                "confidence": crcs_signal.get("Confidence"),
                "evidence_count": crcs_signal.get("Evidence_Count")
            }
        else:
            final_response["verdict"] = {
                "label": final_classification,
                "truth_breakdown": None,
                "confidence": None,
                "evidence_count": 0
            }
    except Exception:
        # UI adapter must never break backend execution
        pass

# --- CDVM (observe-only, per-claim) ---
    if debug:
        cdvm_signals =[
        analyze_cdvm (
            verb=c.get("verb") or c.get("main_verb"),
            verb_native_domain="physical",  # temp
            subject_domain=map_ccl_to_cdvm_domain(
                c.get("ccl_result", {}).get("primary_domain")
            ),
            object_domain="society"
        )
            for c in claims_data.get("claims", [])
        
        ]
    
    rhetorical_profiles = [
        analyze_option3(sig) if sig else {"present": False}
        for sig in cdvm_signals
    ]
    debug_box["cdvm"] = cdvm_signals
    debug_box["rhetorical_profile"] = rhetorical_profiles
    
    return final_response
# --- ORCHESTRATOR: ANALYZE CONTENT ---
async def analyze_content(
    text: str,
    source_type: str,
    url=None,
    image_bytes=None,
    file_type=None,
    debug: bool = False,
    mode: str = SAFE_MODE
):
    # --- CONFIDENCE ADJUSTMENT (FULL_ACT ONLY) ---
    confidence_adjustment = 0.0
    """
    Minimal orchestrator.
    No globals. No new logic. No refactors.
    """

    # Normalize locals
    debug = bool(debug)
    execution_mode = mode if mode in ALLOWED_MODES else SAFE_MODE


    # Minimal adapter to existing aggregate_signals contract
    claims_data = {
        "claims": [{
            "claim_text": text,
            "signals": {},
            "fkc_result": {
    "violation": False,
    "warning": None,
    "debug": {}
},
            "fkc_flag": False,
            "admission_info": None
        }]
    }
    
    # --- GOOGLE-FIRST EVIDENCE RETRIEVAL (Day 7) ---
    evidence_list, evidence_source = retrieve_evidence(text)
    # Attach retrieved evidence to claim signals (non-breaking)
    claims_data["claims"][0]["signals"]["retrieved_evidence"] = evidence_list
    
    # --- Verb Extraction (shared) ---
    for c in claims_data.get("claims", []):
        c["verb"] = extract_main_verb(c["claim_text"])
        
        
    # --- NLI LEVEL-2: IMPLICIT CONTRADICTION (Plan-2) ---
    for claim in claims_data.get("claims", []):
        claim["implicit_result"] = detect_implicit_conflict(
            claim.get("claim_text", "")
        )

    # --- CCL (Observe-only) ---
    for claim in claims_data.get("claims", []):
        claim_text = claim.get("claim_text", "")

        claim["ccl_result"] = minimal_ccl_decision(claim_text)
                
    # Ensure effective_domain always exists
        claim["effective_domain"] = claim.get("effective_domain") or claim.get("ccl_result", {}).get("primary_domain")
        
    
        # --- METAPHOR GATE (Day-2 / Test-11 SAFE) ---
        
        ccl_notes = claim.get("ccl_result", {}).get("notes", {})
        ccl_metaphor = bool(ccl_notes.get("possible_metaphor", False))

        # Minimal CDVM-style metaphor check (observe-only, no scoring)
        cdvm_metaphor = False
        try:
            verb = extract_main_verb(claim_text)
            subject_domain = map_ccl_to_cdvm_domain(
                claim.get("ccl_result", {}).get("primary_domain")
            )

            cdvm_signal = analyze_cdvm(
                verb=verb,
                verb_native_domain="physical",
                subject_domain=subject_domain,
                object_domain="society"
            )

            if (
                cdvm_signal
                and cdvm_signal.get("present") is True
                and isinstance(cdvm_signal.get("domain_transfer"), str)
                and cdvm_signal["domain_transfer"].startswith("physical->")
            ):
                cdvm_metaphor = True

        except Exception:
            cdvm_metaphor = False

        is_metaphor = bool(ccl_metaphor and cdvm_metaphor)
        claim["metaphor_flag"] = is_metaphor
        
        # --- PASS C: DOMAIN NORMALIZATION (Day 5) ---
        ccl_res = claim.get("ccl_result", {})
        primary_domain = ccl_res.get("primary_domain")
        
        claim["effective_domain"] = primary_domain  # default
        
        if (
            primary_domain == "Opinion / Interpretation"
            and claim.get("metaphor_flag") is False
):
            text = claim_text.lower()
            has_number = any(ch.isdigit() for ch in text)
            has_entity = any(
                token in text
                for token in ["india", "population", "million", "billion"]
            )
            if has_number and has_entity:
                ccl_res["primary_domain"] = "General Knowledge"
                claim["effective_domain"] = "General Knowledge"
                ccl_res.setdefault("notes", {})["domain_corrected"] = True
                

        # --- FKC (GATED BY METAPHOR, NO SCORE CHANGE) ---
    # --- FKC (GATED BY METAPHOR, MODE-AWARE) ---
        if is_metaphor and execution_mode != FULL_ACT_MODE:
            # SAFE / OBSERVE: gate literal FKC
            fkc_result = {
                "skipped": True,
                "violation": False,
                "reason": "Metaphorical claim — literal FKC gated",
                "debug": {
                    "metaphor_gate": True,
                    "source": "CCL" if ccl_metaphor else "CDVM",
                    "execution_mode": execution_mode
                }
            }
        else:
            # Literal claim OR FULL_ACT mode → allow FKC
            fkc_result = FKC.check_claim(claim_text)

        claim["fkc_result"] = fkc_result
        claim["fkc_flag"] = bool(fkc_result.get("violation"))
#        # --- FIX 1: RUN FUNDAMENTAL KNOWLEDGE CHECKER (FKC) ---

#    fkc_result = FKC.check_claim(text)

#    claims_data["claims"][0]["fkc_result"] = fkc_result
#    claims_data["claims"][0]["fkc_flag"] = bool(fkc_result.get("violation"))
    
        # --- FIX 2: RUN CARS RETRIEVAL ---
    cars_domain = CARS.identify_domain(text)
    cars_context = CARS.retrieve_context(text, cars_domain)
    
        # --- FIX 3: RUN CRCS (ACTIVATES NLI) ---
    crcs_signal = None
    if cars_context and cars_context.get("retrieved"):
        crcs_signal = CRCS.evaluate_consensus(
            claim_text=text,
            cars_output=cars_context,
            nli_provider=FKC,
            metaphor_flag=claim.get("metaphor_flag", False),
            primary_domain=claim.get("effective_domain")
        )
    # --- DAY-2: CRCS INTERPRETATION GATE (Metaphor / Rhetorical Claims) ---
    try:
        # If the primary claim is metaphorical, CRCS should not act as a
        # physical contradiction signal
        claims_list = claims_data.get("claims") or []
        primary_claim = claims_list[0] if claims_list else {}
        is_metaphor = bool(primary_claim.get("metaphor_flag", False))

        if is_metaphor and crcs_signal:
            # Preserve evidence & raw NLI, but neutralize contradiction effect
            crcs_signal = {
                **crcs_signal,
                "OverCautious": False,
                "ContradictionPenalty": False,
                "Confidence_Adjustment": 0.0,
                "Notes": "CRCS contradiction gated due to metaphorical / rhetorical claim"
            }

    except Exception:
        # Absolute safety: never allow CRCS gating to crash pipeline
        pass        
            # --- FIX 4: RUN REBUTTAL DETECTION ---
    rebuttal_findings = REBUTTAL_DETECTOR.scan(
        text=text,
        claims=[text]
    )

    # Neutral NN signals (no influence)
    nn_signals_01 = [0.5, 0.5, 0.5, 0.5]

    final_response = aggregate_signals(

    claims_data=claims_data,
    nn_signals_01=nn_signals_01,
    source_type="text",
    url=None,
    crcs_signal=crcs_signal,
    cars_context=cars_context,
    debug=debug
)
    # --- ADAPT GOOGLE EVIDENCE INTO CARS CONTEXT (Day 7) ---
    if evidence_list:
        cars_context = {
            "retrieved": True,
            "sources": [],
            "contexts": []
        }
    for item in evidence_list:
        cars_context["contexts"].append({
            "text": item.get("snippet", ""),
            "source": item.get("source", evidence_source),
            "tier": "T1" if evidence_source == "google" else "T2"
        })        

    # --- APPLY FULL_ACT CONFIDENCE ADJUSTMENT ---
    if execution_mode == FULL_ACT_MODE:
        base_score = final_response.get("confidence_score", 50)
        adjusted_score = base_score + int(confidence_adjustment * 100)
        final_response["confidence_score"] = max(
            0, min(100, adjusted_score)
        )
        final_response["full_act_adjustment"] = confidence_adjustment
    else:
        final_response["full_act_adjustment"] = 0.0
    
    # --- DAY-3 STEP-1: CONFIDENCE INTERPRETATION (EXPLANATION ONLY) ---
    try:
        claims_list = claims_data.get("claims") or []
        primary_claim = claims_list[0] if claims_list else {}

        metaphor_flag = bool(primary_claim.get("metaphor_flag", False))
        rebuttal_info = final_response.get("rebuttal_detection")
        crcs_info = final_response.get("crcs_consensus")

        explanation_lines = []
        interpretation_mode = "literal"

        if metaphor_flag:
            interpretation_mode = "metaphorical"
            explanation_lines.append(
                "This claim uses metaphorical or rhetorical language to express causality or emphasis."
            )

            if crcs_info:
                explanation_lines.append(
                    "Available evidence contradicts a literal reading, but the claim is not asserting a physical fact."
                )

        if rebuttal_info:
            explanation_lines.append(
                "The claim appears in a corrective or rebuttal context."
            )

        if not explanation_lines:
            explanation_lines.append(
                "The claim is interpreted using a literal factual reading based on available evidence."
            )

        final_response["interpretation_mode"] = interpretation_mode
        final_response["confidence_explanation"] = " ".join(explanation_lines)

    except Exception:
        # Explanation layer must never affect execution
        pass
    
    # --- DAY-3 STEP-2: UI / OUTPUT MAPPING (DISPLAY ONLY) ---
    try:
        claims_list = claims_data.get("claims") or []
        primary_claim = claims_list[0] if claims_list else {}

        metaphor_flag = bool(primary_claim.get("metaphor_flag", False))
        rebuttal_info = final_response.get("rebuttal_detection")
        crcs_info = final_response.get("crcs_consensus")

        display_category = "Literal Fact Claim"
        display_summary = "The claim is evaluated as a literal factual statement."

        if metaphor_flag:
            display_category = "Metaphorical / Rhetorical Claim"
            display_summary = (
                "The claim uses figurative language to express emphasis or causality, "
                "and should not be interpreted as a literal physical assertion."
            )

        elif rebuttal_info:
            display_category = "Corrective / Rebuttal Context"
            display_summary = (
                "The claim appears in a corrective or rebuttal context rather than asserting a new fact."
            )

        elif not crcs_info:
            display_category = "Insufficient Evidence"
            display_summary = (
                "There is not enough reliable evidence to fully evaluate this claim."
            )

        final_response["display_category"] = display_category
        final_response["display_summary"] = display_summary

    except Exception:
        # UI mapping must never affect execution
        pass


    # --- DAY-3: SURFACE REBUTTAL DETECTION (OBSERVE-ONLY) ---
    try:
        if rebuttal_findings:
            final_response["rebuttal_detection"] = rebuttal_findings
        else:
            final_response["rebuttal_detection"] = None
    except Exception:
        final_response["rebuttal_detection"] = None

    return final_response
# FINAL INITIALIZATION (module import side-effects)
# --------------------------------------------------
        
def analyze(text: str) -> dict:
    """
    Full analysis entry point for runtime usage.
    """

    if not isinstance(text, str) or not text.strip():
        return {
            "classification": "Invalid",
            "credibility_score": 0,
            "confidence_label": "No input provided",
            "claims_data": {"claims": []}
        }

    import asyncio

    return asyncio.run(
        analyze_content(
            text=text,
            source_type="text",
            url=None,
            image_bytes=None,
            file_type=None,
            debug=False   # ✅ FIX: explicitly set
        )
    )

def build_runtime():
    
    global FKC

    # 1. create FKC
    FKC = FundamentalKnowledgeChecker()

    # 2. start background NN loading
    start_nn_model_loader()

    # 3. return runtime exports
    # ✅ THIS RETURN IS FOR build_runtime(), NOT analyze()
    return {
    "analyze": analyze,
    "run_nn_prediction": run_nn_prediction,
    "extract_text_from_image": extract_text_from_image_local
}
def analyze_text(
    text: str,
    debug: bool = False,
    mode: str = "default"
) -> dict:
    """
    Pure engine entrypoint.
    No FastAPI, no HTTP, no request objects.
    """
    return analyze(payload={
        "text": text,
        "debug": debug,
        "mode": mode
    })
    