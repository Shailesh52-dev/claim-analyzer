from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
# FIX: Corrected CORSMiddleware import name (lowercase 'w')
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests 
import json
import os
import urllib.parse
from urllib.parse import urlparse
from bs4 import BeautifulSoup 
from typing import List, Dict, Any, Union
import time 
import torch
import re
from io import BytesIO
from PIL import Image, ImageEnhance

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
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    # Also attempt to import TrOCR components globally
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("FATAL WARNING: Core 'transformers' library not found. NN/TrOCR models will fail.")

# Attempt to import PaddleOCR
PADDLEOCR_AVAILABLE = False
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    print("WARNING: PaddleOCR not found. Will attempt TrOCR fallback.")

# --- CRITICAL CONFIGURATION ---
API_KEY =  
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
LOCAL_NN_MODEL_PATH = "./local_nn_model" 
NUM_OUTPUT_HEADS = 4 

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
        self.enabled = False
        self.tokenizer = None
        self.model = None
        # --- FIX 3: INVARIANT SEVERITY TAXONOMY ---
        # Axioms defined with strict severity levels.
        # "absolute": Immutable logical invariants (Calendar, Definitions, Composition). PENALTY APPLIED.
        # "physical": Laws of nature/physics. WARNING ONLY.
        # "soft": Context-dependent or high-level scientific concepts. IGNORED.
        self.axioms = [
            # ABSOLUTE: Identity, Chemical Composition, Definitions
            {"text": "Tuesday comes before Wednesday.", "severity": "absolute"}, 
            {"text": "Water is chemically composed of hydrogen and oxygen.", "severity": "absolute"},
            {"text": "A triangle has three sides.", "severity": "absolute"}, 
            {"text": "A square has four sides.", "severity": "absolute"},

            # PHYSICAL: Laws of Physics/Nature (Warning Only)
            {"text": "The Earth is a sphere or spheroid.", "severity": "physical"},
            {"text": "Gravity attracts objects towards the center of large masses like Earth.", "severity": "physical"},
            {"text": "Perpetual motion machines are impossible due to thermodynamics.", "severity": "physical"},
            {"text": "The sun is a star at the center of the solar system.", "severity": "physical"},
            
            # SOFT: Contextual/Medical (Ignored)
            {"text": "Vaccines generally work by stimulating the immune system.", "severity": "soft"}
        ]
        
    def load_model(self):
        if TRANSFORMERS_AVAILABLE:
            try:
                print(f"Initializing FKC with NLI model: {FKC_MODEL_NAME}...")
                self.tokenizer = AutoTokenizer.from_pretrained(FKC_MODEL_NAME)
                self.model = AutoModelForSequenceClassification.from_pretrained(FKC_MODEL_NAME)
                self.enabled = True
                print("Fundamental Knowledge Checker (FKC) loaded successfully.")
            except Exception as e:
                print(f"FKC Initialization Failed: {e}")
                self.enabled = False

    def check_claim(self, claim_text: str) -> dict:
        """
        Checks a claim against axioms using NLI. 
        Returns structured data about violations.
        """
        default_result = {'violation': False, 'is_hard': False, 'warning': None, 'confidence': 0.0}

        # --- FIX 2: CONTEXT & BELIEF SHIELDING ---
        # Before running NLI, detect belief / historical framing.
        # Ensure FKC does NOT trigger for historical/belief/hypothetical statements.
        PROTECTED_CONTEXT_TERMS = [
            "believed", "thought", "according to", "historically",
            "in the past", "medieval", "ancient", "once believed",
            "claims that", "hypothetically", "if ", "suppose"
        ]
        if any(term in claim_text.lower() for term in PROTECTED_CONTEXT_TERMS):
            return default_result

        # FIX 1 (from previous query) is handled in the caller (analyze_content) to prevent calls on placeholders, 
        # but we retain a length check here as a fallback safety.
        if not self.enabled or not claim_text or len(claim_text.split()) < 6:
            return default_result

        lower_claim = claim_text.lower()

        # --- FIX 1 & 2 (Previous Turn): GLOBAL ABSOLUTE INVARIANT PRECHECKS (SYMBOLIC) ---
        # These checks are context-independent and trigger on specific phrasing 
        # that contradicts absolute axioms (Identity, Composition).

        # 1. Composition Precheck (Absolute)
        COMPOSITION_INDICATORS = ["made of", "composed of", "consists of", "made from", "created from"]
        IMPOSSIBLE_MATERIALS = ["fire", "stone", "magic", "nothing", "pure energy"]
        
        if any(ind in lower_claim for ind in COMPOSITION_INDICATORS):
            if any(mat in lower_claim for mat in IMPOSSIBLE_MATERIALS):
                 # Find relevant absolute axiom to cite (e.g. Water)
                 comp_axiom = next((a["text"] for a in self.axioms if "Water" in a["text"] and a["severity"] == "absolute"), None)
                 if not comp_axiom:
                     comp_axiom = next((a["text"] for a in self.axioms if a["severity"] == "absolute"), "Material Composition Laws")
                 
                 return {
                    'violation': True, 
                    'is_hard': True,  # Sets severity=absolute
                    'confidence': 1.0, 
                    'warning': f"Scientific Conflict (ABSOLUTE): Invalid composition claim detected against '{comp_axiom}'",
                    'axiom_text': comp_axiom
                }

        # 2. Identity Precheck (Absolute)
        IDENTITY_PHRASES = ["is the same as", "equals", "is identical to", "is the same day as"]
        if any(p in lower_claim for p in IDENTITY_PHRASES):
            # Targeted check for Tuesday/Wednesday contradiction
            if "tuesday" in lower_claim and "wednesday" in lower_claim:
                 tues_axiom = next((a["text"] for a in self.axioms if "Tuesday" in a["text"]), "Calendar Invariants")
                 return {
                    'violation': True, 'is_hard': True, 'confidence': 1.0,
                    'warning': f"Logical Contradiction (ABSOLUTE): Identity error detected against '{tues_axiom}'",
                    'axiom_text': tues_axiom
                }

        # --- FIX 5 (Previous): PHYSICAL PRECHECK (MINIMAL & SAFE) ---
        PHYSICAL_PHRASES = ["fall upward", "gravity repels", "pushes objects away", "earth is flat"]
        if any(p in lower_claim for p in PHYSICAL_PHRASES):
             phys_axiom = next((a["text"] for a in self.axioms if a["severity"] == "physical"), "Physical Laws")
             return {
                'violation': True, 'is_hard': False, 'confidence': 1.0,
                'warning': f"Scientific Conflict (PHYSICAL): Claim contains anti-physical phrasing.",
                'axiom_text': phys_axiom
            }

        # --- NLI CHECK (GLOBAL / CONTEXT-INDEPENDENT) ---
        # Iterate all axioms and check for contradiction against the claim.
        strongest_contradiction = 0.0
        violated_axiom_text = ""
        violated_axiom_severity = "soft"

        try:
            for axiom in self.axioms:
                # NLI: Premise = Axiom, Hypothesis = Claim
                # Checked unconditionally against every valid claim
                inputs = self.tokenizer(axiom["text"], claim_text, return_tensors="pt", truncation=True)
                
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                
                probs = logits.softmax(dim=1)
                
                # Use dynamic label mapping from config
                label_map = self.model.config.id2label
                contradiction_index = None
                
                for idx, label in label_map.items():
                    if label.lower() == "contradiction":
                        contradiction_index = idx
                        break
                
                if contradiction_index is None:
                    return default_result

                contradiction_prob = probs[0][contradiction_index].item()
                
                # Track strongest contradiction globally across all axioms
                if contradiction_prob > strongest_contradiction:
                    strongest_contradiction = contradiction_prob
                    violated_axiom_text = axiom["text"]
                    violated_axiom_severity = axiom["severity"]
            
            # --- STRICT THRESHOLD CHECK ---
            if strongest_contradiction >= 0.97:
                is_hard_violation = (violated_axiom_severity == "absolute")
                
                return {
                    'violation': True,
                    'is_hard': is_hard_violation,
                    'confidence': strongest_contradiction,
                    'warning': f"Scientific Conflict ({violated_axiom_severity.upper()}): Claim contradicts '{violated_axiom_text}'",
                    'axiom_text': violated_axiom_text
                }
            
        except Exception as e:
            # print(f"FKC Inference Error: {e}")
            pass
            
        return default_result

FKC = FundamentalKnowledgeChecker()


try:
    if not TRANSFORMERS_AVAILABLE:
         raise ImportError("Cannot initialize models: Core transformers library is missing.")

    # 1. NN Model Loading (Common to both PaddleOCR and TrOCR paths)
    if os.path.exists(LOCAL_NN_MODEL_PATH):
        LOCAL_NN_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        LOCAL_NN_MODEL = AutoModelForSequenceClassification.from_pretrained(LOCAL_NN_MODEL_PATH, num_labels=NUM_OUTPUT_HEADS)
        NN_MODEL_AVAILABLE = True
        print(f"Local Multi-Head NN Signal Model loaded.")
    else:
        print(f"NN Model not found. Running in Heuristic only mode.")

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


except Exception as e:
    print(f"FATAL ERROR during model initialization: {e}")
    # Set NN and VLM availability to False if any initialization failed
    NN_MODEL_AVAILABLE = False
    PADDLEOCR_AVAILABLE = False
# ---------------------------------------------


# --- Initialize App ---
app = FastAPI(title="FactCheck AI Backend (NN Signal Hybrid)")

# --- CORS and Data Models (Unchanged) ---
origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

class UrlRequest(BaseModel):
    url: str
    
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
    

# --- CORE LOGIC: SIGNAL AGGREGATION AND FINAL JUDGMENT (NN + HCDM PURE) ---
def aggregate_signals(claims_data: dict, nn_signals_01: List[float], source_type: str, url: str) -> tuple:
    """
    Refactored HCDM Logic (Dynamic & Proportional).
    Eliminates hard cliffs by using scaled penalties and a signal-driven baseline.
    """
    
    factors = []
    
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
        fkc_result = claim.get('fkc_result', {})
        
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


    # --- PART 2: SCALED PENALTIES (NN-BASED) ---
    
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


    # FIX 9 & FIX 3: CLEAN CLAIM BONUS (CONTROLLED & GUARDED)
    # Applied LAST.
    # Conditions: No FKC warnings (any kind), Low Manipulation, Low Bias
    if not any_fkc_violation and avg_manipulation_score < 30 and (nn_bias * 100) < 30:
        base_score += 8
        factors.append("✅ Clean Claim Bonus: +8 (No logical contradictions or manipulation detected).")


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

    # Clamp final score to 0-100
    final_score = max(0, min(100, base_score))
    
    factors.insert(0, "✅ Decision Method: Dynamic HCDM Aggregation.")

    # ----------------------------------------------------
    # (Rest of the function remains compatible with original return format)
    
    score_int = int(final_score)
    
    if final_score >= 80:
        final_classification = 'Real'
        explanation = f"Rated as REAL ({get_confidence_label(final_score)}). The content uses authoritative sources, confirmed language, and lacks manipulative framing."
    elif final_score <= 30: 
        final_classification = 'Fake'
        explanation = f"Flagged as FAKE ({get_confidence_label(final_score)}). The content exhibits signs of manipulation, lacks evidence, or contains significant contradictions."
    else:
        final_classification = 'Unsure'
        explanation = f"Rated as UNSURE ({get_confidence_label(final_score)}). The content is plausible but lacks sufficient verification or authoritative confirmation. Speculative language may be present."
        
    conf_real = final_score 
    conf_fake = 100 - final_score
    
    return final_score, factors, claims_data 


# --- CORE ANALYZE FUNCTION (Re-enabled image handling) ---
# Updated signature to pass file_type
async def analyze_content(text: str, source_type: str = "text", url: str = None, image_bytes: bytes = None, file_type: str = None):
    
    # NEW LOG: Confirming analysis start and input type
    print(f"\n--- Analysis Started (Source Type: {source_type.upper()}) ---")

    nn_input_text = text
    document_metadata = {}
    
    # 1. Determine Input Text (Text/URL or Image OCR)
    if source_type == "document":
        if image_bytes:
            # Pass image_bytes AND file_type for local OCR processing
            document_data = extract_text_from_image_local(image_bytes, file_type) 
            nn_input_text = document_data['text']
            document_metadata = {
                "document_type": document_data['document_type'],
                "text_extraction_method": document_data['text_extraction_method']
            }
        else:
            # CRITICAL FIX: If document source is selected but image bytes are missing, raise an error.
            raise HTTPException(status_code=400, detail="Document data is missing.")
    elif source_type == "image":
        # Treating 'image' source_type as 'document' for the API consistency
        if image_bytes:
            document_data = extract_text_from_image_local(image_bytes, file_type) 
            nn_input_text = document_data['text']
            document_metadata = {
                "document_type": document_data['document_type'],
                "text_extraction_method": document_data['text_extraction_method']
            }
        else:
            raise HTTPException(status_code=400, detail="Image data is missing.")
    elif source_type in ["text", "url"]:
        # If source is text or url, nn_input_text is already set to 'text' from the request.
        # This explicit 'elif' ensures we skip the VLM/Document error paths above.
        pass


    # 2. NN Prediction on the determined text (Input or Extracted)
    
    # --- TEMPORARY AGGREGATE SCORES FOR URL/EXPLANATION LOGIC ---
    # Need to run NN prediction on the input first to determine the overall score and explanation.
    nn_signals_for_aggregate = run_nn_prediction(nn_input_text)
    
    
    # --- DECOMPOSITION STEP ---
    MIN_CLAIM_WORDS = 6 # New Minimum Word Count Filter
    
    # Generate explanation is now handled in aggregate_signals, so we just need decomposed claims for UI
    
    if source_type == "url":
        # Special handling for URL claims (synthetic decomposition)
        decomposed_claims = ["Source analysis: " + nn_input_text]
        
    else:
        # For TEXT and DOCUMENT inputs, split the original text into claims (sentences)
        if "Document processed, but no substantial" in nn_input_text:
             decomposed_claims = [nn_input_text]
        else:
             decomposed_claims = re.split(r'(?<=[.!?])\s*', nn_input_text.strip())
             
             # Apply cleaning and length filter
             filtered_claims = []
             for c in decomposed_claims:
                 stripped_claim = c.strip()
                 # CRITICAL IMPROVEMENT: Filter claims shorter than MIN_CLAIM_WORDS
                 if stripped_claim and len(stripped_claim.split()) >= MIN_CLAIM_WORDS:
                     filtered_claims.append(stripped_claim)

             decomposed_claims = filtered_claims
             
             if not decomposed_claims:
                 decomposed_claims = ["No substantial claim found after text extraction or after filtering (min 6 words)."]


    # 3. Iterate and Score Each Claim
    claims_data_list = []
    
    def map_nn_to_llm_signal(nn_score):
        if nn_score >= SIGNAL_THRESHOLDS["high"]: return "high"
        if nn_score >= SIGNAL_THRESHOLDS["medium"]: return "medium"
        if nn_score >= SIGNAL_THRESHOLDS["low"]: return "low"
        return "low" 

    
    print(f"DEBUG: Found {len(decomposed_claims)} claims for scoring.")

    for claim_text in decomposed_claims:
        # Run NN Prediction for each individual claim 
        # (For URL, we just reuse the aggregate signals to save compute, for text/doc we re-run)
        if source_type == "url":
             current_nn_signals = nn_signals_for_aggregate
        else:
             current_nn_signals = run_nn_prediction(claim_text)
        
        # --- NEW: RUN FKC ON CLAIM ---
        # FIX 1: STRICT FKC GATING (CRITICAL)
        # Only run FKC if claim is valid (not a placeholder, sufficiently long)
        is_placeholder = "No substantial claim found" in claim_text
        is_too_short = len(claim_text.split()) < MIN_CLAIM_WORDS
        
        if not is_placeholder and not is_too_short:
             # Capture the full result object for HCDM
             fkc_result = FKC.check_claim(claim_text)
        else:
             # Skip FKC for invalid claims/placeholders
             fkc_result = {'violation': False, 'is_hard': False, 'warning': None, 'confidence': 0.0}
        
        claims_data_list.append({
            "claim_text": claim_text,
            "signals": {
                "plausibility": map_nn_to_llm_signal(current_nn_signals[0]),
                "evidence_specificity": map_nn_to_llm_signal(current_nn_signals[1]),
                "bias_level": map_nn_to_llm_signal(current_nn_signals[2]),
                "uncertainty_present": map_nn_to_llm_signal(current_nn_signals[3])
            },
            # Attach FKC results (used by UI and HCDM)
            "fkc_result": fkc_result, # Full structured object
            # Keep legacy flags for UI compatibility
            "fkc_flag": fkc_result['violation'],
            "fkc_reason": fkc_result['warning']
        })
    
    claims_data = {"claims": claims_data_list}
    
    # 4. FINAL JUDGMENT (using the aggregated score calculated earlier)
    
    # Rerun aggregate_signals using the original aggregate score and classification generated above
    final_score, factors, claims_data = aggregate_signals(claims_data, nn_signals_for_aggregate, source_type, url)
    
    # Logic copied from aggregate_signals to ensure consistency in response object
    score_int = int(final_score)
    confidence_label = get_confidence_label(final_score)
    
    if final_score >= 80:
        final_classification = 'Real'
        explanation = f"Rated as REAL ({confidence_label}). The content uses authoritative sources, confirmed language, and lacks manipulative framing."
    elif final_score <= 30: 
        final_classification = 'Fake'
        explanation = f"Flagged as FAKE ({confidence_label}). The content exhibits signs of manipulation, lacks evidence, or contains significant contradictions."
    else:
        final_classification = 'Unsure'
        explanation = f"Rated as UNSURE ({confidence_label}). The content is plausible but lacks sufficient verification or authoritative confirmation. Speculative language may be present."
        
    conf_real = final_score 
    conf_fake = 100 - final_score
    
    # --- BUG FIX: DEFINE VARIABLES BEFORE USE IN FINAL_RESPONSE ---
    suggestion = "To improve credibility, ensure claims reference specific, verifiable entities (journals, official bodies) and use neutral language."

    related_news = []
    verification_tools = [
        {"source": "Google Fact Check", "url": "https://toolbox.google.com/factcheck/explorer"},
        {"source": "Snopes", "url": "https://www.snopes.com/"}
    ]
    # ----------------------------------------------------------------

    # Check if ANY claim triggered the FKC (Hard or Soft) for UI Flag
    fundamental_contradiction_detected = any(c.get('fkc_flag') for c in claims_data.get('claims', []))

    # Prepare final metadata for the document case
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
        "fundamental_contradiction": fundamental_contradiction_detected # UI Flag
    }
    
    # EXPOSE DOCUMENT METADATA IN RESPONSE
    if source_type in ["document", "image"]:
        final_response.update(document_metadata)
    
    return final_response


# --- API Endpoints ---

@app.post("/analyze", response_model=Dict[str, Any])
async def analyze_text(request: TextRequest):
    """
    Endpoint for analyzing plain text input.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    return await analyze_content(
        text=request.text, 
        source_type="text",
        url=None,
        image_bytes=None,
        file_type=None
    )

@app.post("/predict_url", response_model=Dict[str, Any])
async def predict_url_endpoint(request: UrlRequest):
    """
    Endpoint for analyzing a URL.
    NOTE: Currently treats the URL as a text source for the NN.
    """
    if not request.url or not request.url.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="Invalid URL format.")

    # In a full version, this would perform web scraping or fetch the article text.
    # For this demo, we'll just analyze the URL itself and rely on the Domain Bonus heuristic.
    
    # 1. Extract domain
    domain = get_domain_from_url(request.url)
    
    # 2. Construct synthetic input text (meaningful for NN scoring)
    nn_input_text = f"Source domain: {domain}. Article credibility assessment."
    
    return await analyze_content(
        text=nn_input_text, 
        source_type="url",
        url=request.url,
        image_bytes=None,
        file_type=None
    )


# IMAGE/DOCUMENT Analysis Endpoint
@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint for document/image analysis using local OCR (PaddleOCR).
    """
    
    # Check for valid image file types OR PDF types (MIME and extension)
    is_image = file.content_type.startswith('image/') or file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))
    is_pdf = file.content_type == 'application/pdf' or file.filename.lower().endswith('.pdf')
    
    if not file.content_type or not (is_image or is_pdf):
        # Updated error message to include PDF
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an image or PDF file.")

    contents = await file.read()
    
    # Get file type before passing to analysis function
    file_type = file.content_type
    
    # We use 'document' as the source_type for clear distinction
    return await analyze_content(
        text="", 
        source_type="document", 
        image_bytes=contents,
        file_type=file_type # Pass file type here
    )


if __name__ == "__main__":
    # Ensure necessary ML dependencies are installed for local TrOCR usage
    required_libs = ['transformers', 'torch', 'Pillow', 'timm']
    # NOTE: The check below is removed as it causes errors if run in a non-native environment.
    # The assumption is that the user has installed dependencies via pip install -r requirements.txt
    
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
