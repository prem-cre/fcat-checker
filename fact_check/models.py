from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# --- Basic Models ---
class FactCheckRequest(BaseModel):
    query: str
    language: str = "en"

# --- Internal Models for Pipeline ---

class LegalClaim(BaseModel):
    sentence_id: int = 0
    sentence_text: str
    claim_type: str  # 'statute', 'case_law', 'date_fact', 'legal_fact', 'general_fact'
    search_query: str 
    requires_verification: bool = True
    confidence: float = 1.0 
    location: str = "" 

class VerificationEvidence(BaseModel):
    """Container for search results associated with a claim"""
    claim_id: int
    google_fact_checks: List[Dict[str, Any]] = []
    news_results: List[Dict[str, Any]] = []
    academic_results: List[Dict[str, Any]] = []
    indian_kanoon_results: List[Dict[str, Any]] = []  # <--- Added this

# --- Final Output Models ---

class LegalFactCheck(BaseModel): 
    sentence_id: int
    sentence: str
    claim: str 
    verdict: str  # accurate, inaccurate, partial, unverifiable
    source: str
    source_type: str
    corrected_sentence: Optional[str] = None
    checked_on: datetime = Field(default_factory=lambda: datetime.now())
    legal_authority_level: str
    citation_provided: Optional[str] = None

class LegalVerificationReport(BaseModel):
    blog_id: str
    total_claims_extracted: int
    verifiable_claims: int
    verified_claims: int
    accuracy_score: float
    claims_breakdown: Dict[str, int]
    fact_checks: List[LegalFactCheck]
    recommendations: List[str] = []
    timestamp: datetime