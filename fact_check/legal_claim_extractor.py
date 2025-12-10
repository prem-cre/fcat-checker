import logging
import json
import os
import google.generativeai as genai
from typing import List
from models import LegalClaim

logger = logging.getLogger(__name__)

class LegalClaimExtractor:
    """
    Production-grade Extractor using Google Gemini.
    Extracts ONLY sentences requiring verification (Dates, Statutes, Case Law).
    """
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash-lite"):
        # Load API Key from env if not passed
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for LegalClaimExtractor")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"response_mime_type": "application/json"}
        )

    async def extract_claims(self, content: str) -> List[LegalClaim]:
        """
        Scans content using Gemini to find verifiable legal claims.
        """
        logger.info("Sending content to Gemini for claim extraction...")
        
        prompt = (
            "You are an expert Legal Data Analyst. Analyze the text below and extract "
            "verifiable factual claims.\n\n"
            "TARGET SENTENCES CONTAINING:\n"
            "1. Dates, Timelines, or Times.\n"
            "2. Case Law Citations (e.g., 'State vs X', 'AIR 2024 SC...').\n"
            "3. Statutory References (e.g., 'Section 302 IPC', 'Article 21').\n"
            "4. Specific Legal Authority claims (e.g., 'Supreme Court held', 'High Court ruled').\n"
            "5. Document Numbers, FIR numbers, or Digital Rights IDs.\n\n"
            "Ignore opinions, generic theories, or introductions.\n\n"
            "OUTPUT JSON SCHEMA:\n"
            "{\n"
            "  \"claims\": [\n"
            "    {\n"
            "      \"sentence_id\": 1,\n"
            "      \"sentence_text\": \"The exact sentence from text.\",\n"
            "      \"claim_type\": \"statute\" | \"case_law\" | \"date_fact\" | \"legal_fact\",\n"
            "      \"search_query\": \"Optimized Google search query for this fact\"\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"TEXT TO ANALYZE:\n{content}"
        )

        try:
            # Async call to Gemini
            response = await self.model.generate_content_async(prompt)
            
            # Parse JSON output
            data = json.loads(response.text)
            claims_list = data.get("claims", [])

            extracted = []
            for item in claims_list:
                extracted.append(LegalClaim(
                    sentence_id=item.get('sentence_id', 0),
                    sentence_text=item.get('sentence_text', ''),
                    claim_type=item.get('claim_type', 'legal_fact'),
                    search_query=item.get('search_query', item.get('sentence_text')),
                    requires_verification=True,
                    confidence=1.0, # Default high confidence since LLM selected it
                    location=f"id_{item.get('sentence_id')}"
                ))
            
            logger.info(f"Gemini extracted {len(extracted)} verifiable claims.")
            return extracted

        except Exception as e:
            logger.error(f"Gemini Extraction failed: {e}")
            # Fallback: Return empty list or handle gracefully
            return []