import logging
import asyncio
import json
import os
import google.generativeai as genai
from datetime import datetime, timezone
from contextlib import AsyncExitStack
from typing import List, Dict, Any, Optional

# FIXED IMPORTS
from fact_checking_clients import (
    GoogleFactCheckClient,
    GoogleCustomSearchClient,
    SemanticScholarClient,
    IndianKanoonClient
)
from models import (
    LegalClaim, 
    VerificationEvidence,
    LegalVerificationReport, 
    LegalFactCheck
)
from legal_claim_extractor import LegalClaimExtractor

logger = logging.getLogger(__name__)

class LegalFactCheckService:
    def __init__(
            self,
            google_api_key: str,
            cse_id: str,
            gemini_api_key: str,
            indian_kanoon_key: str,  # <--- New Argument
            academic_key: Optional[str] = None,
            llm_model: str = "gemini-2.5-flash-lite"
    ):
        # 1. Validation
        if not google_api_key: raise ValueError("Google API key is required.")
        if not cse_id: raise ValueError("CSE ID is required.")
        if not gemini_api_key: raise ValueError("Gemini API key is required.")
        if not indian_kanoon_key: raise ValueError("Indian Kanoon API key is required.")

        # 2. Search Clients
        self.fact_check_client = GoogleFactCheckClient(google_api_key)
        self.news_client = GoogleCustomSearchClient(google_api_key, cse_id)
        self.indian_kanoon_client = IndianKanoonClient(indian_kanoon_key) # <--- Init Client
        
        self.academic_client = None
        if academic_key:
            self.academic_client = SemanticScholarClient(academic_key)

        # 3. LLM Configuration (Gemini)
        genai.configure(api_key=gemini_api_key)
        self.llm_model_name = llm_model
        self.judge_model = genai.GenerativeModel(
            model_name=llm_model,
            generation_config={"response_mime_type": "application/json"}
        )
        
        # 4. Extractor
        self.claim_extractor = LegalClaimExtractor(api_key=gemini_api_key, model_name=llm_model)

        logger.info(f"LegalFactCheckService initialized (Model: {llm_model})")

    async def verify_legal_blog(self, blog_content: str, blog_id: str) -> LegalVerificationReport:
        logger.info(f"Starting verification for blog: {blog_id}")
        
        # STEP 1: LLM Extraction
        extracted_claims = await self.claim_extractor.extract_claims(blog_content)
        
        if not extracted_claims:
            logger.warning("No verifiable claims found.")
            return self._generate_empty_report(blog_id, 0)

        # STEP 2: Parallel API Execution
        logger.info(f"Verifying {len(extracted_claims)} claims concurrently...")
        
        async with AsyncExitStack() as stack:
            # Init all clients in context
            await stack.enter_async_context(self.fact_check_client)
            await stack.enter_async_context(self.news_client)
            await stack.enter_async_context(self.indian_kanoon_client) # <--- Enter Context
            if self.academic_client:
                await stack.enter_async_context(self.academic_client)

            tasks = [self._gather_evidence_for_claim(claim) for claim in extracted_claims]
            evidence_results = await asyncio.gather(*tasks)

        # STEP 3: LLM Verification (The Judge)
        logger.info("Sending evidence to Gemini Judge for final verdict...")
        final_report = await self._generate_final_verdict_llm(
            blog_id, extracted_claims, evidence_results
        )
        
        logger.info(f"Verification complete. Accuracy: {final_report.accuracy_score}")
        return final_report

    async def _gather_evidence_for_claim(self, claim: LegalClaim) -> VerificationEvidence:
        """
        Intelligently routes requests to specific APIs based on Claim Type.
        This prevents 'Google Search' facts from getting 'Supreme Court' citations.
        """
        evidence = VerificationEvidence(claim_id=claim.sentence_id)
        query = claim.search_query
        claim_type = claim.claim_type.lower()

        # --- Helper Functions ---
        async def fetch_fact():
            try:
                res = await self.fact_check_client.search(query)
                return res.get("claims", [])
            except Exception as e:
                logger.error(f"FactCheck Error ({claim.sentence_id}): {e}")
                return []

        async def fetch_news():
            try:
                res = await self.news_client.search(query, num_results=4)
                return res.get("items", [])
            except Exception as e:
                logger.error(f"News/Google Error ({claim.sentence_id}): {e}")
                return []

        async def fetch_academic():
            if not self.academic_client: return []
            try:
                res = await self.academic_client.search(query, limit=2)
                return res.get("data", [])
            except Exception as e:
                logger.error(f"Academic Error ({claim.sentence_id}): {e}")
                return []
        
        async def fetch_kanoon():
            try:
                # Indian Kanoon returns a 'docs' array inside the JSON
                res = await self.indian_kanoon_client.search(query)
                return res.get("docs", [])
            except Exception as e:
                logger.error(f"Indian Kanoon Error ({claim.sentence_id}): {e}")
                return []

        # --- Routing Logic ---
        tasks = []

        # 1. Indian Kanoon: ONLY for Legal facts, Case Laws, Statutes
        if claim_type in ["case_law", "statute", "legal_fact"]:
            tasks.append(fetch_kanoon())
        else:
            # If it's a date or general fact, return empty list immediately
            tasks.append(asyncio.sleep(0, result=[]))

        # 2. Google Custom Search (News/Web): For ALL claims (context is always good)
        tasks.append(fetch_news())

        # 3. Google Fact Check: Mostly for General/Date facts, less useful for obscure case law
        if claim_type in ["date_fact", "general_fact", "legal_fact"]:
            tasks.append(fetch_fact())
        else:
            tasks.append(asyncio.sleep(0, result=[]))

        # 4. Academic: Mostly for Legal/Statute
        if claim_type in ["case_law", "statute"]:
            tasks.append(fetch_academic())
        else:
            tasks.append(asyncio.sleep(0, result=[]))

        # Execute selected tasks
        kanoon_res, news_res, fc_res, acad_res = await asyncio.gather(*tasks)

        evidence.indian_kanoon_results = kanoon_res
        evidence.news_results = news_res
        evidence.google_fact_checks = fc_res
        evidence.academic_results = acad_res
        
        return evidence

    async def _generate_final_verdict_llm(
        self, 
        blog_id: str, 
        claims: List[LegalClaim], 
        evidence_list: List[VerificationEvidence]
    ) -> LegalVerificationReport:
        
        context_data = []
        for claim, ev in zip(claims, evidence_list):
            
            # Compress Evidence for Prompt
            simplified_evidence = {
                "indian_kanoon": [
                    {
                        "title": x.get("title"),
                        "headline": x.get("headline"),
                        "doc_id": x.get("tid"),
                        "source": x.get("docsource"),
                        "url": f"https://indiankanoon.org/doc/{x.get('tid')}/"
                    } for x in ev.indian_kanoon_results[:3] # Top 3 only
                ],
                "fact_check_db": [
                    {
                        "verdict": x.get("claimReview", [{}])[0].get("textualRating"), 
                        "url": x.get("claimReview", [{}])[0].get("url")
                    } for x in ev.google_fact_checks
                ],
                "web_search": [
                    {
                        "title": x.get("title"), 
                        "snippet": x.get("snippet", "")[:200], 
                        "link": x.get("link")
                    } for x in ev.news_results
                ],
                "academic": [
                    {"title": x.get("title")} for x in ev.academic_results
                ]
            }

            context_data.append({
                "sentence_id": claim.sentence_id,
                "original_sentence": claim.sentence_text,
                "claim_type": claim.claim_type,
                "evidence": simplified_evidence
            })

        system_instruction = (
            "You are a Supreme Court Legal Verification AI. "
            "Analyze Claims against Evidence and generate a strict JSON report.\n\n"
            "SOURCE SELECTION RULES (CRITICAL):\n"
            "1. IF claim_type is 'case_law' OR 'statute': You MUST prioritize 'indian_kanoon' evidence. "
            "If Indian Kanoon confirms it, cite the Indian Kanoon URL.\n"
            "2. IF claim_type is 'date_fact' OR 'general_fact': You MUST prioritize 'web_search' or 'fact_check_db'. "
            "DO NOT cite a court judgment for a simple date or general fact unless explicitly mentioned in the judgment.\n"
            "3. If Indian Kanoon returns no results for a legal claim, mark as 'unverifiable' or check web_search for news coverage of the case.\n\n"
            "VERDICT RULES:\n"
            "- 'accurate': Evidence explicitly confirms the claim.\n"
            "- 'inaccurate': Evidence explicitly contradicts the claim.\n"
            "- 'unverifiable': No relevant evidence found.\n\n"
            "OUTPUT JSON STRUCTURE:\n"
            "{\n"
            "  \"total_claims_extracted\": int,\n"
            "  \"verifiable_claims\": int,\n"
            "  \"verified_claims\": int,\n"
            "  \"accuracy_score\": float (0-100),\n"
            "  \"claims_breakdown\": { \"legal_fact\": int, \"statute\": int, \"case_law\": int, \"date_fact\": int },\n"
            "  \"fact_checks\": [\n"
            "    {\n"
            "      \"sentence_id\": int,\n"
            "      \"sentence\": \"string\",\n"
            "      \"claim\": \"string\",\n"
            "      \"verdict\": \"accurate|inaccurate|unverifiable\",\n"
            "      \"source\": \"Name of source\",\n"
            "      \"source_type\": \"legal_official|news|academic\",\n"
            "      \"corrected_sentence\": \"string or null give a corrected sentence if verdict is inaccurate or unverifiable but when accurate provide null\",\n"
            "      \"legal_authority_level\": \"supreme_court|statute|secondary|none\",\n"
            "      \"citation_provided\": \"URL\"\n"
            "    }\n"
            "  ]\n"
            "}"
        )

        try:
            response = await self.judge_model.generate_content_async(
                f"{system_instruction}\n\nDATA TO VERIFY:\n{json.dumps(context_data)}"
            )
            
            result_json = json.loads(response.text)
            
            fact_checks_objs = []
            for item in result_json.get("fact_checks", []):
                verdict = item.get("verdict", "unverifiable").lower()
                
                fc_obj = LegalFactCheck(
                    sentence_id=item.get("sentence_id", 0),
                    sentence=item.get("sentence", "") or item.get("claim", ""),
                    claim=item.get("claim", ""),
                    verdict=verdict,
                    source=item.get("source") or "Unknown",
                    source_type=item.get("source_type", "secondary"),
                    checked_on=datetime.now(timezone.utc),
                    legal_authority_level=item.get("legal_authority_level", "secondary"),
                    citation_provided=item.get("citation_provided"),
                    corrected_sentence=item.get("corrected_sentence")
                )
                fact_checks_objs.append(fc_obj)

            report = LegalVerificationReport(
                blog_id=blog_id,
                total_claims_extracted=result_json.get("total_claims_extracted", len(claims)),
                verifiable_claims=result_json.get("verifiable_claims", len(claims)),
                verified_claims=result_json.get("verified_claims", len(fact_checks_objs)),
                accuracy_score=result_json.get("accuracy_score", 0.0),
                claims_breakdown=result_json.get("claims_breakdown", {}),
                fact_checks=fact_checks_objs,
                recommendations=[], 
                timestamp=datetime.now(timezone.utc)
            )
            return report

        except Exception as e:
            logger.error(f"Gemini Judge Logic Failed: {e}")
            return self._generate_empty_report(blog_id, len(claims))

    def _generate_empty_report(self, blog_id: str, count: int) -> LegalVerificationReport:
        return LegalVerificationReport(
            blog_id=blog_id,
            total_claims_extracted=count,
            verifiable_claims=count,
            verified_claims=0,
            accuracy_score=0.0,
            claims_breakdown={},
            fact_checks=[],
            recommendations=["Analysis failed."],
            timestamp=datetime.now(timezone.utc)
        )