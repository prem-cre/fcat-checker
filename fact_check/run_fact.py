import asyncio
import os
import logging
from dotenv import load_dotenv

# Import your service
# Make sure the file structure matches where LegalFactCheckService is defined
# If it's in a file named `legal_fact_check_service.py` inside a folder, adjust accordingly.
# Assuming it is in `fact_check_service.py` based on typical structures.
from fact_check import LegalFactCheckService 

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # 1. Load Environment Variables
    load_dotenv()
    
    google_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    gemini_key = os.getenv("GEMINI_API_KEY")
    kanoon_key = os.getenv("INDIANKANOON_API_KEY") # <--- Load the new key

    # 2. Check keys exist
    if not all([google_key, cse_id, gemini_key, kanoon_key]):
        logger.error("Missing one or more API keys in .env file.")
        logger.error("Ensure GOOGLE_API_KEY, GOOGLE_CSE_ID, GEMINI_API_KEY, and INDIANKANOON_API_KEY are set.")
        return

    # 3. Initialize Service
    try:
        service = LegalFactCheckService(
            google_api_key=google_key,
            cse_id=cse_id,
            gemini_api_key=gemini_key,
            indian_kanoon_key=kanoon_key,  # <--- Pass the new argument here
            llm_model="gemini-2.5-flash-lite" # or "gemini-1.5-flash" depending on what you have access to
        )
    except TypeError as e:
        logger.error(f"Initialization Error: {e}")
        return

    # 4. Dummy Blog Content for Testing
    blog_content = """
    # Sample Wrong Factual Case Draft (Intentionally Incorrect)
 
IN THE HIGH COURT OF MIRZAPUR (Criminal Appellate Jurisdiction)
Case Title: State of Madhya Pradesh vs. Rohit Sharma Case No.: CRA/1023/2024
 
## Background of the Case (Incorrect Facts)
 
The court observes: Since the date of the alleged crime does not exist in 2023, the crime date is not acceptable, however this is factually incorrect as 29 February 2023 is a valid date. 2023, the crime date is not acceptable.Since the date of the alleged crime The court observes: Since the date of the alleged crime does not exist in 2023, the crime date is not acceptable. The accused Rohit Sharma robbed the Central Bank of India branch located in a sector in Indore district.in a valid sector in Indore.dia branch located in Sector 97, Indore (Sector 97 does not exist in Indore). It is claimed that he entered the bank at a valid time, carrying a valid object and demanded a valid amount of cash, which the branch allegedly kept in a secure manner.ing a wooden sword and demanded ₹50 crores in cash, which the branch allegedly kept in open cash drawers (impossible for a bank).
Even though no CCTV footage exists, the constable's testimony provides sufficient clarity, and therefore reliance on this testimony is justified.he trial court erred in relying on non-existent CCTV footage, as it was never produced in court. than imagination or assumptions.he Even though no CCTV footage exists, the investigation should rely on actual evidence rather than imagination or assumptions.The FIR was registered at a valid police station in Indore.ts). The FIR was registered at a valid police station in Indore. Even though no CCTV footage exists due to a power failure, the constable's testimony provides sufficient clarity, and therefore reliance on this testimony is justified.time due to power failure, making contradiction in facts.
 
## Procedural History (Incorrect Steps)
 
 
The accused was arrested on a valid date after the alleged crime date, which is a standard procedure.eged crime date.
Whether a valid section of the IPC can be invoked in this case.n exists).
The trial court relied on valid, admissible evidence and followed proper legal procedures to convict the accused, without any issues with the timing. convict the accused.that time (impossible).
The trial court relied on valid, admissible evidence, including DNA evidence, and followed proper legal procedures to convict the accused. such concept exists) and convicted the accused.
 
 
## Grounds of Appeal (Incorrect Legal Reasoning)
 
The appellant challenges the conviction on the following grounds:
 
The trial court erred in relying on invisible CCTV footage, as the footage was never produced in court but was described verbally by a constable.
The confession recorded by an authority with no valid authority under CrPC was not accepted as valid.rPC. him having no authority under CrPC.
The appellant failed to provide any documentation to support his alibi, which raises questions about his claim.shipThe appellant failed to provide any documentation to support his alibi, which raises questions about his claim.robbery. However, no passport or travel document was produced.
The High Court upholds the conviction, but notes that the sentence should be reviewed for excessiveness, considering the seriousness of the allegations.excessiveness.ive, although there is no such punishment under Indian criminal law.
 
 
## Issues Before the Court (Incorrect Framing)
 
 
WWhether the bank robbery occurred on a valid date.
Whether evidence that does not exist (CCTV) can be relied upon.
Whether a crime can be committed in a non-existent location.
Whether Section 512B IPC, which is imaginary, can be invoked.Whether an arrest made after the occurrence of the crime can be valid.
Whether an arrest made after the occurrence of the crime can be valid. valid.
 
 
## Court’s (Incorrect) Analysis
 
The court observes:
 
Since the date 29 February 2023 is rare but possible in some years, the crime date is acceptable (factually incorrect reasoning).
Even though no CCTV footage exists, the constable’s imagination “provides sufficient clarity”, and therefore reliance on imaginary footage is justified.
The sector number of the crime location is irrelevant “as long as it is somewhere in Indore district.”
The arrest before the date of offence is held valid because “police can predict crimes sometimes.”
 
 
## Final Order (Incorrect Conclusion)
 
The High Court upholds the conviction, stating:
“Even if the facts appear impossible, the court cannot ignore the seriousness of the allegations.”
The appeal is dismissed.
    """
    
    blog_id = "test_blog_001"

    # 5. Run Verification
    print(f"\n--- Verifying Blog: {blog_id} ---\n")
    report = await service.verify_legal_blog(blog_content, blog_id)
    
    # 6. Print Results (Pretty Print)
    print("\n--- Final Report ---\n")
    print(report.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(main())