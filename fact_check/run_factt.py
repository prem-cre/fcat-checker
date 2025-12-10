import sys
import json
import os
import asyncio
import logging
from dotenv import load_dotenv
from pathlib import Path

# --- 1. SETUP LOGGING (To see API outputs) ---
# This makes sure you see the "Step 1...", "Found 5 results..." messages in the terminal
logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG for even more detail
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FactCheckRunner")

# --- 2. SETUP PATHS ---
# Ensure we can import the modules in the current directory
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Load .env file
load_dotenv()

# --- 3. SAMPLE INPUT ---
sample_text = ('''
IN THE HIGH COURT OF MIRZAPUR (Criminal Appellate Jurisdiction)
Case Title: State of Madhya Pradesh vs. Rohit Sharma Case No.: CRA/1023/2024

The court observes: Since the date of the alleged crime does not exist in 2023, the crime date is not acceptable, however this is factually incorrect as 29 February 2023 is a valid date. 
The accused Rohit Sharma robbed the Central Bank of India branch located in Sector 97, Indore (Sector 97 does not exist in Indore).
The trial court relied on invisible CCTV footage, as the footage was never produced in court but was described verbally by a constable.
The arrest before the date of offence is held valid because "police can predict crimes sometimes."
''')

async def main():
    # 4. GET KEYS
    google_key = os.environ.get("GOOGLE_API_KEY")
    cse_id = os.environ.get("GOOGLE_CSE_ID")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    academic_key = os.environ.get("SEMANTIC_SCHOLAR_KEY")

    if not google_key or not cse_id or not gemini_key:
        logger.error("❌ MISSING API KEYS. Please check your .env file.")
        return

    try:
        # 5. IMPORT SERVICE
        # Using simple import since we are in the same folder
        from fact_check import LegalFactCheckService
        
        # 6. INITIALIZE SERVICE
        logger.info("🚀 Initializing Legal Fact Check Service...")
        service = LegalFactCheckService(
            google_api_key=google_key,
            cse_id=cse_id,
            gemini_api_key=gemini_key,
            academic_key=academic_key,
            llm_model="gemini-2.5-flash-lite"
        )

        # 7. RUN VERIFICATION
        logger.info("🔍 Starting Verification Pipeline...")
        report = await service.verify_legal_blog(
            blog_content=sample_text, 
            blog_id='test_run_001'
        )

        # 8. PRINT RESULT
        print("\n" + "="*50)
        print("✅ FINAL JSON REPORT")
        print("="*50)
        
        # Convert Pydantic model to JSON string safely
        json_output = report.model_dump_json(indent=2)
        print(json_output)

    except ImportError as e:
        logger.error(f"❌ Import Error: {e}")
        logger.error("Make sure 'fact_check.py', 'models.py', etc. are in the same folder.")
    except Exception as e:
        logger.exception(f"❌ Execution Failed: {e}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())