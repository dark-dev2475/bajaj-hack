# client.py
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- PINECONE SETUP ----------------
import pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g., "gcp-starter" or "us-west1-gcp"

if not PINECONE_API_KEY or not PINECONE_ENV:
    logger.error("Missing Pinecone API key or environment in .env file")
    raise ValueError("Missing Pinecone configuration")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

logger.info("✅ Pinecone initialized successfully.")

# You can now use `pinecone.Index("your_index_name")` elsewhere.


# ---------------- GEMINI SETUP ----------------
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.error("Missing Gemini API key in .env file")
    raise ValueError("Missing Gemini API key")

genai.configure(api_key=GEMINI_API_KEY)

logger.info("✅ Gemini client configured.")

# Optional: Create a default Gemini model object to reuse
try:
    gemini_model = genai.GenerativeModel("gemini-flash")  # Or another model name
    logger.info("✅ Gemini model initialized.")
except Exception as e:
    logger.error(f"❌ Failed to initialize Gemini model: {e}")
    raise
