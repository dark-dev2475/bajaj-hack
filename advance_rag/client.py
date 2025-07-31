# client.py
import os
import logging
from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec
import pinecone as Pinecone
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- PINECONE SETUP ----------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g., "gcp-starter" or "us-west1-gcp"

if not PINECONE_API_KEY or not PINECONE_ENV:
    logger.error("Missing Pinecone API key or environment in .env file")
    raise ValueError("Missing Pinecone configuration")

# ✅ New Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Optional: check if connected or list indexes
try:
    index_names = pc.list_indexes().names()
    logger.info(f"✅ Pinecone connected. Existing indexes: {index_names}")
except Exception as e:
    logger.error(f"❌ Pinecone connection failed: {e}")
    raise

# ---------------- GEMINI SETUP ----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.error("Missing Gemini API key in .env file")
    raise ValueError("Missing Gemini API key")

genai.configure(api_key=GEMINI_API_KEY)

# Optional: Create default Gemini model object
try:
    gemini_model = genai.GenerativeModel("gemini-flash")
    logger.info("✅ Gemini model initialized.")
except Exception as e:
    logger.error(f"❌ Failed to initialize Gemini model: {e}")
    raise
