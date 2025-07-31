# ml_service/clients.py — Client Initialization Module

import os
import logging
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import AsyncOpenAI

# Load environment variables from .env
load_dotenv()
# --- NEW IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# --- END NEW IMPORTS ---

# Retrieve API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")

# Validate presence of required API keys
if not PINECONE_API_KEY:
    logging.error("FATAL: PINECONE_API_KEY not found in environment or .env.")
    raise KeyError("PINECONE_API_KEY is required")
if not OPENAI_API_KEY:
    logging.error("FATAL: OPENAI_API_KEY not found in environment or .env.")
    raise KeyError("OPENAI_API_KEY is required")

# --- Initialize Clients ---
pinecone_client = None
openai_async_client = None



# --- NEW: Gemini Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Validate presence of the key
if not GOOGLE_API_KEY:
    logging.error("FATAL: GOOGLE_API_KEY not found in environment or .env.")
    raise KeyError("GOOGLE_API_KEY is required")

# --- Initialize Clients ---
gemini_embeddings = None
gemini_llm = None

# Initialize Gemini Embeddings client once
try:
    # The recommended model for free-tier embeddings
    logging.info("Initializing Gemini embeddings client...")
    gemini_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )
    logging.info("Gemini embeddings client initialized successfully.")
except Exception as e:
    logging.error(f"FATAL: Unable to initialize Gemini embeddings client: {e}")
    raise

# Initialize Gemini LLM client once
try:
    # Gemini 1.5 Flash is the fastest and most cost-effective model
    logging.info("Initializing Gemini LLM client...")
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        convert_system_message_to_human=True # Helps with compatibility
    )
    logging.info("Gemini LLM client initialized successfully.")
except Exception as e:
    logging.error(f"FATAL: Unable to initialize Gemini LLM client: {e}")
    raise

# Initialize Pinecone client once
try:
    logging.info("Initializing Pinecone client...")
    # The 'environment' parameter is deprecated in modern clients
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    logging.info("Pinecone client initialized successfully.")
except Exception as e:
    logging.error(f"FATAL: Unable to initialize Pinecone client: {e}")
    raise

# Initialize Async OpenAI client once
try:
    logging.info("Initializing OpenAI async client...")
    openai_async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    logging.info("OpenAI async client initialized successfully.")
except Exception as e:
    logging.error(f"FATAL: Unable to initialize OpenAI client: {e}")
    raise