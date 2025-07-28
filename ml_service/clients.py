# ml_service/clients.py — Client Initialization Module

import os
import logging
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import AsyncOpenAI

# Load environment variables from .env
load_dotenv()

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