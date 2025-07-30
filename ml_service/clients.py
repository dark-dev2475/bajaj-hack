# ml_service/clients.py — Client Initialization Module

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import AsyncOpenAI

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Retrieve API keys with fallbacks
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate presence of required API keys
if not PINECONE_API_KEY:
    logging.error("FATAL: PINECONE_API_KEY not found in environment or .env.")
    raise KeyError("PINECONE_API_KEY is required")
if not OPENAI_API_KEY:
    logging.error("FATAL: OPENAI_API_KEY not found in environment or .env.")
    raise KeyError("OPENAI_API_KEY is required")

# --- Initialize Clients ---
pinecone_client: Optional[Pinecone] = None
openai_async_client: Optional[AsyncOpenAI] = None

# Validate required API keys
def validate_api_keys():
    """Validate the presence of required API keys"""
    if not PINECONE_API_KEY:
        raise KeyError("PINECONE_API_KEY is required in .env")
    
    if not OPENAI_API_KEY:
        raise KeyError("OPENAI_API_KEY is required in .env")
    
    logger.info("API key validation successful")

# Initialize clients
def initialize_clients():
    """Initialize Pinecone and OpenAI clients"""
    global pinecone_client, openai_async_client
    
    # Initialize Pinecone
    try:
        logger.info("Initializing Pinecone client...")
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        logger.info("✓ Pinecone client initialized successfully")
    except Exception as e:
        logger.error(f"FATAL: Unable to initialize Pinecone client: {e}")
        raise

    # Initialize OpenAI
    try:
        logger.info("Initializing OpenAI async client...")
        openai_async_client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            timeout=30.0,  # Increased timeout for reliability
            max_retries=3  # Added retries for better reliability
        )
        logger.info("✓ OpenAI async client initialized successfully")
    except Exception as e:
        logger.error(f"FATAL: Unable to initialize OpenAI client: {e}")
        raise

# Run initialization
validate_api_keys()
initialize_clients()