# ml_service/clients.py — Client Initialization Module

import os
import logging
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import AsyncOpenAI
# from openai import AsyncOpenAI
from openai import OpenAI


# Load environment variables from .env
load_dotenv()

# Retrieve API keys and environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV      = os.getenv("PINECONE_ENVIRONMENT")  # e.g. "aws-us-west-2"
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Validate presence
if not PINECONE_API_KEY:
    logging.error("FATAL: PINECONE_API_KEY not found in environment or .env.")
    raise KeyError("PINECONE_API_KEY is required")
if not OPENAI_API_KEY:
    logging.error("FATAL: OPENAI_API_KEY not found in environment or .env.")
    raise KeyError("OPENAI_API_KEY is required")

# Initialize Pinecone client once
try:
    logging.info(f"Initializing Pinecone client in env '{PINECONE_ENV}'...")
    pinecone_client = Pinecone(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
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
