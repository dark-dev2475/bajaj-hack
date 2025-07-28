import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

# Import the main pipeline handler
from submission_handler.handler import handle_submission

# --- Configuration ---
# It's good practice to load these from environment variables for production
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "temp_docs")
INDEX_NAME = os.getenv("INDEX_NAME", "polisy-search")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Logging Setup ---
# Using a structured format can be helpful for log analysis tools
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Insurance RAG Pipeline API",
    description="An API to process insurance documents and answer questions using a RAG pipeline.",
    version="1.0.0"
)

# --- CORS Middleware ---
# For production, replace "*" with your specific frontend URL, e.g., ["https://your-frontend.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Request Models ---
class RAGRequest(BaseModel):
    # Corrected field name to match the handler's expectation
    doc_url: str
    questions: List[str]

# --- API Endpoints ---

@app.get("/", tags=["Health Check"])
async def read_root():
    """
    Root endpoint for health checks. Returns a simple status message.
    """
    return {"status": "Insurance RAG API is running"}


@app.post("/hackrx/run", tags=["RAG Pipeline"])
async def run_rag(data: RAGRequest) -> Dict[str, Any]:
    """
    Main endpoint to run the full RAG pipeline.
    - Downloads a document from a URL.
    - Ingests, chunks, and embeds the content into a vector store.
    - Answers a list of questions based on the document.
    - Cleans up all temporary resources.
    """
    if not data.questions or not isinstance(data.questions, list):
        raise HTTPException(status_code=400, detail="'questions' must be a non-empty list of strings.")

    logger.info(f"Received RAG request for doc: {data.doc_url} with {len(data.questions)} questions.")

    try:
        # Call the main orchestration handler
        answers = await handle_submission(data.doc_url, data.questions, UPLOAD_FOLDER, INDEX_NAME)
        
        logger.info(f"Successfully generated {len(answers)} answers for the request.")
        return {"answers": answers}
    
    except Exception as e:
        # Log the full exception traceback for easier debugging in production
        logger.exception(f"An unhandled error occurred in the RAG pipeline for doc: {data.doc_url}")
        raise HTTPException(status_code=500, detail="An internal server error occurred. Please check the logs.")

