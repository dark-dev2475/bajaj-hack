import os
import logging
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

# Import the main pipeline handler
from rag_pipeline2.handler import handle_rag_request
from rag_pipeline.pts_handler import handle_pts_rag_request

# --- Configuration ---
# It's good practice to load these from environment variables for production
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "temp_docs")
INDEX_NAME = os.getenv("INDEX_NAME", "policy-index")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Logging Setup ---
# This more advanced configuration ensures we see logs from all libraries.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# --- THIS IS THE FIX ---
# Force the langchain library to show INFO-level logs, which includes the verbose output.
logging.getLogger("langchain").setLevel(logging.INFO)
logging.getLogger("langchain_core").setLevel(logging.INFO)
# --- END OF FIX ---

logger = logging.getLogger(__name__)

# --- In-Memory Job Store (for demonstration) ---
# In a real production environment, use a persistent store like Redis or a database.
job_results: Dict[str, Any] = {}


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
    documents: str
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
    Main endpoint to run the full RAG pipeline with timeout handling.
    - Downloads a document from a URL.
    - Ingests, chunks, and embeds the content into a vector store.
    - Answers a list of questions based on the document.
    - Cleans up all temporary resources.
    """
    if not data.questions or not isinstance(data.questions, list):
        raise HTTPException(status_code=400, detail="'questions' must be a non-empty list of strings.")

    logger.info(f"Received RAG request for doc: {data.documents} with {len(data.questions)} questions.")

    try:
        # Set a timeout for the entire pipeline (5 minutes max)
        timeout_seconds = 300  # 5 minutes
        
        # Run the pipeline with timeout
        answers = await asyncio.wait_for(
            handle_rag_request(data.documents, data.questions, UPLOAD_FOLDER, INDEX_NAME),
            timeout=timeout_seconds
        )
        
        logger.info(f"Successfully generated {len(answers)} answers for the request.")
        return {"answers": answers}
    
    except asyncio.TimeoutError:
        logger.error(f"RAG pipeline timed out after {timeout_seconds} seconds for doc: {data.documents}")
        raise HTTPException(
            status_code=408, 
            detail=f"Request timed out after {timeout_seconds} seconds. Try with a smaller document or fewer questions."
        )
    
    except Exception as e:
        # Log the full exception traceback for easier debugging in production
        logger.exception(f"An unhandled error occurred in the RAG pipeline for doc: {data.documents}")
        raise HTTPException(status_code=500, detail="An internal server error occurred. Please check the logs.")

