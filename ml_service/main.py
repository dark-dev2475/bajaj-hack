import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import aiofiles
import uvicorn

from submission_handler.handler import handle_submission as submission_handler

# Configuration
UPLOAD_FOLDER = 'temp_docs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MAX_CONTENT_SIZE = 100 * 1024 * 1024  # 100MB
INDEX_NAME = "polisy-search"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema for /hackrx/run
class RAGRequest(BaseModel):
    documents: str
    questions: List[str]

# --- Endpoint 1: Run RAG ---
@app.post("/hackrx/run")
async def run_rag(data: RAGRequest):
    if not data.questions or not isinstance(data.questions, list):
        raise HTTPException(status_code=400, detail="'questions' must be a non-empty list of strings.")

    logging.info(f"Received RAG request for docs: {data.documents} with {len(data.questions)} questions.")

    try:
        # --- LOG 1: Log the inputs to the handler ---
        logging.info(f"Calling submission_handler with index_name: '{INDEX_NAME}' and upload_folder: '{UPLOAD_FOLDER}'")
        
        answers = await submission_handler(data.documents, data.questions, UPLOAD_FOLDER, INDEX_NAME)

        # --- LOG 2: Log the summary of the successful result ---
        logging.info(f"Successfully generated {len(answers)} answers.")
        
        return {"answers": answers}
    
    except Exception as e:
        # --- LOG 3 (CRITICAL): Log the full exception traceback ---
        logging.exception(f"An unhandled error occurred in the RAG pipeline: {e}")
        
        raise HTTPException(status_code=500, detail="An internal server error occurred.")