import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import aiofiles
import uvicorn
import asyncio
from submission_handler.handler import handle_submission as submission_handler
from document_parser.document_ingestion import ingest_documents
from document_parser.document_chunks import chunk_documents
from document_parser.document_genembedd import generate_embeddings
import search

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
    document_url: str
    questions: List[str]

# --- Endpoint 1: Run RAG ---
@app.post("/hackrx/run")
async def run_rag(data: RAGRequest):
    if not data.questions or not isinstance(data.questions, list):
        raise HTTPException(status_code=400, detail="'questions' must be a non-empty list of strings.")

    logging.info(f"Received RAG request for doc: {data.document_url} with {len(data.questions)} questions.")

    try:
        answers = await submission_handler(data.document_url, data.questions, UPLOAD_FOLDER,INDEX_NAME)
        return {"answers": answers}
    except Exception as e:
        logging.error(f"[RAG Error] {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# --- Endpoint 2: Ingest File ---
@app.post("/ingest-file")
async def ingest_file(document: UploadFile = File(...)):
    logging.info(f"Received file: {document.filename} of size: {document.size or 'unknown'}")

    if not document.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    filename = os.path.basename(document.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        # Save uploaded file
        async with aiofiles.open(save_path, 'wb') as out_file:
            content = await document.read()
            if len(content) > MAX_CONTENT_SIZE:
                raise HTTPException(status_code=413, detail="File too large")
            await out_file.write(content)

        logging.info(f"Saved file: {filename}")

        # Step 1: Parse
        ingested_docs = ingest_documents(specific_file=save_path)
        if not ingested_docs:
            raise ValueError("No documents parsed.")

        # Step 2: Chunk
        chunked_docs = chunk_documents(ingested_docs)
        if not chunked_docs:
            raise ValueError("No chunks created.")

        # Step 3: Embeddings
        final_data = generate_embeddings(chunked_docs)
        if not final_data or not isinstance(final_data, list):
            raise ValueError("Embedding generation failed or returned wrong format.")

        # Step 4: Pinecone Upsert
        vectors = [{
            "id": f"{filename}_chunk_{i}",
            "values": item["embedding"],
            "metadata": {"text": item["chunk_text"], "source": filename}
        } for i, item in enumerate(final_data)]

        if not vectors or any(isinstance(v["values"], str) for v in vectors):
            raise ValueError("Invalid vector format")

        search.pc.Index(INDEX_NAME).upsert(vectors=vectors, batch_size=100)
        logging.info(f"Uploaded {len(vectors)} vectors to Pinecone.")

        return {
            "message": f"Successfully processed {filename}",
            "chunks_uploaded": len(vectors)
        }

    except Exception as e:
        logging.error(f"[Ingest Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

# Run using: uvicorn main:app --reload --port 5000
