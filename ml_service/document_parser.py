import os
import fitz  # PyMuPDF
import docx
import logging
from pathlib import Path
from langdetect import detect
from typing import Optional, List, Dict, Any
from openai import OpenAI, AsyncOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI clients
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)
async_client = AsyncOpenAI(api_key=OPENAI_KEY)

# --- Document Ingestion ---
def ingest_documents(specific_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extracts and cleans text from PDF or DOCX files.
    """
    documents = []
    file_paths = [Path(specific_file)] if specific_file else list(Path("./data").glob("*"))

    for file_path in file_paths:
        text = ""
        try:
            if file_path.suffix.lower() == ".pdf":
                with fitz.open(file_path) as doc:
                    text = "".join(page.get_text() for page in doc)
            elif file_path.suffix.lower() == ".docx":
                doc = docx.Document(file_path)
                text = "\n".join(para.text for para in doc.paragraphs)
            else:
                logging.warning(f"Unsupported file format: {file_path.name}")
                continue

            clean_text = " ".join(text.split())
            lang = detect(clean_text)

            documents.append({
                "source_file": file_path.name,
                "raw_text": clean_text,
                "language": lang
            })

        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {e}")
    
    logging.info(f"Ingested {len(documents)} document(s)")
    return documents

# --- Chunking ---
def chunk_documents(documents: list, chunk_size: int = 800, overlap: int = 100) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len
    )

    all_chunks = []
    for doc in documents:
        chunks = splitter.split_text(doc["raw_text"])
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "metadata": {
                    "source_file": doc["source_file"],
                    "language": doc["language"],
                    "chunk_number": i + 1
                },
                "chunk_text": chunk_text
            })

    logging.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks

# --- Embedding: Sync ---
def generate_embeddings(chunks_with_metadata: list) -> list:
    logging.info("Generating embeddings (sync)...")
    texts = [chunk["chunk_text"] for chunk in chunks_with_metadata]

    try:
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        for i, chunk in enumerate(chunks_with_metadata):
            chunk["embedding"] = response.data[i].embedding
        return chunks_with_metadata
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        return []

# --- Embedding: Async ---
async def generate_embeddings_async(chunks_with_metadata: list) -> list:
    logging.info("Generating embeddings (async)...")
    texts = [chunk["chunk_text"] for chunk in chunks_with_metadata]

    try:
        response = await async_client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        for i, chunk in enumerate(chunks_with_metadata):
            chunk["embedding"] = response.data[i].embedding
        return chunks_with_metadata
    except Exception as e:
        logging.error(f"Async embedding failed: {e}")
        return []

# --- Test Entry Point ---
if __name__ == "__main__":
    logging.info("Running document parser demo...")
    docs = ingest_documents()
    chunks = chunk_documents(docs)
    embedded = generate_embeddings(chunks)

    if embedded:
        logging.info(f"Example Chunk Metadata: {embedded[0]['metadata']}")
        logging.info(f"Example Chunk Preview: {embedded[0]['chunk_text'][:120]}...")
        logging.info(f"Embedding dimension: {len(embedded[0]['embedding'])}")
