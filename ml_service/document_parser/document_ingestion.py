import logging
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF
import docx
from langdetect import detect
from clients import client
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)

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


async def ingest_documents_async(specific_file: Optional[str] = None) -> List[Dict[str, Any]]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, ingest_documents, specific_file)
