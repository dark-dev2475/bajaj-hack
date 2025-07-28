import logging
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF
import docx
from langdetect import detect, lang_detect_exception

# This function processes just ONE file. It's our unit of work.
def process_single_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Processes a single document file."""
    logging.info(f"Starting processing for: {file_path.name}")
    try:
        text = ""
        if file_path.suffix.lower() == ".pdf":
            with fitz.open(file_path) as doc:
                text = "".join(page.get_text() for page in doc)
        elif file_path.suffix.lower() == ".docx":
            doc = docx.Document(file_path)
            text = "\n".join(para.text for para in doc.paragraphs)
        else:
            logging.warning(f"Unsupported file format: {file_path.name}")
            return None

        # --- ADDED LOG: Confirm text was extracted ---
        logging.info(f"Extracted {len(text)} characters from {file_path.name}.")

        # The aggressive cleaning is still here, which might lose paragraph data.
        clean_text = " ".join(text.split())

        if not clean_text.strip():
            logging.warning(f"Skipping empty document: {file_path.name}")
            return None

        try:
            lang = detect(clean_text)
        except lang_detect_exception.LangDetectException:
            lang = "unknown"

        logging.info(f"Successfully processed {file_path.name}.")
        return {
            "source_file": file_path.name,
            "raw_text": clean_text,
            "language": lang
        }
    except Exception as e:
        # --- IMPROVED LOG: Use .exception() to get the full traceback ---
        logging.exception(f"An unexpected error occurred while processing {file_path.name}: {e}")
        return None

# The async function now acts as a coordinator.
async def ingest_documents_parallel(specific_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extracts text from documents in parallel.
    """
    file_paths = [Path(specific_file)] if specific_file else list(Path("./data").glob("*"))
    if not file_paths:
        logging.warning("No documents found in the target directory.")
        return []
    
    # --- ADDED LOG: Confirm which files are being processed ---
    logging.info(f"Found {len(file_paths)} files to ingest: {[fp.name for fp in file_paths]}")
    
    # Create a separate "to_thread" task for each file.
    tasks = [asyncio.to_thread(process_single_file, fp) for fp in file_paths]
    
    # asyncio.gather runs all the tasks concurrently and waits for them to finish.
    results = await asyncio.gather(*tasks)
    
    # Filter out any files that failed (returned None).
    documents = [res for res in results if res is not None]
    
    logging.info(f"Successfully ingested {len(documents)} out of {len(file_paths)} document(s).")
    return documents