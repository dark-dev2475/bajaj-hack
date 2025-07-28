import logging
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF
import docx
from langdetect import detect, lang_detect_exception
import re # Import the regular expression module

# This function processes just ONE file. It's our unit of work.
def process_single_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Processes a single document file with improved text cleaning."""
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

        logging.info(f"Extracted {len(text)} characters from {file_path.name}.")

        # --- IMPROVED TEXT CLEANING ---
        # 1. Replace multiple newlines with a single one.
        clean_text = re.sub(r'\n\s*\n', '\n', text)
        # 2. Replace multiple spaces with a single space.
        clean_text = re.sub(r'[ \t]+', ' ', clean_text)
        # 3. Remove leading/trailing whitespace from the whole text.
        clean_text = clean_text.strip()
        # --- END OF IMPROVEMENT ---

        if not clean_text:
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
        logging.exception(f"An unexpected error occurred while processing {file_path.name}: {e}")
        return None

# The async function now acts as a coordinator.
async def ingest_documents_parallel(specific_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extracts text from documents in parallel using improved cleaning.
    """
    file_paths = [Path(specific_file)] if specific_file else list(Path("./data").glob("*"))
    if not file_paths:
        logging.warning("No documents found in the target directory.")
        return []
    
    logging.info(f"Found {len(file_paths)} files to ingest: {[fp.name for fp in file_paths]}")
    
    tasks = [asyncio.to_thread(process_single_file, fp) for fp in file_paths]
    
    results = await asyncio.gather(*tasks)
    
    documents = [res for res in results if res is not None]
    
    logging.info(f"Successfully ingested {len(documents)} out of {len(file_paths)} document(s).")
    return documents
