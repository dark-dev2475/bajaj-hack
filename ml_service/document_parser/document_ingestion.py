import logging
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF
import docx
from langdetect import detect

# This function processes just ONE file. It's our unit of work.
def process_single_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Processes a single document file."""
    logging.info(f"Processing {file_path.name}...")
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

        # The aggressive cleaning is still here, which might lose paragraph data.
        clean_text = " ".join(text.split())

        if not clean_text.strip():
            logging.warning(f"Skipping empty document: {file_path.name}")
            return None

        try:
            lang = detect(clean_text)
        except Exception:
            lang = "unknown"

        return {
            "source_file": file_path.name,
            "raw_text": clean_text,
            "language": lang
        }
    except Exception as e:
        logging.error(f"Error processing {file_path.name}: {e}")
        return None

# The async function now acts as a coordinator.
async def ingest_documents_parallel(specific_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extracts text from documents in parallel.
    """
    file_paths = [Path(specific_file)] if specific_file else list(Path("./data").glob("*"))
    if not file_paths:
        logging.info("No documents found.")
        return []
    
    # Create a separate "to_thread" task for each file.
    tasks = [asyncio.to_thread(process_single_file, fp) for fp in file_paths]
    
    # asyncio.gather runs all the tasks concurrently and waits for them to finish.
    results = await asyncio.gather(*tasks)
    
    # Filter out any files that failed (returned None).
    documents = [res for res in results if res is not None]
    
    logging.info(f"Successfully ingested {len(documents)} out of {len(file_paths)} document(s).")
    return documents

# Example of how to run it
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     # Assuming you have a 'data' folder with some files
#     documents = asyncio.run(ingest_documents_parallel())
#     print(f"Total documents ingested: {len(documents)}")