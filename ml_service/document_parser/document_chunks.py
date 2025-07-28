from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging


def _chunk_single_document(doc, chunk_size, overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len
    )
    chunks = splitter.split_text(doc["raw_text"])

    return [
        {
            "metadata": {
                "source_file": doc["source_file"],
                "language": doc["language"],
                "chunk_number": i + 1
            },
            "chunk_text": chunk
        }
        for i, chunk in enumerate(chunks)
    ]

def chunk_documents_parallel(documents: list, chunk_size: int = 800, overlap: int = 100) -> list:
    all_chunks = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_chunk_single_document, doc, chunk_size, overlap) for doc in documents]
        for future in futures:
            try:
                # Safely get the result from each future
                all_chunks.extend(future.result())
            except Exception as e:
                # Log the error and continue with the other documents
                logging.error(f"Failed to chunk a document: {e}")

    logging.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks
