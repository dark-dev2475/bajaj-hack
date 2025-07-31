from concurrent.futures import ThreadPoolExecutor
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
import logging
import os

# --- Client Initialization ---
# The SemanticChunker needs an embedding model to understand the text's meaning.
# Ensure your OPENAI_API_KEY is set as an environment variable.
try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
except Exception as e:
    logging.error(f"Failed to initialize OpenAIEmbeddings. Ensure OPENAI_API_KEY is set. Error: {e}")
    # Fallback to a dummy or handle error as needed
    embeddings = None


def _chunk_single_document(doc: dict):
    """
    Chunks a single document based on semantic meaning using SemanticChunker.
    
    Args:
        doc: A dictionary containing the document's raw_text and metadata.

    Returns:
        A list of chunk dictionaries, or an empty list if chunking fails.
    """
    if not embeddings:
        logging.error("Embeddings client not available. Cannot perform semantic chunking.")
        return []
        
    # Initialize the SemanticChunker.
    # It uses the embedding model to find "natural" breakpoints in the text.
    # 'percentile' is a good starting point for the threshold type.
    semantic_splitter = SemanticChunker(
        embeddings, breakpoint_threshold_type="percentile"
    )

    # The splitter returns a list of strings.
    chunks = semantic_splitter.split_text(doc["raw_text"])

    # Re-structure the output to include metadata with each chunk.
    return [
        {
            "metadata": {
                "source_file": doc.get("source_file", "unknown"),
                "language": doc.get("language", "unknown"),
                "chunk_number": i + 1
            },
            "chunk_text": chunk_text
        }
        for i, chunk_text in enumerate(chunks)
    ]


def chunk_documents_parallel(documents: list) -> list:
    """
    Chunks a list of documents in parallel using the semantic chunking strategy.

    Args:
        documents: A list of document dictionaries.

    Returns:
        A flat list of all chunks from all documents.
    """
    all_chunks = []
    # Using ThreadPoolExecutor to run the CPU-bound chunking tasks in parallel.
    with ThreadPoolExecutor() as executor:
        # Submit each document to be chunked as a separate job.
        futures = [executor.submit(_chunk_single_document, doc) for doc in documents]
        
        for future in futures:
            try:
                # Safely get the result from each future.
                result = future.result()
                if result:
                    all_chunks.extend(result)
            except Exception as e:
                # Log any error that occurs during a chunking task and continue.
                logging.exception(f"Failed to process a document for chunking: {e}")

    logging.info(f"Total semantic chunks created: {len(all_chunks)}")
    return all_chunks

