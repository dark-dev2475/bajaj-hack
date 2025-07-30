from concurrent.futures import ThreadPoolExecutor
import logging
import os
import re
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings

# Initialize OpenAI embeddings
from clients import openai_async_client

# Constants for chunking
CHUNK_SIZE = 100          # Target chunk size
CHUNK_OVERLAP = 20        # Overlap between chunks
MIN_CHUNK_LENGTH = 50      # Minimum chunk size to process

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_embeddings():
    """Initialize OpenAI embeddings"""
    if not openai_async_client:
        raise RuntimeError("OpenAI client not configured. Please set up OpenAI credentials.")
    
    try:
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            client=openai_async_client
        )
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI embeddings: {e}")
        raise

try:
    embeddings = get_embeddings()
except Exception as e:
    logging.error(f"Failed to initialize OpenAI embeddings: {e}")
    embeddings = None


def _chunk_single_document(doc: dict) -> List[Dict[str, Any]]:
    """
    Chunks a single document using smart text splitting with paragraph awareness.
    
    Args:
        doc: A dictionary containing the document's raw_text and metadata.

    Returns:
        A list of chunk dictionaries, or an empty list if chunking fails.
    """
    if not doc.get("raw_text"):
        logging.error("Document has no raw_text field")
        return []

    text = doc["raw_text"]
    
    # Clean the text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        # If paragraph is too long, split it further
        if len(para) > CHUNK_SIZE:
            # Add any existing chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split long paragraph into sentences
            sentences = [s.strip() for s in re.split(r'[.!?]+', para) if s.strip()]
            
            temp_chunk = []
            temp_length = 0
            
            for sentence in sentences:
                if temp_length + len(sentence) > CHUNK_SIZE:
                    if temp_chunk:  # Save current temp chunk
                        chunks.append(' '.join(temp_chunk))
                    temp_chunk = [sentence]
                    temp_length = len(sentence)
                else:
                    temp_chunk.append(sentence)
                    temp_length += len(sentence)
            
            if temp_chunk:  # Save any remaining sentences
                chunks.append(' '.join(temp_chunk))
                
        # Normal paragraph handling
        elif current_length + len(para) > CHUNK_SIZE:
            if current_chunk:  # Save current chunk
                chunks.append(' '.join(current_chunk))
            current_chunk = [para]
            current_length = len(para)
        else:
            current_chunk.append(para)
            current_length += len(para)
    
    # Add any remaining text
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Filter out any chunks that are too small
    chunks = [chunk for chunk in chunks if len(chunk) >= MIN_CHUNK_LENGTH]
    
    logging.info(f"Created {len(chunks)} chunks from document")
    
    # Format chunks with metadata
    return [
        {
            "metadata": {
                "source_file": doc.get("source_file", "unknown"),
                "language": doc.get("language", "unknown"),
                "chunk_number": i + 1,
                "total_chunks": len(chunks)
            },
            "chunk_text": chunk_text
        }
        for i, chunk_text in enumerate(chunks)
    ]



def chunk_documents_parallel(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Chunks a list of documents in parallel using smart text splitting.

    Args:
        documents: A list of document dictionaries, each containing raw_text and metadata.

    Returns:
        A flat list of all chunks from all documents with metadata.
    """
    if not documents:
        logging.warning("No documents provided for chunking")
        return []

    logging.info(f"Starting parallel chunking of {len(documents)} documents")
    
    # Filter out any empty documents
    valid_docs = [
        doc for doc in documents 
        if doc.get("raw_text") and len(doc["raw_text"].strip()) > MIN_CHUNK_LENGTH
    ]
    
    if len(valid_docs) < len(documents):
        logging.warning(f"Filtered out {len(documents) - len(valid_docs)} empty or invalid documents")
    
    if not valid_docs:
        logging.error("No valid documents to process after filtering")
        return []

    all_chunks = []
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

