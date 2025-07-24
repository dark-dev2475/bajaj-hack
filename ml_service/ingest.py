import os
import logging
import pinecone
import document_parser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- ONE-TIME INGESTION PIPELINE (DAY 2 & 3) ---
def run_ingestion_pipeline():
    """
    Runs the full one-time data ingestion pipeline.
    """
    INDEX_NAME = "polisy-search"
    
    # Initialize Pinecone client
    pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    logging.info("--- Starting Data Ingestion Pipeline ---")
    
    ingested_docs = document_parser.ingest_documents()
    if not ingested_docs:
        logging.warning("No documents found in 'data' folder.")
        return

    chunked_docs = document_parser.chunk_documents(ingested_docs)
    final_data = document_parser.generate_embeddings(chunked_docs)

    logging.info(f"Uploading {len(final_data)} vectors to Pinecone index '{INDEX_NAME}'...")
    index = pc.Index(INDEX_NAME)
    index.upsert(vectors=[{
        "id": f"chunk_{i}",
        "values": item["embedding"],
        "metadata": {"text": item["chunk_text"], "source": item["metadata"]["source_file"]}
    } for i, item in enumerate(final_data)], batch_size=100)
    
    logging.info("--- Data Ingestion Pipeline Complete ---")

if __name__ == "__main__":
    run_ingestion_pipeline()