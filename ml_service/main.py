import document_parser
import os
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import logging
from query_parser.main_parser import get_structured_query

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize clients
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def upload_to_pinecone(data: list, index_name: str):
    print(f"\n--- 4. Uploading to Pinecone Index '{index_name}' ---")
    index = pc.Index(index_name)
    vectors_to_upsert = []
    for i, item in enumerate(data):
        vectors_to_upsert.append({
            "id": f"chunk_{i}",
            "values": item["embedding"],
            "metadata": {
                "text": item["chunk_text"],
                "source": item["metadata"]["source_file"]
            }
        })
    index.upsert(vectors=vectors_to_upsert, batch_size=100)
    print("  > Upload complete.")

def test_query_parser():
    """
    Tests the structured query extraction functionality using the new parser.
    """
    sample_query = "I have a personal accident policy. I'm 32 years old. Yesterday, I fell down the stairs at my home in Lucknow and fractured my arm."
    
    logging.info(f"Original Query: \"{sample_query}\"")
    structured_query = get_structured_query(sample_query)
    
    if structured_query:
        print("\n--- Structured Query Output (JSON) ---")
        print(structured_query.model_dump_json(indent=2))
    else:
        print("\n--- Failed to parse the query ---")

def run_ingestion_pipeline():
    index_name = "polisy-search"
    ingested_docs = document_parser.ingest_documents()
    if not ingested_docs: return
    chunked_docs = document_parser.chunk_documents(ingested_docs)
    final_data = document_parser.generate_embeddings(chunked_docs)
    upload_to_pinecone(final_data, index_name)

    # Test query parsing after upload
    test_query_parser()

# ✅ Single entry point
if __name__ == "__main__":
    run_ingestion_pipeline()
