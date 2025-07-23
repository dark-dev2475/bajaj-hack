# ml_service/main.py

import document_parser
import os
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# Initialize clients (ensure environment variables are set)
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
            "metadata": {"text": item["chunk_text"], "source": item["metadata"]["source_file"]}
        })
    index.upsert(vectors=vectors_to_upsert, batch_size=100)
    print("  > Upload complete.")

def test_query(index_name: str):
    """
    Performs a test query against the vector DB to confirm relevance.
    """
    print("\n--- 5. Testing Query ---")
    index = pc.Index(index_name)

    # Create an embedding for a test query
    query_text = "what was turning point of john?"
    query_embedding = openai_client.embeddings.create(
        input=[query_text],
        model="text-embedding-3-small"
    ).data[0].embedding

    # Query the index
    results = index.query(
        vector=query_embedding,
        top_k=3,  # Get the top 3 most relevant results
        include_metadata=True
    )

    print(f"Query: '{query_text}'")
    print("Results:")
    for match in results['matches']:
        print(f"  - Score: {match['score']:.4f}")
        print(f"    Source: {match['metadata']['source']}")
        print(f"    Text: {match['metadata']['text']}\n")


def run_ingestion_pipeline():
    index_name = "polisy-search"
    ingested_docs = document_parser.ingest_documents()
    if not ingested_docs: return
    chunked_docs = document_parser.chunk_documents(ingested_docs)
    final_data = document_parser.generate_embeddings(chunked_docs)
    upload_to_pinecone(final_data, index_name)

    # Test a query after uploading
    test_query(index_name)

if __name__ == "__main__":
    run_ingestion_pipeline()
