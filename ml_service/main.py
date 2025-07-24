import os
import logging
import pinecone
from openai import OpenAI

# Import our custom query parser

from query_parser.main_parser import get_structured_query

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- QUERY PROCESSING & SEARCH PIPELINE (DAY 4 & 5) ---
def process_and_search(index_name: str, raw_query: str):
    """
    Processes a raw query and searches the vector DB.
    """
    # Initialize clients
    pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    logging.info("--- Starting Query Processing & Search ---")
    
    # Step 1: Parse the raw query to get structured data (Day 4)
    structured_query = get_structured_query(raw_query)
    if structured_query:
        print("\n--- Structured Query (from Day 4) ---")
        print(structured_query.model_dump_json(indent=2))
    else:
        logging.error("Failed to parse query.")
        return

    # Step 2: Perform semantic search on the vector DB (Day 5)
    logging.info("Performing semantic search with the original query...")
    index = pc.Index(index_name)
    
    query_embedding = openai_client.embeddings.create(
        input=[raw_query],
        model="text-embedding-3-small"
    ).data[0].embedding
    
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    
    print("\n--- Semantic Search Results (from Day 5) ---")
    for match in results['matches']:
        print(f"  - Score: {match['score']:.4f}")
        print(f"    Source: {match['metadata']['source']}")
        print(f"    Text: {match['metadata']['text']}\n")


if __name__ == "__main__":
    INDEX_NAME = "polisy-search"
    
    # Define the query you want to test
    test_query = "I'm 32 years old and I fractured my arm at home in Lucknow, is it covered under my personal accident policy?"
    
    # Run the query processing and search
    process_and_search(index_name=INDEX_NAME, raw_query=test_query)