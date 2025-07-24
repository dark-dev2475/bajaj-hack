# ml_service/main.py
import logging
from query_parser import main_parser as query_parser
import search
from answer import answer_generator # Import our new answer generator module

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to run the full RAG query pipeline.
    """
    INDEX_NAME = "polisy-search"
    raw_query = "I'm 32 years old and I fractured my arm at home in Lucknow, is it covered under my personal accident policy?"
    logging.info(f"Processing query: \"{raw_query}\"")

    # Day 4: Parse query
    structured_query = query_parser.get_structured_query(raw_query)
    if not structured_query: return
    print("\n--- Structured Query (Day 4) ---")
    print(structured_query.model_dump_json(indent=2))

    # Day 5: Retrieve relevant documents
    search_results = search.perform_search(
        raw_query=raw_query,
        structured_query=structured_query,
        index_name=INDEX_NAME
    )
    if not search_results:
        print("No relevant documents found.")
        return
    print("\n--- Retrieved Documents (Day 5) ---")
    for r in search_results: print(f"  - Source: {r['metadata']['source']}, Score: {r['score']:.4f}")

    # Day 6: Generate final answer
    final_answer = answer_generator.generate_answer(raw_query, search_results)
    if not final_answer:
        print("Could not generate a final answer from the LLM.")
        return
        
    print("\n--- Final Answer (Day 6) ---")
    print(final_answer.model_dump_json(indent=2))

if __name__ == "__main__":
    # Ensure your data is already ingested by running ingest.py separately
    main()