# ml_service/main.py

import document_parser

def run_ingestion_pipeline():
    """
    Runs the full document ingestion pipeline by calling the functions
    from the document_parser module.
    """
    # [cite_start]Step 1: Ingest raw documents from files [cite: 1]
    ingested_docs = document_parser.ingest_documents()

    if not ingested_docs:
        print("No documents found in the 'data' folder. Please add some sample files and try again.")
        return

    # [cite_start]Step 2: Chunk the documents into smaller pieces [cite: 1]
    chunked_docs = document_parser.chunk_documents(ingested_docs)
    #  # --- NEW CODE TO VIEW CHUNKS ---
    # print("\n--- Displaying Generated Chunks ---")
    # for i, chunk in enumerate(chunked_docs):
    #     print(f"--- Chunk {i+1} ---")
    #     # Print the metadata tagged with each chunk for later justification
    #     print(f"Metadata: {chunk['metadata']}")
    #     print("Text:")
    #     print(chunk['chunk_text'])
    #     print("-" * 20) # Separator for readability
    # # --- END OF NEW CODE ---

    # [cite_start]Step 3: Generate embeddings for each chunk [cite: 1]
    final_data_with_embeddings = document_parser.generate_embeddings(chunked_docs)

     # --- NEW CODE TO VIEW A SAMPLE EMBEDDING ---
    if final_data_with_embeddings:
        print("\n--- Inspecting a Sample Embedding ---")
        first_chunk = final_data_with_embeddings[0]
        embedding_vector = first_chunk.get("embedding")
        
        if embedding_vector:
            print(f"Full embedding dimension: {len(embedding_vector)}")
            print(f"First 5 values of the vector: {embedding_vector[:5]}")
        else:
            print("Embedding vector not found for the first chunk.")
    # --- END OF NEW CODE ---

    print("\n--- Day 2 Complete ---")
    print(f"Successfully processed {len(final_data_with_embeddings)} chunks.")
    print("This data is now ready to be loaded into a vector database on Day 3.")

if __name__ == "__main__":
    run_ingestion_pipeline()