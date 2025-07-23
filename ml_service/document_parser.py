# ml_service/document_parser.py
import fitz  # PyMuPDF
import docx
from pathlib import Path
from langdetect import detect

def ingest_documents():
    """
    Reads all documents from the 'data' folder, extracts text,
    and identifies the language.
    """
    data_path = Path("./data")
    documents = []
    
    for file_path in data_path.glob("*"):
        raw_text = ""
        if file_path.suffix == ".pdf":
            with fitz.open(file_path) as doc:
                raw_text = "".join(page.get_text() for page in doc)
        elif file_path.suffix == ".docx":
            doc = docx.Document(file_path)
            raw_text = "\n".join(para.text for para in doc.paragraphs)
        
        if raw_text:
            # Normalize text by removing extra whitespace
            clean_text = " ".join(raw_text.split())
            
            # Identify language per doc 
            lang = detect(clean_text)
            
            documents.append({
                "source_file": str(file_path.name),
                "raw_text": clean_text,
                "language": lang
            })
            print(f"Successfully ingested {file_path.name} ({lang})")

    return documents

if __name__ == "__main__":
    ingested_docs = ingest_documents()
    print(f"\nTotal documents ingested: {len(ingested_docs)}")
    # Print a snippet of the first document's text
    if ingested_docs:
        print(f"Snippet from {ingested_docs[0]['source_file']}:")
        print(ingested_docs[0]['raw_text'][:500] + "...")



# Add this to ml_service/document_parser.py
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents: list) -> list:
    """
    Takes ingested documents and splits them into smaller, overlapping chunks
    with associated metadata.
    """
    # This splitter tries to split on paragraphs, then sentences, etc.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Aim for chunks of this many characters 
        chunk_overlap=100, # Overlap chunks to preserve context 
        length_function=len,
    )
    
    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["raw_text"])
        
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                # Add metadata to each chunk 
                "metadata": {
                    "source_file": doc["source_file"],
                    "language": doc["language"],
                    "chunk_number": i + 1
                },
                "chunk_text": chunk_text
            })
    
    print(f"\nTotal chunks created: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    ingested_docs = ingest_documents()
    chunked_docs = chunk_documents(ingested_docs)
    
    if chunked_docs:
        print("\n--- Example Chunk ---")
        print(chunked_docs[0])


# Add this to ml_service/document_parser.py
import os
from openai import OpenAI

# It's best to set your API key as an environment variable
# export OPENAI_API_KEY='your-key-here'
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_embeddings(chunks_with_metadata: list) -> list:
    """
    Computes embeddings for each chunk using the chosen OpenAI model.
    """
    print("\n--- Generating Embeddings ---")
    
    # Extract just the text for the API call
    texts_to_embed = [item["chunk_text"] for item in chunks_with_metadata]
    
    # Make the API call to OpenAI
    response = client.embeddings.create(
        input=texts_to_embed,
        model="text-embedding-3-small" # Or your chosen model
    )
    
    # Add the resulting embedding vector to each chunk object
    for i, item in enumerate(chunks_with_metadata):
        item["embedding"] = response.data[i].embedding
        
    print("Embeddings generated successfully.")
    return chunks_with_metadata

if __name__ == "__main__":
    ingested_docs = ingest_documents()
    chunked_docs = chunk_documents(ingested_docs)
    final_data = generate_embeddings(chunked_docs)
    
    if final_data:
        print("\n--- Example of Final Data Point (ready for Vector DB) ---")
        # We won't print the whole embedding vector as it's very long
        print(f"Metadata: {final_data[0]['metadata']}")
        print(f"Text: {final_data[0]['chunk_text'][:100]}...")
        print(f"Embedding Vector Dimension: {len(final_data[0]['embedding'])}")        