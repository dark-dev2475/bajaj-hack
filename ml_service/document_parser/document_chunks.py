import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from clients import client
# --- Chunking ---
def chunk_documents(documents: list, chunk_size: int = 800, overlap: int = 100) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len
    )

    all_chunks = []
    for doc in documents:
        chunks = splitter.split_text(doc["raw_text"])
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "metadata": {
                    "source_file": doc["source_file"],
                    "language": doc["language"],
                    "chunk_number": i + 1
                },
                "chunk_text": chunk_text
            })

    logging.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks