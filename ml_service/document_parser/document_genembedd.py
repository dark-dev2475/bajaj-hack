import logging

def generate_embeddings(chunks_with_metadata: list, batch_size=100) -> list:
    logging.info("Generating embeddings in batches...")

    embedded_chunks = []

    for i in range(0, len(chunks_with_metadata), batch_size):
        batch = chunks_with_metadata[i:i+batch_size]
        texts = [chunk["chunk_text"] for chunk in batch]

        try:
            response = client.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
            for j, chunk in enumerate(batch):
                chunk["embedding"] = response.data[j].embedding
                embedded_chunks.append(chunk)

        except Exception as e:
            logging.error(f"Embedding batch {i//batch_size} failed: {e}")

    logging.info(f"Total embedded chunks: {len(embedded_chunks)}")
    return embedded_chunks
