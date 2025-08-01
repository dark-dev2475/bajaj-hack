# /rag_pipeline/embedder.py

import logging
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
from pinecone import Pinecone

# LlamaIndex Imports
from llama_index.core import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HierarchicalEmbedder:
    """Embeds leaf nodes and stores them in Pinecone."""

    def __init__(self, pinecone_api_key: str, index_name: str, namespace: str = "default"):
        self.namespace = namespace
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        
        # --- FIX: Use the globally configured embed_model ---
        self.embed_model = Settings.embed_model
        if not self.embed_model:
            raise ValueError("Embedding model not found in LlamaIndex Settings. Please configure it first.")
        
        dimension = len(self.embed_model.get_text_embedding("test"))
        
        if index_name not in self.pc.list_indexes().names():
            logger.info(f"Creating new Pinecone index: {index_name} with dimension {dimension}")
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )
        self.index = self.pc.Index(index_name)

    # ... (the rest of your embedder class is the same) ...
    def _create_pinecone_vectors(self, nodes: List[Any], embeddings: np.ndarray) -> List[Dict]:
        # ... (implementation is correct)
        vectors = []
        for node, embedding in zip(nodes, embeddings):
            parent_id = node.parent_node.node_id if node.parent_node else None
            
            metadata_for_pinecone = {
                **node.metadata,
                "parent_id": parent_id,
                "text": node.text 
            }
            metadata_for_pinecone = {k: v for k, v in metadata_for_pinecone.items() if v is not None}

            vectors.append({
                "id": node.id_,
                "values": embedding.tolist(),
                "metadata": metadata_for_pinecone
            })
        return vectors

    def embed_and_store(self, leaf_nodes: List[Any], batch_size: int = 32):
        # ... (implementation is correct)
        logger.info(f"Embedding and storing {len(leaf_nodes)} leaf nodes into Pinecone.")
        
        for i in tqdm(range(0, len(leaf_nodes), batch_size), desc="Embedding Batches"):
            batch_nodes = leaf_nodes[i:i + batch_size]
            batch_texts = [node.text for node in batch_nodes]
            
            try:
                embeddings = self.embed_model.get_text_embedding_batch(batch_texts, show_progress=False)
                vectors = self._create_pinecone_vectors(batch_nodes, np.array(embeddings))
                
                if vectors:
                    self.index.upsert(vectors=vectors, namespace=self.namespace)
            except Exception as e:
                logger.error(f"Error processing batch starting at index {i}: {e}")
                continue
        
        logger.info("Successfully embedded and stored all leaf nodes.")