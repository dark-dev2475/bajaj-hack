import logging
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
# Assumes your HierarchicalNode is in this file
from .parser import HierarchicalNode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HierarchicalEmbedder:
    """
    Embeds hierarchical leaf nodes and stores them in Pinecone,
    while ensuring all nodes are accessible in a document store.
    """
    def __init__(
        self,
        pinecone_api_key: str,
        index_name: str,
        # NEW: Pass in a document store that holds ALL nodes.
        docstore: Dict[str, HierarchicalNode],
        namespace: str = "default",
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 16
    ):
        self.index_name = index_name
        self.namespace = namespace
        self.batch_size = batch_size
        # NEW: Store the document store
        self.docstore = docstore

        # Initialize the embedding model
        logger.info(f"Loading embedding model {model_name}")
        self.embed_model = SentenceTransformer(model_name)
        
        # NEW: Get embedding dimension dynamically
        self.dimension = self.embed_model.get_sentence_embedding_dimension()

        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Get or create Pinecone index
        if index_name not in self.pc.list_indexes().names():
            logger.info(f"Creating new Pinecone index: {index_name} with dimension {self.dimension}")
            self.pc.create_index(
                name=index_name,
                dimension=self.dimension, # Use dynamic dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.index = self.pc.Index(index_name)

    def _create_pinecone_vectors(
        self,
        nodes: List[HierarchicalNode],
        embeddings: np.ndarray
    ) -> List[Dict]:
        """Create vector records for Pinecone insertion."""
        vectors = []
        for node, embedding in zip(nodes, embeddings):
            node_id = node.id_
            if not node_id:
                logger.warning("Node missing stable ID, skipping.")
                continue

            # --- CRITICAL FIX: Add parent_id to metadata ---
            parent_id = node.parent_node.node_id if node.parent_node else None
            
            # --- BEST PRACTICE: Don't store full text in vector metadata ---
            # The text can be retrieved from the docstore using the ID.
            metadata_for_pinecone = {
                # Copy existing metadata
                **node.metadata,
                # Add parent_id for the auto-merging retriever
                "parent_id": parent_id,
                # Remove redundant fields if they exist to save space
                "children": None,
                "parent": None
            }
            # Clean up any None values
            metadata_for_pinecone = {k: v for k, v in metadata_for_pinecone.items() if v is not None}

            vectors.append({
                "id": node_id,
                "values": embedding.tolist(),
                "metadata": metadata_for_pinecone
            })
        return vectors

    def embed_and_store(self, leaf_nodes: List[HierarchicalNode]):
        """Embeds ONLY leaf nodes and stores their vectors in Pinecone."""
        logger.info(f"Embedding and storing {len(leaf_nodes)} leaf nodes into Pinecone.")
        
        for i in tqdm(range(0, len(leaf_nodes), self.batch_size), desc="Embedding Batches"):
            batch_nodes = leaf_nodes[i:i + self.batch_size]
            # It's better to embed the full, untruncated content of the leaf node
            batch_texts = [node.text for node in batch_nodes]
            
            try:
                # Generate embeddings
                embeddings = self.embed_model.encode(
                    batch_texts,
                    normalize_embeddings=True
                )
                
                # Create Pinecone vector records with correct metadata
                vectors = self._create_pinecone_vectors(batch_nodes, embeddings)
                
                if not vectors:
                    continue

                # Upsert to Pinecone
                self.index.upsert(vectors=vectors, namespace=self.namespace)
            except Exception as e:
                logger.error(f"Error processing batch starting at index {i}: {e}")
                continue
        
        logger.info("Successfully embedded and stored all leaf nodes.")