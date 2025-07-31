import asyncio
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PTSEmbedder:
    """Parent to Sentence Embedder - Handles embedding generation and storage for PTS nodes."""
    
    def __init__(
        self,
        pinecone_api_key: str,
        pinecone_environment: str,
        index_name: str,
        namespace: str = "pts_default",
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32
    ):
        """
        Initialize the PTS Embedder.
        
        Args:
            pinecone_api_key: Pinecone API key
            pinecone_environment: Pinecone environment
            index_name: Name of the Pinecone index
            namespace: Namespace within the index
            model_name: HuggingFace model name for embeddings
            batch_size: Batch size for embedding generation
        """
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_environment = pinecone_environment
        self.index_name = index_name
        self.namespace = namespace
        self.batch_size = batch_size
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Get or create Pinecone index
        dimension = 384  # BGE-small dimension
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        self.index = self.pc.Index(index_name)
        
        # Initialize the embedding model
        logger.info(f"Loading embedding model {model_name}")
        self.embed_model = SentenceTransformer(model_name)
        
        logger.info("PTSEmbedder initialized successfully")
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def _create_pinecone_vectors(
        self,
        sentence_nodes: List[Dict[str, Any]],
        embeddings: np.ndarray
    ) -> List[Dict]:
        """Create vector records for Pinecone insertion."""
        vectors = []
        for node_data, embedding in zip(sentence_nodes, embeddings):
            # Normalize the embedding
            normalized_embedding = self._normalize_vector(embedding)
            
            # Create vector record
            vector = {
                "id": node_data["id"],
                "values": normalized_embedding.tolist(),
                "metadata": {
                    "text": node_data["text"],
                    "parent_id": node_data.get("parent_id", ""),
                    "node_type": "sentence",
                    "model": "bge-small-en-v1.5",
                    **node_data.get("metadata", {})
                }
            }
            vectors.append(vector)
        return vectors
    
    async def embed_sentence_nodes(
        self, 
        sentence_nodes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Embed sentence nodes and store them in Pinecone.
        
        Args:
            sentence_nodes: List of sentence node dictionaries with 'id', 'text', 'parent_id'
            
        Returns:
            Dictionary with embedding results
        """
        logger.info(f"Embedding and storing {len(sentence_nodes)} sentence nodes")
        
        total_stored = 0
        failed_batches = 0
        
        # Process in batches
        for i in tqdm(range(0, len(sentence_nodes), self.batch_size)):
            batch_nodes = sentence_nodes[i:i + self.batch_size]
            batch_texts = [node["text"] for node in batch_nodes]
            
            try:
                # Generate embeddings
                embeddings = self.embed_model.encode(
                    batch_texts,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                
                # Create Pinecone vector records
                vectors = self._create_pinecone_vectors(batch_nodes, embeddings)
                
                # Upsert to Pinecone
                self.index.upsert(
                    vectors=vectors,
                    namespace=self.namespace
                )
                
                total_stored += len(vectors)
                
            except Exception as e:
                logger.error(f"Error processing batch {i}: {str(e)}")
                failed_batches += 1
                continue
        
        logger.info(f"Successfully embedded {total_stored} sentence nodes")
        
        return {
            "total_processed": len(sentence_nodes),
            "total_stored": total_stored,
            "failed_batches": failed_batches,
            "status": "success" if failed_batches == 0 else "partial_success"
        }
    
    async def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Normalized embedding vector
        """
        try:
            embedding = self.embed_model.encode(text, normalize_embeddings=True)
            return self._normalize_vector(embedding)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    async def clear_namespace(self) -> bool:
        """
        Clear all vectors from the namespace.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
            logger.info(f"Successfully cleared namespace: {self.namespace}")
            return True
        except Exception as e:
            logger.error(f"Error clearing namespace: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "namespace_stats": stats.namespaces,
                "dimension": stats.dimension
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {"error": str(e)}

# Factory function for easy creation
def create_pts_embedder(
    pinecone_api_key: str,
    pinecone_environment: str,
    index_name: str,
    **kwargs
) -> PTSEmbedder:
    """
    Create a PTS Embedder instance.
    
    Args:
        pinecone_api_key: Pinecone API key
        pinecone_environment: Pinecone environment
        index_name: Pinecone index name
        **kwargs: Additional arguments for PTSEmbedder
        
    Returns:
        PTSEmbedder instance
    """
    return PTSEmbedder(
        pinecone_api_key=pinecone_api_key,
        pinecone_environment=pinecone_environment,
        index_name=index_name,
        **kwargs
    )
