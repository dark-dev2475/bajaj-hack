import logging
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm
import pinecone
from sentence_transformers import SentenceTransformer
from parser import HierarchicalNode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HierarchicalEmbedder:
    """Embeds hierarchical nodes using BGE embeddings and stores them in Pinecone."""
    
    def __init__(
        self,
        pinecone_api_key: str,
        pinecone_environment: str,
        index_name: str,
        namespace: str = "default",
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32
    ):
        """
        Initialize the embedder with Pinecone and HuggingFace settings.
        
        Args:
            pinecone_api_key: Pinecone API key
            pinecone_environment: Pinecone environment
            index_name: Name of the Pinecone index
            namespace: Namespace within the index
            model_name: HuggingFace model name for embeddings
            batch_size: Batch size for embedding generation
        """
        self.index_name = index_name
        self.namespace = namespace
        self.batch_size = batch_size
        
        # Initialize Pinecone
        pinecone.init(
            api_key=pinecone_api_key,
            environment=pinecone_environment
        )
        
        # Get or create Pinecone index
        dimension = 384  # BGE-small dimension
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine"
            )
        self.index = pinecone.Index(index_name)
        
        # Initialize the embedding model
        logger.info(f"Loading embedding model {model_name}")
        self.embed_model = SentenceTransformer(model_name)
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def _create_pinecone_vectors(
        self,
        nodes: List[HierarchicalNode],
        batch_texts: List[str],
        embeddings: np.ndarray
    ) -> List[Dict]:
        """Create vector records for Pinecone insertion."""
        vectors = []
        for node, text, embedding in zip(nodes, batch_texts, embeddings):
            # Normalize the embedding
            normalized_embedding = self._normalize_vector(embedding)
            
            # Create vector record
            vector = {
                "id": f"node_{hash(text)}",
                "values": normalized_embedding.tolist(),
                "metadata": {
                    **node.metadata,
                    "text": text,
                    "level": node.level,
                    "model": "bge-small-en-v1.5"
                }
            }
            vectors.append(vector)
        return vectors
    
    async def embed_and_store(self, leaf_nodes: List[HierarchicalNode]) -> None:
        """
        Embed leaf nodes and store them in Pinecone.
        
        Args:
            leaf_nodes: List of leaf nodes to embed and store
        """
        logger.info(f"Embedding and storing {len(leaf_nodes)} leaf nodes")
        
        # Process in batches
        for i in tqdm(range(0, len(leaf_nodes), self.batch_size)):
            batch_nodes = leaf_nodes[i:i + self.batch_size]
            batch_texts = [node.content for node in batch_nodes]
            
            try:
                # Generate embeddings
                embeddings = self.embed_model.encode(
                    batch_texts,
                    normalize_embeddings=True  # Ensure unit length
                )
                
                # Create Pinecone vector records
                vectors = self._create_pinecone_vectors(
                    batch_nodes,
                    batch_texts,
                    embeddings
                )
                
                # Upsert to Pinecone
                self.index.upsert(
                    vectors=vectors,
                    namespace=self.namespace
                )
                
            except Exception as e:
                logger.error(f"Error processing batch {i}: {str(e)}")
                continue
        
        logger.info("Successfully embedded and stored all leaf nodes")
    
    def similarity_search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search using the query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of matching documents with scores
        """
        # Generate query embedding
        query_embedding = self.embed_model.encode(
            query,
            normalize_embeddings=True
        )
        
        # Search Pinecone
        results = self.index.query(
            vector=query_embedding.tolist(),
            namespace=self.namespace,
            top_k=top_k,
            include_metadata=True
        )
        
        return [
            {
                "score": match.score,
                "text": match.metadata["text"],
                "metadata": match.metadata
            }
            for match in results.matches
        ]
