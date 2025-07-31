import logging
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from .parser import HierarchicalNode

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
        batch_size: int = 10,  # Increased for faster processing
        max_text_length: int = 400  # Reduced for faster embedding
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
        self.max_text_length = max_text_length
        
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
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to maximum length while preserving complete sentences."""
        if len(text) <= self.max_text_length:
            return text.strip()
        
        # Clean up the text first
        text = text.strip()
        
        # Try to find a good sentence boundary within the limit
        truncated = text[:self.max_text_length]
        
        # Look for sentence endings in order of preference
        sentence_endings = ['. ', '! ', '? ']
        
        for delimiter in sentence_endings:
            last_pos = truncated.rfind(delimiter)
            if last_pos > self.max_text_length * 0.6:  # At least 60% of target length
                # Include the delimiter and return clean sentence
                result = truncated[:last_pos + 1].strip()
                # Remove any trailing incomplete parts
                if result and not result.endswith(('.', '!', '?')):
                    result += '.'
                return result
        
        # If no sentence boundary found, try paragraph breaks
        last_para = truncated.rfind('\n')
        if last_para > self.max_text_length * 0.5:
            result = truncated[:last_para].strip()
            if result and not result.endswith(('.', '!', '?')):
                result += '.'
            return result
        
        # Fallback: find last complete word and ensure proper ending
        words = truncated.split()
        if len(words) > 1:
            # Remove last word to avoid cutoffs
            result = ' '.join(words[:-1]).strip()
            # Ensure it ends properly
            if result and not result.endswith(('.', '!', '?')):
                result += '.'
            return result
        
        # Last resort: return truncated text with proper ending
        result = truncated.strip()
        if result and not result.endswith(('.', '!', '?')):
            result += '.'
        return result
    
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
            
            # Clean the text one more time before storing
            clean_text = text.strip()
            
            # Create vector record
            vector = {
                "id": f"node_{hash(clean_text)}",
                "values": normalized_embedding.tolist(),
                "metadata": {
                    **node.metadata,
                    "text": clean_text,
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
            batch_texts = [self._truncate_text(node.content) for node in batch_nodes]  # Truncate texts
            
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
        truncated_query = self._truncate_text(query)  # Truncate query too
        query_embedding = self.embed_model.encode(
            truncated_query,
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