import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pinecone
from sentence_transformers import SentenceTransformer
from parser import HierarchicalNode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseRetriever:
    """Base retriever that performs similarity search on Pinecone."""
    
    def __init__(
        self,
        pinecone_api_key: str,
        pinecone_environment: str,
        index_name: str,
        namespace: str = "default",
        model_name: str = "BAAI/bge-small-en-v1.5",
        similarity_top_k: int = 3
    ):
        """
        Initialize the base retriever.
        
        Args:
            pinecone_api_key: Pinecone API key
            pinecone_environment: Pinecone environment
            index_name: Name of the Pinecone index
            namespace: Namespace to search in
            model_name: HuggingFace model for query embedding
            similarity_top_k: Number of similar nodes to retrieve
        """
        self.namespace = namespace
        self.similarity_top_k = similarity_top_k
        
        # Initialize Pinecone
        pinecone.init(
            api_key=pinecone_api_key,
            environment=pinecone_environment
        )
        self.index = pinecone.Index(index_name)
        
        # Initialize embedding model
        self.embed_model = SentenceTransformer(model_name)
        logger.info(f"Base retriever initialized with top_k={similarity_top_k}")
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve similar nodes based on query.
        
        Args:
            query: Query string
            
        Returns:
            List of retrieved nodes with metadata
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
            top_k=self.similarity_top_k,
            include_metadata=True
        )
        
        # Format results
        retrieved_nodes = []
        for match in results.matches:
            node_data = {
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "metadata": match.metadata,
                "level": match.metadata.get("level", 0),
                "chunk_size": match.metadata.get("chunk_size", 256)
            }
            retrieved_nodes.append(node_data)
        
        logger.info(f"Retrieved {len(retrieved_nodes)} nodes for query")
        return retrieved_nodes

class AutoMergingRetriever:
    """
    Auto-merging retriever that combines retrieved nodes based on similarity threshold.
    Mimics the LlamaIndex AutoMergingRetriever functionality.
    """
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        simple_ratio_thresh: float = 0.0,
        storage_context: Optional[Dict] = None,
        verbose: bool = True
    ):
        """
        Initialize the auto-merging retriever.
        
        Args:
            base_retriever: Base retriever instance
            simple_ratio_thresh: Threshold for merging nodes (0.0 means merge all)
            storage_context: Storage context (for compatibility)
            verbose: Whether to print verbose logs
        """
        self.base_retriever = base_retriever
        self.simple_ratio_thresh = simple_ratio_thresh
        self.storage_context = storage_context
        self.verbose = verbose
        
        logger.info(f"AutoMergingRetriever initialized with threshold={simple_ratio_thresh}")
    
    def _should_merge_nodes(self, node1: Dict, node2: Dict) -> bool:
        """
        Determine if two nodes should be merged based on similarity threshold.
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            Whether nodes should be merged
        """
        # If threshold is 0.0, merge all nodes (as in your notebook)
        if self.simple_ratio_thresh == 0.0:
            return True
        
        # Check if nodes are from the same source and adjacent
        same_source = (
            node1["metadata"].get("source") == node2["metadata"].get("source")
        )
        
        # Check score similarity
        score_diff = abs(node1["score"] - node2["score"])
        score_similarity = 1.0 - score_diff
        
        should_merge = same_source and score_similarity > self.simple_ratio_thresh
        
        if self.verbose and should_merge:
            logger.info(f"Merging nodes: score_similarity={score_similarity:.3f}")
        
        return should_merge
    
    def _merge_nodes(self, nodes: List[Dict]) -> List[Dict]:
        """
        Merge similar nodes based on the threshold.
        
        Args:
            nodes: List of retrieved nodes
            
        Returns:
            List of merged nodes
        """
        if not nodes or self.simple_ratio_thresh == 0.0:
            # If threshold is 0.0, combine all text (like in notebook)
            if nodes:
                merged_text = " ".join([node["text"] for node in nodes])
                merged_node = {
                    **nodes[0],  # Take metadata from first node
                    "text": merged_text,
                    "merged_count": len(nodes),
                    "original_scores": [node["score"] for node in nodes]
                }
                if self.verbose:
                    logger.info(f"Merged {len(nodes)} nodes into single context")
                return [merged_node]
            return nodes
        
        # Group nodes that should be merged
        merged_nodes = []
        used_indices = set()
        
        for i, node1 in enumerate(nodes):
            if i in used_indices:
                continue
                
            merged_group = [node1]
            used_indices.add(i)
            
            for j, node2 in enumerate(nodes[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                if self._should_merge_nodes(node1, node2):
                    merged_group.append(node2)
                    used_indices.add(j)
            
            # Create merged node
            if len(merged_group) > 1:
                merged_text = " ".join([node["text"] for node in merged_group])
                merged_node = {
                    **merged_group[0],
                    "text": merged_text,
                    "merged_count": len(merged_group),
                    "original_scores": [node["score"] for node in merged_group]
                }
            else:
                merged_node = merged_group[0]
                merged_node["merged_count"] = 1
            
            merged_nodes.append(merged_node)
        
        if self.verbose:
            logger.info(f"Auto-merged {len(nodes)} nodes into {len(merged_nodes)} contexts")
        
        return merged_nodes
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve and auto-merge similar nodes.
        
        Args:
            query: Query string
            
        Returns:
            List of merged retrieved nodes
        """
        if self.verbose:
            logger.info(f"AutoMergingRetriever processing query: '{query[:50]}...'")
        
        # Get base retrieval results
        base_results = self.base_retriever.retrieve(query)
        
        # Apply auto-merging
        merged_results = self._merge_nodes(base_results)
        
        if self.verbose:
            logger.info(f"Final result: {len(merged_results)} merged contexts")
        
        return merged_results

# Factory function to create retriever (matches your notebook pattern)
def create_auto_merging_retriever(
    pinecone_api_key: str,
    pinecone_environment: str,
    index_name: str,
    namespace: str = "default",
    similarity_top_k: int = 3,
    simple_ratio_thresh: float = 0.0,
    storage_context: Optional[Dict] = None,
    verbose: bool = True
) -> AutoMergingRetriever:
    """
    Create an AutoMergingRetriever (equivalent to your notebook logic).
    
    Your notebook:
        base_retriever = base_index.as_retriever(similarity_top_k=3)
        retriever = AutoMergingRetriever(base_retriever, simple_ratio_thresh=0.0, 
                                       storage_context=storage_context, verbose=True)
    
    Equivalent:
        retriever = create_auto_merging_retriever(
            pinecone_api_key="your-key",
            pinecone_environment="your-env", 
            index_name="your-index",
            similarity_top_k=3,
            simple_ratio_thresh=0.0,
            verbose=True
        )
    """
    # Create base retriever (equivalent to base_index.as_retriever())
    base_retriever = BaseRetriever(
        pinecone_api_key=pinecone_api_key,
        pinecone_environment=pinecone_environment,
        index_name=index_name,
        namespace=namespace,
        similarity_top_k=similarity_top_k
    )
    
    # Create auto-merging retriever
    retriever = AutoMergingRetriever(
        base_retriever=base_retriever,
        simple_ratio_thresh=simple_ratio_thresh,
        storage_context=storage_context,
        verbose=verbose
    )
    
    return retriever
