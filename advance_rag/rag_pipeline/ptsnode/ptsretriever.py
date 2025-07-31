import asyncio
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from .ptsembedder import PTSEmbedder
from .ptsvector import PTSVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PTSRetriever:
    """Parent to Sentence Retriever - Implements recursive retrieval algorithm."""
    
    def __init__(
        self,
        embedder: PTSEmbedder,
        vector_store: PTSVectorStore,
        parent_nodes_dict: Dict[str, Any],
        similarity_top_k: int = 5,
        parent_retrieval_factor: float = 1.5
    ):
        """
        Initialize the PTS Retriever.
        
        Args:
            embedder: PTSEmbedder instance for query embedding
            vector_store: PTSVectorStore instance for vector operations
            parent_nodes_dict: Dictionary mapping parent IDs to parent nodes
            similarity_top_k: Number of sentence nodes to retrieve
            parent_retrieval_factor: Factor to expand parent node retrieval
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.parent_nodes_dict = parent_nodes_dict
        self.similarity_top_k = similarity_top_k
        self.parent_retrieval_factor = parent_retrieval_factor
        
        logger.info(f"PTSRetriever initialized with top_k={similarity_top_k}")
    
    async def retrieve_sentence_nodes(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve sentence nodes based on query.
        
        Args:
            query: Query string
            top_k: Number of results (uses class default if None)
            filter_dict: Optional metadata filter
            
        Returns:
            List of sentence node results with scores
        """
        if top_k is None:
            top_k = self.similarity_top_k
        
        try:
            # Generate query embedding
            query_embedding = await self.embedder.get_embedding(query)
            query_vector = query_embedding.tolist()
            
            # Retrieve similar sentence nodes
            sentence_results = await self.vector_store.query_vectors(
                query_vector=query_vector,
                top_k=top_k,
                filter_dict=filter_dict,
                include_metadata=True
            )
            
            logger.info(f"Retrieved {len(sentence_results)} sentence nodes for query")
            return sentence_results
            
        except Exception as e:
            logger.error(f"Error retrieving sentence nodes: {str(e)}")
            return []
    
    async def retrieve_parent_nodes(
        self,
        sentence_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve parent nodes from sentence results using recursive algorithm.
        
        Args:
            sentence_results: List of sentence node results
            
        Returns:
            List of parent node results with aggregated scores
        """
        try:
            # Group sentence nodes by parent ID
            parent_groups = {}
            for result in sentence_results:
                parent_id = result.get("parent_id", "")
                if parent_id and parent_id in self.parent_nodes_dict:
                    if parent_id not in parent_groups:
                        parent_groups[parent_id] = []
                    parent_groups[parent_id].append(result)
            
            # Create parent results with aggregated scores
            parent_results = []
            for parent_id, sentence_group in parent_groups.items():
                parent_node = self.parent_nodes_dict[parent_id]
                
                # Calculate aggregated score (max, mean, or weighted)
                scores = [result["score"] for result in sentence_group]
                aggregated_score = max(scores)  # Use max score as primary indicator
                avg_score = sum(scores) / len(scores)
                
                parent_result = {
                    "id": parent_id,
                    "text": getattr(parent_node, 'text', str(parent_node)),
                    "score": aggregated_score,
                    "avg_score": avg_score,
                    "sentence_count": len(sentence_group),
                    "sentence_scores": scores,
                    "metadata": getattr(parent_node, 'metadata', {}),
                    "node": parent_node
                }
                
                parent_results.append(parent_result)
            
            # Sort by aggregated score
            parent_results.sort(key=lambda x: x["score"], reverse=True)
            
            logger.info(f"Retrieved {len(parent_results)} parent nodes from recursive algorithm")
            return parent_results
            
        except Exception as e:
            logger.error(f"Error retrieving parent nodes: {str(e)}")
            return []
    
    async def retrieve(
        self,
        query: str,
        return_type: str = "parent",
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Main retrieval method implementing recursive retrieval algorithm.
        
        Args:
            query: Query string
            return_type: "sentence", "parent", or "both"
            top_k: Number of results
            filter_dict: Optional metadata filter
            
        Returns:
            List of retrieved nodes based on return_type
        """
        try:
            # Step 1: Retrieve sentence nodes
            sentence_results = await self.retrieve_sentence_nodes(
                query=query,
                top_k=top_k,
                filter_dict=filter_dict
            )
            
            if not sentence_results:
                return []
            
            if return_type == "sentence":
                return sentence_results
            
            # Step 2: Retrieve parent nodes using recursive algorithm
            parent_results = await self.retrieve_parent_nodes(sentence_results)
            
            if return_type == "parent":
                return parent_results
            
            elif return_type == "both":
                return {
                    "sentence_nodes": sentence_results,
                    "parent_nodes": parent_results
                }
            
            else:
                raise ValueError(f"Invalid return_type: {return_type}")
                
        except Exception as e:
            logger.error(f"Error in retrieve method: {str(e)}")
            return []
    
    async def batch_retrieve(
        self,
        queries: List[str],
        return_type: str = "parent",
        top_k: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve results for multiple queries in batch.
        
        Args:
            queries: List of query strings
            return_type: "sentence", "parent", or "both"
            top_k: Number of results per query
            
        Returns:
            List of result lists for each query
        """
        try:
            batch_results = []
            
            # Process each query
            for query in queries:
                results = await self.retrieve(
                    query=query,
                    return_type=return_type,
                    top_k=top_k
                )
                batch_results.append(results)
            
            logger.info(f"Completed batch retrieval for {len(queries)} queries")
            return batch_results
            
        except Exception as e:
            logger.error(f"Error in batch retrieve: {str(e)}")
            return []
    
    async def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retrieval system.
        
        Returns:
            Dictionary with retrieval statistics
        """
        try:
            # Get vector store stats
            vector_stats = await self.vector_store.get_namespace_stats()
            
            # Get parent nodes count
            parent_count = len(self.parent_nodes_dict)
            
            return {
                "sentence_nodes_count": vector_stats.get("vector_count", 0),
                "parent_nodes_count": parent_count,
                "similarity_top_k": self.similarity_top_k,
                "parent_retrieval_factor": self.parent_retrieval_factor,
                "index_dimension": vector_stats.get("dimension", 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting retrieval stats: {str(e)}")
            return {"error": str(e)}
    
    def update_parent_nodes_dict(self, new_parent_nodes_dict: Dict[str, Any]) -> None:
        """
        Update the parent nodes dictionary.
        
        Args:
            new_parent_nodes_dict: New parent nodes dictionary
        """
        self.parent_nodes_dict = new_parent_nodes_dict
        logger.info(f"Updated parent nodes dictionary with {len(new_parent_nodes_dict)} nodes")
    
    def set_similarity_top_k(self, new_top_k: int) -> None:
        """
        Update the similarity top_k parameter.
        
        Args:
            new_top_k: New top_k value
        """
        self.similarity_top_k = new_top_k
        logger.info(f"Updated similarity_top_k to {new_top_k}")

# Factory function for easy creation
def create_pts_retriever(
    embedder: PTSEmbedder,
    vector_store: PTSVectorStore,
    parent_nodes_dict: Dict[str, Any],
    **kwargs
) -> PTSRetriever:
    """
    Create a PTS Retriever instance.
    
    Args:
        embedder: PTSEmbedder instance
        vector_store: PTSVectorStore instance
        parent_nodes_dict: Dictionary of parent nodes
        **kwargs: Additional arguments for PTSRetriever
        
    Returns:
        PTSRetriever instance
    """
    return PTSRetriever(
        embedder=embedder,
        vector_store=vector_store,
        parent_nodes_dict=parent_nodes_dict,
        **kwargs
    )
