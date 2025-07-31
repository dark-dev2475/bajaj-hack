import asyncio
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PTSVectorStore:
    """Parent to Sentence Vector Store - Handles vector storage operations."""
    
    def __init__(
        self,
        pinecone_api_key: str,
        pinecone_environment: str,
        index_name: str,
        namespace: str = "pts_default"
    ):
        """
        Initialize the PTS Vector Store.
        
        Args:
            pinecone_api_key: Pinecone API key
            pinecone_environment: Pinecone environment
            index_name: Name of the Pinecone index
            namespace: Namespace within the index
        """
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_environment = pinecone_environment
        self.index_name = index_name
        self.namespace = namespace
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
        
        logger.info(f"PTSVectorStore initialized for index: {index_name}, namespace: {namespace}")
    
    async def store_vectors(
        self,
        vectors: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Store vectors in Pinecone.
        
        Args:
            vectors: List of vector dictionaries with 'id', 'values', 'metadata'
            
        Returns:
            Storage result dictionary
        """
        try:
            # Upsert vectors
            upsert_response = self.index.upsert(
                vectors=vectors,
                namespace=self.namespace
            )
            
            logger.info(f"Successfully stored {len(vectors)} vectors")
            
            return {
                "stored_count": len(vectors),
                "upserted_count": upsert_response.upserted_count,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error storing vectors: {str(e)}")
            return {
                "stored_count": 0,
                "error": str(e),
                "status": "failed"
            }
    
    async def query_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query vectors from Pinecone.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            include_metadata: Whether to include metadata
            
        Returns:
            List of matching vectors with scores
        """
        try:
            query_response = self.index.query(
                vector=query_vector,
                namespace=self.namespace,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=include_metadata
            )
            
            results = []
            for match in query_response.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                }
                if include_metadata and match.metadata:
                    result["metadata"] = match.metadata
                    result["text"] = match.metadata.get("text", "")
                    result["parent_id"] = match.metadata.get("parent_id", "")
                
                results.append(result)
            
            logger.info(f"Retrieved {len(results)} vectors for query")
            return results
            
        except Exception as e:
            logger.error(f"Error querying vectors: {str(e)}")
            return []
    
    async def get_vector_by_id(
        self,
        vector_id: str,
        include_metadata: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific vector by ID.
        
        Args:
            vector_id: Vector ID to retrieve
            include_metadata: Whether to include metadata
            
        Returns:
            Vector data or None if not found
        """
        try:
            fetch_response = self.index.fetch(
                ids=[vector_id],
                namespace=self.namespace
            )
            
            if vector_id in fetch_response.vectors:
                vector_data = fetch_response.vectors[vector_id]
                result = {
                    "id": vector_id,
                    "values": vector_data.values
                }
                
                if include_metadata and vector_data.metadata:
                    result["metadata"] = vector_data.metadata
                    result["text"] = vector_data.metadata.get("text", "")
                    result["parent_id"] = vector_data.metadata.get("parent_id", "")
                
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching vector {vector_id}: {str(e)}")
            return None
    
    async def delete_vectors(
        self,
        vector_ids: Optional[List[str]] = None,
        delete_all: bool = False,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Delete vectors from the namespace.
        
        Args:
            vector_ids: List of specific vector IDs to delete
            delete_all: Whether to delete all vectors in namespace
            filter_dict: Optional metadata filter for deletion
            
        Returns:
            Deletion result dictionary
        """
        try:
            if delete_all:
                self.index.delete(delete_all=True, namespace=self.namespace)
                logger.info(f"Deleted all vectors from namespace: {self.namespace}")
                return {"status": "success", "action": "delete_all"}
            
            elif vector_ids:
                self.index.delete(ids=vector_ids, namespace=self.namespace)
                logger.info(f"Deleted {len(vector_ids)} specific vectors")
                return {"status": "success", "action": "delete_ids", "count": len(vector_ids)}
            
            elif filter_dict:
                self.index.delete(filter=filter_dict, namespace=self.namespace)
                logger.info("Deleted vectors matching filter")
                return {"status": "success", "action": "delete_filter"}
            
            else:
                return {"status": "error", "message": "No deletion criteria provided"}
                
        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_namespace_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the namespace.
        
        Returns:
            Namespace statistics dictionary
        """
        try:
            stats = self.index.describe_index_stats()
            namespace_stats = stats.namespaces.get(self.namespace, {})
            
            return {
                "namespace": self.namespace,
                "vector_count": namespace_stats.get("vector_count", 0),
                "total_index_vectors": stats.total_vector_count,
                "dimension": stats.dimension
            }
            
        except Exception as e:
            logger.error(f"Error getting namespace stats: {str(e)}")
            return {"error": str(e)}
    
    async def batch_query_vectors(
        self,
        query_vectors: List[List[float]],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Query multiple vectors in batch.
        
        Args:
            query_vectors: List of query embedding vectors
            top_k: Number of results per query
            filter_dict: Optional metadata filter
            
        Returns:
            List of result lists for each query
        """
        try:
            batch_results = []
            
            for query_vector in query_vectors:
                results = await self.query_vectors(
                    query_vector=query_vector,
                    top_k=top_k,
                    filter_dict=filter_dict
                )
                batch_results.append(results)
            
            logger.info(f"Completed batch query for {len(query_vectors)} vectors")
            return batch_results
            
        except Exception as e:
            logger.error(f"Error in batch query: {str(e)}")
            return []

# Factory function for easy creation
def create_pts_vector_store(
    pinecone_api_key: str,
    pinecone_environment: str,
    index_name: str,
    **kwargs
) -> PTSVectorStore:
    """
    Create a PTS Vector Store instance.
    
    Args:
        pinecone_api_key: Pinecone API key
        pinecone_environment: Pinecone environment
        index_name: Pinecone index name
        **kwargs: Additional arguments for PTSVectorStore
        
    Returns:
        PTSVectorStore instance
    """
    return PTSVectorStore(
        pinecone_api_key=pinecone_api_key,
        pinecone_environment=pinecone_environment,
        index_name=index_name,
        **kwargs
    )