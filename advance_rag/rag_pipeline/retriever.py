import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# LlamaIndex imports for true auto-merging
try:
    from llama_index.core.retrievers import AutoMergingRetriever as LlamaAutoMergingRetriever
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# This class remains useful as a fallback
class BasicRetriever:
    """A basic retriever that concatenates text from retrieved nodes."""
    def __init__(self, pinecone_api_key, pinecone_environment, index_name, namespace, similarity_top_k):
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
        self.namespace = namespace
        self.similarity_top_k = similarity_top_k
        self.embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve similar nodes with vector DB hit logging."""
        logger.info(f"BasicRetriever - Starting vector search for query: '{query[:50]}...'")
        
        query_embedding = self.embed_model.encode(query, normalize_embeddings=True)
        
        logger.info(f"Vector DB Query - Requesting top_{self.similarity_top_k} from namespace '{self.namespace}'")
        results = self.index.query(
            vector=query_embedding.tolist(),
            namespace=self.namespace,
            top_k=self.similarity_top_k,
            include_metadata=True
        )
        
        # Log vector database hit details
        hits_count = len(results.matches)
        logger.info(f"Vector DB Hit - Retrieved {hits_count}/{self.similarity_top_k} matches from namespace '{self.namespace}'")
        
        # In fallback mode, we just merge all retrieved text
        if not results.matches:
            logger.warning("Vector DB Hit - No matches found in vector database")
            return []

        merged_text = " ".join([match.metadata.get("text", "") for match in results.matches])
        
        # Return a single merged node
        merged_node = {
            "id": results.matches[0].id,
            "text": merged_text,
            "score": results.matches[0].score,
            "metadata": {"source": results.matches[0].metadata.get("source")},
        }
        
        logger.info(f"BasicRetriever Summary - {hits_count} nodes merged into 1 context")
        return [merged_node]


class AutoMergingRetriever:
    """
    Wrapper around LlamaIndex AutoMergingRetriever to maintain function name compatibility.
    This ensures your existing code continues to work while using true hierarchical auto-merging.
    """
    
    def __init__(self, llama_retriever, verbose: bool = True):
        """Initialize with a LlamaIndex AutoMergingRetriever."""
        self.llama_retriever = llama_retriever
        self.verbose = verbose
        # Add base_retriever attribute for compatibility with clear_vector_store
        self.base_retriever = getattr(llama_retriever, 'base_retriever', llama_retriever)
        
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve using LlamaIndex AutoMergingRetriever with vector DB hit logging.
        
        Args:
            query: Query string
            
        Returns:
            List of retrieved and auto-merged nodes
        """
        if self.verbose:
            logger.info(f"LlamaIndex AutoMergingRetriever - Processing query: '{query[:50]}...'")
        
        try:
            # Use LlamaIndex AutoMergingRetriever
            results = self.llama_retriever.retrieve(query)
            
            # Convert LlamaIndex results to your expected format
            converted_results = []
            for node in results:
                converted_node = {
                    "id": getattr(node, 'id_', str(hash(node.text))),
                    "text": node.text,
                    "score": getattr(node, 'score', 1.0),
                    "metadata": node.metadata,
                    "level": node.metadata.get("level", 0)
                }
                converted_results.append(converted_node)
            
            if self.verbose:
                logger.info(f"LlamaIndex AutoMerging Summary - Retrieved {len(converted_results)} hierarchically merged contexts")
                if converted_results:
                    total_chars = sum(len(node["text"]) for node in converted_results)
                    logger.info(f"LlamaIndex Context Size - {total_chars} total characters across {len(converted_results)} merged documents")
            
            return converted_results
            
        except Exception as e:
            logger.error(f"LlamaIndex AutoMergingRetriever failed: {e}")
            return []


# --- This becomes your main factory function ---
def create_auto_merging_retriever(
    pinecone_api_key: str,
    pinecone_environment: str,
    index_name: str,
    namespace: str = "default",
    similarity_top_k: int = 5,  # Use your accuracy-optimized setting
    simple_ratio_thresh: float = 0.0,  # Keep your exact parameter
    storage_context: Optional[Any] = None,
    verbose: bool = True
) -> Any:
    """
    Creates AutoMergingRetriever using LlamaIndex when available.
    Maintains exact function signature for compatibility with your existing code.
    
    Args:
        pinecone_api_key: Pinecone API key
        pinecone_environment: Pinecone environment (maintained for compatibility)
        index_name: Pinecone index name
        namespace: Namespace to search in
        similarity_top_k: Number of documents to retrieve
        simple_ratio_thresh: Threshold for merging (0.0 = merge all)
        storage_context: Storage context for hierarchical relationships
        verbose: Enable verbose logging
        
    Returns:
        AutoMergingRetriever instance with LlamaIndex hierarchical auto-merging
    """
    if LLAMAINDEX_AVAILABLE:
        try:
            logger.info("LlamaIndex available - Creating true hierarchical AutoMergingRetriever")
            
            # 1. Setup Pinecone connection for LlamaIndex
            pc = Pinecone(api_key=pinecone_api_key)
            pinecone_index = pc.Index(index_name)
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace=namespace)
            
            # 2. Create the StorageContext and VectorStoreIndex
            if not storage_context:
                storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)

            # 3. Create the LlamaIndex base retriever and auto-merging retriever
            base_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
            llama_auto_merger = LlamaAutoMergingRetriever(
                base_retriever, 
                storage_context=storage_context, 
                verbose=verbose
            )
            
            # 4. Wrap in our compatibility class
            retriever = AutoMergingRetriever(llama_auto_merger, verbose=verbose)
            
            logger.info(f"Successfully created LlamaIndex AutoMergingRetriever with top_k={similarity_top_k}")
            return retriever

        except Exception as e:
            logger.error(f"Failed to create LlamaIndex AutoMergingRetriever: {e}")
            logger.warning("Falling back to BasicRetriever")
    
    # Fallback if LlamaIndex is not available or setup failed
    if not LLAMAINDEX_AVAILABLE:
        logger.warning("LlamaIndex not available - using BasicRetriever fallback")
    
    # Create BasicRetriever wrapped as AutoMergingRetriever for compatibility
    basic_retriever = BasicRetriever(
        pinecone_api_key=pinecone_api_key,
        pinecone_environment=pinecone_environment,
        index_name=index_name,
        namespace=namespace,
        similarity_top_k=similarity_top_k
    )
    
    # Wrap BasicRetriever to maintain AutoMergingRetriever interface
    class BasicAutoMergingRetriever:
        def __init__(self, basic_retriever, verbose=True):
            self.basic_retriever = basic_retriever  # Add this for compatibility
            self.base_retriever = basic_retriever   # Add this for clear_vector_store compatibility
            self.verbose = verbose
            
        def retrieve(self, query: str):
            return self.basic_retriever.retrieve(query)
    
    return BasicAutoMergingRetriever(basic_retriever, verbose=verbose)