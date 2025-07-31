import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# LlamaIndex imports for true auto-merging with local embeddings
try:
    from llama_index.core.retrievers import AutoMergingRetriever as LlamaAutoMergingRetriever
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    LLAMAINDEX_AVAILABLE = True
except ImportError as e:
    LLAMAINDEX_AVAILABLE = False
    print(f"LlamaIndex import error: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoMergingRetriever:
    """
    LlamaIndex AutoMergingRetriever with local embeddings.
    Uses your local BGE embeddings instead of OpenAI.
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
            logger.info(f"Using AutoMergingRetriever for retrieval: '{query[:50]}...'")
        
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
    Creates AutoMergingRetriever using LlamaIndex with your local BGE embeddings.
    No more fallback - uses only LlamaIndex AutoMergingRetriever.
    
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
    if not LLAMAINDEX_AVAILABLE:
        raise ImportError("LlamaIndex is required but not available. Please install: pip install llama-index llama-index-vector-stores-pinecone llama-index-embeddings-huggingface")
    
    try:
        logger.info("Creating LlamaIndex AutoMergingRetriever with local BGE embeddings")
        
        # 1. Setup local BGE embeddings (your embeddings, not OpenAI)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # 2. Setup Pinecone connection for LlamaIndex
        pc = Pinecone(api_key=pinecone_api_key)
        pinecone_index = pc.Index(index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace=namespace)
        
        # 3. Create the StorageContext and VectorStoreIndex with your local embeddings
        if not storage_context:
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, 
            storage_context=storage_context,
            embed_model=embed_model  # Use your local BGE embeddings
        )

        # 4. Create the LlamaIndex base retriever and auto-merging retriever
        base_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        llama_auto_merger = LlamaAutoMergingRetriever(
            base_retriever, 
            storage_context=storage_context, 
            verbose=verbose
        )
        
        # 5. Wrap in our compatibility class
        retriever = AutoMergingRetriever(llama_auto_merger, verbose=verbose)
        
        logger.info(f"Successfully created LlamaIndex AutoMergingRetriever with local BGE embeddings, top_k={similarity_top_k}")
        return retriever

    except Exception as e:
        logger.error(f"Failed to create LlamaIndex AutoMergingRetriever: {e}")
        raise RuntimeError(f"AutoMergingRetriever creation failed: {e}")