import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone

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


# --- THIS CLASS IS REMOVED ---
# The AutoMergingRetriever wrapper class is no longer needed and was causing conflicts.
# We will now use the LlamaAutoMergingRetriever directly.


# --- This becomes your main factory function ---
# LlamaIndex now uses a global Settings object, which is cleaner.
from llama_index.core import Settings

def create_auto_merging_retriever(
    pinecone_api_key: str, # This is now required
    index_name: str,
    storage_context: StorageContext,
    namespace: str = "default",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    similarity_top_k: int = 12,
    verbose: bool = True
) -> LlamaAutoMergingRetriever: # Return the real LlamaIndex retriever
    """
    Creates a LlamaIndex AutoMergingRetriever correctly configured.
    """
    if not LLAMAINDEX_AVAILABLE:
        raise ImportError("LlamaIndex is required for auto-merging.")
        
    if not storage_context or not storage_context.docstore:
        raise ValueError("A StorageContext with a docstore containing all parent nodes is required.")

    logger.info("Creating LlamaIndex AutoMergingRetriever...")
    
    # 1. Setup Pinecone vector store from the provided index name
    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace=namespace)

    # 2. Create the index object from the existing vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, 
        storage_context=storage_context,
        embedding_model=HuggingFaceEmbedding(model_name=embedding_model, api_key=pinecone_api_key)
    )

    # 3. Create the LlamaIndex base retriever and auto-merging retriever
    base_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    
    llama_auto_merger = LlamaAutoMergingRetriever(
        base_retriever, 
        storage_context=storage_context, 
        verbose=verbose
    )
    
    logger.info(f"Successfully created LlamaIndex AutoMergingRetriever, top_k={similarity_top_k}")
    return llama_auto_merger
