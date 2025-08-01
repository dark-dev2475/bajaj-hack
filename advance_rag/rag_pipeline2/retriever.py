# /rag_pipeline/retriever.py

import logging
from typing import Any
from pinecone import Pinecone

# LlamaIndex Imports
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.vector_stores.pinecone import PineconeVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_auto_merging_retriever(
    pinecone_api_key: str,
    index_name: str,
    storage_context: StorageContext,
    namespace: str = "default",
    similarity_top_k: int = 5,
    verbose: bool = True
) -> AutoMergingRetriever:
    """
    Creates an AutoMergingRetriever connected to a Pinecone index.
    """
    if not storage_context.docstore:
        raise ValueError("The provided StorageContext must contain a docstore.")

    logger.info("Creating LlamaIndex AutoMergingRetriever...")
    
    # Setup Pinecone vector store
    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace=namespace)

    # Create the index object from the existing vector store and docstore
    base_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, 
        storage_context=storage_context
    )

    # Create the base retriever and the auto-merging retriever
    base_retriever = base_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, 
        storage_context=storage_context, 
        verbose=verbose
    )
    
    logger.info(f"Successfully created AutoMergingRetriever.")
    return retriever