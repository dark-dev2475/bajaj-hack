# /rag_pipeline/handler.py

import os
import logging
from typing import List

# LlamaIndex Imports
from llama_index.core import Settings, StorageContext
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Import our custom RAG pipeline components
from .document_loader import DocumentLoader
from .parser import HierarchicalParser
from .embedder import HierarchicalEmbedder
from .answer import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_global_models(google_api_key: str):
    """Initializes and sets the global LLM and embedding models."""
    logger.info("Setting up global LLM and embedding models...")
    
    # Set the LLM (Gemini)
    Settings.llm = Gemini(model="models/gemini-1.5-flash-latest", api_key=google_api_key)
    
    # Set the embedding model (Hugging Face)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    
    logger.info("✅ Global models configured successfully.")


async def handle_rag_request(document_url: str, questions: List[str], upload_folder: str, index_name: str) -> List[str]:
    """
    Handles the full RAG flow from download to answering questions.
    """
    rag_pipeline = None
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    namespace = "documents"

    if not all([pinecone_api_key, google_api_key]):
        raise ValueError("Missing required environment variables: PINECONE_API_KEY, GOOGLE_API_KEY")

    try:
        # --- CRITICAL FIX: Configure models BEFORE any other operation ---
        setup_global_models(google_api_key)

        # Step 1: Download the document
        logger.info(f"Downloading document from: {document_url}")
        document_loader = DocumentLoader(data_dir=upload_folder)
        download_results = await document_loader.download_files([document_url])
        
        if not download_results or download_results[0]["status"] != "success":
            raise Exception(f"Failed to download document: {download_results[0].get('reason', 'Unknown error')}")
        
        file_path = download_results[0]["file_path"]
        
        # Step 2: Parse document
        parser = HierarchicalParser()
        all_nodes = parser.parse_document(file_path)
        leaf_nodes = get_leaf_nodes(all_nodes)

        # Step 2.5: Create Storage Context
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(all_nodes)
        logger.info(f"Created StorageContext with {len(all_nodes)} total nodes.")

        # Step 3: Embed and store leaf nodes
        embedder = HierarchicalEmbedder(pinecone_api_key, index_name, namespace)
        embedder.embed_and_store(leaf_nodes)

        # Step 4: Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            pinecone_api_key, index_name, google_api_key, storage_context, namespace
        )
        
        # Step 5: Answer questions
        answers = []
        for question in questions:
            answer_result = await rag_pipeline.query(question)
            answers.append(answer_result["answer"])
        
        return answers
        
    except Exception as e:
        logger.exception("Error in RAG handler")
        raise Exception(f"RAG pipeline error: {str(e)}")
    finally:
        # Step 6: Clean up the vector store
        if rag_pipeline:
            logger.info("Clearing vector store namespace...")
            rag_pipeline.clear_vector_store(pinecone_api_key, index_name, namespace)