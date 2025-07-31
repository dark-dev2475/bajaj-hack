# handler.py

import os
import logging
from typing import Dict, Any, List
from pathlib import Path
import pinecone

# Import our custom RAG pipeline components
from .document_loader import DocumentLoader
from .parser import HierarchicalParser
from .embedder import HierarchicalEmbedder
from .retriever import create_auto_merging_retriever
from .answer import RAGPipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def handle_rag_request(document_url: str, questions: List[str], upload_folder: str, index_name: str) -> List[str]:
    """
    Handles the full RAG flow:
    1. Download the document from URL
    2. Parse document into hierarchical chunks
    3. Embed leaf nodes and store in Pinecone
    4. For each question, retrieve relevant context and generate answer
    
    Args:
        document_url: URL of the document to process
        questions: List of questions to answer
        upload_folder: Folder to store downloaded documents
        index_name: Pinecone index name
        
    Returns:
        List of answers for each question
    """
    
    try:
        # Get environment variables
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_env = os.getenv("PINECONE_ENV")
        google_api_key = os.getenv("GEMINI_API_KEY")
        
        if not google_api_key:
            logger.error(f"Missing GEMINI_API_KEY environment variable")
            raise ValueError("Missing required environment variables")
        
        # Step 1: Download the document
        logger.info(f"Downloading document from: {document_url}")
        document_loader = DocumentLoader(data_dir=upload_folder)
        download_results = await document_loader.download_files([document_url])
        
        if not download_results or download_results[0]["status"] != "success":
            raise Exception(f"Failed to download document: {document_url}")
        
        file_path = download_results[0]["file_path"]
        logger.info(f"Document downloaded to: {file_path}")
        
        # Step 2: Parse document into hierarchical structure
        logger.info("Parsing document into hierarchical chunks")
        parser = HierarchicalParser(
            chunk_sizes=[800, 400, 200],  # Accuracy-optimized values
            chunk_overlap=50  # Added overlap for better continuity
        )
        hierarchical_nodes = parser.parse_document(file_path)
        leaf_nodes = parser.get_leaf_nodes(hierarchical_nodes)
        logger.info(f"Created {len(hierarchical_nodes)} top-level nodes, {len(leaf_nodes)} leaf nodes")
        
        # Step 3: Embed and store leaf nodes (for hierarchical auto-merging)
        logger.info("Embedding and storing leaf nodes in Pinecone")
        embedder = HierarchicalEmbedder(
            pinecone_api_key=pinecone_api_key,
            pinecone_environment=pinecone_env,
            index_name=index_name,
            namespace="documents"
        )
        # Remove await since embed_and_store is no longer async
        embedder.embed_and_store(leaf_nodes)
        logger.info("Successfully embedded and stored all nodes")
        
        # Step 4: Initialize RAG pipeline (it already creates AutoMergingRetriever internally)
        logger.info("Initializing RAG pipeline with built-in LlamaIndex AutoMergingRetriever")
        
        rag_pipeline = RAGPipeline(
            pinecone_api_key=pinecone_api_key,
            pinecone_environment=pinecone_env,
            index_name=index_name,
            google_api_key=google_api_key,
            namespace="documents",
            similarity_top_k=5  # Accuracy-optimized setting
        )
        
        logger.info("RAG pipeline initialized with enhanced auto-merging capabilities")
        
        # Step 5: Answer each question with enhanced logging
        answers = []
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
            
            try:
                answer_result = rag_pipeline.query(question)
                # Extract only the answer text for the simplified response format
                answers.append(answer_result["answer"])
                
                # Log retrieval stats for monitoring
                if "retrieval_count" in answer_result:
                    logger.info(f"Question {i+1} - Retrieved {answer_result['retrieval_count']} contexts")
                    
            except Exception as e:
                logger.error(f"Error answering question {i+1}: {str(e)}")
                # Return error message as a simple string to maintain consistency
                answers.append(f"Error processing question: {str(e)}")
       
      


        
        # Step 6: Clear the vector store after all questions are processed
        logger.info("Clearing vector store...")
        try:
            if rag_pipeline.clear_vector_store():
                logger.info("Vector store cleared successfully")
            else:
                logger.warning("Failed to clear vector store")
        except Exception as e:
            logger.warning(f"Error clearing vector store: {e}")
        
        logger.info(f"Successfully processed {len(questions)} questions with enhanced auto-merging")
        return answers
        
    except Exception as e:
        logger.exception("Error in RAG handler")
        raise Exception(f"RAG pipeline error: {str(e)}")