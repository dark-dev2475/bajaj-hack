# pts_handler.py

import os
import logging
from typing import Dict, Any, List
from pathlib import Path

# Import PTS components
from .ptsnode.parent_to_sentence_node_parser import ParentToSentenceNodeParser
from .ptsnode.ptsembedder import PTSEmbedder, create_pts_embedder
from .ptsnode.ptsvector import PTSVectorStore, create_pts_vector_store
from .ptsnode.ptsretriever import PTSRetriever, create_pts_retriever
from .ptsnode.ptsanswer import PTSAnswerGenerator, create_pts_answer_generator

# Import existing components for document loading
from .document_loader import DocumentLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def handle_pts_rag_request(
    document_url: str, 
    questions: List[str], 
    upload_folder: str, 
    index_name: str
) -> List[str]:
    """
    Handles the full PTS RAG flow:
    1. Download the document from URL
    2. Parse document into parent and sentence nodes using PTS parser
    3. Embed sentence nodes and store in Pinecone
    4. For each question, retrieve relevant parent context and generate answer
    5. Clean up vector store
    
    Args:
        document_url: URL of the document to process
        questions: List of questions to answer
        upload_folder: Folder to store downloaded documents
        index_name: Pinecone index name
        
    Returns:
        List of answers (strings only) for each question
    """
    
    try:
        # Get environment variables
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_env = os.getenv("PINECONE_ENV")
        google_api_key = os.getenv("GEMINI_API_KEY")
        
        if not all([pinecone_api_key, google_api_key]):
            logger.error("Missing required environment variables")
            raise ValueError("Missing required environment variables")
        
        # Step 1: Download the document
        logger.info(f"Downloading document from: {document_url}")
        document_loader = DocumentLoader(data_dir=upload_folder)
        download_results = await document_loader.download_files([document_url])
        
        if not download_results or download_results[0]["status"] != "success":
            raise Exception(f"Failed to download document: {document_url}")
        
        file_path = download_results[0]["file_path"]
        logger.info(f"Document downloaded to: {file_path}")
        
        # Step 2: Parse document using PTS parser
        logger.info("Parsing document with Parent to Sentence parser")
        pts_parser = ParentToSentenceNodeParser.from_defaults(
            chunk_size=1024,
            chunk_overlap=50
        )
        
        # Load document and parse
        from llama_index.core import SimpleDirectoryReader
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        
        # Get sentence nodes and parent nodes dict
        sentence_nodes, parent_nodes_dict = pts_parser.get_nodes_from_documents(documents)
        logger.info(f"Created {len(sentence_nodes)} sentence nodes from {len(parent_nodes_dict)} parent nodes")
        
        # Step 3: Initialize PTS components
        logger.info("Initializing PTS components")
        
        # Create embedder
        pts_embedder = create_pts_embedder(
            pinecone_api_key=pinecone_api_key,
            pinecone_environment=pinecone_env,
            index_name=index_name,
            namespace="pts_documents",
            batch_size=32
        )
        
        # Create vector store
        pts_vector_store = create_pts_vector_store(
            pinecone_api_key=pinecone_api_key,
            pinecone_environment=pinecone_env,
            index_name=index_name,
            namespace="pts_documents"
        )
        
        # Step 4: Convert sentence nodes to embedding format and embed
        logger.info("Converting nodes and embedding sentence nodes")
        sentence_node_data = []
        for i, node in enumerate(sentence_nodes):
            # Extract text and metadata from the node
            node_text = getattr(node, 'text', str(node))
            node_metadata = getattr(node, 'metadata', {})
            
            # Get parent ID from relationships or generate one
            parent_id = ""
            if hasattr(node, 'relationships'):
                # Try to get parent from relationships
                parent_ref = node.relationships.get('1')  # PARENT relationship type
                if parent_ref:
                    parent_id = parent_ref.node_id
            
            if not parent_id:
                # Fallback: assign to a parent based on index
                parent_keys = list(parent_nodes_dict.keys())
                if parent_keys:
                    parent_id = parent_keys[i % len(parent_keys)]
            
            sentence_node_data.append({
                "id": f"sentence_{i}",
                "text": node_text,
                "parent_id": parent_id,
                "metadata": node_metadata
            })
        
        # Embed and store sentence nodes
        embed_results = await pts_embedder.embed_sentence_nodes(sentence_node_data)
        logger.info(f"Embedded {embed_results['total_stored']} sentence nodes")
        
        # Step 5: Create retriever
        logger.info("Creating PTS retriever")
        pts_retriever = create_pts_retriever(
            embedder=pts_embedder,
            vector_store=pts_vector_store,
            parent_nodes_dict=parent_nodes_dict,
            similarity_top_k=5
        )
        
        # Step 6: Create answer generator
        logger.info("Creating PTS answer generator")
        pts_answer_generator = create_pts_answer_generator(
            pts_retriever=pts_retriever,
            google_api_key=google_api_key,
            llm_model="gemini-1.5-flash"
        )
        
        # Step 7: Answer each question
        logger.info(f"Answering {len(questions)} questions")
        answers = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
            
            try:
                # Get answer using PTS system
                answer_result = await pts_answer_generator.answer_question(
                    question=question,
                    top_k=5,
                    return_context=False  # We only need the answer text
                )
                
                # Extract just the answer text
                answer_text = answer_result.get("answer", "Error processing question")
                answers.append(answer_text)
                
            except Exception as e:
                logger.error(f"Error answering question {i+1}: {str(e)}")
                answers.append(f"Error processing question: {str(e)}")
        
        # Step 8: Clean up vector store
        logger.info("Cleaning up vector store")
        cleanup_success = await pts_answer_generator.clear_vector_store()
        if cleanup_success:
            logger.info("Vector store cleaned up successfully")
        else:
            logger.warning("Failed to clean up vector store")
        
        logger.info(f"Successfully processed {len(questions)} questions using PTS system")
        return answers
        
    except Exception as e:
        logger.exception("Error in PTS RAG handler")
        # Return error messages for all questions
        error_message = f"PTS RAG pipeline error: {str(e)}"
        return [error_message] * len(questions)

# Alternative factory function for direct PTS pipeline creation
async def create_pts_pipeline(
    pinecone_api_key: str,
    pinecone_environment: str,
    index_name: str,
    google_api_key: str,
    namespace: str = "pts_documents",
    chunk_size: int = 1024,
    chunk_overlap: int = 50,
    similarity_top_k: int = 5
) -> PTSAnswerGenerator:
    """
    Create a complete PTS pipeline ready for question answering.
    
    Args:
        pinecone_api_key: Pinecone API key
        pinecone_environment: Pinecone environment
        index_name: Pinecone index name
        google_api_key: Google API key
        namespace: Pinecone namespace
        chunk_size: Parent chunk size
        chunk_overlap: Chunk overlap
        similarity_top_k: Number of results to retrieve
        
    Returns:
        PTSAnswerGenerator ready for use
    """
    try:
        # Create components
        pts_embedder = create_pts_embedder(
            pinecone_api_key=pinecone_api_key,
            pinecone_environment=pinecone_environment,
            index_name=index_name,
            namespace=namespace
        )
        
        pts_vector_store = create_pts_vector_store(
            pinecone_api_key=pinecone_api_key,
            pinecone_environment=pinecone_environment,
            index_name=index_name,
            namespace=namespace
        )
        
        # Note: parent_nodes_dict needs to be provided separately after parsing
        pts_retriever = create_pts_retriever(
            embedder=pts_embedder,
            vector_store=pts_vector_store,
            parent_nodes_dict={},  # Empty initially
            similarity_top_k=similarity_top_k
        )
        
        pts_answer_generator = create_pts_answer_generator(
            pts_retriever=pts_retriever,
            google_api_key=google_api_key
        )
        
        logger.info("PTS pipeline created successfully")
        return pts_answer_generator
        
    except Exception as e:
        logger.error(f"Error creating PTS pipeline: {str(e)}")
        raise
