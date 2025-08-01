# /rag_pipeline/answer.py

import logging
from typing import List, Dict, Any
from pinecone import Pinecone
from .retriever import create_auto_merging_retriever

# LlamaIndex Imports
from llama_index.core import Settings, StorageContext, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.gemini import Gemini
from llama_index.core.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Complete RAG pipeline that retrieves context and generates answers."""

    def __init__(
        self,
        pinecone_api_key: str,
        index_name: str,
        google_api_key: str,
        storage_context: StorageContext,
        namespace: str = "default",
        similarity_top_k: int = 12,
    ):
        if not hasattr(Settings, 'llm'):
            Settings.llm = Gemini(model="models/gemini-1.5-flash-latest", api_key=google_api_key)

        retriever = create_auto_merging_retriever(
            pinecone_api_key=pinecone_api_key,
            index_name=index_name,
            storage_context=storage_context,
            namespace=namespace,
            similarity_top_k=similarity_top_k
        )
        
        # --- THIS IS THE FIX ---
        # Define a strict prompt template
        qa_prompt_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "You are a helpful assistant. Based ONLY on the provided context information and not on any prior knowledge, "
            "answer the query.\n"
            "If the context does not contain the answer, you MUST state that you cannot answer with the given information.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

        # Use the 'compact' response mode to ensure a single API call
        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=qa_prompt_tmpl,
            use_async=True
        )
        # ----------------------

        # Update the query engine to use the new synthesizer
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer
        )
        
        logger.info("RAG Pipeline initialized successfully with 'compact' mode.")

    async def query(self, question: str) -> Dict[str, Any]:
        """
        Asynchronously queries the RAG pipeline.
        """
        logger.info(f"Processing query: '{question[:50]}...'")
        response = await self.query_engine.aquery(question)
        
        return {
            "answer": str(response),
            "source_nodes": [
                {"text": node.get_content(), "score": node.get_score()}
                for node in response.source_nodes
            ]
        }

    def clear_vector_store(self, pinecone_api_key: str, index_name: str, namespace: str) -> bool:
        """Deletes all vectors from the specified namespace."""
        try:
            pc = Pinecone(api_key=pinecone_api_key)
            index = pc.Index(index_name)
            index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Successfully cleared vector store namespace: {namespace}")
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False
