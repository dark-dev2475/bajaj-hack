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
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

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
        # --- CONFIGURE THE LLM WITH AN OUTPUT LIMIT ---
        # Set a hard limit on the number of tokens. 200 words is roughly 256 tokens.
        generation_config = {"max_output_tokens": 256, "temperature": 0.2}
        
        if not hasattr(Settings, 'llm'):
            Settings.llm = Gemini(
                model="models/gemini-1.5-flash-latest", 
                api_key=google_api_key,
                generation_config=generation_config
            )
        # ----------------------------------------------------

        retriever = create_auto_merging_retriever(
            pinecone_api_key=pinecone_api_key,
            index_name=index_name,
            storage_context=storage_context,
            namespace=namespace,
            similarity_top_k=similarity_top_k
        )
        
        reranker = SentenceTransformerRerank(top_n=4, model="BAAI/bge-reranker-base")

        # --- ENHANCED PROMPT FOR COMPLEX REASONING AND CONCISE ANSWERS ---
        qa_prompt_tmpl_str = (
            "You are an expert analyst. Your task is to provide a precise and factual answer based *only* on the provided context. "
            "Think step-by-step to arrive at the correct conclusion.\n"
            "---------------------\n"
            "CONTEXT:\n{context_str}\n"
            "---------------------\n"
            "INSTRUCTIONS:\n"
            "1.  **Reason Step-by-Step:** First, understand the query. Then, synthesize information from all relevant parts of the context to construct your answer.\n"
            "2.  **Be Factual and Concise:** Your answer must be based exclusively on the provided text. Do not add outside information. The final answer should be around 200 words.\n"
            "3.  **Handle Missing Information:** If the context does not contain the answer, you must state: 'The provided documents do not contain enough information to answer this question.'\n\n"
            "QUERY: {query_str}\n\n"
            "PRECISE ANSWER:\n"
        )
        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
        # ----------------------------------------------------

        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=qa_prompt_tmpl,
            use_async=True
        )

        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[reranker]
        )
        
        logger.info("RAG Pipeline initialized with Reranker and a Precise Prompt.")

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
