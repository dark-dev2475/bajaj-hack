import asyncio
import logging
from typing import List, Dict, Any
import google.generativeai as genai
from .ptsretriever import PTSRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PTSAnswerGenerator:
    """PTS Answer Generator - Generates answers using PTS retrieval system."""
    
    def __init__(
        self,
        pts_retriever: PTSRetriever,
        google_api_key: str,
        llm_model: str = "gemini-1.5-flash",
        max_output_tokens: int = 500,
        temperature: float = 0.1
    ):
        """
        Initialize the PTS Answer Generator.
        
        Args:
            pts_retriever: PTSRetriever instance for document retrieval
            google_api_key: Google API key for Gemini Flash
            llm_model: Gemini model for answer generation
            max_output_tokens: Maximum tokens for response
            temperature: Temperature for generation
        """
        self.pts_retriever = pts_retriever
        self.llm_model = llm_model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        
        # Initialize Gemini
        genai.configure(api_key=google_api_key)
        self.model = genai.GenerativeModel(llm_model)
        
        logger.info(f"PTSAnswerGenerator initialized with model: {llm_model}")
    
    def _format_context(self, retrieved_nodes: List[Dict[str, Any]]) -> str:
        """
        Format retrieved parent nodes into context string.
        
        Args:
            retrieved_nodes: List of parent nodes from PTS retriever
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, node in enumerate(retrieved_nodes, 1):
            # Extract text from parent node
            text = node.get('text', '')
            score = node.get('score', 0.0)
            sentence_count = node.get('sentence_count', 0)
            
            context_part = f"Document {i} (Score: {score:.3f}, Sentences: {sentence_count}):\n{text}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    async def _generate_answer_async(self, query: str, context: str) -> Dict[str, Any]:
        """
        Generate answer using Gemini Flash asynchronously.
        
        Args:
            query: User question
            context: Formatted context from retrieval
            
        Returns:
            Answer result dictionary
        """
        prompt = f"""You are an expert assistant that answers questions based on the provided context.

Context:
{context}

Question: {query}

Instructions:
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Be specific and cite relevant parts of the context when possible
4. Keep your answer clear, concise, and helpful
5. Provide a complete answer without referring to "the context" or "the document"

Answer:"""

        try:
            # Use asyncio to run the generation in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                    )
                )
            )
            
            answer = response.text.strip()
            return {
                "answer": answer,
                "model": self.llm_model,
                "status": "success",
                "token_count": len(answer.split())  # Rough token estimate
            }
            
        except Exception as e:
            logger.error(f"Error generating answer with Gemini: {str(e)}")
            return {
                "answer": "Sorry, I encountered an error while generating the answer. Please try again.",
                "model": self.llm_model,
                "status": "error",
                "error": str(e)
            }
    
    async def answer_question(
        self,
        question: str,
        top_k: int = 5,
        return_context: bool = True,
        filter_dict: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Complete question answering pipeline using PTS system.
        
        Args:
            question: User's question
            top_k: Number of parent nodes to retrieve
            return_context: Whether to include context in response
            filter_dict: Optional metadata filter for retrieval
            
        Returns:
            Complete answer response dictionary
        """
        logger.info(f"Processing question: '{question[:50]}...'")
        
        try:
            # Step 1: Retrieve relevant parent nodes using PTS system
            retrieved_nodes = await self.pts_retriever.retrieve(
                query=question,
                return_type="parent",
                top_k=top_k,
                filter_dict=filter_dict
            )
            
            if not retrieved_nodes:
                return {
                    "question": question,
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "context": [] if return_context else None,
                    "retrieval_count": 0,
                    "status": "no_context"
                }
            
            # Step 2: Format context from parent nodes
            context = self._format_context(retrieved_nodes)
            
            # Step 3: Generate answer using formatted context
            answer_result = await self._generate_answer_async(question, context)
            
            # Step 4: Prepare complete response
            response = {
                "question": question,
                "answer": answer_result["answer"],
                "retrieval_count": len(retrieved_nodes),
                "model": answer_result["model"],
                "status": answer_result["status"],
                "generation_info": {
                    "token_count": answer_result.get("token_count", 0),
                    "temperature": self.temperature,
                    "max_tokens": self.max_output_tokens
                }
            }
            
            # Include context and retrieval details if requested
            if return_context:
                response["context"] = retrieved_nodes
                response["formatted_context"] = context
                response["retrieval_scores"] = [node.get("score", 0.0) for node in retrieved_nodes]
            
            # Include error info if present
            if "error" in answer_result:
                response["error"] = answer_result["error"]
            
            logger.info(f"Successfully generated answer using {len(retrieved_nodes)} parent nodes")
            return response
            
        except Exception as e:
            logger.error(f"Error in answer_question: {str(e)}")
            return {
                "question": question,
                "answer": "An unexpected error occurred while processing your question.",
                "context": [] if return_context else None,
                "retrieval_count": 0,
                "status": "error",
                "error": str(e)
            }
    
    async def batch_answer_questions(
        self,
        questions: List[str],
        top_k: int = 5,
        return_context: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Answer multiple questions in batch.
        
        Args:
            questions: List of questions
            top_k: Number of parent nodes to retrieve per question
            return_context: Whether to include context in responses
            
        Returns:
            List of answer response dictionaries
        """
        logger.info(f"Processing {len(questions)} questions in batch")
        
        try:
            # Process all questions concurrently
            tasks = [
                self.answer_question(
                    question=question,
                    top_k=top_k,
                    return_context=return_context
                )
                for question in questions
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions in results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing question {i}: {str(result)}")
                    processed_results.append({
                        "question": questions[i],
                        "answer": "Error processing this question.",
                        "status": "error",
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)
            
            logger.info(f"Completed batch processing of {len(questions)} questions")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in batch_answer_questions: {str(e)}")
            # Return error responses for all questions
            return [
                {
                    "question": question,
                    "answer": "Batch processing error occurred.",
                    "status": "error",
                    "error": str(e)
                }
                for question in questions
            ]
    
    async def clear_vector_store(self) -> bool:
        """
        Clear the vector store through PTS retriever.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            result = await self.pts_retriever.vector_store.delete_vectors(delete_all=True)
            success = result.get("status") == "success"
            
            if success:
                logger.info("Successfully cleared PTS vector store")
            else:
                logger.error(f"Failed to clear PTS vector store: {result}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error clearing PTS vector store: {str(e)}")
            return False
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the PTS answer generation system.
        
        Returns:
            System statistics dictionary
        """
        try:
            # Get retrieval stats
            retrieval_stats = await self.pts_retriever.get_retrieval_stats()
            
            return {
                "model_info": {
                    "llm_model": self.llm_model,
                    "max_output_tokens": self.max_output_tokens,
                    "temperature": self.temperature
                },
                "retrieval_system": retrieval_stats,
                "status": "operational"
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {"error": str(e), "status": "error"}

# Factory function for easy creation
def create_pts_answer_generator(
    pts_retriever: PTSRetriever,
    google_api_key: str,
    **kwargs
) -> PTSAnswerGenerator:
    """
    Create a PTS Answer Generator instance.
    
    Args:
        pts_retriever: PTSRetriever instance
        google_api_key: Google API key for Gemini
        **kwargs: Additional arguments for PTSAnswerGenerator
        
    Returns:
        PTSAnswerGenerator instance
    """
    return PTSAnswerGenerator(
        pts_retriever=pts_retriever,
        google_api_key=google_api_key,
        **kwargs
    )
