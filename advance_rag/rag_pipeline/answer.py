import logging
from typing import List, Dict, Any
from .retriever import create_auto_merging_retriever
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Complete RAG pipeline that retrieves context and generates answers."""
    
    def __init__(
        self,
        pinecone_api_key: str,
        pinecone_environment: str,
        index_name: str,
        google_api_key: str,
        namespace: str = "default",
        similarity_top_k: int = 5,  # Increased for more comprehensive context
        llm_model: str = "gemini-1.5-flash"
    ):
        """
        Initialize the complete RAG pipeline.
        
        Args:
            pinecone_api_key: Pinecone API key
            pinecone_environment: Pinecone environment
            index_name: Pinecone index name
            google_api_key: Google API key for Gemini Flash
            namespace: Pinecone namespace
            similarity_top_k: Number of documents to retrieve
            llm_model: Gemini model for answer generation
        """
        # Initialize retriever
        self.retriever = create_auto_merging_retriever(
            pinecone_api_key=pinecone_api_key,
            pinecone_environment=pinecone_environment,
            index_name=index_name,
            namespace=namespace,
            similarity_top_k=similarity_top_k,
            simple_ratio_thresh=0.0,
            verbose=True
        )
        
        # Initialize Gemini
        genai.configure(api_key=google_api_key)
        self.model = genai.GenerativeModel(llm_model)
        self.llm_model = llm_model
        
        logger.info("RAG Pipeline initialized successfully with Gemini Flash")
    
    def clear_vector_store(self) -> bool:
        """
        Delete all vectors from the specified namespace in the vector store.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Access Pinecone through the base retriever
            base_retriever = self.retriever.base_retriever
            index = base_retriever.index  # Use existing connection
            
            # Delete all vectors in the namespace
            index.delete(delete_all=True, namespace=base_retriever.namespace)
            logger.info(f"Successfully cleared vector store namespace: {base_retriever.namespace}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            return False
    
    def _format_context(self, retrieved_nodes: List[Dict]) -> str:
        """Format retrieved nodes into context string with clean sentence boundaries."""
        context_parts = []
        total_length = 0
        max_context_length = 3000  # Increased for richer context and better accuracy
        
        for i, node in enumerate(retrieved_nodes, 1):
            node_text = node['text'].strip()
            
            # Clean truncation for long nodes while preserving sentence boundaries
            if len(node_text) > 800:
                # Find the last complete sentence within 800 chars
                truncated = node_text[:800]
                for delimiter in ['. ', '! ', '? ']:
                    last_pos = truncated.rfind(delimiter)
                    if last_pos > 600:  # At least 75% of target length
                        node_text = truncated[:last_pos + 1].strip()
                        break
                else:
                    # Fallback: find last word boundary and add period
                    words = truncated.split()
                    if len(words) > 1:
                        node_text = ' '.join(words[:-1]).strip()
                        if not node_text.endswith(('.', '!', '?')):
                            node_text += '.'
                    else:
                        node_text = truncated.strip()
                        if not node_text.endswith(('.', '!', '?')):
                            node_text += '.'
            
            context_part = f"Document {i}:\n{node_text}\n"
            
            if total_length + len(context_part) > max_context_length:
                # Add partial content with clean boundaries
                remaining_space = max_context_length - total_length
                if remaining_space > 200:
                    # Find last sentence boundary within remaining space
                    partial_limit = remaining_space - 50
                    partial_text = node_text[:partial_limit]
                    
                    for delimiter in ['. ', '! ', '? ']:
                        last_pos = partial_text.rfind(delimiter)
                        if last_pos > partial_limit * 0.7:
                            partial_text = partial_text[:last_pos + 1].strip()
                            break
                    else:
                        # Word boundary fallback
                        words = partial_text.split()
                        if len(words) > 1:
                            partial_text = ' '.join(words[:-1]).strip()
                            if not partial_text.endswith(('.', '!', '?')):
                                partial_text += '.'
                    
                    context_parts.append(f"Document {i} [Partial]:\n{partial_text}\n")
                break
            
            context_parts.append(context_part)
            total_length += len(context_part)
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, query: str, context: str) -> Dict[str, Any]:
        """Generate answer using Gemini Flash."""
        prompt = f"""
### ROLE ###
You are an expert analyst. Your primary goal is to synthesize a complete and direct answer to the user's question based on the provided sources.

### SOURCES ###
{context}

### QUESTION ###
{query}

### CORE INSTRUCTION ###
Analyze the sources to construct the most accurate and comprehensive answer possible. Your default behavior should be to directly answer the question by synthesizing the information you find.

### RESPONSE PROTOCOL ###

1.  **Direct Answer First (Primary Goal):**
    * Thoroughly analyze all sources. If the information is present, even if spread across multiple parts, synthesize it into a confident, direct answer.
    * Support your answer with integrated details and brief quotes from the text. This should be your approach for 90% of questions.

2.  **Fallback for Incomplete Information (Secondary Goal):**
    * **Only if a direct, synthesized answer is impossible to create**, should you use this fallback protocol.
    * **A. State the Limitation:** Begin by clearly stating *which specific part* of the question cannot be answered. (e.g., "While the text explains Kavery's actions, it does not state her internal motivations.")
    * **B. Provide Helpful Context:** Then, add a section like **`Related Findings:`** and provide the most relevant facts you *did* find that might help the user.

### CRITICAL CONSTRAINTS ###
-   NEVER invent or assume information.
-   Prioritize synthesizing a direct answer over defaulting to the "incomplete information" fallback.

### FINAL ANSWER ###
"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Increased for more nuanced responses
                    max_output_tokens=800,  # Increased for more detailed answers
                )
            )
            
            answer = response.text.strip()
            return {
                "answer": answer,
                "model": self.llm_model,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error generating answer with Gemini: {str(e)}")
            return {
                "answer": "Sorry, I encountered an error while generating the answer.",
                "model": self.llm_model,
                "status": "error",
                "error": str(e)
            }
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Complete RAG query - retrieve context and generate answer.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer, context, and metadata
        """
        logger.info(f"Processing query: '{question[:50]}...'")
        
        # Step 1: Retrieve relevant context
        retrieved_nodes = self.retriever.retrieve(question)
        
        if not retrieved_nodes:
            return {
                "question": question,
                "answer": "I couldn't find any relevant information to answer your question.",
                "context": [],
                "status": "no_context"
            }
        
        # Step 2: Format context
        context = self._format_context(retrieved_nodes)
        
        # Step 3: Generate answer
        answer_result = self._generate_answer(question, context)
        
        # Step 4: Return complete response
        return {
            "question": question,
            "answer": answer_result["answer"],
            "context": retrieved_nodes,
            "formatted_context": context,
            "retrieval_count": len(retrieved_nodes),
            "model": answer_result["model"],
            "status": answer_result["status"]
        }

# Simple usage function
def create_rag_pipeline(
    pinecone_api_key: str,
    pinecone_environment: str,
    index_name: str,
    google_api_key: str,
    **kwargs
) -> RAGPipeline:
    """Create a complete RAG pipeline ready to answer questions."""
    return RAGPipeline(
        pinecone_api_key=pinecone_api_key,
        pinecone_environment=pinecone_environment,
        index_name=index_name,
        google_api_key=google_api_key,
        **kwargs
    )

# Example usage:
"""
# Initialize pipeline with Gemini Flash
rag = create_rag_pipeline(
    pinecone_api_key="your-pinecone-key",
    pinecone_environment="your-env",
    index_name="your-index",
    google_api_key="your-google-api-key"
)

# Ask questions
result = rag.query("What is covered under my health insurance policy?")
print(f"Answer: {result['answer']}")
print(f"Based on {result['retrieval_count']} retrieved documents")
"""