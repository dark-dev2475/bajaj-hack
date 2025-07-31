"""
Accuracy-Optimized Configuration for RAG Pipeline
================================================

This configuration prioritizes accuracy over speed while maintaining reasonable performance.
It balances context richness, chunk granularity, and retrieval quality.
"""

# Embedding Configuration - Accuracy Focused
EMBEDDING_CONFIG = {
    "model_name": "BAAI/bge-small-en-v1.5",
    "batch_size": 16,  # Increased from 10 for better efficiency, reduced from 32 for stability
    "max_text_length": 800,  # Increased from 400 to capture more context per chunk
    "device": "cpu",  # VM optimized for CPU
    "normalize_embeddings": True,
    "show_progress": True
}

# Document Parsing Configuration - Better Granularity
PARSER_CONFIG = {
    "chunk_sizes": [800, 400, 200],  # Increased from [400, 200, 100] for richer context
    "chunk_overlap": 50,  # Increased from 0 for better continuity
    "separators": ["\n\n", "\n", ". ", " ", ""],  # Added sentence separator for better boundaries
    "max_tokens_per_chunk": 600  # Increased from 300 for more complete thoughts
}

# Retrieval Configuration - Enhanced Context
RETRIEVAL_CONFIG = {
    "similarity_top_k": 5,  # Increased from 3 for more comprehensive context
    "similarity_threshold": 0.7,  # Filter out less relevant chunks
    "rerank_top_k": 5,  # Keep top results after reranking
    "namespace": "default"
}

# Answer Generation Configuration - Quality Focused
ANSWER_CONFIG = {
    "max_context_length": 5000,  # Increased from 3000 for richer context
    "temperature": 0.3,  # Increased from 0.1 for more nuanced responses
    "max_output_tokens": 800,  # Increased from 500 for more detailed answers
    "model": "gemini-1.5-flash"
}

# Timeout Configuration - Balanced Performance
TIMEOUT_CONFIG = {
    "embedding_timeout": 180,  # 3 minutes for embedding
    "retrieval_timeout": 120,  # 2 minutes for retrieval
    "generation_timeout": 300,  # 5 minutes for answer generation
    "total_timeout": 300  # 5 minutes total pipeline timeout
}

# Vector Store Configuration
VECTOR_CONFIG = {
    "metric": "cosine",
    "dimension": 384,  # BGE model dimension
    "cloud": "aws",
    "region": "us-east-1"
}

# Performance vs Accuracy Trade-offs
QUALITY_SETTINGS = {
    "prioritize_accuracy": True,
    "enable_context_reranking": True,
    "use_sentence_boundaries": True,
    "preserve_document_structure": True,
    "enable_overlap": True
}

def get_optimized_config():
    """Get the complete optimized configuration."""
    return {
        "embedding": EMBEDDING_CONFIG,
        "parser": PARSER_CONFIG,
        "retrieval": RETRIEVAL_CONFIG,
        "answer": ANSWER_CONFIG,
        "timeout": TIMEOUT_CONFIG,
        "vector": VECTOR_CONFIG,
        "quality": QUALITY_SETTINGS
    }

def get_accuracy_improvements():
    """
    Summary of accuracy improvements over vm_optimized_config.py:
    
    1. Increased chunk sizes from [400,200,100] to [800,400,200]
    2. Added chunk overlap (50) for better continuity
    3. Increased max_text_length from 400 to 800
    4. Increased similarity_top_k from 3 to 5
    5. Increased max_context_length from 3000 to 5000
    6. Increased temperature from 0.1 to 0.3 for nuanced responses
    7. Increased max_output_tokens from 500 to 800
    8. Added sentence separator for better chunk boundaries
    9. Balanced batch_size at 16 (between 10 and 32)
    10. Added similarity threshold for quality filtering
    """
    return {
        "chunk_size_improvement": "100% larger chunks for richer context",
        "context_improvement": "67% more context in answers",
        "retrieval_improvement": "67% more documents retrieved",
        "response_improvement": "60% longer, more detailed responses",
        "overlap_improvement": "Added continuity between chunks",
        "temperature_improvement": "More nuanced, less robotic responses"
    }
