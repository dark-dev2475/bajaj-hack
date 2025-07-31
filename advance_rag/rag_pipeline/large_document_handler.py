# large_document_handler.py

import logging
from typing import List, Dict, Any, Optional, Tuple
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LargeDocumentHandler:
    """Handles very large documents that exceed token limits."""
    
    def __init__(
        self,
        max_tokens_per_chunk: int = 400,  # Conservative limit for embeddings
        max_context_tokens: int = 3000,   # For final answer generation
        overlap_ratio: float = 0.1,      # 10% overlap
        encoding_name: str = "cl100k_base"  # GPT-4 tokenizer
    ):
        """
        Initialize the large document handler.
        
        Args:
            max_tokens_per_chunk: Maximum tokens per embedding chunk
            max_context_tokens: Maximum tokens for context in answer generation
            overlap_ratio: Overlap between chunks as ratio
            encoding_name: Tokenizer encoding name
        """
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.max_context_tokens = max_context_tokens
        self.overlap_ratio = overlap_ratio
        
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(f"Could not load tokenizer {encoding_name}: {e}")
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: rough estimation (4 chars per token)
            return len(text) // 4
    
    def smart_chunk_large_document(
        self,
        document_content: str,
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Smart chunking that respects token limits and content boundaries.
        
        Args:
            document_content: Large document text
            metadata: Document metadata
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if metadata is None:
            metadata = {}
        
        total_tokens = self.count_tokens(document_content)
        logger.info(f"Processing document with {total_tokens} tokens")
        
        if total_tokens <= self.max_tokens_per_chunk:
            # Document is small enough, return as single chunk
            return [{
                "text": document_content,
                "metadata": {**metadata, "chunk_index": 0, "total_chunks": 1},
                "token_count": total_tokens
            }]
        
        # Calculate chunk size in characters (approximate)
        avg_chars_per_token = len(document_content) / total_tokens if total_tokens > 0 else 4
        target_chunk_chars = int(self.max_tokens_per_chunk * avg_chars_per_token)
        overlap_chars = int(target_chunk_chars * self.overlap_ratio)
        
        # Use hierarchical splitting for better content preservation
        splitters = [
            RecursiveCharacterTextSplitter(
                chunk_size=target_chunk_chars,
                chunk_overlap=overlap_chars,
                separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
        ]
        
        chunks = []
        for splitter in splitters:
            try:
                text_chunks = splitter.split_text(document_content)
                break
            except Exception as e:
                logger.warning(f"Splitter failed: {e}")
                continue
        else:
            # Fallback: simple character-based splitting
            text_chunks = self._fallback_split(document_content, target_chunk_chars, overlap_chars)
        
        # Process and validate chunks
        valid_chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_tokens = self.count_tokens(chunk_text)
            
            if chunk_tokens > self.max_tokens_per_chunk:
                # Further split oversized chunks
                sub_chunks = self._force_split_chunk(chunk_text, chunk_tokens)
                for j, sub_chunk in enumerate(sub_chunks):
                    valid_chunks.append({
                        "text": sub_chunk["text"],
                        "metadata": {
                            **metadata,
                            "chunk_index": len(valid_chunks),
                            "parent_chunk": i,
                            "sub_chunk": j,
                            "total_chunks": "TBD"  # Will update later
                        },
                        "token_count": sub_chunk["token_count"]
                    })
            else:
                valid_chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": len(valid_chunks),
                        "total_chunks": "TBD"  # Will update later
                    },
                    "token_count": chunk_tokens
                })
        
        # Update total chunks count
        for chunk in valid_chunks:
            chunk["metadata"]["total_chunks"] = len(valid_chunks)
        
        logger.info(f"Split document into {len(valid_chunks)} chunks")
        return valid_chunks
    
    def _fallback_split(
        self,
        text: str,
        target_size: int,
        overlap: int
    ) -> List[str]:
        """Fallback character-based splitting."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + target_size
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to find a good break point
            break_point = end
            for delimiter in ["\n\n", "\n", ". ", " "]:
                candidate = text.rfind(delimiter, start, end)
                if candidate > start:
                    break_point = candidate + len(delimiter)
                    break
            
            chunks.append(text[start:break_point])
            start = break_point - overlap
        
        return chunks
    
    def _force_split_chunk(
        self,
        chunk_text: str,
        chunk_tokens: int
    ) -> List[Dict[str, Any]]:
        """Force split a chunk that's still too large."""
        target_chars = len(chunk_text) * self.max_tokens_per_chunk // chunk_tokens
        
        sub_chunks = []
        start = 0
        
        while start < len(chunk_text):
            end = min(start + target_chars, len(chunk_text))
            sub_text = chunk_text[start:end]
            
            # Validate token count
            sub_tokens = self.count_tokens(sub_text)
            if sub_tokens > self.max_tokens_per_chunk and len(sub_text) > 100:
                # Further reduce if still too large
                reduction_factor = self.max_tokens_per_chunk / sub_tokens
                new_end = start + int((end - start) * reduction_factor)
                sub_text = chunk_text[start:new_end]
                sub_tokens = self.count_tokens(sub_text)
            
            sub_chunks.append({
                "text": sub_text,
                "token_count": sub_tokens
            })
            
            start = end
        
        return sub_chunks
    
    def optimize_retrieval_context(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        query: str
    ) -> Tuple[str, int]:
        """
        Optimize retrieved context to fit within token limits.
        
        Args:
            retrieved_chunks: List of retrieved chunk dictionaries
            query: Original query
            
        Returns:
            Tuple of (optimized_context, token_count)
        """
        query_tokens = self.count_tokens(query)
        available_tokens = self.max_context_tokens - query_tokens - 200  # Buffer for prompt
        
        if available_tokens <= 0:
            logger.warning("Query too long, using minimal context")
            return retrieved_chunks[0]["text"][:500] if retrieved_chunks else "", 500
        
        # Sort chunks by relevance score if available
        sorted_chunks = sorted(
            retrieved_chunks,
            key=lambda x: x.get("score", 0.0),
            reverse=True
        )
        
        context_parts = []
        current_tokens = 0
        
        for chunk in sorted_chunks:
            chunk_text = chunk.get("text", "")
            chunk_tokens = chunk.get("token_count") or self.count_tokens(chunk_text)
            
            if current_tokens + chunk_tokens > available_tokens:
                # Try to fit partial chunk
                remaining_tokens = available_tokens - current_tokens
                if remaining_tokens > 100:  # Only if meaningful space left
                    chars_per_token = len(chunk_text) / chunk_tokens if chunk_tokens > 0 else 4
                    partial_chars = int(remaining_tokens * chars_per_token)
                    partial_text = chunk_text[:partial_chars]
                    context_parts.append(f"[Partial] {partial_text}...")
                break
            
            context_parts.append(chunk_text)
            current_tokens += chunk_tokens
        
        final_context = "\n\n".join(context_parts)
        final_tokens = self.count_tokens(final_context)
        
        logger.info(f"Optimized context: {final_tokens} tokens from {len(sorted_chunks)} chunks")
        return final_context, final_tokens

# Factory function
def create_large_document_handler(**kwargs) -> LargeDocumentHandler:
    """Create a large document handler with custom settings."""
    return LargeDocumentHandler(**kwargs)
