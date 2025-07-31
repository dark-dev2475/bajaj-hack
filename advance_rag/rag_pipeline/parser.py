import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# Document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredHTMLLoader
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HierarchicalNode:
    """Represents a node in the hierarchical document structure."""
    content: str
    metadata: Dict[str, Any]
    level: int
    children: List['HierarchicalNode'] = None
    parent: Optional['HierarchicalNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

class HierarchicalParser:
    """
    A parser that creates a hierarchical representation of documents with
    multiple levels of granularity.
    """
    
    def __init__(
        self,
        chunk_sizes: List[int] = [800, 400, 200],  # Accuracy-optimized for richer context
        chunk_overlap: int = 50,  # Added overlap for better continuity
        separators: List[str] = ["\n\n", "\n", ". ", " ", ""],  # Added sentence boundary
        max_tokens_per_chunk: int = 600  # Increased for complete thoughts
    ):
        """
        Initialize the parser with specified chunk sizes for each level.
        
        Args:
            chunk_sizes: List of chunk sizes for each level (from largest to smallest)
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use for text splitting
        """
        self.chunk_sizes = chunk_sizes
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        
        # Create text splitters for each level
        self.splitters = [
            RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=chunk_overlap,
                separators=separators,
                length_function=len,
            )
            for size in chunk_sizes
        ]
    
    def _load_document(self, file_path: str) -> Document:
        """Load document based on file type."""
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        try:
            if suffix == '.pdf':
                loader = PyPDFLoader(str(path))
            elif suffix in ['.docx', '.doc']:
                loader = Docx2txtLoader(str(path))
            elif suffix == '.txt':
                loader = TextLoader(str(path))
            elif suffix in ['.eml', '.msg']:
                loader = UnstructuredEmailLoader(str(path))
            elif suffix == '.html':
                loader = UnstructuredHTMLLoader(str(path))
            else:
                raise ValueError(f"Unsupported file type: {suffix}")
            
            return loader.load()
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def _create_hierarchical_nodes(
        self,
        text: str,
        metadata: Dict[str, Any],
        level: int = 0,
        parent: Optional[HierarchicalNode] = None
    ) -> List[HierarchicalNode]:
        """Recursively create hierarchical nodes from text."""
        # Base case: if we've reached the maximum level
        if level >= len(self.chunk_sizes):
            return []
        
        # Split text using current level's splitter
        splits = self.splitters[level].split_text(text)
        
        # Create nodes for this level
        nodes = []
        for i, split in enumerate(splits):
            node = HierarchicalNode(
                content=split,
                metadata={
                    **metadata,
                    "chunk_size": self.chunk_sizes[level],
                    "chunk_index": i,
                    "level": level
                },
                level=level,
                parent=parent
            )
            
            # Recursively create child nodes
            if level < len(self.chunk_sizes) - 1:
                children = self._create_hierarchical_nodes(
                    text=split,
                    metadata=metadata,
                    level=level + 1,
                    parent=node
                )
                node.children.extend(children)
            
            nodes.append(node)
        
        return nodes
    
    def parse_document(self, file_path: str) -> List[HierarchicalNode]:
        """
        Parse a document into a hierarchical structure.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of top-level hierarchical nodes
        """
        try:
            # Load the document
            docs = self._load_document(file_path)
            
            # Process each document page/section
            all_nodes = []
            for doc in docs:
                metadata = {
                    "source": file_path,
                    **doc.metadata
                }
                
                # Create hierarchical structure
                nodes = self._create_hierarchical_nodes(
                    text=doc.page_content,
                    metadata=metadata
                )
                all_nodes.extend(nodes)
            
            logger.info(f"Created hierarchical structure with {len(all_nodes)} top-level nodes")
            return all_nodes
            
        except Exception as e:
            logger.exception(f"Error parsing document {file_path}: {str(e)}")
            raise
    
    def get_leaf_nodes(self, nodes: List[HierarchicalNode]) -> List[HierarchicalNode]:
        """Get all leaf nodes (nodes without children) from the hierarchy."""
        leaves = []
        
        def collect_leaves(node):
            if not node.children:
                leaves.append(node)
            for child in node.children:
                collect_leaves(child)
        
        for node in nodes:
            collect_leaves(node)
        
        return leaves
    
    def get_nodes_by_level(self, nodes: List[HierarchicalNode], level: int) -> List[HierarchicalNode]:
        """Get all nodes at a specific level in the hierarchy."""
        level_nodes = []
        
        def collect_level(node):
            if node.level == level:
                level_nodes.append(node)
            for child in node.children:
                collect_level(child)
        
        for node in nodes:
            collect_level(node)
        
        return level_nodes