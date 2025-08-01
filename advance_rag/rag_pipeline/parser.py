import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import uuid  # For generating stable unique IDs

# Document processing
from langchain.docstore.document import Document
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredHTMLLoader
)

# LlamaIndex imports for hierarchical parsing
try:
    from llama_index.core.node_parser import HierarchicalNodeParser
    from llama_index.core.node_parser import get_leaf_nodes
    from llama_index.core.schema import TextNode, BaseNode, Document as LlamaDocument
    LLAMAINDEX_AVAILABLE = True
except ImportError as e:
    LLAMAINDEX_AVAILABLE = False
    print(f"LlamaIndex import error: {e}")
    raise ImportError("LlamaIndex is required. Please install: pip install llama-index")

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
    Simplified parser that uses only LlamaIndex HierarchicalNodeParser.
    No fallback - requires LlamaIndex to be installed.
    """
    
    def __init__(
        self,
        chunk_sizes: List[int] = [800, 400, 200],  # Accuracy-optimized for richer context
        chunk_overlap: int = 50,  # Added overlap for better continuity
    ):
        """
        Initialize the parser with LlamaIndex HierarchicalNodeParser.
        
        Args:
            chunk_sizes: List of chunk sizes for each level (from largest to smallest)
            chunk_overlap: Number of characters to overlap between chunks
        """
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError("LlamaIndex is required but not available. Please install: pip install llama-index")
        
        self.chunk_sizes = chunk_sizes
        self.chunk_overlap = chunk_overlap
        
        # Initialize LlamaIndex HierarchicalNodeParser
        try:
            self.llama_parser = HierarchicalNodeParser.from_defaults(
                chunk_sizes=chunk_sizes,
                chunk_overlap=chunk_overlap
            )
            logger.info("Using LlamaIndex HierarchicalNodeParser for optimal hierarchical parsing")
        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndex HierarchicalNodeParser: {e}")
            raise RuntimeError(f"LlamaIndex HierarchicalNodeParser initialization failed: {e}")
    
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
    
   
    
    def parse_document(self, file_path: str) -> List[HierarchicalNode]:
        """
        Parse a document into a hierarchical structure using LlamaIndex HierarchicalNodeParser.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of top-level hierarchical nodes
        """
        try:
            # Load the document
            docs = self._load_document(file_path)
            
            # Convert LangChain documents to LlamaIndex documents
            llama_docs = []
            for doc in docs:
                llama_doc = LlamaDocument(
                    text=doc.page_content,
                    metadata={
                        "source": file_path,
                        **doc.metadata
                    }
                )
                llama_docs.append(llama_doc)
            
            # Parse with LlamaIndex HierarchicalNodeParser
            llama_nodes = self.llama_parser.get_nodes_from_documents(llama_docs)
            
            # Convert LlamaIndex nodes to our HierarchicalNode format for compatibility
           
                
                
            return llama_nodes
            
        except Exception as e:
            logger.exception(f"Error parsing document {file_path}: {str(e)}")
            raise
    
   
    
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
    
    def to_llamaindex_nodes(self, nodes: List[HierarchicalNode]) -> List[Dict[str, Any]]:
        """
        Convert hierarchical nodes to LlamaIndex-compatible format with parent-child relationships.
        Since we're using LlamaIndex HierarchicalNodeParser, nodes are already optimized.
        """
        llamaindex_nodes = []
        
        def process_node(node: HierarchicalNode, parent_id: Optional[str] = None):
            # Use the stable ID already assigned during parsing
            node_id = node.metadata.get("id", str(uuid.uuid4()))
            
            # Create LlamaIndex-compatible node
            text_node = {
                "id": node_id,
                "text": node.content,
                "metadata": {
                    **node.metadata,
                    "node_type": "hierarchical",
                    "level": node.level,
                    "parser_type": "llamaindex_hierarchical"
                },
                "level": node.level,
                "parent_id": parent_id
            }
            
            llamaindex_nodes.append(text_node)
            
            # Process children with this node as parent
            for child in node.children:
                process_node(child, parent_id=node_id)
        
        # Process all top-level nodes
        for node in nodes:
            process_node(node)
        
        logger.info(f"Converted {len(llamaindex_nodes)} nodes from LlamaIndex HierarchicalNodeParser for AutoMergingRetriever")
        return llamaindex_nodes