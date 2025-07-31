import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import uuid  # For generating stable unique IDs

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

# LlamaIndex imports for better auto-merging
try:
    from llama_index.core.node_parser import HierarchicalNodeParser
    from llama_index.core.schema import TextNode, BaseNode
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False

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
    A parser that creates hierarchical representation using LlamaIndex HierarchicalNodeParser when available,
    falls back to custom implementation otherwise.
    """
    
    def __init__(
        self,
        chunk_sizes: List[int] = [800, 400, 200],  # Accuracy-optimized for richer context
        chunk_overlap: int = 20,  # Added overlap for better continuity
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
        
        # Try to use LlamaIndex HierarchicalNodeParser
        if LLAMAINDEX_AVAILABLE:
            try:
                self.llama_parser = HierarchicalNodeParser.from_defaults(
                    chunk_sizes=chunk_sizes,
                    chunk_overlap=chunk_overlap
                )
                logger.info("Using LlamaIndex HierarchicalNodeParser for optimal hierarchical parsing")
                self.use_llamaindex = True
            except Exception as e:
                logger.warning(f"Failed to initialize LlamaIndex HierarchicalNodeParser: {e}")
                self.use_llamaindex = False
        else:
            logger.info("LlamaIndex not available, using custom hierarchical parser")
            self.use_llamaindex = False
        
        # Fallback: Create text splitters for each level (custom implementation)
        if not self.use_llamaindex:
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
            # Generate stable unique ID for this node
            node_id = str(uuid.uuid4())
            
            # Add parent_id to metadata if this node has a parent
            node_metadata = {
                **metadata,
                "id": node_id,
                "chunk_size": self.chunk_sizes[level],
                "chunk_index": i,
                "level": level
            }
            
            if parent:
                node_metadata["parent_id"] = parent.metadata["id"]
            
            node = HierarchicalNode(
                content=split,
                metadata=node_metadata,
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
        Parse a document into a hierarchical structure using LlamaIndex when available.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of top-level hierarchical nodes
        """
        try:
            # Load the document
            docs = self._load_document(file_path)
            
            if self.use_llamaindex:
                # Use LlamaIndex HierarchicalNodeParser
                return self._parse_with_llamaindex(docs, file_path)
            else:
                # Use custom hierarchical parser
                return self._parse_with_custom(docs, file_path)
            
        except Exception as e:
            logger.exception(f"Error parsing document {file_path}: {str(e)}")
            raise
    
    def _parse_with_llamaindex(self, docs: List[Document], file_path: str) -> List[HierarchicalNode]:
        """Parse using LlamaIndex HierarchicalNodeParser."""
        try:
            from llama_index.core.schema import Document as LlamaDocument
            
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
            
            # Parse with LlamaIndex
            llama_nodes = self.llama_parser.get_nodes_from_documents(llama_docs)
            
            # Convert LlamaIndex nodes to our HierarchicalNode format
            hierarchical_nodes = []
            for llama_node in llama_nodes:
                # Extract level from metadata or node type
                level = llama_node.metadata.get('level', 0)
                
                # Ensure stable ID exists
                node_id = getattr(llama_node, 'id_', str(uuid.uuid4()))
                
                node_metadata = {
                    **llama_node.metadata,
                    "id": node_id,
                    "level": level,
                    "source": file_path
                }
                
                hierarchical_node = HierarchicalNode(
                    content=llama_node.text,
                    metadata=node_metadata,
                    level=level
                )
                hierarchical_nodes.append(hierarchical_node)
            
            logger.info(f"LlamaIndex parsed {len(hierarchical_nodes)} hierarchical nodes")
            return hierarchical_nodes
            
        except Exception as e:
            logger.error(f"LlamaIndex parsing failed: {e}, falling back to custom parser")
            return self._parse_with_custom(docs, file_path)
    
    def _parse_with_custom(self, docs: List[Document], file_path: str) -> List[HierarchicalNode]:
        """Parse using custom hierarchical implementation."""
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
        
        logger.info(f"Custom parser created {len(all_nodes)} top-level nodes")
        return all_nodes
    
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
    
    def to_llamaindex_nodes(self, nodes: List[HierarchicalNode]) -> List[Dict[str, Any]]:
        """
        Convert hierarchical nodes to LlamaIndex-compatible format with parent-child relationships.
        If LlamaIndex was used for parsing, return nodes in their native format.
        """
        if not LLAMAINDEX_AVAILABLE:
            # Fallback to current format if LlamaIndex not available
            return self._to_dict_nodes(nodes)
        
        if self.use_llamaindex:
            # If we used LlamaIndex for parsing, nodes are already in optimal format
            logger.info("Nodes already parsed with LlamaIndex HierarchicalNodeParser - optimal for auto-merging")
            
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
                    "parser_type": "llamaindex" if self.use_llamaindex else "custom"
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
        
        parser_type = "LlamaIndex HierarchicalNodeParser" if self.use_llamaindex else "custom parser"
        logger.info(f"Converted {len(llamaindex_nodes)} nodes from {parser_type} to LlamaIndex format")
        return llamaindex_nodes
    
    def _to_dict_nodes(self, nodes: List[HierarchicalNode]) -> List[Dict[str, Any]]:
        """Convert hierarchical nodes to dictionary format (fallback)."""
        dict_nodes = []
        
        def process_node(node: HierarchicalNode):
            dict_node = {
                "text": node.content,
                "metadata": node.metadata,
                "level": node.level
            }
            dict_nodes.append(dict_node)
            
            for child in node.children:
                process_node(child)
        
        for node in nodes:
            process_node(node)
        
        return dict_nodes