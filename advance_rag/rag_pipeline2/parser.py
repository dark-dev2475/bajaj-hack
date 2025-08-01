# /rag_pipeline/parser.py

import logging
from typing import List
from pathlib import Path

# LlamaIndex Imports
from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.readers.file import PyMuPDFReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HierarchicalParser:
    """Parses documents into a hierarchical structure using LlamaIndex."""

    def __init__(self, chunk_sizes: List[int] = [800, 400, 200]):
        self.node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=chunk_sizes,
            chunk_overlap=40
        )
        self.loader = PyMuPDFReader()

    def parse_document(self, file_path: str) -> List:
        """
        Loads a PDF and parses it into hierarchical nodes.
        
        Returns:
            List of all nodes (parents and leaves).
        """
        try:
            # Load the document
            docs0 = self.loader.load_data(file_path=Path(file_path))
            
            # Combine text content, excluding the first two pages (table of contents)
            full_text = "\n\n".join([doc.get_content() for doc in docs0])
            documents = [Document(text=full_text)]

            
            # Get nodes
            nodes = self.node_parser.get_nodes_from_documents(documents)
            logger.info(f"Successfully parsed document into {len(nodes)} nodes.")
            return nodes
        except Exception as e:
            logger.error(f"Error parsing document {file_path}: {e}")
            raise