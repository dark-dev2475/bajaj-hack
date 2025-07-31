from typing import Any, Dict, List, Optional, Sequence, Tuple

from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.schema import BaseNode, Document, NodeRelationship, IndexNode, TextNode
from llama_index.core.utils import get_tqdm_iterable
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer

class ParentToSentenceNodeParser(NodeParser):
    """Parent to Sentence node parser.

    Splits a document into a Parent nodes and sentence nodes referring to parent node
    using a NodeParser.
    """

    chunk_size: int = Field(
        default=None,
        description=(
            "The chunk size to use when splitting documents"
        ),
    )
    
    node_parser: SentenceSplitter = Field(
        default=SentenceSplitter,
        description=(
 "The chunk size to use when splitting documents"
        ),
    )

    @classmethod
    def from_defaults(
        cls,
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        node_parser: SentenceSplitter = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "ParentToSentenceNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        
        node_parser = SentenceSplitter(
                    chunk_size=chunk_size,
                    callback_manager=callback_manager,
                    chunk_overlap=chunk_overlap,
                    include_metadata=include_metadata,
                    include_prev_next_rel=include_prev_next_rel,
                    )

        return cls(
            chunk_size=chunk_size,
            node_parser=node_parser,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        return "ParentToSentenceNodeParser"

    def _get_nodes_from_doc(
        self,
        docs: List[Document],
        show_progress: bool = False,
    ) -> Tuple[List[BaseNode],Dict[str, BaseNode]]:
        """get nodes from docs."""

        # first split current nodes into sub-nodes
        docs_with_progress = get_tqdm_iterable(
            docs, show_progress, "Parsing documents into nodes"
        )
        base_nodes = []
        for node in docs_with_progress:
            cur_sub_nodes = self.node_parser.get_nodes_from_documents([node])
            base_nodes.extend(cur_sub_nodes)
        
        for idx, node in enumerate(base_nodes):
            node.id_ = f"node-{idx}"
        
        nodes_with_progress = get_tqdm_iterable(
            base_nodes, show_progress, "Parsing nodes into sub nodes"
        )
        sent_node_parser = split_by_sentence_tokenizer()
        all_nodes = []
        for node in nodes_with_progress:
            sub_nodes = sent_node_parser(node.text)
            sub_inodes = [
                IndexNode.from_text_node(TextNode(text=sn), node.node_id) for sn in sub_nodes
            ]
            all_nodes.extend(sub_inodes)
            
        return all_nodes, {n.node_id: n for n in base_nodes}

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> Tuple[List[BaseNode], Dict[str, BaseNode]]:
        """Parse document into nodes.

        Args:
            documents (Sequence[Document]): documents to parse
            show_progress (bool): whether to show progress

        Returns:
            Tuple of (sentence_nodes, parent_nodes_dict)
        """
        with self.callback_manager.event(
            CBEventType.NODE_PARSING, payload={EventPayload.DOCUMENTS: documents}
        ) as event:
            all_nodes: List[BaseNode] = []
            all_base_nodes_dict: Dict[str, BaseNode] = dict()
            documents_with_progress = get_tqdm_iterable(
                documents, show_progress, "Parsing documents into nodes"
            )
            for doc in documents_with_progress:
                nodes_from_doc, base_nodes_dict = self._get_nodes_from_doc([doc])
                all_nodes.extend(nodes_from_doc)
                all_base_nodes_dict.update(base_nodes_dict)
            event.on_end(payload={EventPayload.NODES: all_nodes})

        return all_nodes, all_base_nodes_dict
    
    def get_sentence_nodes_only(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse document into sentence nodes only (for compatibility).

        Args:
            documents (Sequence[Document]): documents to parse
            show_progress (bool): whether to show progress

        Returns:
            List of sentence nodes only
        """
        all_nodes, _ = self.get_nodes_from_documents(documents, show_progress, **kwargs)
        return all_nodes

    # Unused abstract method
    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        return list(nodes)

# Example usage (commented out - uncomment when you have documents to parse):
# p2s_node_parser = ParentToSentenceNodeParser.from_defaults(chunk_size=1024, chunk_overlap=0)
# all_nodes_sent, all_base_nodes_dict_sent = p2s_node_parser.get_nodes_from_documents(docs)
# print(f"Number of sentence nodes: {len(all_nodes_sent)}")