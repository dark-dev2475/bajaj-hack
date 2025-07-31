#!/bin/bash
# Install LlamaIndex packages for enhanced hierarchical auto-merging

echo "Installing LlamaIndex core packages..."
pip install llama-index-core
pip install llama-index
pip install llama-index-vector-stores-pinecone
pip install llama-index-embeddings-huggingface
pip install llama-index-llms-gemini

echo "Verifying installation..."
python -c "
try:
    from llama_index.core.retrievers import AutoMergingRetriever
    from llama_index.core.node_parser import HierarchicalNodeParser
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    print('✅ LlamaIndex packages installed successfully!')
    print('✅ AutoMergingRetriever available')
    print('✅ HierarchicalNodeParser available') 
    print('✅ PineconeVectorStore available')
except ImportError as e:
    print(f'❌ Import error: {e}')
"

echo "Installation complete! Your RAG pipeline now has enhanced hierarchical auto-merging capabilities."
