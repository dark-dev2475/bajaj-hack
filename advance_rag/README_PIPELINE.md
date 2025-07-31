# Advanced RAG Pipeline

A complete Retrieval-Augmented Generation (RAG) pipeline using hierarchical document processing, BGE embeddings, Pinecone vector storage, and Gemini Flash for answer generation.

## 🚀 Quick Start

### 1. Setup Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENV=us-west1-gcp
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API Server

```bash
uvicorn main:app --reload
```

### 4. Test the Pipeline

Send a POST request to `http://localhost:8000/hackrx/run`:

```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is this document about?",
    "What are the key points?"
  ]
}
```

## 📁 Architecture

### Pipeline Components

```
📥 main.py (FastAPI Server)
    ↓
📋 rag_pipeline/handler.py (Main Orchestrator)
    ↓
📁 document_loader.py → Downloads files from URLs
    ↓  
📄 parser.py → Creates hierarchical chunks [1024, 512, 256]
    ↓
🧠 embedder.py → BGE embeddings + Pinecone storage
    ↓
🔍 retriever.py → Auto-merging retrieval
    ↓
🤖 answer.py → Gemini Flash answer generation
    ↓
📤 Structured response with answers + context
```

### Key Features

- **Hierarchical Document Processing**: Multi-level chunking (1024, 512, 256 tokens)
- **BGE Embeddings**: High-quality semantic embeddings using `BAAI/bge-small-en-v1.5`
- **Pinecone Vector Storage**: Scalable vector database with cosine similarity
- **Auto-Merging Retrieval**: Intelligent context merging for better answers
- **Gemini Flash LLM**: Fast and cost-effective answer generation
- **Async Processing**: Concurrent document downloading and processing

## 🔧 Component Details

### Document Loader (`document_loader.py`)
- Downloads documents from URLs asynchronously
- Supports: PDF, DOCX, TXT, EML, HTML
- MIME type detection and validation
- Error handling and retry logic

### Hierarchical Parser (`parser.py`)
- Creates multi-level document chunks
- Uses LangChain text splitters
- Maintains parent-child relationships
- Generates leaf nodes for embedding

### Embedder (`embedder.py`)
- Uses BGE model for high-quality embeddings
- Batch processing for efficiency
- Vector normalization for consistent similarity
- Direct Pinecone integration

### Retriever (`retriever.py`)
- Base retriever for similarity search
- Auto-merging retriever for context combination
- Configurable similarity thresholds
- Verbose logging for debugging

### Answer Generator (`answer.py`)
- Complete RAG pipeline orchestration
- Gemini Flash integration
- Context formatting and prompt engineering
- Structured response generation

### Handler (`handler.py`)
- Main pipeline orchestrator
- Connects all components
- Processes multiple questions per document
- Error handling and logging

## 🛠️ Usage Examples

### Programmatic Usage

```python
from rag_pipeline.handler import handle_rag_request

# Process a document and answer questions
answers = await handle_rag_request(
    document_url="https://example.com/policy.pdf",
    questions=[
        "What is covered under this policy?",
        "What are the exclusions?",
        "How do I file a claim?"
    ],
    upload_folder="temp_docs",
    index_name="insurance-policies"
)

# Process results
for answer in answers:
    print(f"Q: {answer['question']}")
    print(f"A: {answer['answer']}")
    print(f"Status: {answer['status']}")
    print("---")
```

### Direct Component Usage

```python
from rag_pipeline.document_loader import DocumentLoader
from rag_pipeline.parser import HierarchicalParser
from rag_pipeline.embedder import HierarchicalEmbedder
from rag_pipeline.answer import RAGPipeline

# 1. Download document
loader = DocumentLoader("temp_docs")
results = await loader.download_files(["https://example.com/doc.pdf"])

# 2. Parse into hierarchy
parser = HierarchicalParser(chunk_sizes=[1024, 512, 256])
nodes = parser.parse_document(results[0]["file_path"])
leaves = parser.get_leaf_nodes(nodes)

# 3. Embed and store
embedder = HierarchicalEmbedder(
    pinecone_api_key="your-key",
    pinecone_environment="your-env", 
    index_name="your-index"
)
await embedder.embed_and_store(leaves)

# 4. Answer questions
rag = RAGPipeline(
    pinecone_api_key="your-key",
    pinecone_environment="your-env",
    index_name="your-index",
    google_api_key="your-google-key"
)
result = rag.query("What is this document about?")
```

## 🧪 Testing

Run the test suite to verify all components:

```bash
python test_pipeline.py
```

Run simple structure test:

```bash
python test_simple.py
```

## 📦 Dependencies

### Core Libraries
- `fastapi` - Web API framework
- `uvicorn` - ASGI server
- `python-dotenv` - Environment variable management

### Document Processing
- `langchain` - Document loading and text splitting
- `langchain-community` - Additional document loaders
- `aiohttp` - Async HTTP client for downloads
- `pypdf` - PDF processing
- `docx2txt` - Word document processing
- `unstructured[email]` - Email processing

### Embeddings & Vector Storage
- `sentence-transformers` - BGE embedding model
- `pinecone-client` - Vector database client
- `numpy` - Numerical operations

### LLM Integration
- `google-generativeai` - Gemini Flash API

### Utilities
- `tqdm` - Progress bars
- `python-magic` - File type detection (optional)

## 🔐 Environment Configuration

Required environment variables:

```bash
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment  # e.g., us-west1-gcp

# Google Gemini Configuration  
GOOGLE_API_KEY=your_gemini_api_key

# Optional: Custom settings
UPLOAD_FOLDER=temp_docs
INDEX_NAME=rag-index
```

## 📊 Performance Notes

- **Embedding Model**: BGE-small-en-v1.5 (384 dimensions)
- **Chunk Sizes**: 1024 → 512 → 256 tokens (hierarchical)
- **Retrieval**: Top-5 similarity search with auto-merging
- **LLM**: Gemini Flash (fast, cost-effective)
- **Vector DB**: Cosine similarity with normalized vectors

## 🚨 Error Handling

The pipeline includes comprehensive error handling:

- Document download failures
- Parsing errors for unsupported formats
- Embedding generation issues
- Vector storage problems
- LLM API failures

Each component logs detailed information for debugging.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `python test_pipeline.py`
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.
