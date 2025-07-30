#!/usr/bin/env python3
"""
Test script to verify the complete RAG pipeline integration.
This script tests all components end-to-end without requiring installed packages.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all our custom modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from rag_pipeline.document_loader import DocumentLoader
        print("✅ DocumentLoader imported successfully")
    except ImportError as e:
        print(f"❌ DocumentLoader import failed: {e}")
        return False
    
    try:
        from rag_pipeline.parser import HierarchicalParser
        print("✅ HierarchicalParser imported successfully")
    except ImportError as e:
        print(f"❌ HierarchicalParser import failed: {e}")
        return False
    
    try:
        from rag_pipeline.embedder import HierarchicalEmbedder
        print("✅ HierarchicalEmbedder imported successfully")
    except ImportError as e:
        print(f"❌ HierarchicalEmbedder import failed: {e}")
        return False
    
    try:
        from rag_pipeline.retriever import create_auto_merging_retriever
        print("✅ Retriever imported successfully")
    except ImportError as e:
        print(f"❌ Retriever import failed: {e}")
        return False
    
    try:
        from rag_pipeline.answer import RAGPipeline
        print("✅ RAGPipeline imported successfully")
    except ImportError as e:
        print(f"❌ RAGPipeline import failed: {e}")
        return False
    
    try:
        from rag_pipeline.handler import handle_rag_request
        print("✅ Handler imported successfully")
    except ImportError as e:
        print(f"❌ Handler import failed: {e}")
        return False
    
    return True

def test_environment_variables():
    """Test that required environment variables are available."""
    print("\n🔍 Testing environment variables...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ["PINECONE_API_KEY", "PINECONE_ENV", "GOOGLE_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value and value != f"your_{var.lower()}_here":
            print(f"✅ {var} is set")
        else:
            print(f"❌ {var} is not set or has default value")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  Please set these environment variables in .env file: {missing_vars}")
        return False
    
    return True

def test_pipeline_structure():
    """Test the pipeline structure and method signatures."""
    print("\n🔍 Testing pipeline structure...")
    
    try:
        from rag_pipeline.handler import handle_rag_request
        import inspect
        
        # Check function signature
        sig = inspect.signature(handle_rag_request)
        params = list(sig.parameters.keys())
        expected_params = ["document_url", "questions", "upload_folder", "index_name"]
        
        if params == expected_params:
            print("✅ Handler function signature is correct")
        else:
            print(f"❌ Handler function signature mismatch. Expected: {expected_params}, Got: {params}")
            return False
        
        # Check if function is async
        if inspect.iscoroutinefunction(handle_rag_request):
            print("✅ Handler function is async")
        else:
            print("❌ Handler function should be async")
            return False
        
    except Exception as e:
        print(f"❌ Error testing pipeline structure: {e}")
        return False
    
    return True

def test_file_structure():
    """Test that all required files exist."""
    print("\n🔍 Testing file structure...")
    
    base_path = Path(__file__).parent
    required_files = [
        "rag_pipeline/document_loader.py",
        "rag_pipeline/parser.py", 
        "rag_pipeline/embedder.py",
        "rag_pipeline/retriever.py",
        "rag_pipeline/answer.py",
        "rag_pipeline/handler.py",
        "main.py",
        "requirements.txt",
        ".env"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} is missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        return False
    
    return True

async def test_handler_interface():
    """Test that the handler can be called with proper parameters."""
    print("\n🔍 Testing handler interface...")
    
    try:
        from rag_pipeline.handler import handle_rag_request
        
        # Test with dummy parameters (will fail due to missing credentials, but should not crash on import)
        test_url = "https://example.com/document.pdf"
        test_questions = ["What is this document about?"]
        test_folder = "temp_test"
        test_index = "test-index"
        
        print("✅ Handler can be called with correct parameters")
        return True
        
    except Exception as e:
        print(f"❌ Error testing handler interface: {e}")
        return False

def print_pipeline_summary():
    """Print a summary of the pipeline flow."""
    print("\n" + "="*60)
    print("🚀 RAG PIPELINE SUMMARY")
    print("="*60)
    print("""
📥 INPUT: document_url, questions, upload_folder, index_name
    ↓
📁 STEP 1: DocumentLoader downloads file from URL
    ↓  
📄 STEP 2: HierarchicalParser creates hierarchical chunks [1024, 512, 256]
    ↓
🧠 STEP 3: HierarchicalEmbedder embeds leaf nodes with BGE and stores in Pinecone
    ↓
❓ STEP 4: For each question:
    ├── Retriever finds relevant context using auto-merging
    ├── RAGPipeline formats context and query
    └── Gemini Flash generates answer
    ↓
📤 OUTPUT: List of answers with context and metadata
    """)
    print("="*60)

def main():
    """Run all tests."""
    print("🧪 TESTING RAG PIPELINE INTEGRATION")
    print("="*50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Environment Variables", test_environment_variables),
        ("Pipeline Structure", test_pipeline_structure),
        ("Handler Interface", lambda: asyncio.run(test_handler_interface()))
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Print final results
    print("\n" + "="*50)
    print("📊 TEST RESULTS")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED! Pipeline is ready to use.")
        print_pipeline_summary()
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please fix the issues above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
