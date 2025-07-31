# vm_optimized_config.py

# CPU VM-optimized settings for faster performance (No GPU needed!)

VM_OPTIMIZED_SETTINGS = {
    # Embedder settings (CPU optimized)
    "batch_size": 32,  # Increase from 16 (more CPU cores)
    "max_text_length": 400,  # Slightly reduced for speed
    "embedding_device": "cpu",  # CPU with multiple cores
    "cpu_cores": 16,  # Use multi-core processing
    
    # Parser settings
    "chunk_sizes": [400, 200, 100],  # Smaller chunks = faster processing
    "chunk_overlap": 30,  # Reduced overlap for speed
    "parallel_processing": True,  # Enable parallel chunks
    
    # Pinecone settings (cloud to cloud advantage)
    "upsert_batch_size": 100,  # Larger batches for cloud
    "max_connections": 8,  # More concurrent connections
    "timeout": 45,  # Reasonable timeout
    
    # Answer generation
    "max_context_tokens": 3000,  # Balanced for speed
    "gemini_timeout": 20,  # Faster timeout
    "temperature": 0.1,  # Lower for consistent speed
    
    # Memory settings (CPU VM)
    "worker_threads": 8,  # Parallel processing
    "memory_limit": "8GB",  # Standard VM memory
    "cpu_optimization": True,  # Enable CPU-specific optimizations
}

# Expected performance improvements (CPU VM vs Local):
PERFORMANCE_GAINS = {
    "embedding_speed": "3-5x faster (multi-core CPU vs single-core)",
    "network_latency": "4-6x faster (cloud to cloud vs home internet)",
    "batch_processing": "2-3x faster (more RAM + cores)",
    "overall_pipeline": "5-10x faster total",
    "concurrent_requests": "5x more throughput",
    "memory_efficiency": "Better RAM management"
}

# Cost-effective cloud options without GPU:
RECOMMENDED_VMS = {
    "AWS": "c5.2xlarge (8 vCPU, 16GB RAM) - $0.34/hour",
    "Google Cloud": "c2-standard-8 (8 vCPU, 32GB RAM) - $0.35/hour", 
    "Azure": "Standard_F8s_v2 (8 vCPU, 16GB RAM) - $0.33/hour",
    "DigitalOcean": "c-8 (8 vCPU, 16GB RAM) - $0.238/hour"
}

print("🚀 CPU VM Deployment will still give great performance!")
print("Expected speedup: 5-10x faster than local development")
print("Cost: ~$0.30-0.35/hour for high-performance CPU VM")
