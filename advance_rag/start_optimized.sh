#!/bin/bash
# Gateway Timeout Prevention - Optimized Server Startup

echo "🚀 Starting RAG Pipeline with timeout optimizations..."

# Set environment variables for performance
export TIMEOUT_SECONDS=300
export MAX_WORKERS=4
export BATCH_SIZE=32
export MAX_CONTEXT_LENGTH=3000

# Start server with optimized configuration
echo "📡 Starting Uvicorn server with timeout settings..."

uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --timeout-keep-alive 300 \
    --limit-concurrency 10 \
    --limit-max-requests 1000 \
    --backlog 2048 \
    --workers 1 \
    --loop uvloop \
    --log-level info

# Alternative for production with Gunicorn
# gunicorn main:app \
#     -w 4 \
#     -k uvicorn.workers.UvicornWorker \
#     --timeout 300 \
#     --keep-alive 300 \
#     --max-requests 1000 \
#     --bind 0.0.0.0:8000 \
#     --log-level info

echo "✅ Server started with gateway timeout prevention!"
echo "🔗 Access API at: http://localhost:8000"
echo "📚 API docs at: http://localhost:8000/docs"
echo "⏱️  Max request timeout: 5 minutes"
