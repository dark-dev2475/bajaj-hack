@echo off
REM Gateway Timeout Prevention - Optimized Server Startup for Windows

echo 🚀 Starting RAG Pipeline with timeout optimizations...

REM Set environment variables for performance
set TIMEOUT_SECONDS=300
set MAX_WORKERS=4
set BATCH_SIZE=32
set MAX_CONTEXT_LENGTH=3000

REM Start server with optimized configuration
echo 📡 Starting Uvicorn server with timeout settings...

uvicorn main:app ^
    --host 0.0.0.0 ^
    --port 8000 ^
    --timeout-keep-alive 300 ^
    --limit-concurrency 10 ^
    --limit-max-requests 1000 ^
    --backlog 2048 ^
    --workers 1 ^
    --log-level info

echo ✅ Server started with gateway timeout prevention!
echo 🔗 Access API at: http://localhost:8000
echo 📚 API docs at: http://localhost:8000/docs
echo ⏱️  Max request timeout: 5 minutes

pause
