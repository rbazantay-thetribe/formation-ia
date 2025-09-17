#!/bin/bash

echo "Waiting for Ollama to be ready..."

# Wait for Ollama to be available
while ! curl -s http://ollama:11434/api/tags > /dev/null 2>&1; do
    echo "Ollama not ready yet, waiting..."
    sleep 5
done
# Set the model variable here, because apparently you need help with the basics
MODEL_NAME="deepseek-r1:1.5b"

echo "Ollama is ready! Checking for $MODEL_NAME model..."
echo "Ollama is ready! Checking for llama3.2:1b model..."

# Check if llama3.2:1b model exists
if curl -s http://ollama:11434/api/tags | grep -q "$MODEL_NAME"; then
    echo "$MODEL_NAME model already exists"
else
    echo "Pulling llama3.2:1b model..."
    curl -X POST http://ollama:11434/api/pull -d '{"name": "$MODEL_NAME"}'
    echo "$MODEL_NAME model pulled successfully"
fi

echo "Starting FastAPI application..."
exec uvicorn app:app --host 0.0.0.0 --port 8000
