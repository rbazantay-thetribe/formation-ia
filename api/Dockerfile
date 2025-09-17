FROM python:3.13-slim

WORKDIR /app

# System dependencies for LangChain and ML models
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY app.py .

# Wait for Ollama to be ready and pull llama2 model
COPY wait-for-ollama.sh .
RUN chmod +x wait-for-ollama.sh

# Expose LangServe port
EXPOSE 8000

CMD ["./wait-for-ollama.sh"]
