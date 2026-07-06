# Shared image for the FastAPI backend and Streamlit frontend.
# (Ollama runs on the HOST, not in a container — see docker-compose.yml.)
FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; sentence-transformers pulls torch (CPU).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY frontend/ ./frontend/
COPY .streamlit/ ./.streamlit/

# Default command is overridden per-service in docker-compose.yml.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
