# The Keeper — containerized backend + UI.
# Build:  docker build -t keeper .
# Run:    docker run --env-file backend/.env -p 8790:8790 keeper
FROM python:3.13-slim

WORKDIR /app

# deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app
COPY backend/ ./backend/

# persisted memory lives here (mount a volume to keep it across runs)
RUN mkdir -p /app/memory_store

EXPOSE 8790

# 0.0.0.0 so the mapped port is reachable; browser still hits it as localhost,
# which the Host-check middleware allows. For a real remote host, add its domain
# to allowed_hosts in server.py. (MCP tools need node and are disabled here.)
CMD ["python", "-m", "uvicorn", "server:app", "--app-dir", "backend", \
     "--host", "0.0.0.0", "--port", "8790"]
