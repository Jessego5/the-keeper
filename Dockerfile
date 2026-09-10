# The Keeper: containerized backend + UI.
# Build:  docker build -t keeper .
# Run:    docker run --env-file backend/.env -p 8790:8790 keeper
FROM python:3.13-slim

WORKDIR /app

# node for the MCP servers (files / fetch / search / time / git are all `npx`).
# Without it every MCP server fails to spawn and the container silently loses the
# whole tool layer: which is containment by amputation, not isolation.
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# deps first for layer caching. uv provides `uvx`, which the fetch / search / time
# / git MCP servers are launched with (node above covers the `npx` ones).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt uv

# app + the jailed directory the files MCP server is pointed at
COPY backend/ ./backend/
COPY keeper_sandbox/ ./keeper_sandbox/

# mcp.json writes ${KEEPER_ROOT} rather than an absolute path, so the same config
# works here and on a developer machine. The git server still won't start: .git is
# excluded from the image (see .dockerignore), and shipping history into a
# container to read it back would be the wrong trade.
ENV KEEPER_ROOT=/app

# persisted memory lives here (mount a volume to keep it across runs)
RUN mkdir -p /app/memory_store

EXPOSE 8790

# 0.0.0.0 so the mapped port is reachable; browser still hits it as localhost,
# which the Host-check middleware allows. For a real remote host, add its domain
# to allowed_hosts in server.py.
#
# Running here is also the security posture: run_python cannot read the host
# filesystem, and the API key arrives as an env var (never a file, see
# .dockerignore) which sandbox.py strips from the child: so the fetch-a-page ->
# write-code path documented in sandbox.py has no credential and no host to reach.
# Presence sensing does NOT work here: those are macOS APIs. Startup logs say so.
CMD ["python", "-m", "uvicorn", "server:app", "--app-dir", "backend", \
     "--host", "0.0.0.0", "--port", "8790"]
