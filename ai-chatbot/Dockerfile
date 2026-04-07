# syntax=docker/dockerfile:1

# --- base image ---
# python:3.12-slim is a minimal Debian image with Python pre-installed.
# "slim" strips compilers and dev headers — enough for pure-Python apps.
FROM python:3.12-slim

# Set a clean working directory inside the container
WORKDIR /app

# --- install dependencies first (layer cache trick) ---
# Copying requirements.txt separately means Docker reuses this cached layer
# on every rebuild as long as requirements.txt hasn't changed — much faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- copy application code ---
# Done after pip install so code changes don't invalidate the dep layer.
COPY . .

# Streamlit listens on 8501 by default
EXPOSE 8501

# Health check: Docker will mark the container unhealthy if Streamlit stops responding.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# --server.address=0.0.0.0 is required so the port is reachable from outside
# the container (by default Streamlit binds to 127.0.0.1 which is container-local only).
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.address=0.0.0.0", \
            "--server.port=8501", \
            "--server.headless=true"]
