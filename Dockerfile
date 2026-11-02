FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/app/model/best.pt

WORKDIR /app

# CPU-only PyTorch wheels (~200 MB instead of ~2 GB with CUDA on x86). --extra-index-url keeps PyPI
# as the main index (--index-url would replace it and break other dependencies).
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY src ./src
# The trained checkpoint (~400 KB) is versioned in the repo and baked into the image, so hosts that
# cannot mount files (e.g. Render) work. To serve another model: set MODEL_PATH, or mount over this path.
COPY model ./model

# Run as a non-root user.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os, urllib.request as u; u.urlopen('http://localhost:%s/health' % os.environ.get('PORT', '8000'))"

# Hosts like Render inject PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn defect.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
