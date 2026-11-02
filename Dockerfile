FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/models/best.pt

WORKDIR /app

# CPU-only PyTorch wheels (~200 MB instead of ~2 GB with CUDA on x86). --extra-index-url keeps PyPI
# as the main index (--index-url would replace it and break other dependencies).
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY src ./src

# Run as a non-root user.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# The trained checkpoint is NOT baked into the image: mount it at /models/best.pt (see README).
CMD ["uvicorn", "defect.api:app", "--host", "0.0.0.0", "--port", "8000"]
