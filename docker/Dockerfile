FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN adduser --disabled-password --gecos "" appuser

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app

# Ensure runtime directories are writable by the non-root user.
RUN mkdir -p /app/logs /app/artifacts && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501

# Default container command runs the prediction API.
CMD ["python", "-m", "stage_runners.run_api", "--host", "0.0.0.0", "--port", "8000"]

