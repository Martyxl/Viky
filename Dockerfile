# Base image for Viky services (M1).
# All three M1 stubs share the core deps; per-service Dockerfiles / extra
# requirements-*.txt layers get added as milestones bring in heavy ML deps.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl is used by the compose healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Overridden per service in docker-compose.yml.
CMD ["uvicorn", "orchestrator.app:app", "--host", "0.0.0.0", "--port", "8000"]
