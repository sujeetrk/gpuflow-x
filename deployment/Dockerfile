FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY ai ./ai
COPY core ./core
COPY inference ./inference
COPY server ./server
COPY configs ./configs

EXPOSE 8000

CMD ["uvicorn", "server.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
