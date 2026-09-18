FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY server ./server
COPY configs ./configs

EXPOSE 8000

CMD ["uvicorn", "server.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
