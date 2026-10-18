FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

RUN mkdir -p /app/checkpoints /app/outputs

EXPOSE 8000

CMD ["python", "-m", "nova_llm.server.api_server", "--host", "0.0.0.0", "--port", "8000"]
