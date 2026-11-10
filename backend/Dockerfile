# SA Tax Chatbot API (Flask), plus the offline scripts in scripts/.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Tesseract is used by the OCR script that turns the Act's PDF into text.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU build of PyTorch by default (Docker on macOS has no GPU). For an NVIDIA GPU on
# Windows, docker-compose.gpu.yml builds with the CUDA wheels and bitsandbytes instead.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG EXTRA_PIP=""
COPY requirements.txt .
RUN pip install --index-url "$TORCH_INDEX" torch \
 && pip install -r requirements.txt gunicorn $EXTRA_PIP

COPY . .

# backend/data (the Act, the vector index, trained adapters, users and saved chats) is
# mounted from the project folder, so OneDrive keeps syncing it between machines.
EXPOSE 5000

# One worker keeps the loaded model and in-flight requests (for Stop) in one process;
# threads let other requests through while an answer is generated.
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--timeout", "900", "--bind", "0.0.0.0:5000", "run:app"]
