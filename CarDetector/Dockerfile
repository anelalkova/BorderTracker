# YOLO + OpenCV snapshot worker (CPU). Image is large due to PyTorch.
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first — avoids pip pulling huge NVIDIA/CUDA wheels (nvidia_cublas, etc.)
# when ultralytics is installed. Snapshot worker runs YOLO on CPU in this image.
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

# Default: one headless snapshot round (override in compose)
CMD ["python", "snapshot_worker.py", "--once"]
