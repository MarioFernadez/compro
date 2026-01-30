FROM python:3.12-slim

# Dependencias del sistema necesarias para OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ✅ Carpeta fija para modelos de EasyOCR (evita descargas repetidas)
ENV EASYOCR_MODULE_PATH=/app/.easyocr
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

# 🔥 CLAVE: usar índice CPU de PyTorch
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       -r requirements.txt

COPY . .

EXPOSE 8501

# ✅ Railway suele setear PORT; si no, usa 8501
CMD ["bash", "-lc", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
