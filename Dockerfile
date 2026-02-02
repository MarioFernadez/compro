FROM python:3.12-slim

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# PyTorch CPU + deps
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       -r requirements.txt

COPY . .

# ⚡ Pre-descargar modelos EasyOCR en build (evita freeze en primer OCR)
RUN python -c "import easyocr; easyocr.Reader(['es'], gpu=False)"

# Railway te da PORT, no uses fijo 8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8080

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]
