FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       -r requirements.txt

# ✅ directorio fijo para modelos
RUN mkdir -p /app/.easyocr
ENV EASYOCR_MODULE_PATH=/app/.easyocr

# ✅ PRE-DOWNLOAD: esto evita que en runtime “se cuelgue” bajando modelos
RUN python -c "import easyocr; easyocr.Reader(['es'], gpu=False, verbose=False)"

COPY . .

EXPOSE 8501

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
