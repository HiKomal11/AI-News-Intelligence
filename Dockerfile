FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System packages required by Tesseract OCR and PyMuPDF/Pillow
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK resources required by the application
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/share/nltk_data'); nltk.download('wordnet', download_dir='/usr/local/share/nltk_data'); nltk.download('omw-1.4', download_dir='/usr/local/share/nltk_data')"


COPY . .

ENV TESSERACT_CMD=/usr/bin/tesseract

<<<<<<< HEAD
CMD ["sh", "-c", "gunicorn --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:${PORT:-10000} app:app"]
=======
CMD ["sh", "-c", "gunicorn --workers 1 --threads 1 --timeout 120 --bind 0.0.0.0:${PORT:-10000} app:app"]
>>>>>>> 9d23a86 (Optimize TensorFlow LSTM memory usage)
