FROM python:3.10-slim

WORKDIR /app

# Install system deps for nibabel / scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    fonts-wqy-zenhei \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create runtime directories
RUN mkdir -p uploads weights

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
