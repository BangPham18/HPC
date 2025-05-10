FROM python:3.9-slim

# Cài đặt các thư viện cần thiết
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Sao chép file requirements
COPY requirements.txt .

# Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép scripts và configs
COPY scripts/ /app/scripts/
COPY configs/ /app/configs/

# Thiết lập biến môi trường
ENV PYTHONPATH=/app

# Script khởi động mặc định
CMD ["python", "scripts/train.py"]