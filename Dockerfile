FROM python:3.9-slim-bullseye


WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libharfbuzz-dev \
    libfreetype6-dev \
    libfontconfig1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p input output

CMD ["python", "main.py"]
