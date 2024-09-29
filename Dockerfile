FROM python:3.11-slim

# Install gcc, python3-dev, and other necessary build tools
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install pip (ensure it's up to date)
RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .


CMD ["streamlit", "run", "app.py"]