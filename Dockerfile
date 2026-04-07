FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirement files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port 7860 for FastAPI and Hugging Face sync
EXPOSE 7860

# Command to run openenv API or FastAPI
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
