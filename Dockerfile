FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirement files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run openenv API or FastAPI
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
