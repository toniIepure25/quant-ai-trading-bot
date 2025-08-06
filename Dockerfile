# Use a minimal base image for Python
FROM python:3.11-slim-bullseye AS base

# Set working directory
WORKDIR /app

# Copy only requirements to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY modules/ ./modules/

# Create a non-root user and switch to it for security
RUN useradd -m appuser
USER appuser

# Command to run the application
CMD ["python", "modules/modeling/supervised_model.py"]