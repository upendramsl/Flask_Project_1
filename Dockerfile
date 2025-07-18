# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r rqt.txt

# Expose port
EXPOSE 8080

# Run app
CMD ["python", "ml.py"]
