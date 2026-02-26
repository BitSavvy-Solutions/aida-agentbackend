FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for lxml/builds if needed
# RUN apt-get update && apt-get install -y build-essential libxml2-dev libxslt-dev

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port 80
EXPOSE 80

# Run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]