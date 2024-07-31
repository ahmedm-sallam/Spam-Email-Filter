# Start from the Ubuntu base image
FROM ubuntu:latest

# Metadata as described above
LABEL authors="Ahmed M.Sallam"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install Python and necessary tools
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . /app/

# Specify the default command
CMD ["python3", "main.py"]
