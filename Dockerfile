# Start from Python
FROM python:3.10

# Make a folder inside Docker box
WORKDIR /app

# Copy your files
COPY . .

# Install packages
RUN pip install --no-cache-dir -r requirements.txt

# Run your app
CMD ["python", "setup.py"]
