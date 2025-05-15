# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Disables the cache to reduce image size
# --upgrade pip: Ensures pip is up-to-date
# -r requirements.txt: Installs packages listed in the file
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the local code (app.py, ticket_model_api.py) into the container at /app
COPY app.py .
COPY ticket_model_api.py .

# Define environment variables (optional, can be useful)
# ENV FLASK_APP=app.py
# ENV FLASK_RUN_HOST=0.0.0.0

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define the command to run the application
# Uses gunicorn for a more robust production server compared to Flask's built-in server
# Or use python directly if gunicorn is not preferred/installed
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
CMD ["python", "app.py"]

# Optional: Create directories for uploads, models, and data within the image build
# These will be created by app.py/ticket_model_api.py on startup if they don't exist,
# but creating them here can sometimes help with permissions.
# RUN mkdir /app/uploads /app/models /app/data
