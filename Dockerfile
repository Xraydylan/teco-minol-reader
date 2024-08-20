# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Set the working directory in the container
WORKDIR /app

COPY . /app

# Optional: Install any needed packages specified in requirements.txt
# Uncomment the next line and add a requirements.txt file if needed
# Possibly remove the --no-cache-dir flag if you want to cache the packages
RUN pip install --no-cache-dir -r requirements.txt

# Run target.py when the container launches
# Code is run as root user
# "-u" flag is used to avoid buffering of the output
CMD ["python3", "-u", "minol_main.py"]
