# Use an official Python runtime as a parent image
FROM python:3.7-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 4000

# Define environment variable
ENV NAME Modality

# Run app.py when the container launches
#RUN python -m nltk.downloader punkt
#RUN python -m nltk.downloader averaged_perceptron_tagger
CMD ["python", "app.py"]
