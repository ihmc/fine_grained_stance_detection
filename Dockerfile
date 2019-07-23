# Use an official Python runtime as a parent image
FROM python:3.7-stretch

# Get java in order to use java commands
RUN apt-get update && \
	apt-get install -y openjdk-8-jdk && \
	apt-get install -y ant && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/* && \
	rm -rf /var/cache/oracle-jdk8-installer;

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN useradd -ms /bin/bash vault
RUN usermod -u 1001 vault
RUN groupmod -g 1001 vault
RUN chown -R vault:vault /app
USER vault

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 5000


# Define environment variable
ENV NAME Ask_Detection

# Run app.py when the container launches
#RUN python -m nltk.downloader punkt
#RUN python -m nltk.downloader averaged_perceptron_tagger
CMD ["python", "app.py"]
