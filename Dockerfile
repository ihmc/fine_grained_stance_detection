FROM registry.ased.io/panacea/ask_detection:piptainer

USER vault
COPY --chown=vault . /ask_detection
WORKDIR /ask_detection
RUN chown -R vault:vault /ask_detection

# Make port 80 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME Ask_Detection
ENV NLTK_DATA "./nltk_data"

# Run app.py when the container launches
#RUN python -m nltk.downloader punkt
#RUN python -m nltk.downloader averaged_perceptron_tagger
CMD ["python", "app.py"]
