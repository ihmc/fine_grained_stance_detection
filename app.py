from flask import Flask
from flask import request
from flask_basicauth import BasicAuth

# modality.py must be in same directory as this file for import to work
# This is IHMC modality code
import modality
import json


app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'panacea'
app.config['BASIC_AUTH_PASSWORD'] = 'modality'

basic_auth = BasicAuth(app)

print("My name is:\t%s" % __name__)

@app.route("/")
@basic_auth.required
def hello():
	return "Hello World"

@app.route("/modality/email", methods = ['POST','GET'])
@basic_auth.required
def handleModality():
	payload = request.get_json();


	sentence_modalities = modality.getModality(payload['text'])

	return json.dumps(sentence_modalities)

@app.route("/srl/email", methods = ['POST','GET'])
@basic_auth.required
def handleSrl():
	payload = request.get_json();


	sentence_modalities = modality.getSrl(payload['text'])

	return json.dumps(sentence_modalities)

@app.route("/local", methods = ['POST', 'GET'])
def handleLocalRead():
	modality.readLocalFiles()

	return "Finished. What's finsihed? Who knows, just finsihed"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4000)
