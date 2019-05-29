from flask import Flask
from flask import request
from flask_basicauth import BasicAuth

# ask_detection.py must be in same directory as this file for import to work
# This is IHMC ask detction code
import ask_detection
import json


app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'panacea'
app.config['BASIC_AUTH_PASSWORD'] = 'ask_detection'

basic_auth = BasicAuth(app)

print("My name is:\t%s" % __name__)

@app.route("/")
@basic_auth.required
def hello():
	return "Hello World"

@app.route("/modality/email", methods = ['POST','GET'])
@basic_auth.required
def handleModality():
	payload = request.get_json()


	sentence_modalities = ask_detection.getModality(payload['text'])

	return json.dumps(sentence_modalities)

@app.route("/srl/email", methods = ['POST','GET'])
@basic_auth.required
def handleSrl():
	payload = request.get_json();


	sentence_srls = ask_detection.getSrl(payload['text'])

	return json.dumps(sentence_srls)

@app.route("/local", methods = ['POST', 'GET'])
def handleLocalRead():
	ask_detection.readLocalFiles()

	return "Finished. What's finsihed? Who knows, just finsihed"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
