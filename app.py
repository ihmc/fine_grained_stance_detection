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

@app.route("/email", methods = ['POST','GET'])
@basic_auth.required
def handleEmail():
	payload = request.get_json();
	text = payload['text'];
	replace_with_empty = ['(3980)', '(398A)', '(398B)', '(398C)', '(398D)', '(398E)', '(398I)','(398M)', '(398N)', '(3918)']

	if not text:
		return json.dumps({"status_code": 400, "error": "text attribute cannot be empty"}) 
	
	# Attempt to strip some typically encoded characters from the text
	for char in replace_with_empty:
		if char in text:
			text = text.replace(char, '')

	textBytes = text.encode('UTF-8')
	text = textBytes.decode('UTF-8', 'strict')
	text = text.replace('=20', ' ')
	text = text.replace('=', '')
	text = text.replace('\r', '')
	text = text.replace('\n', '')
		
	print(text)
	sentence_modalities = modality.getModality(text)

	return json.dumps(sentence_modalities)

@app.route("/local", methods = ['POST', 'GET'])
def handleLocalRead():
	modality.readLocalFiles()

	return "Finished. What's finsihed? Who knows, just finsihed"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4000)
