from flask import Flask
from flask import request
from flask_basicauth import BasicAuth

# ask_detection.py must be in same directory as this file for import to work
# This is IHMC ask detction code
import ask_detection
import json
from spacy.lang.en import English # updated
from spacy.matcher import Matcher

nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer')) # updated


app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'panacea'
app.config['BASIC_AUTH_PASSWORD'] = 'ask_detection'

basic_auth = BasicAuth(app)

print("My name is:\t%s" % __name__)

@app.route("/health")
def health():
    health = ask_detection.health.getHealth()
    return json.dumps(health)

@app.route("/healthcheck")
def healthCheck():
	return "Healthy", 200

@app.route("/")
@basic_auth.required
def hello():
	return "Hello World Brodie"


@app.route("/srl/email", methods = ['POST','GET'])
@basic_auth.required
def handleEmail():
	payload = request.get_json();

	sentence_srls = ask_detection.getSrl(payload['text'], payload['links'])

	return json.dumps(sentence_srls)

@app.route("/srl/sms", methods = ['POST','GET'])
@basic_auth.required
def handleSMS():
	payload = request.get_json();

	sentence_srls = ask_detection.getSrl(*preprocess_text(payload["text"]))

	return json.dumps(sentence_srls)

@app.route("/srl/linkedin", methods = ['POST','GET'])
@basic_auth.required
def handleLinkedin():
	payload = request.get_json();

	sentence_srls = ask_detection.getSrl(*preprocess_text(payload["text"]))

	return json.dumps(sentence_srls)

def preprocess_text(text):
	url_token_indices = []
	sentences_array = []
	text_to_process = ""
	ask_dict = {}
	ask_id = 0
	tag = "ASKMARKER1234"

	doc = nlp(text)

	for sent in doc.sents:
		sentence_pieces = []
		for token in sent:
			if token.text.lower() in abbrev_mappings:
				sentence_pieces.append(f'{abbrev_mappings.get(token.text)}{token.whitespace_}')
			elif(token.like_url or token.like_email) and tag not in token.text:
				ask_dict[str(ask_id)] = token.text	
				sentence_pieces.append(f'[[[{tag}-{ask_id}-{tag}{token.text}/{tag}-{ask_id}-{tag}]]]{token.whitespace_}')
				ask_id += 1
			else:
				sentence_pieces.append(token.text_with_ws)
		sentences_array.append(sentence_pieces)

	for sent in sentences_array:
		text_to_process += f'{"".join(sent)}\n'

	return(text_to_process, ask_dict)

abbrev_mappings = {
	"txt": "text",
	"sms": "text",
	"msg": "text",
	"im": "text",
	"dm": "text",
	"ansr": "answer",
	"b4": "before",
	"2": "to",
	"l8er": "later",
	"&": "and",
	"r": "are",
	"4fil": "fulfill",
	"unsub": "unsubscribe",
	"u": "you"
}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
