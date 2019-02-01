from flask import Flask
from flask import request
from flask_basicauth import BasicAuth


import json

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import csv

modalityLookup = {}
sentence_modalities = []

with open('./20190130-Dorr-Modality-Baseline-Lexicon.csv') as modalityCSV:
    csvReader = csv.reader(modalityCSV)
    for word, pos, otherWord, owPOS, nextWord, nwPOS, modality, example in csvReader:
        for pos in pos.split("|"):
            if otherWord:
                for owPos in owPOS.split("|"):
                    if nextWord:
                        for nwPos in nwPOS.split("|"):
                            modalityLookup[((word, pos), (otherWord, owPos), (nextWord, nwPos))] = modality
                    else:
                        modalityLookup[((word, pos), (otherWord, owPos))] = modality
            else:
                modalityLookup[(word, pos)] = modality

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

    sentences = nltk.sent_tokenize(payload['text'])
    sentence_modalities = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)
        unigrams = [(morphRoot(tup[0].lower()), tup[1]) for tup in pos_tags]
        bigrams = list(zip(unigrams, unigrams[1:-1]))
        trigrams = list(zip(unigrams, unigrams[1:-1], unigrams[2:-1]))

        modal_words = (unigrams & modalityLookup.keys()) | (bigrams & modalityLookup.keys()) | (trigrams &  modalityLookup.keys())
        modals = [(modal, modalityLookup[modal]) for modal in modal_words]

        sentence_modalities.append({"sentence": sentence, "modals": list(modals), "pos": list(pos_tags)})


    return json.dumps(sentence_modalities)

def morphRoot(word):
    wlem = WordNetLemmatizer()
    return wlem.lemmatize(word,wn.VERB)
