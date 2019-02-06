from flask import Flask
from flask import request
import json

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import csv
import os

modality_lookup = {}
sentence_modalities = []

# When reading in files locally these directorys must be inside the project directory(i.e. mood and modality)
# They can be named whatever you would like just make sure they exists
# All files meant to be read in should be in the text directory.
# The output directory will be filled with the modality results of processing the input files
input_directory = '/text/'
output_directory = '/output/'

with open('./20190205-Dorr-Modality-Lexicon.v2.csv') as modalityCSV:
    csvReader = csv.reader(modalityCSV)
    for word, pos, otherWord, owPOS, nextWord, nwPOS, modality in csvReader:
        for pos in pos.split("|"):
            if otherWord:
                for owPos in owPOS.split("|"):
                    if nextWord:
                        for nwPos in nwPOS.split("|"):
                            modality_lookup[((word, pos), (otherWord, owPos), (nextWord, nwPos))] = modality
                    else:
                        modality_lookup[((word, pos), (otherWord, owPos))] = modality
            else:
                modality_lookup[(word, pos)] = modality

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World"

@app.route("/email", methods = ['POST','GET'])
def handleEmail():
    payload = request.get_json();

    
    sentence_modalities = getModality(payload['text'])

    return json.dumps(sentence_modalities)

@app.route("/local", methods = ['POST', 'GET'])
def handleLocalRead():
    readLocalFiles()

    return "Finished. What's finsihed? Who knows, just finsihed"


def getModality(text):
    sentences = nltk.sent_tokenize(text)
    sentence_modalities = []

    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)
        unigrams = [(morphRoot(tup[0].lower()), tup[1]) for tup in pos_tags]
        bigrams = list(zip(unigrams, unigrams[1:-1]))
        trigrams = list(zip(unigrams, unigrams[1:-1], unigrams[2:-1]))

        unigram_matches = unigrams & modality_lookup.keys()
        bigram_matches = bigrams & modality_lookup.keys()
        trigram_matches = trigrams & modality_lookup.keys()

        uni_match_list = list(unigram_matches)
        bi_match_list  = list(bigram_matches)
        tri_match_list = list(trigram_matches)

        
        for index, element in enumerate(tri_match_list):
            if(len(bi_match_list) > 0):
                if(element[0] == bi_match_list[index]):
                    del bi_match_list[index]

        for index, element in enumerate(bi_match_list):
            if(len(uni_match_list) > 0):
                if(element[0] == uni_match_list[index]):
                    del uni_match_list[index]

        unigram_matches = set(uni_match_list)
        bigram_matches = set(bi_match_list)
        trigram_matches = set(tri_match_list)

        modal_words = unigram_matches | bigram_matches | trigram_matches
        modals = [(modal, modality_lookup[modal]) for modal in modal_words]

        sentence_modalities.append({"sentence": sentence, "modals": list(modals), "pos": list(pos_tags)})


    return sentence_modalities

def readLocalFiles():
    path = os.path.abspath(os.path.dirname(__file__))
    inputPath  = path + input_directory
    outputPath = path + output_directory

    for filename in os.listdir(inputPath):
        with open(inputPath + filename, 'r') as input_file:
            text = input_file.read()
            with open(outputPath + 'ouput' + filename + '.json', 'w') as output_file:
                json_modality = getModality(text)
                output_file.write(json.dumps(json_modality, indent=4, sort_keys=False))

    return 

def morphRoot(word):
    wlem = WordNetLemmatizer()
    return wlem.lemmatize(word,wn.VERB)
