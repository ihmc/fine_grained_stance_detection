from flask import Flask
from flask import request
from pattern.en import parse, Sentence, parse
from pattern.en import modality
from pattern.en import mood

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import csv

modalityDict = {}
posDict = {}
modalityOutputDict = []

with open('./20190130-Dorr-Modality-Baseline-Lexicon.csv') as modalityCSV:
    csvReader = csv.reader(modalityCSV)
    for word, pos, otherWord, owPOS, nextWord, nwPOS, modality, example in csvReader:
        posDict[word] = pos
        modalityDict[word] = modality

print(posDict)
print(modalityDict)
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World"

@app.route("/email", methods = ['POST','GET'])
def handleEmail():
    json = request.get_json();

    wordsPerSentence = []
    lemmatizedSentences = [] 
    taggedSentences = []
    sentences = nltk.sent_tokenize(json['text'])

    for sentence in sentences:
        wordsPerSentence.append(nltk.word_tokenize(sentence))
    print(wordsPerSentence)

    for sentence in wordsPerSentence:
        taggedSentences.append(nltk.pos_tag(sentence))
    print(taggedSentences);

    for sentenceWords in taggedSentences:
        lemmatizedTuples = []
        for word in sentenceWords:
            lowercaseWord = word[0].lower()
            lemmatizedTuples.append((morphRoot(lowercaseWord), word[1]))
        lemmatizedSentences.append(lemmatizedTuples)

    print(lemmatizedSentences, "lemmatized sentences")

    

    for lemmatizedWords in lemmatizedSentences:
        for lemmatizedWord in lemmatizedWords:
            partsOfSpeech = posDict.get(lemmatizedWord[0], 'nil').split("|")
            print(partsOfSpeech, "Parts of speech")
            if lemmatizedWord[1] not in partsOfSpeech:
                modalityOutputDict.append((lemmatizedWord[0], 'nil'))
            else:
                modalityOutputDict.append((lemmatizedWord[0], modalityDict[lemmatizedWord[0]]))
    print(modalityOutputDict)
    return 'Emails yay'; 

def morphRoot(word):
    wlem = WordNetLemmatizer()
    return wlem.lemmatize(word,wn.VERB)
