#import torch
#import sklearn
import os, glob
import json
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from fuzzywuzzy import fuzz

def morphRootNoun(word):
    wlem = WordNetLemmatizer()
    return wlem.lemmatize(word.lower(),wn.NOUN)

def morphRootVerb(word):
    wlem = WordNetLemmatizer()
    return wlem.lemmatize(word.lower(),wn.VERB)

import pandas as pd
from operator import itemgetter

import spacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

import enchant
d = enchant.Dict("en_US")

from nltk.corpus import stopwords

from allennlp.predictors.predictor import Predictor
import allennlp_models
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
#print(stopwords.words('english'))

df = pd.read_excel(os.path.join("./", "ModalityLexiconSubcatTagsPITT.xlsx"), keep_default_na=False)
strength_and_sentiment_dict = {}
for row in df.iterrows():
    row = row[1]
    strength_and_sentiment_dict[row["Lexical item"]] = { 
        "strength": row["Belief Value"],
        "sentiment": row["Sentiment Value"],
        "modality": row["Modality"]
    } 

light_verbs = [
        "use",
        "place",
        "take",
        "make",
        "do",
        "give",
        "have",
        "put",
        "be",
        "'s",
        "’s",
    ]

reporting_verbs = []
with open("./reporting_verbs.txt", "r") as verbs_file:
	verbs = verbs_file.readlines()

	for verb in verbs:
		reporting_verbs.append(verb)



'''
extreme_tweets = pd.read_csv("../../Downloads/extremismtweets/tweets.csv")
tweets = extreme_tweets['tweets'].tolist()
'''


'''
extreme_tweets = pd.read_csv("../../Downloads/capitol_riot_tweets_2021-01-06.csv")
tweets = extreme_tweets['text'].tolist()
'''


with open('./20210726_ieee_geo_just_vaccine_tweets.json', "r") as vaccine_tweets:
	tweets = []
	for line in vaccine_tweets:
		tweets.append(json.loads(line)["full_text"])

print(len(tweets))
training_tweets = tweets[-2500:]
'''
held_out_tweets = tweets[:]

with open("./is_extremism_held_out_tweets.json", "w+") as tweets_output:
	for tweet in held_out_tweets:
		tweets_output.write(json.dumps(tweet) + "\n")
'''

'''
keywords_roots = []
for word in keywords:
    if morphRootNoun(word) not in keywords_roots:
        keywords_roots.append(morphRootNoun(word))
    if morphRootVerb(word) not in keywords_roots:
        keywords_roots.append(morphRootVerb(word))
'''


'''
#TODO This is for filtering a dataset by keywords, the english tweets came from a dataset that wasn't exclusively english. I need to change the name and make sure the data passed in is handled generally enough so it doesn't have to be a dataframe.

filtered_en_tweets = []

for row_data in en_rows_df.iterrows():
    row = row_data[1]
    doc = nlp(row["textoriginal"])
    for token in doc:
        if morphRootNoun(token.text) in keywords_roots or morphRootVerb(token.text) in keywords_roots:
            filtered_en_tweets.append(row["textoriginal"])
            break

print(len(filtered_en_tweets))

'''


wordlist = []
wordfreq = {}


for tweet in training_tweets:
    doc = nlp(tweet)
    for sent in doc.sents:
        for token in sent:
            if (token.text not in stopwords.words('english') and d.check(token.text) and token.pos_ == "NOUN" and token.ent_type_ != "DATE" and token.text.isalpha()): #or token.text.lower() in dataset_keywords:
                wordlist.append(morphRootNoun(token.text.lower()))
                    
for word in wordlist:
    count = wordlist.count(word)
    if word not in wordfreq:
        wordfreq[word] = count
        

print(dict(sorted(wordfreq.items(), key = itemgetter(1), reverse = True)[:10]))


result = dict(sorted(wordfreq.items(), key = itemgetter(1), reverse = True)[:25])
print(result)

top_10_content = list(result.keys())
print(top_10_content)


triggers_list = []
triggers_freq = {}
for tweet in training_tweets[:2500]:
    doc = nlp(tweet)
    for sent in doc.sents:
        srl = predictor.predict(sentence=sent.text)
        verbs = srl['verbs']
        words = srl['words']
        for verb in verbs:
            morph_verb = morphRootVerb(verb['verb'].lower())
            if morph_verb not in light_verbs and morph_verb not in strength_and_sentiment_dict and morph_verb not in reporting_verbs and morph_verb.isalpha():
                arg1 = []
                arg1_indices = []

                arg1_has_content = False
                for index, tag in enumerate(verb['tags']):
                    split_tag = tag.split('-')
                    tag_label = '-'.join(split_tag[1:len(split_tag)]) if split_tag else ''

                    if 'ARG1' in tag_label:
                        if morphRootNoun(words[index].lower()) in top_10_content:
                            arg1_has_content = True

                if arg1_has_content:
                    triggers_list.append(morphRootVerb(verb['verb'].lower()))

for word in triggers_list:
    count = triggers_list.count(word)
    if word not in triggers_freq:
        triggers_freq[word] = count


trig_result = dict(sorted(triggers_freq.items(), key = itemgetter(1), reverse = True)[:40])
print(trig_result)

top_40_trigs = list(trig_result.keys())
print(top_40_trigs)

trig_and_freq = {}
for trigger, count in trig_result.items():
    trig_and_freq[trigger] = {"freq": count, "content_words": {}}


trig_content_sentences = {}
for tweet in training_tweets[:2500]:
    doc = nlp(tweet)
    for sent in doc.sents:
        srl = predictor.predict(sentence=sent.text)
        verbs = srl['verbs']
        words = srl['words']
        for verb in verbs:
            morph_verb = morphRootVerb(verb['verb'].lower())
            if morph_verb in top_40_trigs:
                arg1 = []
                arg1_indices = []

                for index, tag in enumerate(verb['tags']):
                    split_tag = tag.split('-')
                    tag_label = '-'.join(split_tag[1:len(split_tag)]) if split_tag else ''

                    if 'ARG1' in tag_label:
                        morph_arg1 = morphRootNoun(words[index].lower())
                        if (words[index].lower() not in stopwords.words('english') and d.check(words[index].lower()) and words[index].lower().isalpha()): #or words[index].lower() in dataset_keywords: #and token.pos_ == "NOUN" and token.ent_type_ != "DATE":
                            if morph_verb in trig_and_freq:
                                if trig_and_freq[morph_verb]["content_words"]:
                                    if morph_arg1 in trig_and_freq[morph_verb]["content_words"]:
                                        trig_and_freq[morph_verb]["content_words"][morph_arg1] +=1
                                        if sent.text not in trig_content_sentences[morph_verb + morph_arg1]:
                                            trig_content_sentences[morph_verb + morph_arg1].append(sent.text)
                                        else:
                                            matches_existing_sent = False
                                            for sentence in trig_content_sentences[morph_verb + morph_arg1]:
                                                if fuzz.token_set_ratio(sent.text, sentence) > 85:
                                                    matches_existing_sent = True
                                                    break
                                            if not matches_existing_sent:
                                                trig_content_sentences[morph_verb + morph_arg1].append(sent.text)
                                    else:
                                        trig_and_freq[morph_verb]["content_words"][morph_arg1] = 1
                                    
                                        trig_content_sentences[morph_verb + morph_arg1] = [sent.text]
                                        
                                else:
                                    trig_and_freq[morph_verb]["content_words"] = {morph_arg1: 1}
                                    trig_content_sentences[morph_verb + morph_arg1] = [sent.text]

trig_content_top_pairs = {}
for trigger, contents in trig_and_freq.items():
    trig_content_top_pairs[trigger] = {"freq": contents["freq"], "content_words": dict(sorted(contents["content_words"].items(), key = itemgetter(1), reverse = True)[:10])}

for trigger, contents in trig_content_top_pairs.items():
    for word in contents["content_words"]:
        #assoc_sentences[trigger + word] = trig_content_sentences[trigger + word]
        trig_content_top_pairs[trigger]["content_words"][word] = {
            "freq": trig_content_top_pairs[trigger]["content_words"][word],
            "sentences": trig_content_sentences[trigger + word]
        }

print(trig_content_top_pairs)
