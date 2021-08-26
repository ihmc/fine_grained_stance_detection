#import torch
#import sklearn
import os, glob
import json
import html
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from fuzzywuzzy import fuzz
from datetime import datetime
from operator import itemgetter
from pathlib import Path

import pandas as pd

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
		"get",
        "'s",
        "’s",
    ]

reporting_verbs = []
with open("./reporting_verbs.txt", "r") as verbs_file:
	verbs = verbs_file.readlines()

	for verb in verbs:
		reporting_verbs.append(verb.strip())


Path("./trig_content_pairs").mkdir(exist_ok=True)

def morphRootNoun(word):
    wlem = WordNetLemmatizer()
    return wlem.lemmatize(word.lower(),wn.NOUN)

def morphRootVerb(word):
    wlem = WordNetLemmatizer()
    return wlem.lemmatize(word.lower(),wn.VERB)

def load_json(file_path):
	dataset_as_list = []

	with open(file_path, "r") as texts:
		for line in texts:
			dataset_as_list.append(json.loads(line))

	return dataset_as_list

def load_csv(file_path):
	dataset_as_list = pd.read_csv(file_path).to_dict(orient="records")

	return dataset_as_list

def load_excel(file_path):
	dataset_as_list = pd.read_excel(file_path).to_dict(orient="records")

	return dataset_as_list

#TODO Test this to make sure it works as expected we want all lines as elements in a list 
# without having to loop through each lines. One option is the readlines() function but this
# keeps the newline character at the end of the line. It could be removed but may affect other potential
# newline characters that were meant to stay. 
def load_text(file_path):
	dataset_as_list = []

	with open(file_path, "r") as texts:
		for line in texts:
			dataset_as_list.append(line.strip())

	return dataset_as_list

#Currently assuming json but will need to handle csv/xlsx as well.
def get_trig_content_pairs(data_filepath, file_type, text_identifier, domain_name, topic_name, topic_keywords = "",  training_text_number = 2500):
	Path("./trig_content_pairs/" + domain_name).mkdir(exist_ok=True)
	Path("./trig_content_pairs/" + domain_name + "/" + topic_name).mkdir(exist_ok=True)
	
	today = datetime.today().strftime('%Y%m%d')	

	if file_type.lower() == "json":
		data = load_json(data_filepath)
	if file_type.lower() == "csv":
		data = load_csv(data_filepath)
	if file_type.lower() == "excel":
		data = load_excel(data_filepath)

	print("Number of texts in entire dataset: ", len(data))

	#NOTE If data was filtered I want to add this to the output file names this variable 
	# will be populated with the word filtered in order to accomplish this
	data_was_filtered = ""

	#Need to take the roots of the keywords and compare them to the roots of all words in each text
	# to make sure no words are missed because of different endings or the like.
	keywords = topic_keywords.split(" ")
	keywords_roots = []

	#If no keywords are provided then don't waste time "filtering" the dataset
	if len(keywords) > 0:
		for word in keywords:
			if morphRootNoun(word) not in keywords_roots:
				keywords_roots.append(morphRootNoun(word))
			if morphRootVerb(word) not in keywords_roots:
				keywords_roots.append(morphRootVerb(word))

		filtered_data = []
		for line in data:
			doc = nlp(line[text_identifier])
			for token in doc:
				if morphRootNoun(token.text) in keywords_roots or morphRootVerb(token.text) in keywords_roots:
					filtered_data.append(line)
					break

		#NOTE In order to keep things consistent and avoid extra if conditions the data list variable is overwritten
		# with the filtered data list if there is any
		if len(filtered_data) > 0:
			data = filtered_data
			data_was_filtered = "_filtered"


	print("Number of texts in keyword filtered dataset: ", len(data))

	#TODO May need to pause in this condition and inform them they will be using the whole dataset
	# for training and if they do not wish to do so they should set a number lower than the total
	# Output the len of the dataset so they can decide what number

	#The dataset may be small then the default or suggested number of training texts,
	# if so set it to the length of the dataset.
	if training_text_number > len(data):
		training_text_number = len(data)

	training_texts = []
	held_out_texts = []

	#TODO Write these outputs in whatever file format they gave them to us. For csv and excel change the load functions
	# above to also return the column names as those well be needed if this is going to happen
	with open("./trig_content_pairs/" + domain_name + "/" + topic_name + "/" + today + data_was_filtered + "_training_data.json", "w+") as training_output:
		for line in data[:training_text_number]:
			training_output.write(json.dumps(line) + "\n")
			training_texts.append(line[text_identifier])

	with open("./trig_content_pairs/" + domain_name + "/" + topic_name + "/" + today + data_was_filtered  + "_held_out_data.json", "w+") as held_out_output:
		for line in data[training_text_number:]:
			held_out_output.write(json.dumps(line) + "\n")
			held_out_texts.append(line[text_identifier])

	wordlist = []
	wordfreq = {}


	for tweet in training_texts:
		doc = nlp(tweet)
		for sent in doc.sents:
			for token in sent:
				lowered_text = token.text.lower()
				morphed_noun = morphRootNoun(lowered_text)
				if (lowered_text not in stopwords.words('english') and d.check(lowered_text) and token.pos_ == "NOUN" and token.ent_type_ != "DATE" and token.text.isalpha() and lowered_text != "amp") or morphed_noun in keywords_roots:
					wordlist.append(morphed_noun)
						
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
	for tweet in training_texts:
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
	for tweet in training_texts:
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

	with open("./trig_content_pairs/" + domain_name + "/" + topic_name + "/" + today + data_was_filtered + "_" + domain + "_" + topic_name + "_top_trig_content_pairs.json", "w+") as trig_content_output:
		trig_content_output.write(json.dumps({"domain": domain_name, "topic": topic_name, "trig_content_top_pairs": trig_content_top_pairs}))
	return trig_content_top_pairs

#if __name__ == "__main__":
print("\n\n\nThis script is meant to process a data file of texts and produce the top trigger content pairs from those texts which will then be categorized into resources for the stance detection tool\n"\
		"the main function to use is as follows get_trig_content_pairs(data_filepath, file_type, text_identifier, domain_name, topic_name, topic_keywords, training_text_number)\n"\
		"Below is a description of each paramter: \n\n"\
		"data_filepath: The path to the location of the data file (ex. '/path/to/datafile.json')\n"\
		"file_type: Type of data file (current options: 'json', 'csv', 'excel')\n"\
		"text_identifier: The name of the attribute (json) or column (csv/excel) that contains the text to be processed (ex. 'full_text')\n"\
		"domain_name: Name of the domain that the data is specific to (ex. 'covid')\n"\
		"topic_name: Name of the topic of concern within the domain (ex 'mask_wearing')\n"\
		"topic_keywords: Space separated list of keywords relevant to the topic used to filter the dataset to pertinent texts (ex. 'mask covering facemask' etc.)\n"\
		"training_text_number: Number of texts that should processed for top trigger content pairs, if no number is provided the default is 2500 as this tends to accurately represent a well specified dataset\n\n"\
		"A full example is as follows: resource_building.get_trig_content_pairs('/path/to/datafile.json', 'json', 'full_text', 'covid', 'mask_wearing', 'mask covering facemask', 2250)\n\n")
