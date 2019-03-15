import json
import nltk
import csv
import os
import re

# This library allows python to make requests out.
# NOTE: There is a difference between this and the built in request variable  
# that FLASK provides do  not confuse the two.
import requests

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

modality_lookup = {}
sentence_modalities = []

# Url for server hosting coreNLP
coreNLP_server = 'http://panacea:nlp_preprocessing@simon.arcc.albany.edu:44444'

# When reading in files locally these directorys must be inside the project directory(i.e. mood and modality)
# They can be named whatever you would like just make sure they exists
# All files meant to be read in should be in the text directory.
# The output directory will be filled with the modality results of processing the input files
input_directory = '/text/'
output_directory = '/output/'
rule_directory = '/generalized-templates_v1_will_change_after_working_on_idiosyncraticRules/'

# Reading a provided CSV as a lexicon and parsing out each word and it's modality
# A sequence of 2 or 3 words can exist as well so those are checked for first
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
print("Lexicon loaded")

project_path = os.path.abspath(os.path.dirname(__file__))
rule_path = project_path + rule_directory
rules = []
for filename in os.listdir(rule_path):
	with open(rule_path + filename, 'r') as rule_file:
		ruleDict = {}
		rule = rule_file.readline()

		ruleDict['rule'] = rule.strip('\n')
		ruleDict['rule_name'] = filename.strip('.txt')
		
		rules.append(ruleDict)

print("Rules Loaded")


def getModality(text):

	# Split input text into sentences
	sentences = nltk.sent_tokenize(text)
	sentence_modalities = []

	for sentence in sentences:
		constituency_parse = parseSentence(sentence)
		# Split each sentence into words
		words = nltk.word_tokenize(sentence)
		# Builds a tuple for each word in the sentence with its corresponding part of speech
		pos_tags = nltk.pos_tag(words)
		# Build keys for the input text that has sets of single, double, and triple word sequences.
		# The word needs to be lowercased and transformed down to it's root form
		unigrams = [(morphRoot(tup[0].lower()), tup[1]) for tup in pos_tags]
		unigrams.append((None, None))
		unigrams.append((None, None))
		unigrams.append((None, None))
		bigrams = list(zip(unigrams, unigrams[1:-1]))
		trigrams = list(zip(unigrams, unigrams[1:-1], unigrams[2:-1]))

		# Match the input word keys with the keys that were read in from the CSV(lexicon)
		unigram_matches = unigrams & modality_lookup.keys()
		bigram_matches = bigrams & modality_lookup.keys()
		trigram_matches = trigrams & modality_lookup.keys()

		# Handles matching on the longest word sequence and not duplicates of chunks from
		# a longest word sequence.
		modals = []
		for index, element in enumerate(trigrams):
			if element in trigram_matches:
				modals.append((element, modality_lookup[element], index))
			elif bigrams[index] in bigram_matches:
				modals.append((bigrams[index], modality_lookup[bigrams[index]], index))
			elif unigrams[index] in unigram_matches:
				modals.append((unigrams[index], modality_lookup[unigrams[index]], index))

		sentence_modalities.append({"sentence": sentence, "matches": constituency_parse})


	return sentence_modalities


def readLocalFiles():
	path = os.path.abspath(os.path.dirname(__file__))
	input_path  = path + input_directory
	output_path = path + output_directory

	for filename in os.listdir(input_path):
		with open(input_path + filename, 'r') as input_file:
			text = input_file.read()
			with open(output_path + 'ouput' + filename + '.json', 'w') as output_file:
				json_modality = getModality(text)
				output_file.write(json.dumps(json_modality, indent=4, sort_keys=False))

	return 

def morphRoot(word):
	wlem = WordNetLemmatizer()
	return wlem.lemmatize(word,wn.VERB)

def extractTriggerWordAndPos(trigger_string):
	trigger_string = trigger_string.replace('\\n', '');
	match = re.search('\(([A-Z]*) ([a-z]+?)\)', trigger_string)
	if not match:
		return ('', '')
	trigger_pos = match.group(1)
	trigger_word = match.group(2)
	
	return (trigger_word, trigger_pos)
	
def getTriggerModality(word_and_pos):
	trigger_tuple = (morphRoot(word_and_pos[0].lower()), word_and_pos[1])
	
	if trigger_tuple in modality_lookup:
		return modality_lookup[trigger_tuple]

def parseSentence(sentence):
	annotators = '/?annotators=tokenize,pos,parse&tokenize.english=true'
	tregex = '/tregex'
	url = coreNLP_server + tregex
	parse = []	
	print("\n\n\nBeginning of sentence print",sentence, '\n\n\n\nEnd of sentence print')
	for rule in rules:
		parse_with_rules = {}
		trigger_string = ''
		trigger_modality = ''
		response = requests.post(url, data=sentence, params={"pattern": rule['rule']})

		if(response.status_code != 200):
			continue

		if(len(response.json()['sentences'][0]) > 0):
			res_sentences = response.json()['sentences'][0]

			for node in res_sentences['0']['namedNodes']:
				if 'trigger' in node:
					node['trigger'] = node['trigger'].strip('\n')
					trigger_string = node['trigger']	
				
				if 'target' in node:
					node['target'] = node['target'].strip('\n')
			
			if(trigger_string != ''):
				trigger_modality = getTriggerModality(extractTriggerWordAndPos(trigger_string))

			parse_with_rules['match'] = res_sentences["0"]['match']
			parse_with_rules['namedNodes'] = res_sentences["0"]['namedNodes']
			parse_with_rules['trigger_modality'] = trigger_modality
			parse_with_rules['rule'] = rule['rule']
			parse_with_rules['rule_name'] = rule['rule_name']
			
			parse.append(parse_with_rules)
			
	return parse;


