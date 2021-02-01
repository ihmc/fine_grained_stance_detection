from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
#predictor = Predictor.from_path("./srl-model-2018.05.25.tar.gz")
#sentiment_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/basic_stanford_sentiment_treebank-2020.06.09.tar.gz")
sentiment_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/sst-roberta-large-2020.06.08.tar.gz")
#predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2020.03.19.tar.gz")
#predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz")
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")

import health
import unicodedata
import json
import nltk
import csv
import os
import re
import subprocess
import health
import spacy
from bertopic import BERTopic


no_filter_version = False
mutually_constrained_version = True
light_verbs_version = True
exist_case_version = True

#model = BERTopic("distilbert-base-nli-mean-tokens", verbose=True)

#spacy_nlp = spacy.load("en_core_web_lg")
spacy_nlp = spacy.load("en_core_web_sm")

# This library allows python to make requests out.
# NOTE: There is a difference between this and the built in request variable  
# that FLASK provides do not confuse the two.
import requests

#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

from load_resources import catvar_dict, lcs_dict, strength_and_sentiment_dict, trigger_buckets, content_buckets, perform_verbs, give_verbs, lose_verbs, gain_verbs#, protect_verbs, reject_verbs
#TODO Fix the variables imported from this line once it's sorted out.
#from ask_mappings import sashank_categories_sensitive, alan_ask_types, sashanks_ask_types, tomeks_ask_types, perform_verbs, give_verbs, lose_verbs, gain_verbs
from ask_mappings import sashank_categories, panacea_ask_types, pitt_stance_triggers_and_targets
#pitt_stance_triggers = pitt_stance_triggers_and_targets.get("triggers")
#pitt_stance_targets = pitt_stance_triggers_and_targets.get("targets")
pitt_light_verbs = pitt_stance_triggers_and_targets.get("light_verbs")

#If a specific belief type can't be found from the domain trigger or content buckets
# then sentiment words are tried for a belief type which is the default type below
default_belief_type = "EXIST"

from catvar_v_alternates import v_alternates


# catvar_alternates_dict is a dictionary where each key has an array of verbal words from the catvar file
# that exist on a line with more than 1 verbal form. This is to cover cases when a small spelling change is present
# in catvar or when other verbal words exist but were are not a part of the catvar_dict here
catvar_alternates_dict = v_alternates

# Rule directories
# TODO This list will be thinned out as rule sets are chosen for superiority
lexical_item_rule_directory = '/lexical-item-rules/'
preprocess_rule_directory = '/idiosyncratic/'

# Url for server hosting coreNLP
coreNLP_server = 'http://panacea:nlp_preprocessing@simon.arcc.albany.edu:44444'

# When reading in files locally these directorys must be inside the project directory
# They can be named whatever you would like just make sure they exists
# All files meant to be read in should be in the text directory.
# The output directory will be filled with the results of processing the input files
input_directory = '/text/'
output_directory = '/output/'
rule_directory = '/generalized-templates_v1_will_change_after_working_on_idiosyncraticRules/'

# Paths for the tsurgeon java tool
tregex_directory = '/stanford-tregex-2018-10-16/'
tsurgeon_script = tregex_directory + 'tsurgeon.sh'
tsurgeon_class = 'edu.stanford.nlp.trees.tregex.tsurgeon.Tsurgeon'


# Id for emails for the csv file for each email run through the system
email_id = 0

def getSrl(text, links):
	global email_id 
	email_id += 1
	#TODO For now just set links to '' til we kept proper output
	sentence_srls = []
	framing_matches = []
	ask_matches = []
	last_ask = {}
	last_ask_index = -1
	pattern = '\[\[\[ASKMARKER1234-(\d+)-ASKMARKER1234(.*?)/ASKMARKER1234-\d+-ASKMARKER1234\]\]\]'
	text_to_process = unicodedata.normalize('NFKC',text)

	lines = text_to_process.split('\n')

	for line in lines:
		if not line:
			continue
		line_text = line.strip()
		link_offsets = []
		link_ids = []
		link_strings = []
		match = re.search(pattern, line_text)

		while match:
			link_offsets.append((match.start(0), match.start(0) + len(match.group(2))))
			link_ids.append(match.group(1))
			link_strings.append(match.group(2))
			line_text = line_text.replace(match.group(0), match.group(2))
			match = re.search(pattern, line_text)

		#line_matches = parseSrl(line_text, link_offsets, link_ids, link_strings, links, last_ask, last_ask_index)
		line_matches = parseSrlStanza(line_text, link_offsets, link_ids, link_strings, links, last_ask, last_ask_index)
		
		if line_matches:
			# NOTE The last_ask and last_ask_index are overidding the values initialized at the beginning 
			# of this function. This on purpose so that each time parseSrl is called it will get the 
			# most up to date info
			(framings, asks, asks_to_update, last_ask, last_ask_index) = line_matches

			if framings: 
				framing_matches.extend(framings)
			if asks:	
				ask_matches.extend(filter(lambda ask: True if ask['is_ask_confidence'] != 0 else False, asks))

			# If parseSrl determines that the last ask needs to be update then it will update the appropriate ask
			# in ask_matches with the new information that was altered in last_ask inside parseSrl
			if asks_to_update:
				for ask in asks_to_update:
					# Ask will be a tuple with the first part being the updated ask and the second part being the index in ask_matches that needs updating
					ask_matches[ask[1]] = ask[0]
				last_ask = asks_to_update[-1][0]
				last_ask_index = asks_to_update[-1][1]


	sorted_framing = sorted(framing_matches, key = lambda k: k['is_ask_confidence'] , reverse=True)
	sorted_asks = sorted(filter(lambda ask: True if ask['is_ask_confidence'] != 0 else False, ask_matches), key = lambda k: k['is_ask_confidence'], reverse=True)

	return {'email': text, 'framing': sorted_framing, 'asks': sorted_asks}

def stances(text_array):
	stances = []

	for index, text in enumerate(text_array):
		line_stances = get_stances(index, text[0], text[1], text[2], text[3])
		if line_stances:
			stances.extend(line_stances)

	return {'stances': stances}

def morphRootVerb(word):
	wlem = WordNetLemmatizer()
	return wlem.lemmatize(word.lower(),wn.VERB)

def morphRootNoun(word):
	wlem = WordNetLemmatizer()
	return wlem.lemmatize(word.lower(),wn.NOUN)

def extractVerbs(parse_tree):
	parse_tree = parse_tree.replace('\\n', '')
	match = re.findall('\((VB[A-Z]*) *([a-z]+?)\)', parse_tree)
	
	return match

def extractTriggerWordAndPos(trigger_string):
	trigger_string = trigger_string.replace('\\n', '');
	match = re.search('\(([A-Z]*) *([a-z]+?)\)', trigger_string)
	
	trigger_pos = match.group(1)
	trigger_word = match.group(2)
	
	return (trigger_word, trigger_pos)
	
	
def getTriggerModality(word_and_pos):
	trigger_tuple = (morphRootVerb(word_and_pos[0].lower()), word_and_pos[1])
	
	if trigger_tuple in modality_lookup:
		return modality_lookup[trigger_tuple]
  
def preprocessSentence(tree):
	with open('./tree.txt', 'w+') as tree_file:
		tree_file.write(tree)
		for rule in preprocess_rules_in_order:
			# Have to return to the beginning of the file so that the new tree overwrites the previous one.
			tree_file.seek(0)

			# This command is taken out of the tsurgeon.sh file in the coreNLP tregex tool.
			# The cp option is added so the class will run without being in the same directory 
			# Text is commented out here because that parameter only exists in python 3.7
			result = subprocess.run(['java', '-mx100m', '-cp', '.' + tregex_directory + 'stanford-tregex.jar:$CLASSPATH', tsurgeon_class, '-treeFile', 'tree.txt', '.' + preprocess_rule_directory + rule], stdout = subprocess.PIPE)#, text=True)

			string_tree = result.stdout.decode("utf-8")
			#print(string_tree)
			tree_file.write(string_tree)

	return string_tree


# This function runs regex on a parse tree in order to extract potential trigger and target
# labels that may have been placed on the tree.
def extractTrigsAndTargs(tree):
	trigs_and_targs = []
	
	# Remove new lines so the regex is easier to handle
	tree_no_new_lines = tree.replace('\n', '')
	trig_regex = '(\([A-Z]* *(Trig\w+) *[A-Z]* *([a-z]*)\))'
	targ_regex = '(\([A-Z]* *(Targ\w+) *\(*[A-Z]* *[A-Z]* *([a-z]*)\))'

	# The result of a findall is an array of tuples, where each part of the tuple
	# is a group from the regex, denoted by non escaped parentheses.
	# If a larger group encompasses smaller groups I believe the larger group is first
	# in the tuple
	trig_match = re.findall(trig_regex, tree_no_new_lines)
	targ_match = re.findall(targ_regex, tree_no_new_lines)

	if not trig_match:
		#print('Trigger did not match, investigate tree and regex')
		return None
	if not targ_match:
		#print('Target did not match, investigate tree and regex')
		return None

	# Extract the trigger and target from the tree, remove "Trig" and "Targ" so the part of speech 
	# can be kept with the word, and extract the modality
	for index, (entire_match, modality, trig) in enumerate(trig_match):
		# TODO keeping this for now as it parses out the trigger and target
		# with their corresponding part of speech
		# May not be needed later on.
		modality = modality.replace('Trig', '')
		ask = targ_match[index][2]
		trig_word = trig
		trigs_and_targs.append((trig, targ_match[index][2], modality, ask, trig_word))

	return trigs_and_targs	

def buildParseDict(sentence, trigger, target, modality, ask_who, ask, ask_recipient, ask_when, ask_action, ask_procedure, ask_negation, ask_negation_dep_based, is_ask_confidence, confidence, descriptions, s_ask_types, t_ask_types, a_ask_types, t_ask_confidence,  base_word, rule, rule_name, link_id, links):
	parse_dict = {}
	if modality:
		parse_dict['trigger'] = trigger
		parse_dict['target'] = target
		parse_dict['trigger_modality'] = modality
	if ask_negation:
		parse_dict['ask_rep'] = f'<{t_ask_types[0]}[NOT {ask_action}[{ask}({",".join(link_id)}){s_ask_types}]]>'
	else:
		parse_dict['ask_rep'] = f'<{t_ask_types[0]}[{ask_action}[{ask}({",".join(link_id)}){s_ask_types}]]>'
	parse_dict['evidence'] = sentence
	#parse_dict['base_word'] = base_word
	#parse_dict['ask_who'] = ask_who
	parse_dict['ask_action'] = ask_action
	parse_dict['ask_target'] = ask
	#parse_dict['ask_recipient'] = ask_recipient
	#parse_dict['ask_when'] = ask_when
	parse_dict['ask_negation'] = ask_negation
	parse_dict['is_ask_confidence'] = is_ask_confidence
	parse_dict['link_id'] = link_id
	#NOTE Link_id is now an array of link ids. So we need to loop through them
	parse_dict['url'] = {}
	for link in link_id:
		if link:
			parse_dict['url'].update({link: links.get(link)})
	#parse_dict['ask_negation_dep_based'] = ask_negation_dep_based
	#parse_dict['ask_info_confidence'] = confidence
	parse_dict['t_ask_type'] = t_ask_types
	#parse_dict['t_ask_confidence'] = t_ask_confidence
	parse_dict['s_ask_type'] = s_ask_types
	#parse_dict['a_ask_type'] = a_ask_types
	#parse_dict['a_ask_procedure'] = ask_procedure
	#parse_dict['semantic_roles'] = descriptions
	
	
	# Commented for now but useful if debugging which rules are being used
	#parse_dict['rule'] = rule
	#parse_dict['rule_name'] = rule_name
	return parse_dict
	
# Ask types or classes have been provided by Tomek and Sashank, hence s_ask_types and t_ask_types
# This function maps the target word(ask) to a catvar word (the base of a word) which is 
# mapped to LCS (lexical conceptual structures) and eventually those are mapped to the ask types
def getAskTypes(ask, word_pos):
	verb_types = []
	t_ask_types = []
	catvar_object = catvar_dict.get(ask)

	if catvar_object != None:
		catvar_word = catvar_object['catvar_value']
	elif word_pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
		catvar_word = ask	
	else:
		catvar_word = ''

	if catvar_word in perform_verbs:
		return(['PERFORM'])
	else:
		catvar_word_alternates = catvar_alternates_dict.get(ask)
		if catvar_word_alternates:
			for alternate in catvar_word_alternates:
				if alternate in perform_verbs:
					return(['PERFORM'])
	if catvar_word in give_verbs:
		return(['GIVE'])
	else:
		catvar_word_alternates = catvar_alternates_dict.get(ask)
		if catvar_word_alternates:
			for alternate in catvar_word_alternates:
				if alternate in give_verbs:
					return(['GIVE'])
	if catvar_word in lose_verbs:
		return(['LOSE'])
	else:
		catvar_word_alternates = catvar_alternates_dict.get(ask)
		if catvar_word_alternates:
			for alternate in catvar_word_alternates:
				if alternate in lose_verbs:
					return(['LOSE'])
	if catvar_word in gain_verbs:
		return(['GAIN'])
	else:
		catvar_word_alternates = catvar_alternates_dict.get(ask)
		if catvar_word_alternates:
			for alternate in catvar_word_alternates:
				if alternate in gain_verbs:
					return(['GAIN'])

	return t_ask_types

def getBeliefType(word, word_pos):
	belief_types = []
	catvar_object = catvar_dict.get(word)

	if catvar_object != None:
		catvar_word = catvar_object['catvar_value']
	elif word_pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
		catvar_word = word
	else:
		catvar_word = ''

	if catvar_word in trigger_buckets:
		trigger_belief_types = trigger_buckets.get(catvar_word).get("belief_types")
		for trigger_belief_type in trigger_belief_types:
			belief_types.append((trigger_belief_type["belief_type"], trigger_belief_type["strength"], trigger_belief_type["sentiment"]))
			#return(([trigger_belief_type["belief_type"]], trigger_belief_type["sentiment"], trigger_belief_type["strength"]))
	else:
		catvar_word_alternates = catvar_alternates_dict.get(word)
		if catvar_word_alternates:
			for alternate in catvar_word_alternates:
				if alternate in trigger_buckets:
					trigger_belief_types = trigger_buckets.get(alternate).get("belief_types")
					for trigger_belief_type in trigger_belief_types:
						belief_types.append((trigger_belief_type["belief_type"], trigger_belief_type["strength"], trigger_belief_type["sentiment"]))
						return belief_types
						#return(([trigger_belief_type["belief_type"]], trigger_belief_type["sentiment"], trigger_belief_type["strength"]))

	return belief_types

def getBeliefTypeFromContent(content_word):
	belief_types = []

	if content_word in content_buckets:
		content_belief_types = content_buckets.get(content_word).get("belief_types")

		for content_belief_type in content_belief_types:
			belief_types.append((content_belief_type["belief_type"], content_belief_type["sentiment"]))

	if belief_types:
		return belief_types
	else:
		return [('', '')]

def getTAskType(ask):
	verb_types = []
	t_ask_types = []
	catvar_object = catvar_dict.get(ask)

	if catvar_object != None:
		catvar_word = catvar_object['catvar_value']

		for verb_type, words in lcs_dict.items():
			if catvar_word in words:
				verb_types.append(verb_type)
			else:
				catvar_word_alternates = catvar_alternates_dict.get(ask)
				if catvar_word_alternates:
					for alternate in catvar_word_alternates:
						if alternate in words:
							verb_types.append(verb_type)
							#TODO Ask if this break should be there or if I should get a type for each alternate
							break

	#TODO Clean up tonmek ask types, name is changed to Panacea_ask_types
	for vb_type in verb_types:
		for tomek_ask_type, types in panacea_ask_types.items():
				if vb_type in types and tomek_ask_type not in t_ask_types:
					t_ask_types.append(tomek_ask_type)

	return t_ask_types

# This functions checks to see if the items in a list already exist 
# in the original list and if not then add them.
def appendListNoDuplicates(list_to_append, original_list):
	for item in list_to_append:
		if item not in original_list:
			original_list.append(item)

	return original_list

def getNLPParse(sentence):
	annotators = '/?annotators=ssplit,tokenize,pos,parse,depparse&tokenize.english=true'
	tregex = '/tregex'
	coreNLP_ased = 'http://localhost:9000'
	#coreNLP_ased = 'https://corenlp.run'
	url = coreNLP_ased + annotators

	return requests.post(url, data=sentence.encode(encoding='UTF-8',errors='ignore'))

def getLemmaWords(sentence):
	sentence = sentence.lower()
	words = nltk.word_tokenize(sentence)
	return [(morphRootVerb(word.lower())) for word in words]

def extractAskInfoFromDependencies(base_word, dependencies, t_ask_types):
	base_word = base_word.lower()
	ask_who = ''
	ask = ''
	ask_recipient = ''
	ask_when = ''
	ask_negation_dep_based = False
	ask_actor_is_recipient = False
	confidence = ''
	root = ''
	nsubj = ''
	dobj = ''
	iobj = ''
	nsubj_gov_gloss = ''
	dobj_gov_gloss = ''
	iobj_gov_gloss = ''
	neg_gov_gloss = ''
	dep_neg_exists = False
	
	for dependency in dependencies:
		dep = dependency['dep']
		if dependency['governorGloss'] == base_word:
			if dep == 'root':
				root = dependency['dependentGloss']
			if dep == 'neg':
				neg_gov_gloss = dependency['governorGloss']
				dep_neg_exists = True
			if dep == 'nsubj':
				nsubj = dependency['dependentGloss']
				nsubj_gov_gloss = dependency['governorGloss']
			if dep == 'dobj':
				dobj = dependency['dependentGloss']
				dobj_gov_gloss = dependency['governorGloss']
			if dep == 'iobj':
				iobj = dependency['dependentGloss']
				iobj_gov_gloss = dependency['governorGloss']
			
	if 'GIVE' in t_ask_types:
		if nsubj_gov_gloss == base_word and dobj_gov_gloss == base_word and iobj_gov_gloss == base_word:
			ask_who = nsubj
			ask = dobj
			ask_recipient = iobj
			ask_action = root
			#ask_when = ''
			confidence = 'high'
		else:
			ask_who = nsubj
			ask = dobj
			ask_action = root
			confidence = 'low'
	else:
		if nsubj_gov_gloss == base_word and dobj_gov_gloss == base_word and iobj_gov_gloss == base_word:
			ask_who = nsubj
			ask = dobj
			ask_recipient = iobj
			ask_action = root
			confidence = 'low'
		elif 'LOSE' in t_ask_types or 'GAIN' in t_ask_types or 'PERFORM' in t_ask_types:
			ask = dobj
			ask_recipient = nsubj
			ask_action = root
			confidence ='high'
		else:
			ask_who = nsubj
			ask = dobj
			ask_action = root
			confidence = 'low'

	if neg_gov_gloss == base_word:
		ask_negation_dep_based = dep_neg_exists
	
	return(ask_who, ask, ask_recipient, ask_when, ask_negation_dep_based, base_word, confidence)

def extractStanceFromSrl(sentence, srl, base_word):
	tags_for_verb = ''
	arg0_with_indices = []
	arg1_with_indices = []
	arg2_with_indices = []
	arg3_with_indices = []
	verbs = srl['verbs']
	words = [word.lower() for word in srl['words']]

	#TODO if the same verb is in the sentence twice this will always take the second version of it 
	# This needs to be fixed, maybe through deleting the verb once it is used
	for verb in verbs:
		if verb['verb'].lower() == base_word:
			selected_verb = verb['verb']
			tags_for_verb = verb['tags']

	if tags_for_verb:
		for index, tag in enumerate(tags_for_verb):
			tag_label = tag.split('-')[1:2][0] if tag.split('-')[1:2] else ''

			if tag_label == 'ARG0':
				arg0_with_indices.append((words[index], index))
			elif tag_label == 'ARG1':
				arg1_with_indices.append((words[index], index))
			elif tag_label == 'ARG2':
				arg2_with_indices.append((words[index], index))
			elif tag_label == 'ARG3':
				arg3_with_indices.append((words[index], index))

	return (arg0_with_indices, arg1_with_indices, arg2_with_indices, arg3_with_indices)

def extractAskFromSrl(sentence, srl, base_word, t_ask_types):
	ask_who = ''
	ask = ''
	ask_recipient = ''
	ask_when = ''
	confidence = ''
	t_ask_confidence = ''
	selected_verb = ''
	tags_for_verb = ''
	improved_t_ask_types = ''
	ask_actor_is_recipient = False
	arg0 = []
	arg1 = []
	arg2 = []
	arg3 = []
	arg_tmp = []
	arg_mnr = []
	word_number = []
	arg0_with_indices = []
	arg1_with_indices = []
	arg2_with_indices = []
	arg3_with_indices = []
	verbs = srl['verbs']
	words = [word.lower() for word in srl['words']]
	descriptions = []


	#TODO if the same verb is in the sentence twice this will always take the second version of it 
	# This needs to be fixed, maybe through deleting the verb once it is used
	for verb in verbs:
		if verb['verb'].lower() == base_word:
			selected_verb = verb['verb']
			tags_for_verb = verb['tags']
			
		descriptions.append(verb['description'])

	if tags_for_verb:
		for index, tag in enumerate(tags_for_verb):
			tag_label = tag.split('-')[1:2][0] if tag.split('-')[1:2] else ''

			if tag_label == 'ARG0':
				arg0.append(words[index])
				arg0_with_indices.append((words[index], index))
			elif tag_label == 'ARG1':
				arg1.append(words[index])
				arg1_with_indices.append((words[index], index))
				#The placement of the word within the sentence
				word_number.append(index)
			elif tag_label == 'ARG2':
				arg2.append(words[index])
				arg2_with_indices.append((words[index], index))
			elif tag_label == 'ARG3':
				arg3.append(words[index])
				arg3_with_indices.append((words[index], index))
			elif 'ARGM-TMP' in tag:
				arg_tmp.append(words[index])
			elif 'ARGM-MNR' in tag:
				arg_mnr.append(words[index])

	#if not t_ask_types:
	#	#NOTE loop through agr1 words to get potential ask type
	#	arg1.split()
	#	t_ask_types = getTAskTypes(arg1)

	# Handling cases (seems like bugs in allennlp) where there is no arg1 but and arg2 and it seems like the arg2 should be arg1
	if not arg1 and arg2:
		arg1 = arg2
		arg2 = arg_mnr
		arg_mnr = []
	
	if 'GIVE' in t_ask_types:
		if arg0 and arg1 and arg2:
			ask_who = ' '.join(arg0)
			ask = ' '.join(arg1)
			ask_recipient = ' '.join(arg2)
			ask_when = ' '.join(arg_tmp)
			confidence = 'high'
		else:
			ask_who = ' '.join(arg0)
			ask = ' '.join(arg1)
			ask_when = ' '.join(arg_tmp)
			confidence = 'low'
	else:
		if arg0 and arg1 and arg2:
			ask_who = ' '.join(arg2)
			ask = ' '.join(arg1)
			ask_recipient = ' '.join(arg0)
			ask_when = ' '.join(arg_tmp)
			confidence = 'low'
		elif 'LOSE' in t_ask_types or 'GAIN' in t_ask_types:
			ask = ' '.join(arg1)
			ask_recipient = ' '.join(arg0)
			ask_when = ' '.join(arg_tmp)
			confidence = 'high'
		else:
			ask_who = ' '.join(arg0)
			ask = ' '.join(arg1)
			ask_when = ' '.join(arg_tmp)
			confidence = 'low'


	# If this is used again we need a case for LOSE
	'''
	if 'GIVE' in t_ask_types:
		if 'you' in arg2:
			t_ask_types = ["GAIN"]
		elif 'you' in arg0:
			t_ask_types = ["GIVE"]
		elif 'GAIN' in t_ask_types:
			if 'you' in arg0:
				t_ask_types = ["GAIN"]
			if 'i' in arg0 or 'we' in arg0:
				t_ask_types = ["GIVE"]
				t_ask_confidence = 'low'
	elif 'GAIN' in t_ask_types:
		if 'you' in arg0:
			t_ask_types = ["GAIN"]
		elif 'i' in arg0 or 'we' in arg0:
			t_ask_types = ["GIVE"]
			t_ask_confidence = 'low'
	elif 'OTHER' in t_ask_types:
		if arg0 and arg1 and arg2:
			if 'you' in arg2:
				t_ask_types = ["GIVE"]
			elif 'you' in arg0:
				t_ask_types = ["GAIN"]
	'''

	return(ask_who, ask, ask_recipient, ask_when, selected_verb, confidence, descriptions, t_ask_types, t_ask_confidence, word_number, arg2, arg0_with_indices, arg1_with_indices, arg2_with_indices, arg3_with_indices)

def processWord(word, word_pos, sentence, ask_procedure, ask_negation, dependencies, link_in_sentence, link_exists, link_strings, link_ids, link_id, links, srl, is_cop_dep, cop_ask_target, cop_gov_ask_target, big_root_is_nn, big_root_nn_ask_target, is_wh_advmod, advmod_ask_target, is_det_or_nmod, det_ask_target, nmod_poss_ask_target):
	ask_negation_dep_based = False
	is_past_tense = False
	s_ask_types = [] 
	a_ask_types = []
	arg2 = ''
	word = word.lower()
	lem_word = morphRootVerb(word)
	t_ask_types = getAskTypes(word, word_pos)
	lem_t_ask_types = getAskTypes(lem_word, word_pos)

	t_ask_types = appendListNoDuplicates(lem_t_ask_types, t_ask_types)

	(ask_who, ask, ask_recipient, ask_when, ask_action, confidence, descriptions, t_ask_types, t_ask_confidence, word_number, arg2, arg0_with_indices, arg1_with_indices, arg2_with_indices, arg3_with_indices) = extractAskFromSrl(sentence, srl, word, t_ask_types)

	if not ask_action:
		(ask_who, ask, ask_recipient, ask_when, ask_negation_dep_based, ask_action, confidence) = extractAskInfoFromDependencies(word, dependencies, t_ask_types)

	#NOTE This should only be commented out when running NO VB
	if word_pos in ['VBD', 'VBN', 'VBG']:
		is_past_tense = True
		# 8/13/19 Bonnie said for now we can ignore past tense and leave it out of asks, may change later
		#return


	if arg2:
		arg2 = ' '.join(arg2)

	for ask_type, keywords in sashank_categories.items():
		for keyword in keywords:
			if ask_type not in s_ask_types:
				left_boundary_regex = r'\b' + keyword + ' '
				right_boundary_regex = ' ' + keyword + r'\b'
				if keyword in ['$', '£', '€', '₹']:
					if keyword == '$':
						left_boundary_regex = r'\$'
						right_boundary_regex = r'\$'

					left_boundary_regex = keyword
					right_boundary_regex = keyword

				if keyword in ask:
					if len(keyword) == len(ask):
						s_ask_types.append(ask_type)
					elif re.search(left_boundary_regex, ask):
						s_ask_types.append(ask_type)
					elif re.search(right_boundary_regex, ask):
						s_ask_types.append(ask_type)
				if keyword.lower() == ask_action.lower():
					s_ask_types.append(ask_type)
				if keyword in arg2:
					if len(keyword) == len(arg2):
						s_ask_types.append(ask_type)
					elif re.search(left_boundary_regex, arg2):
						s_ask_types.append(ask_type)
					elif re.search(right_boundary_regex, arg2):
						s_ask_types.append(ask_type)

				#NOTE/TODO This is kind of a workaround, if the word is a dep of cop then we want to check 
				# the whole sentence for potential s_ask_type words. Later might want to extract the nsubj
				# of the WP attached to the cop but for now this will do 03/22/20
				if not s_ask_types:
					if is_cop_dep or big_root_is_nn or is_wh_advmod:
						if keyword in sentence:
							if len(keyword) == len(arg2):
								s_ask_types.append(ask_type)
							elif re.search(left_boundary_regex, arg2):
								s_ask_types.append(ask_type)
							elif re.search(right_boundary_regex, arg2):
								s_ask_types.append(ask_type)
			'''
			if (keyword in ask or keyword in ask_action or keyword in arg2) and ask_type not in s_ask_types:
				s_ask_types.append(ask_type)
			'''



	#NOTE only for case 2b original
	'''
	if len(t_ask_types) > 1:
		if 'GIVE' in t_ask_types:
			t_ask_types = ['GIVE']
		elif 'PERFORM' in t_ask_types:
			t_ask_types = ['PERFORM']
		elif 'LOSE' in t_ask_types:
			t_ask_types = ['LOSE']
		elif 'GAIN' in t_ask_types:
			t_ask_types = ['GAIN']
	'''


	# NOTE THIS IS ONLY COMMENTED OUT FOR CASE 2b ORIGINAL so that we can mimic prioritizing the ask types
	if 'PERFORM' in t_ask_types:
		if link_exists:
			t_ask_types = ['PERFORM']
		#Remove PERFORM if GIVE or GAIN has also been chosen.
		#NOTE It should not be the case that GIVE and GAIN are both present at this time.
		elif len(t_ask_types) > 1: 
			t_ask_types.remove('PERFORM')

	
	'''
	if 'GIVE' not in t_ask_types and 'LOSE' not in t_ask_types and 'GAIN' not in t_ask_types and 'PERFORM' not in t_ask_types:
		if (link_exists or link_in_sentence) and ask:
			t_ask_types = ['PERFORM']
	'''

	#TODO if this gets uncommented and used work out link_id as it will probably need to be an array
	'''
	if t_ask_types == ['PERFORM'] and link_in_sentence and not link_exists and ask and word_number:
		for index, link_string in enumerate(link_strings):
			if ask == link_string.lower():
				link_id = [link_ids[index]]
				link_exists = True
		ask_pieces = ask.split(' ')
		for index, ask_piece in enumerate(ask_pieces):
			dependent_number = word_number[index] + 1
			for dependency in dependencies:
				if dependency['dependent'] == dependent_number:
					gov_gloss = dependency['governorGloss']
					dep_gloss = dependency['dependentGloss']
					dependent = dependency['dependent']
					governor = dependency['governor']
			for dependency in dependencies:	
				# TODO governor == dependency['governor'] will at least always match itself may need to fix this
				if dependent == dependency['governor'] or governor == dependency['governor']:
					for index, link_string in enumerate(link_strings):
						#TODO Look into this more as this will always take the last dependency that matched.
						if dependency['dependentGloss'].lower() in link_string.lower():
							link_id = [link_ids[index]]
							link_exists = True
	'''

	'''
	with open("/Users/brodieslab/antiscam_sashank/s_ask_types.txt", "a") as s_types:
		s_types.write(sentence)
		s_types.write("\n")
		s_types.write(json.dumps(s_ask_types, indent=4, sort_keys=True))
		s_types.write("\n\n\n")
	'''
			
	#TODO Need to check on this and make sure this is actually what we wish to do.
	if ask_negation or ask_negation_dep_based:
		ask_negation = True

	if is_cop_dep or big_root_is_nn or is_wh_advmod or is_det_or_nmod:
		t_ask_types.append("GIVE")
		if big_root_is_nn:
			ask = big_root_nn_ask_target + " " + word
		elif is_wh_advmod or is_det_or_nmod:
			if advmod_ask_target:
				ask = word + " " + advmod_ask_target
			elif det_ask_target:
				ask = word + " " + det_ask_target
			elif nmod_poss_ask_target:
				ask = word + " " + nmod_poss_ask_target
		elif is_cop_dep and cop_ask_target:
			ask = cop_gov_ask_target + " " + cop_ask_target
		else:
			ask = "information"
		ask_action = "give"

	if t_ask_types and ask:
		if 'GIVE' in t_ask_types or 'PERFORM' in t_ask_types:
			is_ask_confidence = evaluateAskConfidence(is_past_tense, link_exists, ask, s_ask_types, t_ask_types)
		elif 'GAIN' in t_ask_types or 'LOSE' in t_ask_types:
			is_ask_confidence = 0.9

		return buildParseDict(sentence, '', '', '', ask_who, ask, ask_recipient, ask_when, ask_action, ask_procedure, ask_negation, ask_negation_dep_based, is_ask_confidence, confidence, descriptions, s_ask_types, t_ask_types, a_ask_types, t_ask_confidence, word, '', '', link_id, links)



def evaluateAskConfidence(is_past_tense, link_exists, ask, s_ask_types, t_ask_types):
	confidence_score = 0
	tense_score = 0
	hyper_link_score = 0

	if is_past_tense:
		return 0
	elif link_exists:
		return 0.9
	elif 'PERFORM' in t_ask_types:
		if s_ask_types:
			return 0.8
		else:
			return 0.7
	elif ask and s_ask_types:
		return 0.75
	elif ask:
		return 0.6
	elif s_ask_types:
		return 0.5
	else:
		return 0.1

def getBaseWordsPos(base_word_dependents, tokens):
	base_words_pos = []
	for base_word_dependent_num in base_word_dependents:
		base_words_pos.append(tokens[base_word_dependent_num - 1]["pos"])
		'''
		for token in tokens:
			if token['index'] == base_word_dependent_num:
				base_words_pos.append(token['pos'])
		'''

	return base_words_pos

def isVerbNegated(verb, dependencies):
	for dependency in dependencies:
		if dependency['dep'] == 'neg' and dependency['governorGloss'] == verb:
			return True
	return False

def combineVerbAndPosListsNoDups(base_words, base_word_dependents, parse_verbs, parse_verbs_pos, tokens):
	verbs_and_pos = []

	base_words_pos = getBaseWordsPos(base_word_dependents, tokens)

	for index, verb in enumerate(base_words):
		verbs_and_pos.append((verb, base_words_pos[index], base_word_dependents[index]))

	for index, verb in enumerate(parse_verbs):
		if verb not in base_words:
			verbs_and_pos.append((verb, parse_verbs_pos[index], None))	

	return verbs_and_pos

def parseSrl(line, link_offsets, link_ids, link_strings, links, last_ask, last_ask_index):
	line_framing_matches = []
	line_ask_matches = []
	asks_to_update = []
	link_id = []
	ask_negation = False
	base_word = ''
	conj_base_word = ''
	dep_base_word = ''
	ccomp_base_word = ''
	xcomp_base_word = ''
	
	
	response = getNLPParse(line)
	core_nlp_sentences = response.json()['sentences']

	'''
	with open("/Users/brodieslab/antiscam_sashank/corenlp.txt", "a") as corenlp_output:
		corenlp_output.write(json.dumps(core_nlp_sentences, indent=4, sort_keys=True))
		corenlp_output.write("\n\n\n")
	'''

	for nlp_sentence in core_nlp_sentences:
		ask_procedure = ''
		update_last_ask = False
		link_in_sentence = False
		link_exists = False
		sentence_link_ids = []
		parse_verbs_pos = []
		parse_verbs = []
		sentence_link_ids = []

		#TODO Investigate if this is needed
		#words = getLemmaWords(sentence)
		parse_tree = nlp_sentence['parse']
		dependencies = nlp_sentence['basicDependencies']
		tokens = nlp_sentence['tokens']	
		sentence_begin_char_offset = tokens[0]['characterOffsetBegin']
		sentence_end_char_offset = tokens[len(tokens) - 1]['characterOffsetEnd']

		# Extract all verbs and their parts of speech from the constituency parse to be used for fallback if all verbs are not found in the dependencies
		parse_verb_matches = extractVerbs(parse_tree)
		for parse_verb_match in parse_verb_matches:
			parse_verbs.append(parse_verb_match[1])
			parse_verbs_pos.append(parse_verb_match[0])
			

		#srl = predictor.predict(passage=rebuilt_sentence, question="")
		srl = predictor.predict(sentence=sentence.text)
		'''
		with open("/Users/brodieslab/antiscam_sashank/srloutput.txt", "a") as srloutput:
			srloutput.write(json.dumps(srl, indent=4, sort_keys=True))
			srloutput.write("\n\n\n")
		'''
		
		#print(srl)
		root_dep_is_nn = False
		small_root = ''
		root_dependent_gloss = ''
		aux_dependent = ''
		aux_governor  = ''
		advmod_governor_gloss = ''
		advmod_dependent_gloss = ''
		dep_governor_gloss = ''
		dep_dep = ''
		cop_with_wp_index = '' #This value can not be initialized at 0 because 0 is a viable index
		cop_gov_num = '' #This value can not be initialized at 0 because 0 is a viable governor number
		cop_ask_target = ''
		cop_gov_ask_target = ''
		advmod_ask_target = ''
		det_ask_target = ''
		nmod_poss_ask_target = ''
		punct_dependent = 0
		punctuation_to_match = [':', ';', '-']
		wh_word_pos = ["WP", "WP$", "WDT", "WRB"]
		nsubj_exists = False
		base_words = []
		base_words_pos = []
		base_words_dependents = []

		for index, dependency in enumerate(dependencies):
			'''
			if dependency['dep'] == 'punct' and dependency['dependentGloss'] in punctuation_to_match:
				dependent = dependency['dependent'] + 1
				for dependency2 in dependencies:
					if dependency2['dependent'] == dependent:
						root_dependent_gloss = dependency2['governorGloss']
			'''
			if dependency['dep'] == 'ROOT':
				base_word = dependency['dependentGloss']
				base_words.append(dependency['dependentGloss'])
				base_words_dependents.append(dependency['dependent'])
				if tokens[dependency['dependent'] - 1]['pos'] == "NN":
					root_dep_is_nn = True

			if dependency['dep'] == 'root' and not small_root:
				small_root = dependency['dependentGloss']
				base_words.append(dependency['dependentGloss'])
				base_words_dependents.append(dependency['dependent'])

			#TODO there is a way to simplify this whole operation here. Need to figure it out later.
			'''
			if dependency['governorGloss'] == dependencies[0]['dependentGloss'] or dependency['governorGloss'] == small_root:
				if dependency['dep'] == 'conj':
			'''
				
			if dependency['dep'] == 'conj':
				if dependency['governorGloss'] == dependencies[0]['dependentGloss'] or dependency['governorGloss'] == small_root:
					conj_base_word = dependency['dependentGloss']
					base_words.append(dependency['dependentGloss'])
					base_words_dependents.append(dependency['dependent'])
			if dependency['dep'] == 'dep':
				dep_governor_gloss = dependency['governorGloss']
				dep_dependent_gloss = dependency['dependentGloss']
				dep_dep = dependency['dep']
				if dependency['governorGloss'] == dependencies[0]['dependentGloss'] or dependency['governorGloss'] == small_root:
					dep_base_word = dependency['dependentGloss']
					base_words.append(dependency['dependentGloss'])
					base_words_dependents.append(dependency['dependent'])
			if dependency['dep'] == 'ccomp':
				if dependency['governorGloss'] == dependencies[0]['dependentGloss'] or dependency['governorGloss'] == small_root:
					ccomp_base_word = dependency['dependentGloss']
					base_words.append(dependency['dependentGloss'])
					base_words_dependents.append(dependency['dependent'])
				elif dep_dep == 'dep' and dependency['governorGloss'] == dep_dependent_gloss:
					if dep_governor_gloss == dependencies[0]['dependentGloss']:
						ccomp_base_word = dependency['dependentGloss']
						base_words.append(dependency['dependentGloss'])
						base_words_dependents.append(dependency['dependent'])
			if dependency['dep'] == 'xcomp':
				check_t_ask_types = getTAskType(dependencies[0]['dependentGloss'])
				if dependencies[0]['dependentGloss'] in base_words and 'PERFORM' not in check_t_ask_types:
					base_words.remove(dependencies[0]['dependentGloss'])

					#Added this check because it was trying to remove the same dependent number and it didn't exist anymore
					if dependencies[0]['dependent'] in base_words_dependents:
						base_words_dependents.remove(dependencies[0]['dependent'])
				if dependencies[0]['dependentGloss'] in parse_verbs and 'PERFORM' not in check_t_ask_types:
					parse_verbs_pos.pop(parse_verbs.index(dependencies[0]['dependentGloss']))
					parse_verbs.remove(dependencies[0]['dependentGloss'])
				if dependency['governorGloss'] == dependencies[0]['dependentGloss'] or dependency['governorGloss'] == small_root:
					xcomp_base_word = dependency['dependentGloss']
					base_words.append(dependency['dependentGloss'])
					base_words_dependents.append(dependency['dependent'])

			# We only want to add a cop dependency word if it's governor's POS os a WP
			if dependency['dep'] == 'cop':
				if tokens[dependency['governor'] - 1]["pos"] in wh_word_pos:
					base_words.append(dependency['dependentGloss'])
					base_words_dependents.append(dependency['dependent'])
					cop_gov_num = dependency['governor']
					cop_with_wp_index = index
					cop_gov_ask_target = dependency['governorGloss']

					#Found the big root as NN rule can cause duplicates so we want the other WH word cases to take precedent
					# so the big root nn word must be removed as well as it's dependent
					if dependencies[0]['dependentGloss'] in base_words and root_dep_is_nn:
						base_words.remove(dependencies[0]['dependentGloss'])

						#Added this check because it was trying to remove the same dependent number and it didn't exist anymore
						if dependencies[0]['dependent'] in base_words_dependents:
							base_words_dependents.remove(dependencies[0]['dependent'])
					

			# This check is to provide a more detailed ask target than just "information" but only in the case
			# that a copula situation occurs in the sentence. 
			# NOTE In the current state I believe this assumes that the cop must come before the nsubj in the 
			# dependcy list. This may need to be addressed.
			if dependency['dep'] == 'nsubj':
				if cop_gov_num and cop_gov_num == dependency['governor']:
					cop_ask_target = dependency['dependentGloss']

			# In the three cases below (advmod, det, nmod:poss) we take the governor gloss as well because 
			# thus far it seems combining it together with the wh word makes for a much better ask
			if dependency['dep'] == 'advmod':
				if tokens[dependency['dependent'] - 1]['pos'] in wh_word_pos:
					base_words.append(dependency['dependentGloss'])
					base_words_dependents.append(dependency['dependent'])
					advmod_ask_target = dependency['governorGloss']

					#Found the big root as NN rule can cause duplicates so we want the other WH word cases to take precedent
					# so the big root nn word must be removed as well as it's dependent
					if dependencies[0]['dependentGloss'] in base_words and root_dep_is_nn:
						base_words.remove(dependencies[0]['dependentGloss'])

						#Added this check because it was trying to remove the same dependent number and it didn't exist anymore
						if dependencies[0]['dependent'] in base_words_dependents:
							base_words_dependents.remove(dependencies[0]['dependent'])

			if dependency['dep'] == 'det' or dependency['dep'] == 'nmod:poss':
				if tokens[dependency['dependent'] - 1]['pos'] in ["WP", "WP$", "WDT"]:
					base_words.append(dependency['dependentGloss'])
					base_words_dependents.append(dependency['dependent'])
					if dependency['dep'] == 'det':
						det_ask_target = dependency['governorGloss']
					elif dependency['dep'] == 'nmod:poss':
						nmod_poss_ask_target = dependency['governorGloss']

					#Found the big root as NN rule can cause duplicates so we want the other WH word cases to take precedent
					# so the big root nn word must be removed as well as it's dependent
					if dependencies[0]['dependentGloss'] in base_words and root_dep_is_nn:
						base_words.remove(dependencies[0]['dependentGloss'])

						#Added this check because it was trying to remove the same dependent number and it didn't exist anymore
						if dependencies[0]['dependent'] in base_words_dependents:
							base_words_dependents.remove(dependencies[0]['dependent'])

			# This chunk is for determining if the ask is a request or a directive
			if dependency['dep'] == 'aux':
				aux_dependent = dependency['dependent']
				aux_governor  = dependency['governor']
			if dependency['dep'] == 'nsubj' and aux_dependent:
				nsubj_exists = True
				if dependency['dependent'] > aux_dependent and dependency['governor'] == aux_governor:
					ask_procedure = 'request'

			if dependency['dep'] == 'advmod':
				advmod_governor_gloss = dependency['governorGloss']
				advmod_dependent_gloss = dependency['dependentGloss']

		# Put the verbs with their parts of speech into one list without duplicates
		verbs_and_pos = combineVerbAndPosListsNoDups(base_words, base_words_dependents, parse_verbs, parse_verbs_pos, tokens)


		if not nsubj_exists:
			ask_procedure = 'directive'

		for index, verb_and_pos in enumerate(verbs_and_pos):
			verb = verb_and_pos[0]
			pos = verb_and_pos[1]
			dep_num = verb_and_pos[2]	

			#if dep_num:
			is_det_or_nmod = True if dep_num and tokens[dep_num - 1]['pos'] in  ["WP", "WP$", "WDT"] and tokens[dep_num - 1]['originalText'] == verb else False
			is_wh_advmod = True if dep_num and tokens[dep_num - 1]['pos'] in wh_word_pos and tokens[dep_num - 1]['originalText'] == verb else False
			is_cop_dep = True if dep_num and cop_with_wp_index and dependencies[cop_with_wp_index]["dep"] == "cop" and dependencies[cop_with_wp_index]["dependentGloss"] == verb and  dependencies[cop_with_wp_index]["dependent"] == dep_num else False
			big_root_is_nn  = True if dep_num and dependencies[0]["dependent"] == dep_num and verb == dependencies[0]["dependentGloss"] and tokens[dep_num - 1]["pos"] == "NN" else False
			big_root_nn_ask_target = "what" if big_root_is_nn else ""
			
			# 8/13/19 Bonnie said for now we can ignore VBG and leave it out of asks, may change later
			#if pos == 'VBG':
			#	continue
			link_id = []
			link_exists = False
			ask_negation = False

			ask_negation = isVerbNegated(verb, dependencies)

			for index, link_offset in enumerate(link_offsets):

				if link_offset[0] >= sentence_begin_char_offset and link_offset[1] <= sentence_end_char_offset and link_strings[index] in rebuilt_sentence:
					#NOTE This check is needed because we are looping through each verb so this was sometimes adding the same link id more than once
					if link_ids[index] not in sentence_link_ids:
						sentence_link_ids.append(link_ids[index])
					link_in_sentence = True

					if verb == advmod_governor_gloss and advmod_dependent_gloss in link_strings[index]:
						link_id.append(link_ids[index])
						link_exists = True
						break

					child_dependent_nums = []
					#TODO Need to figure out if breaking here is appropriate or if I should build a list of all the dependencies with the verb as the dependentGloss.
					for dependency in dependencies:
						if verb == dependency['dependentGloss']:
							verb_dependent_num = dependency['dependent']
							break
					for dependency in dependencies:
						if dependency['governor'] == verb_dependent_num:
							child_dependent_nums.append(dependency['dependent'])
					for child_dependent_num in child_dependent_nums:
						for dependency in dependencies:
							if child_dependent_num == dependency['governor'] and dependency['dependentGloss'] in link_strings[index]:
								#NOTE Had to check if link id is already in list. Cases where the whole sentence is the link 
								#causes this to easily overproduce link ids
								if link_ids[index] not in link_id:
									link_id.append(link_ids[index])
									link_exists = True
								break

			ask_details = processWord(verb, pos, rebuilt_sentence, ask_procedure, ask_negation, dependencies, link_in_sentence, link_exists, link_strings, link_ids, link_id, links, srl, is_cop_dep, cop_ask_target, cop_gov_ask_target, big_root_is_nn, big_root_nn_ask_target, is_wh_advmod, advmod_ask_target, is_det_or_nmod, det_ask_target, nmod_poss_ask_target)
			if ask_details:
				if 'GIVE' in ask_details['t_ask_type'] or 'PERFORM' in ask_details['t_ask_type']:
					#TODO Check and see if this line should be in the if condition below. As one of the conditions for advanced url
					#processing is that line ask matches is empty.
					line_ask_matches.append(ask_details)
					if ask_details['is_ask_confidence'] != 0:
						last_ask = ask_details
						last_ask_index += 1
				elif 'GAIN' in ask_details['t_ask_type'] or 'LOSE' in ask_details['t_ask_type']:
					line_framing_matches.append(ask_details)

		if not line_ask_matches and link_in_sentence and last_ask and last_ask['is_ask_confidence'] != 0:
			#TODO IMPORTANT I am just joining all sentence_link_ids at the bottom look into sentence_link_ids to find out why it has duplicate numbers
			for sentence_link_id in sentence_link_ids:
				last_ask['link_id'].append(sentence_link_id)	
				last_ask['url'].update({sentence_link_id: links.get(sentence_link_id)})
			#last_ask['link_id'] = link_ids[0]
			if last_ask['ask_negation']:
				last_ask['ask_rep'] = f'<{last_ask["t_ask_type"][0]}[NOT {last_ask["ask_action"]}[{last_ask["ask_target"]}({",".join(sentence_link_ids)}){last_ask["s_ask_type"]}]]>'
			else:
				last_ask['ask_rep'] = f'<{last_ask["t_ask_type"][0]}[{last_ask["ask_action"]}[{last_ask["ask_target"]}({",".join(sentence_link_ids)}){last_ask["s_ask_type"]}]]>'

			asks_to_update.append((last_ask, last_ask_index))

	if line_framing_matches or line_ask_matches or asks_to_update:
		return (line_framing_matches, line_ask_matches, asks_to_update, last_ask, last_ask_index)

def parseSrlStanza(line, link_offsets, link_ids, link_strings, links, last_ask, last_ask_index):
	line_framing_matches = []
	line_ask_matches = []
	asks_to_update = []
	link_id = []
	ask_negation = False
	base_word = ''
	conj_base_word = ''
	dep_base_word = ''
	ccomp_base_word = ''
	xcomp_base_word = ''
	
	
	stanza_doc = stanza_nlp(line)

	'''
	with open("/Users/brodieslab/antiscam_sashank/corenlp.txt", "a") as corenlp_output:
		corenlp_output.write(json.dumps(core_nlp_sentences, indent=4, sort_keys=True))
		corenlp_output.write("\n\n\n")
	'''

	for sentence in stanza_doc.sentences:
		ask_procedure = ''
		update_last_ask = False
		link_in_sentence = False
		link_exists = False
		sentence_link_ids = []
		rebuilt_sentence = []
		parse_verbs_pos = []
		parse_verbs = []
		sentence_link_ids = []

		#TODO Investigate if this is needed
		#words = getLemmaWords(sentence)
		parse_tree = nlp_sentence['parse']
		dependencies = nlp_sentence['basicDependencies']
		tokens = nlp_sentence['tokens']	
		sentence_begin_char_offset = tokens[0]['characterOffsetBegin']
		sentence_end_char_offset = tokens[len(tokens) - 1]['characterOffsetEnd']

		for token in tokens:
			rebuilt_sentence.append(token['before'])
			rebuilt_sentence.append(token['originalText'])

		rebuilt_sentence = ''.join(rebuilt_sentence)

		# Extract all verbs and their parts of speech from the constituency parse to be used for fallback if all verbs are not found in the dependencies
		parse_verb_matches = extractVerbs(parse_tree)
		for parse_verb_match in parse_verb_matches:
			parse_verbs.append(parse_verb_match[1])
			parse_verbs_pos.append(parse_verb_match[0])
			

		#srl = predictor.predict(passage=rebuilt_sentence, question="")
		srl = predictor.predict(sentence=rebuilt_sentence)
		'''
		with open("/Users/brodieslab/antiscam_sashank/srloutput.txt", "a") as srloutput:
			srloutput.write(json.dumps(srl, indent=4, sort_keys=True))
			srloutput.write("\n\n\n")
		'''
		
		#print(srl)
		root_dep_is_nn = False
		small_root = ''
		root_dependent_gloss = ''
		aux_dependent = ''
		aux_governor  = ''
		advmod_governor_gloss = ''
		advmod_dependent_gloss = ''
		dep_governor_gloss = ''
		dep_dep = ''
		cop_with_wp_index = '' #This value can not be initialized at 0 because 0 is a viable index
		cop_gov_num = '' #This value can not be initialized at 0 because 0 is a viable governor number
		cop_ask_target = ''
		cop_gov_ask_target = ''
		advmod_ask_target = ''
		det_ask_target = ''
		nmod_poss_ask_target = ''
		punct_dependent = 0
		punctuation_to_match = [':', ';', '-']
		wh_word_pos = ["WP", "WP$", "WDT", "WRB"]
		nsubj_exists = False
		base_words = []
		base_words_pos = []
		base_words_dependents = []

		for index, dependency in enumerate(dependencies):
			'''
			if dependency['dep'] == 'punct' and dependency['dependentGloss'] in punctuation_to_match:
				dependent = dependency['dependent'] + 1
				for dependency2 in dependencies:
					if dependency2['dependent'] == dependent:
						root_dependent_gloss = dependency2['governorGloss']
			'''
			if dependency['dep'] == 'ROOT':
				base_word = dependency['dependentGloss']
				base_words.append(dependency['dependentGloss'])
				base_words_dependents.append(dependency['dependent'])
				if tokens[dependency['dependent'] - 1]['pos'] == "NN":
					root_dep_is_nn = True

			if dependency['dep'] == 'root' and not small_root:
				small_root = dependency['dependentGloss']
				base_words.append(dependency['dependentGloss'])
				base_words_dependents.append(dependency['dependent'])

			#TODO there is a way to simplify this whole operation here. Need to figure it out later.
			'''
			if dependency['governorGloss'] == dependencies[0]['dependentGloss'] or dependency['governorGloss'] == small_root:
				if dependency['dep'] == 'conj':
			'''
				
			if dependency['dep'] == 'conj':
				if dependency['governorGloss'] == dependencies[0]['dependentGloss'] or dependency['governorGloss'] == small_root:
					conj_base_word = dependency['dependentGloss']
					base_words.append(dependency['dependentGloss'])
					base_words_dependents.append(dependency['dependent'])
			if dependency['dep'] == 'dep':
				dep_governor_gloss = dependency['governorGloss']
				dep_dependent_gloss = dependency['dependentGloss']
				dep_dep = dependency['dep']
				if dependency['governorGloss'] == dependencies[0]['dependentGloss'] or dependency['governorGloss'] == small_root:
					dep_base_word = dependency['dependentGloss']
					base_words.append(dependency['dependentGloss'])
					base_words_dependents.append(dependency['dependent'])
			if dependency['dep'] == 'ccomp':
				if dependency['governorGloss'] == dependencies[0]['dependentGloss'] or dependency['governorGloss'] == small_root:
					ccomp_base_word = dependency['dependentGloss']
					base_words.append(dependency['dependentGloss'])
					base_words_dependents.append(dependency['dependent'])
				elif dep_dep == 'dep' and dependency['governorGloss'] == dep_dependent_gloss:
					if dep_governor_gloss == dependencies[0]['dependentGloss']:
						ccomp_base_word = dependency['dependentGloss']
						base_words.append(dependency['dependentGloss'])
						base_words_dependents.append(dependency['dependent'])
			if dependency['dep'] == 'xcomp':
				check_t_ask_types = getTAskType(dependencies[0]['dependentGloss'])
				if dependencies[0]['dependentGloss'] in base_words and 'PERFORM' not in check_t_ask_types:
					base_words.remove(dependencies[0]['dependentGloss'])

					#Added this check because it was trying to remove the same dependent number and it didn't exist anymore
					if dependencies[0]['dependent'] in base_words_dependents:
						base_words_dependents.remove(dependencies[0]['dependent'])
				if dependencies[0]['dependentGloss'] in parse_verbs and 'PERFORM' not in check_t_ask_types:
					parse_verbs_pos.pop(parse_verbs.index(dependencies[0]['dependentGloss']))
					parse_verbs.remove(dependencies[0]['dependentGloss'])
				if dependency['governorGloss'] == dependencies[0]['dependentGloss'] or dependency['governorGloss'] == small_root:
					xcomp_base_word = dependency['dependentGloss']
					base_words.append(dependency['dependentGloss'])
					base_words_dependents.append(dependency['dependent'])

			# We only want to add a cop dependency word if it's governor's POS os a WP
			if dependency['dep'] == 'cop':
				if tokens[dependency['governor'] - 1]["pos"] in wh_word_pos:
					base_words.append(dependency['dependentGloss'])
					base_words_dependents.append(dependency['dependent'])
					cop_gov_num = dependency['governor']
					cop_with_wp_index = index
					cop_gov_ask_target = dependency['governorGloss']

					#Found the big root as NN rule can cause duplicates so we want the other WH word cases to take precedent
					# so the big root nn word must be removed as well as it's dependent
					if dependencies[0]['dependentGloss'] in base_words and root_dep_is_nn:
						base_words.remove(dependencies[0]['dependentGloss'])

						#Added this check because it was trying to remove the same dependent number and it didn't exist anymore
						if dependencies[0]['dependent'] in base_words_dependents:
							base_words_dependents.remove(dependencies[0]['dependent'])
					

			# This check is to provide a more detailed ask target than just "information" but only in the case
			# that a copula situation occurs in the sentence. 
			# NOTE In the current state I believe this assumes that the cop must come before the nsubj in the 
			# dependcy list. This may need to be addressed.
			if dependency['dep'] == 'nsubj':
				if cop_gov_num and cop_gov_num == dependency['governor']:
					cop_ask_target = dependency['dependentGloss']

			# In the three cases below (advmod, det, nmod:poss) we take the governor gloss as well because 
			# thus far it seems combining it together with the wh word makes for a much better ask
			if dependency['dep'] == 'advmod':
				if tokens[dependency['dependent'] - 1]['pos'] in wh_word_pos:
					base_words.append(dependency['dependentGloss'])
					base_words_dependents.append(dependency['dependent'])
					advmod_ask_target = dependency['governorGloss']

					#Found the big root as NN rule can cause duplicates so we want the other WH word cases to take precedent
					# so the big root nn word must be removed as well as it's dependent
					if dependencies[0]['dependentGloss'] in base_words and root_dep_is_nn:
						base_words.remove(dependencies[0]['dependentGloss'])

						#Added this check because it was trying to remove the same dependent number and it didn't exist anymore
						if dependencies[0]['dependent'] in base_words_dependents:
							base_words_dependents.remove(dependencies[0]['dependent'])

			if dependency['dep'] == 'det' or dependency['dep'] == 'nmod:poss':
				if tokens[dependency['dependent'] - 1]['pos'] in ["WP", "WP$", "WDT"]:
					base_words.append(dependency['dependentGloss'])
					base_words_dependents.append(dependency['dependent'])
					if dependency['dep'] == 'det':
						det_ask_target = dependency['governorGloss']
					elif dependency['dep'] == 'nmod:poss':
						nmod_poss_ask_target = dependency['governorGloss']

					#Found the big root as NN rule can cause duplicates so we want the other WH word cases to take precedent
					# so the big root nn word must be removed as well as it's dependent
					if dependencies[0]['dependentGloss'] in base_words and root_dep_is_nn:
						base_words.remove(dependencies[0]['dependentGloss'])

						#Added this check because it was trying to remove the same dependent number and it didn't exist anymore
						if dependencies[0]['dependent'] in base_words_dependents:
							base_words_dependents.remove(dependencies[0]['dependent'])

			# This chunk is for determining if the ask is a request or a directive
			if dependency['dep'] == 'aux':
				aux_dependent = dependency['dependent']
				aux_governor  = dependency['governor']
			if dependency['dep'] == 'nsubj' and aux_dependent:
				nsubj_exists = True
				if dependency['dependent'] > aux_dependent and dependency['governor'] == aux_governor:
					ask_procedure = 'request'

			if dependency['dep'] == 'advmod':
				advmod_governor_gloss = dependency['governorGloss']
				advmod_dependent_gloss = dependency['dependentGloss']

		# Put the verbs with their parts of speech into one list without duplicates
		verbs_and_pos = combineVerbAndPosListsNoDups(base_words, base_words_dependents, parse_verbs, parse_verbs_pos, tokens)


		if not nsubj_exists:
			ask_procedure = 'directive'

		for index, verb_and_pos in enumerate(verbs_and_pos):
			verb = verb_and_pos[0]
			pos = verb_and_pos[1]
			dep_num = verb_and_pos[2]	

			#if dep_num:
			is_det_or_nmod = True if dep_num and tokens[dep_num - 1]['pos'] in  ["WP", "WP$", "WDT"] and tokens[dep_num - 1]['originalText'] == verb else False
			is_wh_advmod = True if dep_num and tokens[dep_num - 1]['pos'] in wh_word_pos and tokens[dep_num - 1]['originalText'] == verb else False
			is_cop_dep = True if dep_num and cop_with_wp_index and dependencies[cop_with_wp_index]["dep"] == "cop" and dependencies[cop_with_wp_index]["dependentGloss"] == verb and  dependencies[cop_with_wp_index]["dependent"] == dep_num else False
			big_root_is_nn  = True if dep_num and dependencies[0]["dependent"] == dep_num and verb == dependencies[0]["dependentGloss"] and tokens[dep_num - 1]["pos"] == "NN" else False
			big_root_nn_ask_target = "what" if big_root_is_nn else ""
			
			# 8/13/19 Bonnie said for now we can ignore VBG and leave it out of asks, may change later
			#if pos == 'VBG':
			#	continue
			link_id = []
			link_exists = False
			ask_negation = False

			ask_negation = isVerbNegated(verb, dependencies)

			for index, link_offset in enumerate(link_offsets):

				if link_offset[0] >= sentence_begin_char_offset and link_offset[1] <= sentence_end_char_offset and link_strings[index] in rebuilt_sentence:
					#NOTE This check is needed because we are looping through each verb so this was sometimes adding the same link id more than once
					if link_ids[index] not in sentence_link_ids:
						sentence_link_ids.append(link_ids[index])
					link_in_sentence = True

					if verb == advmod_governor_gloss and advmod_dependent_gloss in link_strings[index]:
						link_id.append(link_ids[index])
						link_exists = True
						break

					child_dependent_nums = []
					#TODO Need to figure out if breaking here is appropriate or if I should build a list of all the dependencies with the verb as the dependentGloss.
					for dependency in dependencies:
						if verb == dependency['dependentGloss']:
							verb_dependent_num = dependency['dependent']
							break
					for dependency in dependencies:
						if dependency['governor'] == verb_dependent_num:
							child_dependent_nums.append(dependency['dependent'])
					for child_dependent_num in child_dependent_nums:
						for dependency in dependencies:
							if child_dependent_num == dependency['governor'] and dependency['dependentGloss'] in link_strings[index]:
								#NOTE Had to check if link id is already in list. Cases where the whole sentence is the link 
								#causes this to easily overproduce link ids
								if link_ids[index] not in link_id:
									link_id.append(link_ids[index])
									link_exists = True
								break

			ask_details = processWord(verb, pos, sentence.text, ask_procedure, ask_negation, dependencies, link_in_sentence, link_exists, link_strings, link_ids, link_id, links, srl, is_cop_dep, cop_ask_target, cop_gov_ask_target, big_root_is_nn, big_root_nn_ask_target, is_wh_advmod, advmod_ask_target, is_det_or_nmod, det_ask_target, nmod_poss_ask_target)
			if ask_details:
				if 'GIVE' in ask_details['t_ask_type'] or 'PERFORM' in ask_details['t_ask_type']:
					#TODO Check and see if this line should be in the if condition below. As one of the conditions for advanced url
					#processing is that line ask matches is empty.
					line_ask_matches.append(ask_details)
					if ask_details['is_ask_confidence'] != 0:
						last_ask = ask_details
						last_ask_index += 1
				elif 'GAIN' in ask_details['t_ask_type'] or 'LOSE' in ask_details['t_ask_type']:
					line_framing_matches.append(ask_details)

		if not line_ask_matches and link_in_sentence and last_ask and last_ask['is_ask_confidence'] != 0:
			#TODO IMPORTANT I am just joining all sentence_link_ids at the bottom look into sentence_link_ids to find out why it has duplicate numbers
			for sentence_link_id in sentence_link_ids:
				last_ask['link_id'].append(sentence_link_id)	
				last_ask['url'].update({sentence_link_id: links.get(sentence_link_id)})
			#last_ask['link_id'] = link_ids[0]
			if last_ask['ask_negation']:
				last_ask['ask_rep'] = f'<{last_ask["t_ask_type"][0]}[NOT {last_ask["ask_action"]}[{last_ask["ask_target"]}({",".join(sentence_link_ids)}){last_ask["s_ask_type"]}]]>'
			else:
				last_ask['ask_rep'] = f'<{last_ask["t_ask_type"][0]}[{last_ask["ask_action"]}[{last_ask["ask_target"]}({",".join(sentence_link_ids)}){last_ask["s_ask_type"]}]]>'

			asks_to_update.append((last_ask, last_ask_index))
	if line_framing_matches or line_ask_matches or asks_to_update:
		return (line_framing_matches, line_ask_matches, asks_to_update, last_ask, last_ask_index)

def get_stances(text_number, text, author = '', timestamp = '', doc_id = ''):
	#bert_docs = text
	#bert_docs = fetch_20newsgroups(subset='all')['data']
	#topics = model.fit_transform(bert_docs)	

	#return model.get_topic(12)
	stances = []

	#May want to move this stuff to load resources later
	positive_modalities = ["want"] 
	negative_modalities = ["notwant"] 

	#stanza_doc = stanza_nlp(text)
	spacy_doc = spacy_nlp(text)

	for sent in spacy_doc.sents:
		sent_stances = []
		possible_triggers = []
		srl = predictor.predict(sentence=sent.text)

		for token in sent:
			if token.dep_ == "ROOT" or token.dep_ == "xcomp" or token.pos_ == "VERB":
				#print(token.text)
				root_negation_children_count = 0
				is_positive_modality = False
				is_negative_modality = False
				word = token.text.lower()
				lem_word = morphRootVerb(word)
				belief_types = getBeliefType(word, token.pos_)
				lem_belief_types = getBeliefType(lem_word, token.pos_)
				#(belief_types, event_sentiment, strength) = getBeliefType(word, token.pos_)
				#(lem_belief_types, lem_event_sentiment, lem_strength) = getBeliefType(lem_word, token.pos_)

				belief_types = appendListNoDuplicates(lem_belief_types, belief_types)
				if not belief_types:
					belief_types = [('','','')]
				##At sometime may be that belief strength could be 0 which is allowable. Since 0 can be falsy
				## this line handles that. 
				#if not strength and strength != 0 and (lem_strength or lem_strength == 0):
				#	strength = lem_strength

				#if not event_sentiment and event_sentiment != 0 and (lem_event_sentiment or lem_event_sentiment == 0):
				#	event_sentiment = lem_event_sentiment

				#TODO Trigger words may be associated with more than one belief type in the future, will need
				# to find a way to handle that without making the strength and sentiment adjusting code run twice
				#For now just acting as though it has only one type per word
				strength = belief_types[0][1]
				event_sentiment = belief_types[0][2]

				#print("belief types: ", belief_types)
				

				belief_strength = 0
				strength_polarity = 1
				strength_word_count = 1
				strength_words_and_indices = []

				sentiment = 1
				sentiment_polarity = 1
				sentiment_word_count = 1
				sentiment_words_and_indices = []

				#In the case the trigger is the light verb the belief strength needs to be set at this point 
				# in the code or it will come out as zero. 
				#EXPERIMENTATION This needs to be commented out unless running experimentation with light verbs
				if word in pitt_light_verbs or lem_word in pitt_light_verbs:
					belief_strength = 3

				#TODO Check if I need to add 1 to strength word count to get correct score, not sure why but 
				# doesn't seem so. Will need to investigate
				if strength or strength == 0:
					if strength < 0:
						belief_strength += strength * -1	
						strength_polarity *= -1
					else:
						belief_strength += strength

				if word not in strength_and_sentiment_dict and lem_word not in strength_and_sentiment_dict:
					if event_sentiment or event_sentiment == 0:
						sentiment_word_count += 1
						if event_sentiment < 0:
							sentiment += event_sentiment * -1
							sentiment_polarity *= -1
						else:
							sentiment += event_sentiment

				for child in token.children:
						(belief_strength, strength_polarity, strength_word_count, strength_adjusted) = adjust_belief_strength(belief_strength, strength_polarity, strength_word_count, child.text)
						if strength_adjusted:
							strength_words_and_indices.append((child.text, child.i - sent.start))

						(sentiment, sentiment_polarity, sentiment_word_count, sentiment_adjusted) = adjust_sentiment(sentiment, sentiment_polarity, sentiment_word_count, child.text)
						if sentiment_adjusted:
							sentiment_words_and_indices.append((child.text, child.i - sent.start))

				if token.dep_ != "ROOT":
					if token.head.dep_ == "ROOT":
						details = {}
						lowered_head = token.head.text.lower()
						if lowered_head in strength_and_sentiment_dict:
							details = strength_and_sentiment_dict.get(lowered_head)
						elif morphRootVerb(lowered_head) in strength_and_sentiment_dict:
							details = strength_and_sentiment_dict.get(morphRootVerb(lowered_head))

						if details:
							if details.get("modality").lower() in positive_modalities:
								is_positive_modality = True
							elif details.get("modality").lower() in negative_modalities:
								is_negative_modality = True

								
							for child in token.head.children:
								child_details = {}
								lowered_child = child.text.lower()
								if lowered_child in strength_and_sentiment_dict:
									child_details = strength_and_sentiment_dict.get(lowered_child)
								elif morphRootNoun(lowered_child) in strength_and_sentiment_dict:
									child_details = strength_and_sentiment_dict.get(morphRootNoun(lowered_child))
								elif morphRootVerb(lowered_child) in strength_and_sentiment_dict:
									child_details = strength_and_sentiment_dict.get(morphRootVerb(lowered_child))

								if child_details:
									if child_details.get("modality").lower() == "negation":
										root_negation_children_count += 1
							
							

					if (token.head.text, token.head.i - sent.start) not in strength_words_and_indices:
						(belief_strength, strength_polarity, strength_word_count, strength_adjusted) = adjust_belief_strength(belief_strength, strength_polarity, strength_word_count, token.head.text)
						if strength_adjusted:
							strength_words_and_indices.append((token.head.text, token.head.i - sent.start))
				
					if (token.head.text, token.head.i - sent.start) not in sentiment_words_and_indices:
						(sentiment, sentiment_polarity, sentiment_word_count, sentiment_adjusted) = adjust_sentiment(sentiment, sentiment_polarity, sentiment_word_count, token.head.text)
						if sentiment_adjusted:
							sentiment_words_and_indices.append((token.head.text, token.head.i - sent.start))
				
				
				for child in token.head.children:
					#if token.text == "use":
					#	#print(child.text, "child index", child.i, "token index: ", token.i)
					#Need to ensure the token itself doesn't get considered more than once for adjusting strength or sentiment which occurs when the token is the ROOT
					if child.i != token.i:
						if (child.text, child.i - sent.start) not in strength_words_and_indices:
							(belief_strength, strength_polarity, strength_word_count, strength_adjusted) = adjust_belief_strength(belief_strength, strength_polarity, strength_word_count, child.text)
							if strength_adjusted:
								strength_words_and_indices.append((child.text, child.i - sent.start))

						if (child.text, child.i - sent.start) not in sentiment_words_and_indices:
							(sentiment, sentiment_polarity, sentiment_word_count, sentiment_adjusted) = adjust_sentiment(sentiment, sentiment_polarity, sentiment_word_count, child.text)
							if sentiment_adjusted:
								sentiment_words_and_indices.append((child.text, child.i - sent.start))

				#if token.text == "use":
				#	#print("calculated strength: ", (belief_strength / strength_word_count) * strength_polarity)
				possible_triggers.append((token.text, token.i - sent.start, token.tag_, (belief_strength / strength_word_count) * strength_polarity, strength_words_and_indices, (sentiment / sentiment_word_count) * sentiment_polarity, sentiment_words_and_indices, belief_types, is_positive_modality, is_negative_modality, root_negation_children_count))

		for trigger, trigger_index, pos, strength, strength_words_and_indices, sentiment, sentiment_words_and_indices, belief_types, is_positive_modality, is_negative_modality, root_negation_children_count in possible_triggers:
				for belief_type, belief_type_strength, belief_type_event_sentiment in belief_types:
					stance_details = process_stance(trigger, pos, sent.text, srl, belief_type, strength, belief_type_event_sentiment)

					if stance_details:
						belief_type = stance_details[0]
						belief_valuation = stance_details[1]
						content_with_indices = stance_details[2]

						content_with_indices.sort(key=lambda x:x[1])
						(allen_prob_sentiment, sentiment_probs) = get_sentiment_score(trigger + " " + ' '.join([x[0] for x in content_with_indices]))
						allen_sentiment = get_valuation_score(allen_prob_sentiment)

						# Check if the belief type came from the sentiment lexicon if so the strength should 
						# be 3.00
						if stance_details[3]:
							strength = 3.00
							#belief_valuation = sentiment
							if sentiment < 0:
								belief_valuation *= -1

						if strength * belief_valuation < 0:
							if is_positive_modality or is_negative_modality:
								is_odd = root_negation_children_count % 2
								if (is_positive_modality and is_odd) or (is_negative_modality and not is_odd):
									strength *= -1
									belief_valuation *= -1
								
						#print("trigger: ", trigger, "strength words: ", strength_words_and_indices)
						#if strength == 0:
							
							#strength = 3.00

						#NOTE targeted sentiment is the quotes. Need to be filled in or removed at some point
						sent_stances.append(build_stance_dict(belief_type, (trigger, trigger_index), content_with_indices, strength_words_and_indices, strength, belief_valuation, '', sentiment_probs, allen_sentiment, sent.text, author, timestamp, doc_id, text_number))

		
		if sent_stances:
			exist_stance_exists = False
			exist_stance_indices = []

			for index, stance in enumerate(sent_stances):
				if stance["belief_type"] == default_belief_type:
					exist_stance_exists = True
					exist_stance_indices.append(index)

			if len(exist_stance_indices) != len(sent_stances):
				for index in sorted(exist_stance_indices, reverse=True):
					del sent_stances[index]

			stances.extend(sent_stances)

	if stances:
		return stances

def adjust_belief_strength(belief_strength, polarity, word_count, word):
	strength_adjusted = False
	lowered_word = word.lower()
	details = {}

	if lowered_word in strength_and_sentiment_dict:
		details = strength_and_sentiment_dict.get(lowered_word)
	elif morphRootNoun(lowered_word) in strength_and_sentiment_dict:
		details = strength_and_sentiment_dict.get(morphRootNoun(lowered_word))
	elif morphRootVerb(lowered_word) in strength_and_sentiment_dict:
		details = strength_and_sentiment_dict.get(morphRootVerb(lowered_word))

	if details:	
		temp_strength = details.get("strength")

		#Strength could be 0 in the future which would come out as false here
		if temp_strength or temp_strength == 0:
			strength_adjusted = True
			word_count += 1
			if temp_strength < 0:
				belief_strength += temp_strength * -1
				polarity *= -1
			else:
				belief_strength += temp_strength

	return (belief_strength, polarity, word_count, strength_adjusted)

def adjust_sentiment(sentiment, polarity, word_count, word):
	sentiment_adjusted = False
	lowered_word = word.lower()
	details = {}

	if lowered_word in strength_and_sentiment_dict:
		details = strength_and_sentiment_dict.get(lowered_word)
	elif morphRootNoun(lowered_word) in strength_and_sentiment_dict:
		details = strength_and_sentiment_dict.get(morphRootNoun(lowered_word))
	elif morphRootVerb(lowered_word) in strength_and_sentiment_dict:
		details = strength_and_sentiment_dict.get(morphRootVerb(lowered_word))

	if details:	
		temp_sentiment = details.get("sentiment")

		#Sentiment could be 0 in the future which would come out as false here
		if temp_sentiment or temp_sentiment == 0:
			sentiment_adjusted  = True
			word_count += 1
			if temp_sentiment < 0:
				sentiment += temp_sentiment * -1
				polarity *= -1
			else:
				sentiment += temp_sentiment

	return (sentiment, polarity, word_count, sentiment_adjusted)

def get_sentiment_score(text):
    probs = sentiment_predictor.predict(sentence=text)['probs']
    if probs[0] > probs[1]:
        sentiment = probs[0]
    elif probs[1] > probs[0]:
        sentiment = -1 * probs[1]
    else:
        sentiment = 0
        
    return (sentiment, probs)

def get_valuation_score(sentiment_score):
	if sentiment_score > .50:
		if sentiment_score <= .70:
			return 1
		elif sentiment_score <= .85:
			return 2
		else:
			return 3
	elif sentiment_score < -.50:
		if sentiment_score >= -.70:
			return -1
		elif sentiment_score >= -.85:
			return -2
		else:
			return -3
	else:
		return 0

def build_stance_dict(belief_type, belief_trigger_with_index, belief_content_with_indices, strength_words_and_indices, belief_strength, belief_valuation, target_sentiment, sentiment_probs, allen_sentiment, sentence, author, timestamp, doc_id, text_number):
	stance_dict = {}
	trigger_and_content_with_indices = belief_content_with_indices

	belief_trigger = morphRootVerb(belief_trigger_with_index[0])

	belief_content_with_indices.sort(key=lambda x:x[1])
	belief_content = ' '.join([x[0] for x in belief_content_with_indices])

	if belief_trigger_with_index not in belief_content_with_indices:
		trigger_and_content_with_indices.append((morphRootVerb(belief_trigger_with_index[0]), belief_trigger_with_index[1]))

	trigger_and_content_with_indices.sort(key=lambda x:x[1])
	sentiment_string = ' '.join([x[0] for x in trigger_and_content_with_indices])

	strength_trigger_and_content_with_indices = trigger_and_content_with_indices
	for word_and_index in strength_words_and_indices:
		if word_and_index not in trigger_and_content_with_indices:
			strength_trigger_and_content_with_indices.append(word_and_index)

	strength_trigger_and_content_with_indices.sort(key=lambda x:x[1])
	belief_string = ' '.join([x[0] for x in strength_trigger_and_content_with_indices])

	attitude = belief_strength * belief_valuation
	belief_strength = f'{belief_strength:.2f}'

	#In cases where the default belief type is used, meaning belief type was gotten from sentiment words,
	#it is inaccurate to have the trigger in the representation
	if belief_type == default_belief_type:
		belief_trigger = ''
			
	stance_dict["stance_rep"] = f'<{belief_type}[{belief_trigger}[{belief_content}]],{belief_strength},{belief_valuation}>'
	stance_dict["belief"] = f'{belief_type}[{belief_trigger}[{belief_content}]]'
	stance_dict["belief_string"] = belief_string
	stance_dict["sentiment_string"] = sentiment_string
	stance_dict["evidence"] = sentence
	stance_dict["belief_strength"] = belief_strength
	stance_dict["belief_valuation"] = belief_valuation
	stance_dict["attitude"] = f'{attitude:.2f}' #NOTE may change to strength times event senti
	stance_dict["attribution"] = {
		"author" : author,
		"timestamp" : timestamp,
		"document_id" : doc_id,
	}
	stance_dict["belief_content"] = belief_content
	stance_dict["belief_trigger"] = belief_trigger
	stance_dict["belief_type"] = belief_type
	stance_dict["allen_sentiment"] = allen_sentiment
	stance_dict["positive_sentiment"] = f'{sentiment_probs[0]:.2f}'
	stance_dict["negative_sentiment"] = f'{sentiment_probs[1]:.2f}'
	stance_dict["target_sentiment"] = target_sentiment
	stance_dict["text_number"] = text_number

	return stance_dict

def process_stance(word, word_pos, sentence, srl, belief_type, strength, event_sentiment):
	is_sentiment_belief_type = False 
	word = word.lower()
	lem_word = morphRootVerb(word)
	'''
	(belief_types, event_sentiment, strength) = getBeliefType(word, word_pos)
	(lem_belief_types, lem_event_sentiment, lem_strength) = getBeliefType(lem_word, word_pos)

	belief_types = appendListNoDuplicates(lem_belief_types, belief_types)
	'''

	(arg0_with_indices, arg1_with_indices, arg2_with_indices, arg3_with_indices) = extractStanceFromSrl(sentence, srl, word)

	'''
	If no belief types are found from the trigger and the trigger is a light verb than each one 
	in the sentences arg1 (from SRL) is checked to see if a belief type can be found from it.
	To prevent overwriting legitimate belief types  with an empty value if the last word in arg1 did 
	not return a belief type, the belief type that is most frequently seen is taken. 
	If all belief types have the same frequency from arg1 then the first one is taken.
	'''
	if not belief_type:
		#NOTE When testing output for just mutual constraint of predicate and content comment out all 
		# of this belief type backoff
		if  ((word in pitt_light_verbs or lem_word in pitt_light_verbs) and light_verbs_version) or no_filter_version: #(catvar_dict.get(word) or catvar_dict.get(lem_word)):
			belief_type_freq = {}
			for arg1_word, word_index in arg1_with_indices:
				lowered_arg1_word = arg1_word.lower()
				lem_arg1_word = morphRootNoun(lowered_arg1_word)
				if not lem_arg1_word:
					lem_arg1_word = morphRootVerb(lowered_arg1_word)

				#(belief_types, event_sentiment) = getBeliefTypeFromContent(word)
				#(lem_belief_types, lem_event_sentiment) = getBeliefTypeFromContent(lem_word)

				#if catvar_dict.get(lowered_arg1_word) or catvar_dict.get(lem_arg1_word):
				belief_types = getBeliefTypeFromContent(lowered_arg1_word)
				lem_belief_types = getBeliefTypeFromContent(lem_arg1_word)

				content_belief_types = appendListNoDuplicates(lem_belief_types, belief_types)

				if content_belief_types:
					for belief_type, sentiment in content_belief_types:
						if belief_type:
							if belief_type in belief_type_freq:
								belief_type_freq[belief_type]["count"] += 1
							else:
								belief_type_freq[belief_type] = {
									"count" : 1,
									"sentiment" : sentiment
								}

			#print("freq types: ", belief_type_freq)
			highest_count = 0
			types_with_same_count = 1
			for content_belief_type, value_dict in belief_type_freq.items():
				if value_dict["count"] == highest_count:
					types_with_same_count += 1
				if value_dict["count"] > highest_count:
					highest_count = value_dict["count"]
					belief_type = content_belief_type
					event_sentiment = value_dict["sentiment"]

			if types_with_same_count == len(belief_type_freq):
				belief_type = list(belief_type_freq)[0]
				event_sentiment = belief_type_freq.get(belief_type).get("sentiment")

			#NOTE This should only be done if belief type back off is turned on with light verb consideration (for experimentation
			#TODO This might not be necessary and probably will be deleted becuase the content bucket types are now the same (i.e. PROTECT) as the trigger buckets
			#if belief_types:
			#	if belief_types[0] in pitt_stance_targets.keys(): 
			#		belief_types = [pitt_stance_targets.get(belief_types[0]).get("counterpart_label")]

		elif word in strength_and_sentiment_dict and exist_case_version:
			details = strength_and_sentiment_dict.get(word)
			if details.get("sentiment"):
				belief_type = default_belief_type
				event_sentiment = details.get("sentiment")
				is_sentiment_belief_type = True
		elif lem_word in strength_and_sentiment_dict and exist_case_version:
			details = strength_and_sentiment_dict.get(lem_word)
			if details.get("sentiment"):
				belief_type = default_belief_type
				event_sentiment = details.get("sentiment")
				is_sentiment_belief_type = True


	#print(belief_type)
	#If there is no belief type at this point there is no point in building the content
	# so the function is cut short in order to be more efficient
	if not belief_type:
		return

	#NOTE This is back off to get details for a target from for an argument that are most appropriate to the specific domain
	# in the current case (12/11/2020) that is PITT/Covid. Here we are checking, from SRL, arg1, then arg0, then arg3
	content = build_content(arg1_with_indices, belief_type, is_sentiment_belief_type)

	if arg0_with_indices:
		content.extend(build_content(arg0_with_indices, belief_type, is_sentiment_belief_type))
	if arg3_with_indices:
		content.extend(build_content(arg3_with_indices, belief_type, is_sentiment_belief_type))
	
	return_tuple = (belief_type, event_sentiment, content, is_sentiment_belief_type)
	#(ask_who, ask, ask_recipient, ask_when, ask_action, confidence, descriptions, belief_type, t_ask_confidence, word_number, arg2, event_sentiment, content, is_sentiment_belief_type)
		
	#if not ask_action:
	#	return_tuple = (ask_who, ask, ask_recipient, ask_when, ask_negation_dep_based, ask_action, confidence) = extractAskInfoFromDependencies(word, dependencies, t_ask_types)

	if belief_type and content:
		return return_tuple

def build_content(potential_content_with_indices, trigger_belief_type, is_sentiment_belief_type): 
	content_words_with_indices = []

	for word, word_index in potential_content_with_indices:
		content_belief_types = []

		#print("word in content Building: ", word)
		if word in content_buckets:
			content_belief_types = content_buckets.get(word).get("belief_types")
		elif morphRootNoun(word) in content_buckets:
			content_belief_types = content_buckets.get(morphRootNoun(word)).get("belief_types")
		elif morphRootVerb(word) in content_buckets:
			content_belief_types = content_buckets.get(morphRootVerb(word)).get("belief_types")

		#print("types in content building: ", content_belief_types)
		#print("this is is sentiment belief type: ", is_sentiment_belief_type)

		if content_belief_types:
			for content_belief_type in content_belief_types:
				#Constrain allowed content by only adding it if the word appears in a counterpart bucket.
				#If the trigger bucket label is PROTECT the word must be in the PROTECT content bucket.
				#Also if they belief type came from the sentiment the check words from any content bucket
				if content_belief_type.get("belief_type") == trigger_belief_type or is_sentiment_belief_type or not mutually_constrained_version: 
					content_words_with_indices.append((word, word_index))
					break

		'''
		for target_label, target_details in pitt_stance_targets.items():
			#Constrain allowed content by only adding it if the word appears in a counterpart bucket.
			#For example if the trigger bucket label is PROTECT the word must be in the PROTECT content bucket.
			#Also if they belief type came from the sentiment the check words from any content bucket
			if target_details["counterpart_label"] == belief_types[0] or is_sentiment_belief_type:
				trigger_details = pitt_stance_triggers.get(target_details["counterpart_label"])
				if target_details["words"]:
					#Need to check if the morph (lemma) of the word or the word itself is in the list
					if morphRootVerb(word) in target_details["words"] or morphRootNoun(word) in target_details["words"] or word in target_details["words"]:
						target_words_with_indices.append((word, word_index))
						break
		'''

	#NOTE To do confidence score for non mutual constraint if the mutual constraint produces no content then check all other buckets for matches. 


	return content_words_with_indices
	
