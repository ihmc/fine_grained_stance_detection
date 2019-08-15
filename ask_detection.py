from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")

import unicodedata
import json
import nltk
import csv
import os
import re
import subprocess

# This library allows python to make requests out.
# NOTE: There is a difference between this and the built in request variable  
# that FLASK provides do  not confuse the two.
import requests

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from load_resources import preprocess_rules_in_order, catvar_dict, lcs_dict
from ask_mappings import sashank_categories_sensitive, alan_ask_types, sashanks_ask_types, tomeks_ask_types
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

def getModality(text):
	sentence_modalities = []
	text = unicodedata.normalize('NFKC',text)

	# Split input text into sentences
	sentences = nltk.sent_tokenize(text)
	sentence_modalities = []

	for sentence in sentences:
		constituency_parse = parseModality(sentence)
		
		sentence_modalities.append({"sentence": sentence, "matches": constituency_parse})


	return sentence_modalities

def getSrl(text, links):
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

		line_matches = parseSrl(line_text, link_offsets, link_ids, link_strings, links, last_ask, last_ask_index)
		if line_matches:
			# NOTE The last_ask and last_ask_index are overidding the values initialized at the beginning 
			# of this function. This on purpose so that each time parseSrl is called it will get the 
			# most up to date info
			(framings, asks, asks_to_update, last_ask, last_ask_index) = line_matches

			if framings: 
				framing_matches.extend(framings)
			if asks:	
				ask_matches.extend(asks)

			# If parseSrl determines that the last ask needs to be update then it will update the appropriate ask
			# in ask_matches with the new information that was altered in last_ask inside parseSrl
			if asks_to_update:
				for ask in asks_to_update:
					# Ask will be a tuple with the first part being the updated ask and the second part being the index in ask_matches that needs updating
					ask_matches[ask[1]] = ask[0]
				last_ask = asks_to_update[-1][0]
				last_ask_index = asks_to_update[-1][1]
			

	filter(lambda ask: True if ask['is_ask_confidence'] != 0 else False, ask_matches)
	sorted_framing = sorted(framing_matches, key = lambda k: k['is_ask_confidence'] , reverse=True)
	sorted_asks = sorted(filter(lambda ask: True if ask['is_ask_confidence'] != 0 else False, ask_matches), key = lambda k: k['is_ask_confidence'], reverse=True)

	return {'email': text, 'framing': sorted_framing, 'asks': sorted_asks}

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
	return wlem.lemmatize(word.lower(),wn.VERB)

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
	trigger_tuple = (morphRoot(word_and_pos[0].lower()), word_and_pos[1])
	
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
		'''
		trig_string = ' '.join(entire_match.split())
		trig_string = re.sub('Trig\w+', '', trig_string)
		targ_string = ' '.join(targ_match[index][0].split())
		targ_string = re.sub('Targ\w+', '', targ_string)
		'''
		modality = modality.replace('Trig', '')
		ask = targ_match[index][2]
		trig_word = trig
		trigs_and_targs.append((trig, targ_match[index][2], modality, ask, trig_word))

	return trigs_and_targs	

def buildParseDict(sentence, trigger, target, modality, ask_who, ask, ask_recipient, ask_when, ask_action, ask_procedure, ask_negation, ask_negation_dep_based, is_ask_confidence, confidence, descriptions, s_ask_types, t_ask_types, a_ask_types, t_ask_confidence, additional_s_ask_type,  base_word, rule, rule_name, link_id, links):
	parse_dict = {}
	if modality:
		parse_dict['trigger'] = trigger
		parse_dict['target'] = target
		parse_dict['trigger_modality'] = modality
	if ask_negation:
		parse_dict['ask_rep'] = f'<{t_ask_types[0]}[NOT {ask_action}[{ask}({link_id}){s_ask_types}]]>'
	else:
		parse_dict['ask_rep'] = f'<{t_ask_types[0]}[{ask_action}[{ask}({link_id}){s_ask_types}]]>'
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
	if link_id:
		parse_dict['url'] = {link_id: links.get(link_id)}
	else:
		parse_dict['url'] = {}
	#parse_dict['ask_negation_dep_based'] = ask_negation_dep_based
	#parse_dict['ask_info_confidence'] = confidence
	parse_dict['t_ask_type'] = t_ask_types
	#parse_dict['t_ask_confidence'] = t_ask_confidence
	parse_dict['s_ask_type'] = s_ask_types
	#parse_dict['additional_s_ask_type'] = additional_s_ask_type
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
def getAskTypes(ask):
	verb_types = []
	s_ask_types = []
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
				
		
		for vb_type in verb_types:
			for sashank_ask_type, types in sashanks_ask_types.items():
				if vb_type in types and sashank_ask_type not in s_ask_types:
					s_ask_types.append(sashank_ask_type)

			for tomek_ask_type, types in tomeks_ask_types.items():
				if vb_type in types and tomek_ask_type not in t_ask_types:
					t_ask_types.append(tomek_ask_type)

	return (s_ask_types, t_ask_types)

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

	for vb_type in verb_types:
		for tomek_ask_type, types in tomeks_ask_types.items():
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
	coreNLP_ased = 'http://10.108.18.14:9000'
	url = coreNLP_ased + annotators

	return requests.post(url, data=sentence.encode(encoding='UTF-8',errors='ignore'))

def getLemmaWords(sentence):
	sentence = sentence.lower()
	words = nltk.word_tokenize(sentence)
	return [(morphRoot(word.lower())) for word in words]

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

def extractAskFromSrl(sentence, base_word, t_ask_types):
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
	arg_tmp = []
	word_number = []
	srl = predictor.predict(sentence=sentence)
	verbs = srl['verbs']
	words = [word.lower() for word in srl['words']]
	descriptions = []

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
			elif tag_label == 'ARG1':
				arg1.append(words[index])
				#The placement of the word within the sentence
				word_number.append(index)
			elif tag_label == 'ARG2':
				arg2.append(words[index])
			elif 'ARGM-TMP' in tag:
				arg_tmp.append(words[index])

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

	return(ask_who, ask, ask_recipient, ask_when, selected_verb, confidence, descriptions, t_ask_types, t_ask_confidence, word_number)

def processWord(word, word_pos, sentence, ask_procedure, ask_negation, dependencies, link_in_sentence, link_exists, link_strings, link_ids, link_id, links):
	ask_negation_dep_based = False
	is_past_tense = False
	s_ask_types = [] 
	a_ask_types = []
	word = word.lower()
	lem_word = morphRoot(word)
	(additional_s_ask_types, t_ask_types) = getAskTypes(word)
	(additional_lem_s_ask_types, lem_t_ask_types) = getAskTypes(lem_word)

	additional_s_ask_types = appendListNoDuplicates(additional_lem_s_ask_types, additional_s_ask_types)
	t_ask_types = appendListNoDuplicates(lem_t_ask_types, t_ask_types)

	(ask_who, ask, ask_recipient, ask_when, ask_action, confidence, descriptions, t_ask_types, t_ask_confidence, word_number) = extractAskFromSrl(sentence, word, t_ask_types)

	if not ask_action:
		(ask_who, ask, ask_recipient, ask_when, ask_negation_dep_based, ask_action, confidence) = extractAskInfoFromDependencies(word, dependencies, t_ask_types)

	'''
	if trig_and_targs:
		for trig_and_targ in trig_and_targs:
			if (trig_and_targ[2] == 'Negation' or trig_and_targ[2] == 'NotSucceed') and word == trig_and_targ[3]:
				ask_negation = True
	'''

	if word_pos in ['VBD', 'VBN', 'VBG']:
		is_past_tense = True
		# 8/13/19 Bonnie said for now we can ignore past tense and leave it out of asks, may change later
		#return


	for ask_type, keywords in sashank_categories_sensitive.items():
		for keyword in keywords:
			if (keyword in ask or keyword in ask_action) and ask_type not in s_ask_types:
				s_ask_types.append(ask_type)

	for s_ask_type in s_ask_types:
		for alan_ask_type, types in alan_ask_types.items():
			if s_ask_type in types and alan_ask_type not in a_ask_types:
				a_ask_types.append(alan_ask_type)

	if 'PERFORM' in t_ask_types:
		if link_exists:
			t_ask_types = ['PERFORM']
		#Remove PERFORM if GIVE or GAIN has also been chosen.
		#NOTE It should not be the case that GIVE and GAIN are both present at this time.
		elif len(t_ask_types) > 1: 
			t_ask_types.remove('PERFORM')

	
	if 'GIVE' not in t_ask_types and 'LOSE' not in t_ask_types and 'GAIN' not in t_ask_types and 'PERFORM' not in t_ask_types:
		if (link_exists or link_in_sentence) and ask:
			t_ask_types = ['PERFORM']

	if t_ask_types == ['PERFORM'] and link_in_sentence and not link_exists and ask and word_number:
		for index, link_string in enumerate(link_strings):
			if ask == link_string.lower():
				link_id = link_ids[index]
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
				if dependent == dependency['governor'] or governor == dependency['governor']:
					for index, link_string in enumerate(link_strings):
						#TODO Look into this more as this will always take the last dependency that matched.
						if dependency['dependentGloss'] in link_string.lower():
							link_id = link_ids[index]
							link_exists = True
			

	for t_ask_type in t_ask_types:
		for alan_ask_type, types in alan_ask_types.items():
			if t_ask_type in types and alan_ask_type not in a_ask_types:
				a_ask_types.append(alan_ask_type)

	#TODO Need to check on this and make sure this is actually what we wish to do.
	if ask_negation or ask_negation_dep_based:
		ask_negation = True

	if t_ask_types and ask:
		if 'GIVE' in t_ask_types or 'PERFORM' in t_ask_types:
			is_ask_confidence = evaluateAskConfidence(is_past_tense, link_exists, ask, s_ask_types, t_ask_types)
		elif 'GAIN' in t_ask_types or 'LOSE' in t_ask_types:
			is_ask_confidence = 0.9

		return buildParseDict(sentence, '', '', '', ask_who, ask, ask_recipient, ask_when, ask_action, ask_procedure, ask_negation, ask_negation_dep_based, is_ask_confidence, confidence, descriptions, s_ask_types, t_ask_types, a_ask_types, t_ask_confidence, additional_s_ask_types, word, '', '', link_id, links)

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
		for token in tokens:
			if token['index'] == base_word_dependent_num:
				base_words_pos.append(token['pos'])

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
		verbs_and_pos.append((verb, base_words_pos[index]))

	for index, verb in enumerate(parse_verbs):
		if verb not in base_words:
			verbs_and_pos.append((verb, parse_verbs_pos[index]))	

	return verbs_and_pos

def parseModality(sentence):
	parse = []
	trigger_string = ''
	target_string = ''
	trigger_modality = ''
	response = getNLPParse(sentence)
	parse_tree = response.json()['sentences'][0]['parse']
	preprocessed_tree = preprocessSentence(parse_tree)

	# Get all words for the sentence and morph them to their root word.
	# Then check each word in the sentence to see if it is in the lexicon and
	# build a list of all the generalized rules that should be tried on the sentence tree
	subsets_per_word = []
	s_ask_types = []
	words = getLemmaWords(sentence)
	for word in words:
		for ask_type, keywords in sashank_categories_sensitive.items():
			if word in keywords and ask_type not in s_ask_types:
				s_ask_types.append(ask_type)	
		if word in lexical_items:
			subset = list(filter(lambda rule: rule['lexical_item'] == word, lexical_specific_rules))
			if subset:
				subsets_per_word.append(subset)

	for s_ask_type in s_ask_types:
		for alan_ask_type, types in alan_ask_types.items():
			if s_ask_type in types and alan_ask_type not in a_ask_types:
				a_ask_types.append(alan_ask_type)
	
		

	# If there are not words from the sentence found in the lexicon then we need to check the 
	# preprocessed tree from and triggers
	if len(subsets_per_word) == 0:
		if "Trig" in preprocessed_tree:
			trigs_and_targs = extractTrigsAndTargs(preprocessed_tree)
			if trigs_and_targs == None:
				return None
			for trig_and_targ in trigs_and_targs:
				# TODO store the portions of the tuple in meaningful names
				(trigger, target, modality, ask, trig_word) = trig_and_targ
				(additional_s_ask_types, t_ask_types) = getAskTypes(trig_word)
				parse.append(buildParseDict(trigger, target, modality, '', ask, '', '', '', '',  '', '', '', s_ask_types, t_ask_types, '', '', additional_s_ask_types, target, 'preprocessed rules', 'preprocess rules'))

			return parse
	# Here we loop through each set of generalized rules that were gathered above and 
	# try all of them for each word of the sentence that was found in the lexicon until 
	# one of the generalized rules produces a match via tsurgeon	
	for rule_subset in subsets_per_word:
		for rule in rule_subset:
			# Tsurgeon is a part of the stanford corenlp tool set. It must be rune on a file as it will not 
			# aceept just a string. NOTE There may be a way to do just a string that I have not discovered yet.
			# Tsurgeon will edit the parse tree and add trigger and target labels according to the rules that match.
			result = subprocess.run(['java', '-mx100m', '-cp', project_path + tregex_directory + 'stanford-tregex.jar:$CLASSPATH', tsurgeon_class, '-treeFile', 'tree.txt', '.' + lexical_item_rule_directory + rule['rule_name'] + '.txt'], stdout = subprocess.PIPE, text=True)

			# TODO make this more accurate. Currently it is unlikely but still possible that a preprocess rule
			# could have added a Trig with the same modality as a generalized rule. And if the generalized rule 
			# does not match then the Trig from the preprocessed will fire the extracting of triggers when it 
			# should attempt more generalized rules
			if 'Trig' + rule['modality'] in result.stdout:
				trigs_and_targs = extractTrigsAndTargs(result.stdout)
				if trigs_and_targs == None:
					return None
				for trig_and_targ in trigs_and_targs: 
					(trigger, target, modality, ask, trig_word) = trig_and_targ
					(additional_s_ask_types, t_ask_types) = getAskTypes(trig_word)
					parse.append(buildParseDict(trigger, target, modality, '', ask, '', '', '', '', '', '', '', s_ask_types, t_ask_types, '', '', additional_s_ask_types, target, rule['rule'], rule['rule_name']))

				return parse



def parseSrl(line, link_offsets, link_ids, link_strings, links, last_ask, last_ask_index):
	line_framing_matches = []
	line_ask_matches = []
	asks_to_update = []
	ask_negation = False
	base_word = ''
	conj_base_word = ''
	dep_base_word = ''
	ccomp_base_word = ''
	xcomp_base_word = ''
	link_id = ''
	
	response = getNLPParse(line)
	core_nlp_sentences = response.json()['sentences']
	#print(core_nlp_sentences, "\n\n")

	for nlp_sentence in core_nlp_sentences:
		ask_procedure = ''
		update_last_ask = False
		link_in_sentence = False
		link_exists = False
		rebuilt_sentence = []
		parse_verbs_pos = []
		parse_verbs = []
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
			

		small_root = ''
		root_dependent_gloss = ''
		aux_dependent = ''
		aux_governor  = ''
		advmod_governor_gloss = ''
		advmod_dependent_gloss = ''
		dep_governor_gloss = ''
		dep_dep = ''
		punct_dependent = 0
		punctuation_to_match = [':', ';', '-']
		nsubj_exists = False
		base_words = []
		base_words_pos = []
		base_words_dependents = []

		for dependency in dependencies:
			'''
			if dependency['dep'] == 'punct' and dependency['dependentGloss'] in punctuation_to_match:
				dependent = dependency['dependent'] + 1
				for dependency2 in dependencies:
					if dependency2['dependent'] == dependent:
						root_dependent_gloss = dependency2['governerGloss']
			'''
			if dependency['dep'] == 'ROOT':
				base_word = dependency['dependentGloss']
				base_words.append(dependency['dependentGloss'])
				base_words_dependents.append(dependency['dependent'])

			if dependency['dep'] == 'root' and not small_root:
				small_root = dependency['dependentGloss']
				base_words.append(dependency['dependentGloss'])
				base_words_dependents.append(dependency['dependent'])

			#TODO there is a way to simplify this whole operation here. Need to figure it out later.
			'''
			if dependency['governorGloss'] == dependencies[0]['dependentGloss'] or dependency['governerGloss'] == small_root:
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
					base_words_dependents.remove(dependencies[0]['dependent'])
				if dependencies[0]['dependentGloss'] in parse_verbs and 'PERFORM' not in check_t_ask_types:
					parse_verbs_pos.pop(parse_verbs.index(dependencies[0]['dependentGloss']))
					parse_verbs.remove(dependencies[0]['dependentGloss'])
				if dependency['governorGloss'] == dependencies[0]['dependentGloss'] or dependency['governorGloss'] == small_root:
					xcomp_base_word = dependency['dependentGloss']
					base_words.append(dependency['dependentGloss'])
					base_words_dependents.append(dependency['dependent'])

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
			# 8/13/19 Bonnie said for now we can ignore VBG and leave it out of asks, may change later
			#if pos == 'VBG':
			#	continue
			link_id = ''
			link_exists = False
			ask_negation = False

			ask_negation = isVerbNegated(verb, dependencies)

			for index, link_offset in enumerate(link_offsets):

				if link_offset[0] >= sentence_begin_char_offset and link_offset[1] <= sentence_end_char_offset and link_strings[index] in rebuilt_sentence:
					link_in_sentence = True

					if verb == advmod_governor_gloss and advmod_dependent_gloss in link_strings[index]:
						link_id = link_ids[index]
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
								link_id = link_ids[index]
								link_exists = True
								break

			ask_details = processWord(verb, pos, rebuilt_sentence, ask_procedure, ask_negation, dependencies, link_in_sentence, link_exists, link_strings, link_ids, link_id, links)
			if ask_details:
				if 'GIVE' in ask_details['t_ask_type'] or 'PERFORM' in ask_details['t_ask_type']:
					line_ask_matches.append(ask_details)
					last_ask = ask_details
					last_ask_index += 1
				elif 'GAIN' in ask_details['t_ask_type'] or 'LOSE' in ask_details['t_ask_type']:
					line_framing_matches.append(ask_details)

		if not line_ask_matches and link_in_sentence and last_ask and last_ask['is_ask_confidence'] != 0:
			last_ask['is_ask_confidence'] = evaluateAskConfidence(False, True, '', '', '')
			last_ask['link_id'] = link_ids[0]
			last_ask['link_url'] = { link_ids[0]: links.get(link_ids[0])}
			if last_ask['ask_negation']:
				last_ask['ask_rep'] = f'<{last_ask["t_ask_type"]}[NOT {last_ask["ask_action"]}[{last_ask["ask_target"]}({link_ids[0]}){last_ask["s_ask_type"]}]]>'
			else:
				last_ask['ask_rep'] = f'<{last_ask["t_ask_type"]}[{last_ask["ask_action"]}[{last_ask["ask_target"]}({link_ids[0]}){last_ask["s_ask_type"]}]]>'

			asks_to_update.append((last_ask, last_ask_index))

	if line_framing_matches or line_ask_matches or asks_to_update:
		return (line_framing_matches, line_ask_matches, asks_to_update, last_ask, last_ask_index)
