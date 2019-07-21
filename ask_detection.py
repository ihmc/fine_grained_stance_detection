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

from ask_mappings import sashank_categories_sensitive, alan_ask_types, sashanks_ask_types, tomeks_ask_types
from catvar_v_alternates import v_alternates


modality_lookup = {}
sentence_modalities = []
word_specific_rules = []


# Url for server hosting coreNLP
coreNLP_server = 'http://panacea:nlp_preprocessing@simon.arcc.albany.edu:44444'

tsurgeon_class = 'edu.stanford.nlp.trees.tregex.tsurgeon.Tsurgeon'

project_path = os.path.abspath(os.path.dirname(__file__))

# Path for files needed for catvar processing
catvar_file = '/catvar.txt'
lcs_file = '/LCS-Bare-Verb-Classes-Final.txt'

# Url for server hosting coreNLP
coreNLP_server = 'http://panacea:nlp_preprocessing@simon.arcc.albany.edu:44444'

# When reading in files locally these directorys must be inside the project directory(i.e. mood and modality)
# They can be named whatever you would like just make sure they exists
# All files meant to be read in should be in the text directory.
# The output directory will be filled with the modality results of processing the input files
input_directory = '/text/'
output_directory = '/output/'
rule_directory = '/generalized-templates_v1_will_change_after_working_on_idiosyncraticRules/'

# Rule directories
# TODO This list will be thinned out as rule sets are chosen for superiority
generalized_rule_directory = '/generalized-templates_v1_will_change_after_working_on_idiosyncraticRules/'
generalized_rule_v2_directory = '/generalized-templates_v2/'
generalized_v3_directory = '/generalized-templates_v3/'
lexical_item_rule_directory = '/lexical-item-rules/'
preprocess_rule_directory = '/idiosyncratic/'

# Paths for the tsurgeon java tool
tregex_directory = '/stanford-tregex-2018-10-16/'
tsurgeon_script = tregex_directory + 'tsurgeon.sh'

lcs_dict = {}
with open('.' + lcs_file) as lcs:
	backup_regex = '\(\s\s:NUMBER \"(.+)\"\s\s:NAME \"(.+)\"\s\s:WORD \((.*)\) \)'
	unit_regex = ':NUMBER \"(.+)\"\s*:NAME \"(.+)\"\s*:WORDS \((.*)\)' 
	lines = lcs.readlines() 
	file_as_string = ''
	for line in lines:
		file_as_string += line

	matches = re.findall(unit_regex, file_as_string, re.MULTILINE)
	#print(matches)
	for match in matches:
		lcs_key = match[0] + ' ' + match[1]
		word_list = match[2].split()
		lcs_dict[lcs_key] = word_list	

#print(lcs_dict)
print('LCS dictionary created')

test_list = []
# catvar_alternates_dict is a dictionary where each key has an array of verbal words from the catvar file
# that exist on a line with more than 1 verbal form. This is to cover cases when a small spelling change is present
# in catvar or when other verbal words exist but were are not a part of the catvar_dict here
catvar_alternates_dict = v_alternates
catvar_dict = {}
with open('.' + catvar_file) as catvar:
	for entry in catvar:
		'''
		v_entries = []
		#if '_V' not in entry_pieces[0]:
		entry_pieces = entry.split('#')
		count = 0
		for index, entry_piece in enumerate(entry_pieces):
			beg_test_word = entry_pieces[0].split('_')[0]
			test_word = entry_piece.split('_')[0]
			if '_V' in entry_piece: #and beg_test_word != test_word:
				v_word = entry_piece.split('_')[0]
				count += 1
				v_entries.append(v_word)
		if count > 1:#and len(v_entries) < 2:
			for word in v_entries:
				alternates_dict[word] = v_entries
		'''
		if '_V' in entry:
			entry_pieces = entry.split('#')

			for entry_piece in entry_pieces:
				if '_V' in entry_piece:
					value_piece_no_POS = entry_piece.split('_')[0]

			for entry_piece in entry_pieces:
				key_piece_no_POS = entry_piece.split('_')[0]	
				catvar_dict[key_piece_no_POS] = {'catvar_value': value_piece_no_POS}

			'''
			if len(entry_pieces) > 1:
				# Must create a key for each piece and it's value the first piece
				for entry_piece in entry_pieces:
					key_piece_no_POS = entry_piece.split('_')[0]
					value_piece_no_POS = entry_pieces[0].split('_')[0]
				catvar_dict[key_piece_no_POS] = {'catvar_value': value_piece_no_POS}
			else:
				# If the entry only has one piece then the key and value are the same 
				piece_no_POS = entry_pieces[0].split('_')[0]
				catvar_dict[piece_no_POS] = {'catvar_value': piece_no_POS}
			'''
		
#print(catvar_dict, 'catvar dicctionary')
print('catvar dictionary create')

preprocess_rules_in_order = []
with open('.' + preprocess_rule_directory + 'ORDER.txt', 'r') as rule_order:
	for rule in rule_order:
		rule = rule.strip('\n')
		preprocess_rules_in_order.append(rule)

print('Preprocess rules loaded')

# Reading a provided CSV as a lexicon and parsing out each word and it's modality
# as well as a list of rules that should apply for each lexical item
# A sequence of 2 or 3 words can exist as well so those are checked for first
lexical_items = []
with open('./ModalityLexiconSubcatTags.csv') as modalityCSV:
	csv_reader = csv.reader(modalityCSV)
	for word, pos, modality, rules in csv_reader:
		lexical_items.append(word)
		for rule in rules.split("|"):
			if rule:
				word_specific_rules.append((word, rule.strip(' '), modality))
		for pos in pos.split("|"):
			modality_lookup[(word, pos)] = modality

print("Lexicon loaded")

if not os.path.exists('.' + lexical_item_rule_directory):
	print('Lexical item rule directory does not exist, creating now');
	os.mkdir('.' + lexical_item_rule_directory)


# Rule here refers to a tuple containing the rule as well as its corresponding lexical item, and modality (lexical item, rule name, modality)
lexical_specific_rules = []
for rule in word_specific_rules:
	if rule[1] + '.txt' not in preprocess_rules_in_order:
		with open('.' + generalized_v3_directory + rule[1] + '.txt') as rule_file:
			filled_in_rule = rule_file.read().replace('**', rule[0])
			rule_name = rule[0] + '-' + rule[2] + '-' + rule[1]

			filled_in_rule = filled_in_rule.replace('TargLabel', 'Targ' + rule[2])
			filled_in_rule = filled_in_rule.replace('TrigLabel', 'Trig' + rule[2])

		# A new file name is built from the combination of the lexical item and the rule
		lexical_specific_rule_file = '.' + lexical_item_rule_directory + rule_name + '.txt'
		#print(lexical_specific_rule_file)
		rule_dict = {}
		rule_dict['rule'] = filled_in_rule
		rule_dict['rule_name'] = rule_name
		rule_dict['modality'] = rule[2]
		rule_dict['lexical_item'] = rule[0]
		lexical_specific_rules.append(rule_dict)
		with open(lexical_specific_rule_file, 'w+') as lexical_rule:
			lexical_rule.write(filled_in_rule)
			
#print(lexical_specific_rules)

def getModality(text):
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
	parse_matches = []
	pattern = '\[\[\[ASKMARKER1234-(\d+)-ASKMARKER1234(.*?)/ASKMARKER1234-\d+-ASKMARKER1234\]\]\]'
	text_to_process = unicodedata.normalize('NFKC',text)

	lines = text_to_process.split('\n')

	
	for line in lines:
		line_text = line.strip()
		if not line:
			continue
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
		#sentence_srls.extend(parseSrl(line_text.lower(), link_offsets, link_ids, link_strings))
		line_matches = parseSrl(line_text.lower(), link_offsets, link_ids, link_strings, links)
		if line_matches: 
			parse_matches.extend(line_matches)

	sorted_matches = sorted(parse_matches, key = lambda k: k['is_ask_confidence'], reverse=True)

	'''
	# Split input text into sentences
	sentences = nltk.sent_tokenize(text)
	sentence_srls = []

	for sentence in sentences:
		constituency_parse = parseSrl(sentence)
		sentence_srls.append({"sentence": sentence, "matches": constituency_parse})


	return sentence_srls
	'''
	return {'email': text, 'matches': sorted_matches}

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
	parse_tree = parse_tree.replace('\\n', '');
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
			#print(rule)
			# Have to return to the beginning of the file so that the new tree overwrites the previous one.
			tree_file.seek(0)

			#TODO remove this chunk
			#with open('.' + preprocess_rule_directory + rule, 'r') as rule_file:
				#print(rule_file.read())

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

# This functions checks to see if the items in a list already exist 
# in the original list and if not then add them.
def appendListNoDuplicates(list_to_append, original_list):
	for item in list_to_append:
		if item not in original_list:
			original_list.append(item)

	return original_list

def getNLPParse(sentence):
	sentence = sentence.lower()
	annotators = '/?annotators=ssplit,tokenize,pos,parse,depparse&tokenize.english=true'
	tregex = '/tregex'
	url = coreNLP_server + annotators

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
		if neg_gov_gloss == base_word:
			ask_negation_dep_based = dep_neg_exists
	else:
		if nsubj_gov_gloss == base_word and dobj_gov_gloss == base_word and iobj_gov_gloss == base_word:
			ask_who = nsubj
			ask = dobj
			ask_recipient = iobj
			ask_action = root
			confidence = 'low'
		elif 'GET' in t_ask_types or 'PERFORM' in t_ask_types:
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
		elif 'GET' in t_ask_types:
			ask = ' '.join(arg1)
			ask_recipient = ' '.join(arg0)
			ask_when = ' '.join(arg_tmp)
			confidence = 'high'
		else:
			ask_who = ' '.join(arg0)
			ask = ' '.join(arg1)
			ask_when = ' '.join(arg_tmp)
			confidence = 'low'


	if 'GIVE' in t_ask_types:
		if 'you' in arg2:
			t_ask_types = ["GET"]
		elif 'you' in arg0:
			t_ask_types = ["GIVE"]
		elif 'GET' in t_ask_types:
			if 'you' in arg0:
				t_ask_types = ["GET"]
			if 'i' in arg0 or 'we' in arg0:
				t_ask_types = ["GIVE"]
				t_ask_confidence = 'low'
	elif 'GET' in t_ask_types:
		if 'you' in arg0:
			t_ask_types = ["GET"]
		elif 'i' in arg0 or 'we' in arg0:
			t_ask_types = ["GIVE"]
			t_ask_confidence = 'low'
	elif 'OTHER' in t_ask_types:
		if arg0 and arg1 and arg2:
			if 'you' in arg2:
				t_ask_types = ["GIVE"]
			elif 'you' in arg0:
				t_ask_types = ["GET"]

	return(ask_who, ask, ask_recipient, ask_when, selected_verb, confidence, descriptions, t_ask_types, t_ask_confidence, word_number)

def processWord(word, sentence, ask_procedure, ask_negation, dependencies, is_past_tense, trig_and_targs, link_in_sentence, link_exists, link_strings, link_ids, link_id, links):
	ask_negation_dep_based = False
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

	if trig_and_targs:
		for trig_and_targ in trig_and_targs:
			if (trig_and_targ[2] == 'Negation' or trig_and_targ[2] == 'NotSucceed') and word == trig_and_targ[3]:
				ask_negation = True


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
		#Remove PERFORM if GIVE or GET has also been chosen.
		#NOTE It should not be the case that GIVE and GET are both present at this time.
		elif len(t_ask_types) > 1: 
			t_ask_types.remove('PERFORM')

	
	if 'GIVE' not in t_ask_types and 'GET' not in t_ask_types and 'PERFORM' not in t_ask_types:
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
		is_ask_confidence = evaluateAskConfidence(is_past_tense, link_exists, ask, s_ask_types)
		return buildParseDict(sentence, '', '', '', ask_who, ask, ask_recipient, ask_when, ask_action, ask_procedure, ask_negation, ask_negation_dep_based, is_ask_confidence, confidence, descriptions, s_ask_types, t_ask_types, a_ask_types, t_ask_confidence, additional_s_ask_types, word, '', '', link_id, links)

def evaluateAskConfidence(is_past_tense, link_exists, ask, s_ask_types):
	confidence_score = 0
	tense_score = 0
	hyper_link_score = 0

	if is_past_tense:
		return 0.1
	elif link_exists:
		return 0.9
	elif ask and s_ask_types:
		return 0.75
	elif ask:
		return 0.6
	elif s_ask_types:
		return 0.5
	else:
		return 0.1

	'''
	tense_score = 0.1 if is_past_tense else 1

	if link_exists:
		return 0.9
	else:
		hyper_link_score = 0.1
	'''
	#confidence_score = float('%.3f'%((tense_score + ask_who_score + ask_recipient_score + hyper_link_score) / 4))

	#return confidence_score

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

def parseSrl(line, link_offsets, link_ids, link_strings, links):
	line_parse_matches = []
	sentences = nltk.sent_tokenize(line)
	is_past_tense = False
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

	previous_token_char_end_offset = 0
	current_token_char_begin_offset = 0
	for nlp_sentence in core_nlp_sentences:
		ask_procedure = ''
		link_in_sentence = False
		link_exists = False
		rebuilt_sentence = []
		parse_verbs = []
		#TODO Investigate if this is needed
		#words = getLemmaWords(sentence)
		parse_tree = nlp_sentence['parse']
		dependencies = nlp_sentence['basicDependencies']
		tokens = nlp_sentence['tokens']	

		for token in tokens:
			rebuilt_sentence.append(token['before'])
			rebuilt_sentence.append(token['originalText'])

		rebuilt_sentence = ''.join(rebuilt_sentence)

		sentence_begin_char_offset = tokens[0]['characterOffsetBegin']
		sentence_end_char_offset = tokens[len(tokens) - 1]['characterOffsetEnd']

		

		preprocessed_tree = preprocessSentence(parse_tree)
		#print(preprocessed_tree)
	
		# Extract all verbs from the constituency parse to be used for fallback if all verbs are not found in the dependencies
		parse_verb_matches = extractVerbs(parse_tree)
		for parse_verb_match in parse_verb_matches:
			parse_verbs.append(parse_verb_match[1])

		# We extract the triggers and targets here to check later on if Negation occurred
		trig_and_targs = extractTrigsAndTargs(preprocessed_tree)
		#print(trig_and_targs, '\n\n')

		#TODO need this to be more accurate for each portion not just the whole sentence
		#If either of these verbs parts of speech exists in the parse tree then the sentence has a past tense part
		if 'VBD' in parse_tree or 'VBN' in parse_tree:
			is_past_tense = True

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

			if dependency['dep'] == 'root' and not small_root:
				small_root = dependency['dependentGloss']
				base_words.append(dependency['dependentGloss'])

			#TODO there is a way to simplify this whole operation here. Need to figure it out later.
			'''
			if dependency['governorGloss'] == dependencies[0]['dependentGloss'] or dependency['governerGloss'] == small_root:
				if dependency['dep'] == 'conj':
			'''
				
			if dependency['dep'] == 'conj':
				if dependency['governorGloss'] == dependencies[0]['dependentGloss'] or dependency['governorGloss'] == small_root:
					conj_base_word = dependency['dependentGloss']
					base_words.append(dependency['dependentGloss'])
			if dependency['dep'] == 'dep':
				dep_governor_gloss = dependency['governorGloss']
				dep_dependent_gloss = dependency['dependentGloss']
				dep_dep = dependency['dep']
				if dependency['governorGloss'] == dependencies[0]['dependentGloss'] or dependency['governorGloss'] == small_root:
					dep_base_word = dependency['dependentGloss']
					base_words.append(dependency['dependentGloss'])
			if dependency['dep'] == 'ccomp':
				if dependency['governorGloss'] == dependencies[0]['dependentGloss'] or dependency['governorGloss'] == small_root:
					ccomp_base_word = dependency['dependentGloss']
					base_words.append(dependency['dependentGloss'])
				elif dep_dep == 'dep' and dependency['governorGloss'] == dep_dependent_gloss:
					if dep_governor_gloss == dependencies[0]['dependentGloss']:
							ccomp_base_word = dependency['dependentGloss']
							base_words.append(dependency['dependentGloss'])
			if dependency['dep'] == 'xcomp':
				if dependency['governorGloss'] == dependencies[0]['dependentGloss'] or dependency['governorGloss'] == small_root:
					xcomp_base_word = dependency['dependentGloss']
					base_words.append(dependency['dependentGloss'])

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

					
		if not nsubj_exists:
			ask_procedure = 'directive'

		for base_word in base_words:
			link_exists = False
			for index, link_offset in enumerate(link_offsets):
				if link_offset[0] >= sentence_begin_char_offset and link_offset[1] <= sentence_end_char_offset and link_strings[index].lower() in rebuilt_sentence:
					link_in_sentence = True
					if base_word == advmod_governor_gloss and advmod_dependent_gloss in link_strings[index].lower():
						link_id = link_ids[index]
						link_exists = True
						break

			ask_details = processWord(base_word, rebuilt_sentence, ask_procedure, ask_negation, dependencies, is_past_tense, trig_and_targs, link_in_sentence, link_exists, link_strings, link_ids, link_id, links)
			if ask_details:
				line_parse_matches.append(ask_details)

		if parse_verbs:
			for verb in parse_verbs:
				if verb not in base_words:
					for index, link_offset in enumerate(link_offsets):
						if link_offset[0] >= sentence_begin_char_offset and link_offset[1] <= sentence_end_char_offset and link_strings[index].lower() in rebuilt_sentence:
							link_in_sentence = True
							if verb == advmod_governor_gloss and advmod_dependent_gloss in link_strings[index].lower():
								link_id = link_ids[index]
								link_exists = True
								break

					ask_details = processWord(verb, rebuilt_sentence, ask_procedure, ask_negation, dependencies, is_past_tense, trig_and_targs, link_in_sentence, link_exists, link_strings, link_ids, link_id, links)
					if ask_details:
						line_parse_matches.append(ask_details)

		#sentence_srls.append({"sentence": rebuilt_sentence, "matches": parse})

	if line_parse_matches:
		return line_parse_matches
