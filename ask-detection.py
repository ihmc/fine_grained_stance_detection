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

sashank_categories_sensitive = {
	"finance_money": [
		"money",
		"cash",
		"fund",
		"check",
		"bit coin",
		"bitcoin"
	],
	"finance_info": [
		"bank",
		"account",
		"transaction",
		"payment",
		"credit card",
		"insurance",
		"transaction",
		"tax",
		"claims",
		"irs",
		"link"
	],
	"personal": [
		"email",
		"e-mail",
		"ssn",
		"social security",
		"birthday",
		"country",
		"citizenship",
		"originally from",
		"family",
		"phone",
		"home address",
		"office address",
		"phone number",
		"offer",
		"contact info",
		"contact information",
		"CV",
		"link",
		"information",
		"available",
		"invite",
		"invitation",
		"schedule",
		"call",
		"message"
	],
	"business": [
		"proposal",
		"meeting",
		"meet",
		"proposition",
		"offer",
		"available",
		"invite",
		"invitation",
		"call",
		"fax"
	],
	"credentials": [
		"credentials",
		"password",
		"account",
		"slack id",
		"email id",
		"id",
		"user id",
		"login"
	],
	"privileged_information": [
		"report",
		"document",
		"attachment",
		"link",
		"intellectual property",
		"brochure",
		"code",
		"form",
		"file",
		"software",
		"project",
		"dropbox"
	],
	"ideology": [
		"human rights",
		"jesus",
		"campaign",
		"health",
		"refugee",
		"humanitarian",
		"climate change",
		"cancer"
	],
	"scam_lottery": [
		"lottery",
		"lucky draw",
		"winner",
		"prize"
	],
	"scam_gift": [
		"gift",
		"present",
		"gift card",
		"gifts card"
	],
	"scam_job": [
		"job",
		"hiring",
		"work experience",
		"interview",
		"candidate"
	]
}

sashanks_ask_types = {
  "finance_money": [
    "11.1 Send Verbs",
    "13.1.a.i Give - No Exchange - Sort of Atelic",
    "13.1.a.ii Give - No Exchange",
    "13.2 Contribute Verbs",
    "13.3 Verbs of Future Having",
    "13.4.1.a Verbs of Fulfilling - Possessional / -to",
    "13.4.1.b Verbs of Fulfilling - Change of State / -with",
    "13.5.1.b.ii Get - Exchange",
    "13.5.2.a Obtain - No Exchange (CAUSE)",
    "13.5.2.b Obtain - No Exchange (LET w/ benefactive)",
	"32.1 Want Verbs"
  ],
  "finance_info": [
    "11.1 Send Verbs",
    "13.3 Verbs of Future Having",
    "13.4.1.a Verbs of Fulfilling - Possessional / -to",
    "13.4.1.b Verbs of Fulfilling - Change of State / -with",
    "13.5.1.b.ii Get - Exchange",
    "13.5.2.b Obtain - No Exchange (LET w/ benefactive)",
    "37.2.a Tell Verbs",
    "37.2.b Tell Verbs / -of/about",
    "37.2.d Tell Verbs / -that/to",
	"32.1 Want Verbs"
  ],
  "personal": [
    "11.1 Send Verbs",
    "13.3 Verbs of Future Having",
    "13.4.1.a Verbs of Fulfilling - Possessional / -to",
    "13.4.1.b Verbs of Fulfilling - Change of State / -with",
    "13.5.2.b Obtain - No Exchange (LET w/ benefactive)",
    "37.2.a Tell Verbs",
    "37.2.b Tell Verbs / -of/about",
    "37.2.d Tell Verbs / -that/to",
	"32.1 Want Verbs"
  ],
  "business": [
    "13.3 Verbs of Future Having"
  ],
  "credentials": [
    "11.1 Send Verbs",
    "13.1.a.ii Give - No Exchange",
    "13.3 Verbs of Future Having",
    "13.4.1.a Verbs of Fulfilling - Possessional / -to",
    "13.4.1.b Verbs of Fulfilling - Change of State / -with",
    "13.5.2.b Obtain - No Exchange (LET w/ benefactive)",
    "37.2.a Tell Verbs",
    "37.2.b Tell Verbs / -of/about",
    "37.2.d Tell Verbs / -that/to",
	"32.1 Want Verbs"
  ],
  "privileged_information": [
    "11.1 Send Verbs",
    "13.1.a.ii Give - No Exchange",
    "13.3 Verbs of Future Having",
    "13.4.1.a Verbs of Fulfilling - Possessional / -to",
    "13.4.1.b Verbs of Fulfilling - Change of State / -with",
    "13.5.2.b Obtain - No Exchange (LET w/ benefactive)",
    "37.2.a Tell Verbs",
    "37.2.b Tell Verbs / -of/about",
    "37.2.d Tell Verbs / -that/to",
	"32.1 Want Verbs"
  ],
  "ideology": [],
  "scam_lottery": [
    "13.5.2.c Obtain - No Exchange (LET w/out benefactive)"
  ],
  "scam_gift": [
    "13.5.2.c Obtain - No Exchange (LET w/out benefactive)"
  ],
  "scam_job": []
}

tomeks_ask_types = {
  "GET": [
    "13.5.1.a Get - No Exchange",
    "13.5.1.b.ii Get - Exchange",
    "13.5.2.a Obtain - No Exchange (CAUSE)",
    "13.5.2.b Obtain - No Exchange (LET w/ benefactive)",
    "13.5.2.c Obtain - No Exchange (LET w/out benefactive)",
    "32.1 Want Verbs"
  ],
  "GIVE": [
    "11.1 Send Verbs",
    "13.1.a.i Give - No Exchange - Sort of Atelic",
    "13.1.a.ii Give - No Exchange",
    "13.2 Contribute Verbs",
    "13.3 Verbs of Future Having",
    "13.4.1.a Verbs of Fulfilling - Possessional / -to",
    "13.4.1.b Verbs of Fulfilling - Change of State / -with",
	"32.1 Want Verbs"
  ],
  "APPLY": [],
  "PERFORM": [
    "9.1 Put Verbs",
    "10.1 Remove Verbs",
    "10.2.b Banish Verbs / -to",
    "10.5 Verbs of Possessional Deprivation : Steal Verbs",
    "10.6.a Verbs of Possessional Deprivation : Cheat Verbs / -of",
    "10.6.c Verbs of Possessional Deprivation : Cheat Verbs / -out of",
    "11.3 Bring and Take Verbs",
    "13.5.2.d Obtain - Exchange",
    "37.1.e Verbs of Transfer of a Message / -that/to",
    "37.2.a Tell Verbs",
    "37.2.b Tell Verbs / -of/about",
    "37.2.d Tell Verbs / -that/to",
    "37.4.a Verbs of Instrument of Communication",
    "42.1.a Murder Verbs - Kill",
    "42.1.b Murder Verbs - Kill / -with",
    "42.1.c Murder Verbs - General",
    "44.a Destroy Verbs / -with",
    "44.b Destroy Verbs / Instrument Subject",
    "54.4 Price Verbs"
  ],
  "OTHER": []
}



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

catvar_dict = {}
with open('.' + catvar_file) as catvar:
	for entry in catvar:
		entry_pieces = entry.split('#')
		#if '_V' in entry_pieces[0]:
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


'''
rule_path = project_path + generalized_rule_directory
rules = []
for filename in os.listdir(rule_path):
	with open(rule_path + filename, 'r') as rule_file:
		ruleDict = {}
		rule = rule_file.readline()

		ruleDict['rule'] = rule.strip('\n')
		ruleDict['rule_name'] = filename.strip('.txt')
		
		rules.append(ruleDict)

print("Generalized Rules Loaded")
'''

def getModality(text):

	# Split input text into sentences
	sentences = nltk.sent_tokenize(text)
	sentence_modalities = []

	for sentence in sentences:
		constituency_parse = parseModality(sentence)
		sentence_modalities.append({"sentence": sentence, "matches": constituency_parse})


	return sentence_modalities

def getSrl(text):

	# Split input text into sentences
	sentences = nltk.sent_tokenize(text)
	sentence_srls = []

	for sentence in sentences:
		constituency_parse = parseSrl(sentence)
		sentence_srls.append({"sentence": sentence, "matches": constituency_parse})


	return sentence_srls


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
			result = subprocess.run(['java', '-mx100m', '-cp', project_path + tregex_directory + 'stanford-tregex.jar:$CLASSPATH', tsurgeon_class, '-treeFile', 'tree.txt', '.' + preprocess_rule_directory + rule], stdout = subprocess.PIPE, text=True)

			tree_file.write(result.stdout)

	return result.stdout


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
		print('Trigger did not match, investigate tree and regex')
		return None
	if not targ_match:
		print('Target did not match, investigate tree and regex')
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

def buildParseDict(trigger, target, modality, ask, s_ask_types, t_ask_types, additional_S_ask_type,  base_word, rule, rule_name):
	parse_dict = {}
	if modality:
		parse_dict['trigger'] = trigger
		parse_dict['target'] = target
		parse_dict['trigger_modality'] = modality
	parse_dict['ask'] = ask
	parse_dict['S_ask_type'] = s_ask_types
	parse_dict['additional_S_ask_type'] = additional_S_ask_type
	parse_dict['T_ask_type'] = t_ask_types
	parse_dict['base_word'] = base_word
	
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
	annotators = '/?annotators=tokenize,pos,parse,depparse&tokenize.english=true'
	tregex = '/tregex'
	url = coreNLP_server + annotators

	return requests.post(url, data=sentence)

def getLemmaWords(sentence):
	words = nltk.word_tokenize(sentence)
	return [(morphRoot(word.lower())) for word in words]

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

	# If there are not words from the sentence found in the lexicon then we need to check the 
	# preprocessed tree from and triggers
	if len(subsets_per_word) == 0:
		if "Trig" in preprocessed_tree:
			trigs_and_targs = extractTrigsAndTargs(preprocessed_tree)
			print(trigs_and_targs)
			if trigs_and_targs == None:
				return None
			for trig_and_targ in trigs_and_targs:
				# TODO store the portions of the tuple in meaningful names
				(trigger, target, modality, ask, trig_word) = trig_and_targ
				print(trig_and_targ[4])
				(additional_s_ask_types, t_ask_types) = getAskTypes(trig_word)
				parse.append(buildParseDict(trigger, target, modality, ask, s_ask_types, t_ask_types, additional_s_ask_types, target, 'preprocessed rules', 'preprocess rules'))

			return parse
	# Here we loop through each set of generalized rules that were gathered above and 
	# try all of them for each word of the sentence that was found in the lexicon until 
	# one of the generalized rules produces a match via tsurgeon	
	for rule_subset in subsets_per_word:
		for rule in rule_subset:
			print(rule['rule_name'])
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
					print(trigger, target, modality, ask, trig_word)
					print(trig_and_targ[4])
					(additional_s_ask_types, t_ask_types) = getAskTypes(trig_word)
					parse.append(buildParseDict(trigger, target, modality, ask, s_ask_types, t_ask_types, additional_s_ask_types, target, rule['rule'], rule['rule_name']))

				return parse

def parseSrl(sentence):
	parse = []
	base_word = ''
	conj_base_word = ''
	additional_s_ask_types = []
	s_ask_types = []
	t_ask_types = []
	conj_t_ask_types = []
	conj_additional_s_ask_types = []
	words = getLemmaWords(sentence)
	response = getNLPParse(sentence)
	dependencies = response.json()['sentences'][0]['basicDependencies']
	tokens = response.json()['sentences'][0]['tokens']

	for word in words:
		for ask_type, keywords in sashank_categories_sensitive.items():
			if word in keywords and ask_type not in s_ask_types:
				s_ask_types.append(ask_type)
	
	for dependency in dependencies:
		if dependency['dep'] == 'ROOT':
			base_word = dependency['dependentGloss']
		if dependency['dep'] == 'conj':
			if dependency['governorGloss'] == dependencies[0]['dependentGloss']:
				conj_base_word = dependency['dependentGloss']
			
	lem_base_word = morphRoot(base_word)
	(additional_s_ask_types, t_ask_types) = getAskTypes(base_word)
	(additional_lem_s_ask_types, lem_t_ask_types) = getAskTypes(lem_base_word)

	addtional_s_ask_types = appendListNoDuplicates(additional_lem_s_ask_types, additional_s_ask_types)
	t_ask_types = appendListNoDuplicates(lem_t_ask_types, t_ask_types)
	if not t_ask_types:
		t_ask_types.append('OTHER')

	parse.append(buildParseDict('', '', '', '', s_ask_types, t_ask_types, additional_s_ask_types, base_word, '', ''))

	if conj_base_word:
		conj_lem_base_word = morphRoot(conj_base_word)
		(conj_additional_s_ask_types, conj_t_ask_types) = getAskTypes(conj_base_word)
		(conj_additional_lem_s_ask_types, conj_lem_t_ask_types) = getAskTypes(conj_lem_base_word)

		conj_additional_s_ask_types = appendListNoDuplicates(conj_additional_lem_s_ask_types, conj_additional_s_ask_types)
		conj_t_ask_types = appendListNoDuplicates(conj_lem_t_ask_types, conj_t_ask_types)
			
		if not conj_t_ask_types:
			conj_t_ask_types.append('OTHER')

		parse.append(buildParseDict('', '', '', '', s_ask_types, conj_t_ask_types, conj_additional_s_ask_types, conj_base_word, '', ''))	

	return parse

