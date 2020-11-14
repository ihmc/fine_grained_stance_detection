import os
import re
import csv
import pandas as pd

from ask_mappings import panacea_ask_types, pitt_ask_types

project_path = os.path.abspath(os.path.dirname(__file__))

ask_and_framing_types = list(panacea_ask_types.keys())
ask_and_framing_types.extend(list(pitt_ask_types.keys()))
#ask_and_framing_types = ["PERFORM", "GIVE", "GAIN", "LOSE"]

# Path for files needed for catvar processing
catvar_file = '/catvar.txt'
lcs_file = '/dev_LCS.txt'
#lcs_file = '/ACL_Archna-Bonnie-Fixed-LCS-Bare-Verb-Classes-for-Panacea.txt'
#lcs_file = '/LCS-Bare-Verb-Classes-Final_27Aug19.txt'

# Rule directories
# TODO This list will be thinned out as rule sets are chosen for superiority
generalized_v3_directory = '/generalized-templates_v3/'
lexical_item_rule_directory = '/lexical-item-rules/'
preprocess_rule_directory = '/idiosyncratic/'

modality_lookup = {}
word_specific_rules = []

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

print('LCS dictionary created')

verb_list_dict = {}
for a_or_f_type in ask_and_framing_types:
	if a_or_f_type in panacea_ask_types:
		lcs_categories = panacea_ask_types[a_or_f_type]
	else:
		lcs_categories = pitt_ask_types[a_or_f_type]
	verb_list_dict[a_or_f_type] = set([])
	for category in lcs_categories:
		verb_list_dict[a_or_f_type].update(lcs_dict[category])

#print(verb_list_dict)
#NOTE can definitely automate this better without having to specify a variable for each list type
verb_list_dict["PERFORM"].add("wear")
verb_list_dict["PERFORM"].add("vaccinate")
perform_verbs = sorted(verb_list_dict["PERFORM"])
give_verbs = sorted(verb_list_dict["GIVE"])
gain_verbs = sorted(verb_list_dict["GAIN"])
lose_verbs = sorted(verb_list_dict["LOSE"])
protect_verbs = sorted(verb_list_dict["PROTECT"])
reject_verbs = sorted(verb_list_dict["REJECT"])



test_list = []

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

'''
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
			
'''
#print(lexical_specific_rules)

df = pd.read_excel("ModalityLexiconSubcatTags.xlsx")
belief_strength_dict = pd.Series(df["Belief Value"].values,index=df["Lexical item"]).to_dict()
	
