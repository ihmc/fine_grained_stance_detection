,# ask_detection.py must be in same directory as this file for import to work
# This is IHMC ask detction code
import ask_detection
import json
import email
import os
import csv
import math
import numpy as np
import lxml.html
from pathlib import Path

from lxml import etree
from spacy.lang.en import English # updated
from spacy.matcher import Matcher
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd

nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer')) # updated

def handleEmail():
	for filename in os.listdir("emls"):
		(root, ext) = os.path.splittext(filename)
		with open("emls/" + filename) as eml_file:
			message = email.message_from_file(eml_file)
		if message.is_multipart():
			for part in message.walk():
				if part.get_content_type() == "text/html":
					html_part = part.get_payload(decode=True).decode('UTF-8')
				elif part.get_content_type() == "text/plain":
					text_part = part.get_payload()

		if html_part:
			sentence_srls = ask_detection.getSrl(*preprocess_html(html_part, True))
		elif text_part:
			sentence_srls = ask_detection.getSrl(*preprocess_text())

		with open("emls_output/" + root + ".txt") as eml_output:
			eml_output.write(json.dumps(sentence_srls))
			eml_output.write("\n")
		
		return json.dumps(sentence_srls)

def handleSMS(text):
	sentence_srls = ask_detection.getSrl(*preprocess_text(text))

	return json.dumps(sentence_srls)

def handleLinkedin(text):
	sentence_srls = ask_detection.getSrl(*preprocess_text(text))

	return json.dumps(sentence_srls)

def preprocess_html(html, replace_hrefs):
	url_token_indices = []
	sentences_array = []
	text_to_process = ""
	ask_dict = {}
	ask_id = 0
	tag = "ASKMARKER1234"

	parsed_html = lxml.html.fromstring(html) 
	body = parsed_html.cssselect('body')[0]

	for img in body.cssselect('img'):
		if img.attrib.has_key('alt'):
			alt = img.attrib['alt']
			replacement_element = etree.Element("p")
			replacement_element.text = f'{alt}'
			img.getparent().replace(img, replacement_element)

	for style in body.cssselect('style'):
		style.remove()

	for script in body.cssselect('script'):
		script.remove()

	for node in body.cssselect('.gmail_quote'):
		node.remove()

	if replace_hrefs:
		for link in body.cssselect('a'):
			if link.text:
				if tag not in link.text:
					ask_dict[ask_id] = link.attrib['href']
					if len(link.text.strip()) > 0 :
						link.text = f' [[[{tag}-{ask_id}-{tag}{link.text}/{tag}-{ask_id}-{tag}]]]'
						ask_id += 1
	'''
	for node in body.cssselect("hr"):
		node.content = "________________________________\n"

	for node in body.cssselect("li"):
		node.content = " #{node.content} \n\n"

	for node in body.cssselect("p"):
		node.content = " #{node.content} \n\n"

	for node in body.cssselect("br"):
		node.content = " #{node.content} \n\n"

	'''
	for node in body.cssselect("div"):
		for child in node.getchildren():
			child.text = f' {child.text} \n\n' if child.text else child.text

	return(body.text_content(), ask_dict)



def preprocess_text(text):
	url_token_indices = []
	sentences_array = []
	text_to_process = ""
	ask_dict = {}
	ask_id = 0
	tag = "ASKMARKER1234"

	doc = nlp(text)

	for sent in doc.sents:
		sentence_pieces = []
		for token in sent:
			if token.text.lower() in abbrev_mappings:
				sentence_pieces.append(f'{abbrev_mappings.get(token.text)}{token.whitespace_}')
			elif(token.like_url or token.like_email) and tag not in token.text:
				ask_dict[str(ask_id)] = token.text	
				sentence_pieces.append(f'[[[{tag}-{ask_id}-{tag}{token.text}/{tag}-{ask_id}-{tag}]]]{token.whitespace_}')
				ask_id += 1
			else:
				sentence_pieces.append(token.text_with_ws)
		sentences_array.append(sentence_pieces)

	for sent in sentences_array:
		text_to_process += f'{"".join(sent)}\n'

	return(text_to_process, ask_dict)

abbrev_mappings = {
	"txt": "text",
	"sms": "text",
	"msg": "text",
	"im": "text",
	"dm": "text",
	"ansr": "answer",
	"b4": "before",
	"2": "to",
	"l8er": "later",
	"&": "and",
	"r": "are",
	"4fil": "fulfill",
	"unsub": "unsubscribe",
	"u": "you"
}

def extract_relevant_text():
	stemmer = PorterStemmer()
	relevant_text = []
	df = list(pd.read_csv('./2020-04-01-tweets.tsv', sep='\\t', engine='python', header=None)[4])
	def has_topic(passage, topics):
		tokens = word_tokenize(passage) # tokenize before lowering
		stemmed_tokens = [stemmer.stem(token).lower() for token in tokens]
		if all([x in stemmed_tokens for x in topics]):
			relevant_text.append(passage)

	for text in df:
		has_topic(text, ['mask'])

	return ask_detection.stances(relevant_text)

def test(text):
	return ask_detection.stances([[text.replace("’", "'"), '', '', '', '']])

def process_mask_tweets(version_description = "", num_to_process = 0):
	tweet_full_texts = []
	tweets_file = "./mask_lines.txt"
	
	with open(tweets_file, 'r') as tweet_file:
		progress_count = 0
		tweets = tweet_file.read().splitlines()
		
		#NOTE User passes in amount of tweets to process, if the amount is 0 then it defaults 
		# to processing the entirety of the tweets
		if num_to_process == 0:
			num_to_process = len(tweets)

		for tweet_text in tweets[:num_to_process]:
			tweet = json.loads(tweet_text)

			#Json.loads seems to automatically convert unicode characters like \u2019 which is a right single 
			# quote, which is not typical and will cause issues when looking up items like n't in the lexicons
			tweet_full_texts.append([tweet["full_text"].replace("’", "'"), tweet["user"]["id"], tweet["created_at"], tweet["id"]])

	output = ask_detection.stances(tweet_full_texts)

	Path("./mask_stances_output").mkdir(exist_ok=True)

	#Add underscore to front of version description if one exist so that the 
	# file name will be nicely underscore separated
	if version_description:
		version_description = "_" + version_description

	with open("./mask_stances_output/mask_lines_stances" + version_description + ".txt", "w+") as mask_stances:
		for stance in output["stances"]:
			mask_stances.write(json.dumps(stance))
			mask_stances.write("\n")
	return output

def process_sf_tweets(version_description = "", num_to_process = 0):
	tweet_full_texts = []
	tweets_file = "./sf_masks.json"
	with open(tweets_file, 'r') as tweet_file:
		progress_count = 0
		tweets = tweet_file.read().splitlines()

		#NOTE User passes in amount of tweets to process, if the amount is 0 then it defaults 
		# to processing the entirety of the tweets
		if num_to_process == 0:
			num_to_process = len(tweets)

		for tweet_text in tweets[:num_to_process]:
			tweet = json.loads(tweet_text)
			#right single quote which is not typical and will cause issues when looking
			# up items like n't in the lexicons
			tweet_full_texts.append([tweet["full_text"].replace("’", "'"), tweet["user"]["id"], tweet["created_at"], tweet["id"]])

	output = ask_detection.stances(tweet_full_texts)
	Path("./san_fran_mask_stances_output").mkdir(exist_ok=True)

	#Add underscore to front of version description if one exist so that the 
	# file name will be nicely underscore separated
	if version_description:
		version_description = "_" + version_description

	with open("./san_fran_mask_stances_output/sf_mask_lines" + version_description + ".jsonl", "w+") as mask_stances:
		for stance in output["stances"]:
			mask_stances.write(json.dumps(stance))
			mask_stances.write("\n")

	return output

def process_mask_chyrons(version_description = "", num_to_process = 0):
	chyron_data = []
	chyrons_file = "./mask_dist_chyrons.txt"
	with open(chyrons_file, 'r') as chyron_file:
		progress_count = 0
		chyrons = chyron_file.read().splitlines()

		#NOTE User passes in amount of chyrons to process, if the amount is 0 then it defaults 
		# to processing the entirety of the file
		if num_to_process == 0:
			num_to_process = len(tweets)

		for chyron_text in chyrons[:num_to_process]:
			chyron = json.loads(chyron_text)
			#right single quote which is not typical and will cause issues when looking
			# up items like n't in the lexicons
			chyron_data.append([chyron["chyron_text"].replace("’", "'"), chyron["author"], chyron["timestamp"], chyron["doc_id"]])

	output = ask_detection.stances(chyron_data)
	Path("./chyron_mask_stances_output").mkdir(exist_ok=True)

	#Add underscore to front of version description if one exist so that the 
	# file name will be nicely underscore separated
	if version_description:
		version_description = "_" + version_description

	with open("./chyron_mask_stances_output/mask_chyron_stances" + version_description + ".jsonl", "w+") as mask_chyron_stances:
		for stance in output["stances"]:
			mask_chyron_stances.write(json.dumps(stance))
			mask_chyron_stances.write("\n")
			
	return output

#User provides a path to a text file for stances. Each line must be single text that the user
# desires to process for stances
def text_to_stances(txt_file_path, version_description = "", num_to_process = 0):
	data = []
	texts_file = txt_file_path
	with open(texts_file, 'r', encoding="utf-8") as text_file:
		progress_count = 0
		texts = text_file.read().splitlines()

		#NOTE User passes in amount of chyrons to process, if the amount is 0 then it defaults 
		# to processing the entirety of the file
		if num_to_process == 0:
			num_to_process = len(texts)

		for text in texts[:num_to_process]:
			#right single quote which is not typical and will cause issues when looking
			# up items like n't in the lexicons
			data.append([text.replace("’", "'").replace("\u2019", "'"), "", "", ""])

	output = ask_detection.stances(data)
	Path("./user_provided_stance_output").mkdir(exist_ok=True)

	#Add underscore to front of version description if one exist so that the 
	# file name will be nicely underscore separated
	if version_description:
		version_description = "_" + version_description

	with open("./user_provided_stance_output/user_provided_text_stances" + version_description + ".jsonl", "w+") as user_text_stances:
		for stance in output["stances"]:
			user_text_stances.write(json.dumps(stance))
			user_text_stances.write("\n")
			
	return output

'''
User provides a path to a json file for stance processing. The extension does not have to be json, 
 but each line must be a single json structure that has an attribute for the text to process,
 some kind of author identifier, a timestamp, and some kind of document identifier.
 For example if the json represented a tweet there should be the author id, timestamp of the tweet,
 and the id of the tweet. User must provide the name of each of these attributes that is found in
 each json structure.

In order to handle neseted json the user should put (in the appropriate order) the attributes comma separated.
'''
def json_to_stances(json_file_path, text_attrb_name, author_attrb_name, timestamp_attrb_name, 
						doc_id_attrb_name, version_description = "", num_to_process = 0):
	data = []
	with open(json_file_path, 'r') as json_file:
		progress_count = 0
		jsons = json_file.read().splitlines()

		#NOTE User passes in amount of chyrons to process, if the amount is 0 then it defaults 
		# to processing the entirety of the file
		if num_to_process == 0:
			num_to_process = len(jsons)

		for line_json in jsons[:num_to_process]:
			line = json.loads(line_json)

			nested_text_attrbs = text_attrb_name.split(",")
			nested_author_attrbs = author_attrb_name.split(",")
			nested_timestamp_attrbs = timestamp_attrb_name.split(",")
			nested_doc_id_attrbs = doc_id_attrb_name.split(",")

			text = handle_nested_json(nested_text_attrbs, line)
			author = handle_nested_json(nested_author_attrbs, line)
			timestamp = handle_nested_json(nested_timestamp_attrbs, line)
			doc_id = handle_nested_json(nested_doc_id_attrbs, line)
			
				

			#\u2019 is unicode for right single quote which is not typical, and will cause issues when looking
			# up items like n't in the lexicons
			#data.append([line[text_attrb_name].replace("’", "'"), line[author_attrb_name], 
			#					line[timestamp_attrb_name], line[doc_id_attrb_name]])
			data.append([text.replace("’", "'"), author, timestamp, doc_id])

	output = ask_detection.stances(data)
	Path("./user_provided_stance_output").mkdir(exist_ok=True)

	#Add underscore to front of version description if one exist so that the 
	# file name will be nicely underscore separated
	if version_description:
		version_description = "_" + version_description

	with open("./user_provided_stance_output/user_provided_json_stances" + version_description + ".jsonl", "w+") as user_json_stances:
		for stance in output["stances"]:
			user_json_stances.write(json.dumps(stance))
			user_json_stances.write("\n")
			
	return output

'''
User provides a path to a csv file for stance processing. The file must have a header row with a label for 
for the text to process, some kind of author identifier, a timestamp, and some kind of document identifier.
For example if each csv line represented a tweet the text of the tweet, the author id, timestamp of the tweet,
and the id of the tweet. User must provide the name of each of these labels that is found in the csv file.

Some csv data might use 2 columns or more to specify the timestamp so when providing the labels for timestamp
separate them with a pipe (|) character. Please note these must be in the correct order, date first then time.
'''
def csv_to_stances(csv_file_path, text_label, author_label, timestamp_label, 
						doc_id_label, version_description = "", num_to_process = 0):
	num_processed = 0
	
	df = pd.read_csv(csv_file_path)

	#NOTE User passes in amount of chyrons to process, if the amount is 0 then it defaults 
	# to processing the entirety of the file
	if num_to_process == 0:
		num_to_process = len(df.index)

	#This is to decide how many chunks to split the data into to process batches of the specified
	# number. I will need to probably do the number as a variable that can be passed in by the uer
	num_chunks = math.floor(len(df.index) / 10)
	if num_chunks == 0:
		num_chunks = 1

	Path("./user_provided_stance_output").mkdir(exist_ok=True)

	#Add underscore to front of version description if one exist so that the 
	# file name will be nicely underscore separated
	if version_description:
		version_description = "_" + version_description

	#total_output = []
	with open("./user_provided_stance_output/user_provided_csv_stances" + version_description + ".jsonl", "w+") as user_json_stances:
		for chunk in np.array_split(df, num_chunks):
			data = []
			chunk_output = {"stances": []}
			for row in chunk.iterrows():
				timestamp_data = ''
				#This step is necessary cause each row of the dataframe is a tuple (index, row info)
				row_info = row[1]
				num_processed += 1

				if num_processed > num_to_process:
					break

				labels = timestamp_label.split("|")

				for label in labels:
					timestamp_data += row_info[label] + " "

				#\u2019 is unicode for right single quote which is not typical, and will cause issues when looking
				# up items like n't in the lexicons
				data.append([row_info[text_label].replace("’", "'").replace("\u2019", "'"), row_info[author_label], 
								timestamp_data, row_info[doc_id_label]])

			chunk_output = ask_detection.stances(data)
			#total_output.extend(chunk_output["stances"])

			for stance in chunk_output["stances"]:
				user_json_stances.write(json.dumps(stance))
				user_json_stances.write("\n")

			print("Batch finished")

	#for row in df.iterrows():
	#	timestamp_data = ''
	#	#This step is necessary cause each row of the dataframe is a tuple (index, row info)
	#	row_info = row[1]
	#	num_processed += 1
	#	if num_processed > num_to_process:
	#		break

	#	labels = timestamp_label.split("|")

	#	for label in labels:
	#		timestamp_data += row_info[label] + " "

	#	#\u2019 is unicode for right single quote which is not typical, and will cause issues when looking
	#	# up items like n't in the lexicons
	#	data.append([row_info[text_label].replace("’", "'").replace("\u2019", "'"), row_info[author_label], 
	#							timestamp_data, row_info[doc_id_label]])

	#output = ask_detection.stances(data)
	#Path("./user_provided_stance_output").mkdir(exist_ok=True)

	##Add underscore to front of version description if one exist so that the 
	## file name will be nicely underscore separated
	#if version_description:
	#	version_description = "_" + version_description

	#with open("./user_provided_stance_output/user_provided_json_stances" + version_description + ".jsonl", "w+") as user_json_stances:
		#for stance in output["stances"]:
		#	user_json_stances.write(json.dumps(stance))
		#	user_json_stances.write("\n")
			
	return total_output


#NOTE Stance version should be a string (underscore separated) that will be append to the file name 
# so it's obvious what version of stances was run (improved_belief_string, improved_belief_score, etc.) 
def stances_to_csv(file_to_convert, stance_version):
	columns = ['Number', 'Belief String', 'Sentiment String', 'Belief Type', 'Trigger', 'Content', 'Belief Strength', 'Belief Valuation', 'Evidence', 
				'Adjudicated Type', 'Adjudicated Trigger', 'Adjudicated Content', 'Adjudicated Belief Stregnth', 'Adjudicated Belief Valuation']

	Path("./stance_output_csvs").mkdir(exist_ok=True)

	if stance_version:
		stance_version = "_" + stance_version
	
	with open('./mask_stances_' + stance_version + '.csv', 'w+') as csvfile:
		csvwriter = csv.writer(csvfile)

		csvwriter.writerow(columns)

		line_number = 1
		evidence = ''
		with open(file_to_convert, 'r') as mask_stances:
			for stance in mask_stances.readlines():
				stance_dict = json.loads(stance)

				# Keep the line number the same if the stance is on the same evidence as the previous stance
				if not evidence:
					evidence = stance_dict['evidence']
				elif evidence != stance_dict['evidence']:
					evidence = stance_dict['evidence']
					line_number += 1

				row = [stance_dict['text_number'], stance_dict['belief_string'], stance_dict['sentiment_string'], stance_dict['belief_type'], stance_dict['belief_trigger'], 
						stance_dict['belief_content'], stance_dict['belief_strength'], stance_dict['belief_valuation'], stance_dict['evidence']]
				csvwriter.writerow(row)
				
def stances_to_csv_tp_and_fp(file_to_convert, stance_version = ""):
	columns = ['Number', 'Belief String', 'Sentiment String', 'Belief Type', 'Trigger', 'Content', 'Belief Strength', 'Belief Valuation', 'Evidence', 
				'TP', 'FP']

	Path("./stance_output_csvs").mkdir(exist_ok=True)

	if stance_version:
		stance_version = "_" + stance_version
	with open('./stance_output_csvs/mask_stances' + stance_version + '.csv', 'w+') as csvfile:
		csvwriter = csv.writer(csvfile)

		csvwriter.writerow(columns)

		line_number = 1
		evidence = ''
		with open(file_to_convert, 'r') as mask_stances:
			for stance in mask_stances.readlines():
				stance_dict = json.loads(stance)

				# Keep the line number the same if the stance is on the same evidence as the previous stance
				if not evidence:
					evidence = stance_dict['evidence']
				elif evidence != stance_dict['evidence']:
					evidence = stance_dict['evidence']
					line_number += 1

				row = [stance_dict['text_number'], stance_dict['belief_string'], stance_dict['sentiment_string'], stance_dict['belief_type'], stance_dict['belief_trigger'], 
						stance_dict['belief_content'], stance_dict['belief_strength'], stance_dict['belief_valuation'], stance_dict['evidence']]
				csvwriter.writerow(row)

def stances_to_csv_mask_attitudes(file_to_convert, stance_version = ""):
	columns = ['Number', 'Belief String', 'Sentiment String', 'Belief Type', 'Trigger', 'Content', 'Belief Strength', 'Belief Valuation', 'Attitude', 'Evidence']

	Path("./stance_output_csvs").mkdir(exist_ok=True)

	if stance_version:
		stance_version = "_" + stance_version
	with open('./stance_output_csvs/mask_stances' + stance_version + '.csv', 'w+') as csvfile:
		csvwriter = csv.writer(csvfile)

		csvwriter.writerow(columns)

		line_number = 1
		evidence = ''
		with open(file_to_convert, 'r') as mask_stances:
			for stance in mask_stances.readlines():
				stance_dict = json.loads(stance)

				# Keep the line number the same if the stance is on the same evidence as the previous stance
				if not evidence:
					evidence = stance_dict['evidence']
				elif evidence != stance_dict['evidence']:
					evidence = stance_dict['evidence']
					line_number += 1

				is_mask_related = False
				if stance_dict['belief_type'] == "PROTECT" or stance_dict['belief_type'] == "RESTRICT" or stance_dict['belief_type'] == "EXIST":
					for word in stance_dict['belief_content'].split(" "):
						if ask_detection.morphRootNoun(word).lower() == "mask" or word.lower() == "mask":
							is_mask_related = True

				if is_mask_related:
					row = [stance_dict['text_number'], stance_dict['belief_string'], stance_dict['sentiment_string'], stance_dict['belief_type'], stance_dict['belief_trigger'], 
							stance_dict['belief_content'], stance_dict['belief_strength'], stance_dict['belief_valuation'], stance_dict['attitude'], stance_dict['evidence']]
					csvwriter.writerow(row)

def handle_nested_json(attributes, json_object):
	value = json_object

	for attrib in attributes:
		value = value[attrib]

	return(value)


