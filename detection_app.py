# ask_detection.py must be in same directory as this file for import to work
# This is IHMC ask detction code
import ask_detection
import json
import email
import os
import lxml.html
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
	return ask_detection.stances([[text, '', '', '', '']])

def process_mask_tweets():
	tweet_full_texts = []
	tweets_file = "./mask_lines.txt"
	with open(tweets_file, 'r') as tweet_file:
		progress_count = 0
		tweets = tweet_file.read().splitlines()
		for tweet_text in tweets[:2500]:
			tweet = json.loads(tweet_text)
			tweet_full_texts.append([tweet["full_text"], tweet["user"]["id"], tweet["created_at"], tweet["id"]])

	output = ask_detection.stances(tweet_full_texts)
	with open("./mask_lines_stances", "w+") as mask_stances:
		mask_stances.write(json.dumps(output))
	return output

def process_mask_chyrons():
	chyron_data = []
	chyrons_file = "./mask_dist_chyrons.txt"
	with open(chyrons_file, 'r') as chyron_file:
		progress_count = 0
		chyrons = chyron_file.read().splitlines()
		for chyron_text in chyrons[:1000]:
			chyron = json.loads(chyron_text)
			chyron_data.append([chyron["chyron_text"], chyron["author"], chyron["timestamp"], chyron["doc_id"]])

	output = ask_detection.stances(chyron_data)
	with open("./mask_chyron_stances.txt", "w+") as mask_chyron_stances:
		mask_chyron_stances.write(json.dumps(output, indent=4, sort_keys=False))
	return output
			
