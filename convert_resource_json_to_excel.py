import csv
import json
import pandas as pd

from allennlp.predictors.predictor import Predictor

sentiment_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/stanford-sentiment-treebank-roberta.2021-03-11.tar.gz")

def build_triggers_excel(filepath, domain_name):
	trigger_rows = []
	fields = ["Lexical item", "POS", "Belief", "Default Belief Value", "Default Sentiment Value"] 	

	with open(filepath, "r") as triggers_json:
		triggers = json.load(triggers_json)
		for category, words in triggers.items():
			sentiment = get_sentiment_score(category)

			if category != "unassigned":
				for word in words:
					trigger_rows.append([word, '', category.upper(), 3, sentiment])

	trigger_df = pd.DataFrame(trigger_rows, columns=fields)
	trigger_df.to_excel("./" + domain_name + "_triggers.xlsx", index=False)

def build_content_excel(filepath, domain_name):
	content_rows = []
	fields = ["Lexical item", "Belief", "Sentiment Value"] 	

	with open(filepath, "r") as content_json:
		content = json.load(content_json)
		for category, words in content.items():
			sentiment = get_sentiment_score(category)

			if category != "unassigned":
				for word in words:
					content_rows.append([word, category.upper(), sentiment])

	content_df = pd.DataFrame(content_rows, columns=fields)
	content_df.to_excel("./" + domain_name + "_contents.xlsx", index=False)

def get_sentiment_score(text):
	probs = sentiment_predictor.predict(sentence=text)['probs']
	if probs[0] > probs[1]:
		return 1
	elif probs[1] > probs[0]:
		return -1
	else:
		return 0
        
