#Native python modules

#Third party modules

#Local modules

temp_predargs_all = []
temp_predargs_from_possible_triggers = []

all_predargs_version = False
no_filter_version = False
mutually_constrained_version = True
light_verbs_version = True
exist_case_version = True

#If a specific belief type can't be found from the domain trigger or content buckets
# then sentiment words are tried for a belief type which is the default type below
default_belief_type = "EXIST"

#spacy_nlp = ''
#srl_predictor = ''
#sentiment_predictor = ''
#morphRootVerb = ''
#morphRootNoun = ''
#catvar_dict = ''
#catvar_alternates_dict = ''

def init_nlp_tools(spacy, srl, senti_predictor, morphVerb, morphNoun, catvar, catvar_alt_dict):
	global spacy_nlp
	global srl_predictor
	global sentiment_predictor
	global morphRootVerb
	global morphRootNoun
	global catvar_dict
	global catvar_alternates_dict

	spacy_nlp = spacy
	srl_predictor = srl
	sentiment_predictor = senti_predictor
	morphRootVerb = morphVerb
	morphRootNoun = morphNoun
	catvar_dict = catvar
	catvar_alternates_dict = catvar_alt_dict

# This functions checks to see if the items in a list already exist 
# in the original list and if not then add them.
def appendListNoDuplicates(list_to_append, original_list):
	for item in list_to_append:
		if item not in original_list:
			original_list.append(item)

	return original_list

def stances(text_array, text_number, domain_config):
	stances = []

	for text in text_array:
		text_number += 1
		line_stances = get_stances(text_number, domain_config, text[0], text[1], text[2], text[3])
		if line_stances:
			stances.extend(line_stances)

	if all_predargs_version:
		print(temp_predargs_all)
		print(temp_predargs_from_possible_triggers)
		print(len(temp_predargs_all))
		print(len(temp_predargs_from_possible_triggers))

	return ({'stances': stances}, text_number)

def getBeliefType(word, word_pos, trigger_buckets):
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

def getBeliefTypeFromContent(content_word, content_buckets):
	belief_types = []

	if content_word in content_buckets:
		content_belief_types = content_buckets.get(content_word).get("belief_types")

		for content_belief_type in content_belief_types:
			belief_types.append((content_belief_type["belief_type"], content_belief_type["sentiment"]))

	if belief_types:
		return belief_types
	else:
		return [('', '')]

def extractStanceFromSrl(sentence, srl, base_word, word_index, text_number):
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
			#TODO Investigate this becuase if there is a tags list it should have a verb
			if 'B-V' in verb['tags']:
				if verb['tags'].index('B-V') == word_index :
					selected_verb = verb['verb']
					tags_for_verb = verb['tags']

					if all_predargs_version:
						if "ARG0" in verb["description"] or  "ARG1" in verb["description"] or  "ARG3" in verb["description"]:
							temp_predargs_from_possible_triggers.append({"srl": verb["description"], "text_number": text_number - 1})
				elif verb['tags'].index('B-V') == word_index - 1:
					#print(base_word, " verb index: ", verb['tags'].index('B-V'), " word index: ", word_index)
					#print(verb["description"])
					#print(sentence)
					selected_verb = verb['verb']
					tags_for_verb = verb['tags']

					if all_predargs_version:
						if "ARG0" in verb["description"] or  "ARG1" in verb["description"] or  "ARG3" in verb["description"]:
							temp_predargs_from_possible_triggers.append({"srl": verb["description"], "text_number": text_number - 1})

				elif all_predargs_version:
					print(base_word, " verb index: ", verb['tags'].index('B-V'), " word index: ", word_index)
					print(verb["description"])
					print(sentence)
				

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

def get_stances(text_number, domain_config, text, author = '', timestamp = '', doc_id = ''):
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
		srl = srl_predictor.predict(sentence=sent.text)

		if all_predargs_version:
			for verb in srl['verbs']:
				if "ARG0" in verb["description"] or  "ARG1" in verb["description"] or  "ARG3" in verb["description"]:
					temp_predargs_all.append({"srl": verb["description"], "text_number": text_number - 1})

		for token in sent:
			if token.dep_ == "ROOT" or token.dep_ == "xcomp" or token.pos_ == "VERB":
				#print(token.text)
				root_negation_children_count = 0
				is_positive_modality = False
				is_negative_modality = False
				word = token.text.lower()
				lem_word = morphRootVerb(word)
				belief_types = getBeliefType(word, token.pos_, domain_config.get("trigger_buckets"))
				lem_belief_types = getBeliefType(lem_word, token.pos_, domain_config.get("trigger_buckets"))
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
				#EXPERIMENTATION  This will not run unless running experimentation with light verbs
				if light_verbs_version:
					if not strength:
						if word in domain_config.get("light_verbs") or lem_word in domain_config.get("light_verbs"):
							belief_strength = 3

				#TODO Check if I need to add 1 to strength word count to get correct score, not sure why but 
				# doesn't seem so. Will need to investigate
				if strength or strength == 0:
					if strength < 0:
						belief_strength += strength * -1	
						strength_polarity *= -1
					else:
						belief_strength += strength


				if word not in domain_config.get("modality_lexicon") and lem_word not in domain_config.get("modality_lexicon"):
					if event_sentiment or event_sentiment == 0:
						sentiment_word_count += 1
						if event_sentiment < 0:
							sentiment += event_sentiment * -1
							sentiment_polarity *= -1
						else:
							sentiment += event_sentiment

				for child in token.children:
						(belief_strength, strength_polarity, strength_word_count, strength_adjusted) = adjust_belief_strength(belief_strength, strength_polarity, strength_word_count, child.text, domain_config)
						if strength_adjusted:
							strength_words_and_indices.append((child.text, child.i - sent.start))

						(sentiment, sentiment_polarity, sentiment_word_count, sentiment_adjusted) = adjust_sentiment(sentiment, sentiment_polarity, sentiment_word_count, child.text, domain_config)
						if sentiment_adjusted:
							sentiment_words_and_indices.append((child.text, child.i - sent.start))

				if token.dep_ != "ROOT":
					if token.head.dep_ == "ROOT":
						details = {}
						lowered_head = token.head.text.lower()
						if lowered_head in domain_config.get("modality_lexicon"):
							details = domain_config.get("modality_lexicon").get(lowered_head)
						elif morphRootVerb(lowered_head) in domain_config.get("modality_lexicon"):
							details = domain_config.get("modality_lexicon").get(morphRootVerb(lowered_head))

						if details:
							if details.get("modality").lower() in positive_modalities:
								is_positive_modality = True
							elif details.get("modality").lower() in negative_modalities:
								is_negative_modality = True

								
							for child in token.head.children:
								child_details = {}
								lowered_child = child.text.lower()
								if lowered_child in domain_config.get("modality_lexicon"):
									child_details = domain_config.get("modality_lexicon").get(lowered_child)
								elif morphRootNoun(lowered_child) in domain_config.get("modality_lexicon"):
									child_details = domain_config.get("modality_lexicon").get(morphRootNoun(lowered_child))
								elif morphRootVerb(lowered_child) in domain_config.get("modality_lexicon"):
									child_details = domain_config.get("modality_lexicon").get(morphRootVerb(lowered_child))

								if child_details:
									if child_details.get("modality").lower() == "negation":
										root_negation_children_count += 1
							
							

					if (token.head.text, token.head.i - sent.start) not in strength_words_and_indices:
						(belief_strength, strength_polarity, strength_word_count, strength_adjusted) = adjust_belief_strength(belief_strength, strength_polarity, strength_word_count, token.head.text, domain_config)
						if strength_adjusted:
							strength_words_and_indices.append((token.head.text, token.head.i - sent.start))
				
					if (token.head.text, token.head.i - sent.start) not in sentiment_words_and_indices:
						(sentiment, sentiment_polarity, sentiment_word_count, sentiment_adjusted) = adjust_sentiment(sentiment, sentiment_polarity, sentiment_word_count, token.head.text, domain_config)
						if sentiment_adjusted:
							sentiment_words_and_indices.append((token.head.text, token.head.i - sent.start))
				
				
				for child in token.head.children:
					#Need to ensure the token itself doesn't get considered more than once for adjusting strength or sentiment which occurs when the token is the ROOT
					if child.i != token.i:
						if (child.text, child.i - sent.start) not in strength_words_and_indices:
							(belief_strength, strength_polarity, strength_word_count, strength_adjusted) = adjust_belief_strength(belief_strength, strength_polarity, strength_word_count, child.text, domain_config)
							if strength_adjusted:
								strength_words_and_indices.append((child.text, child.i - sent.start))

						if (child.text, child.i - sent.start) not in sentiment_words_and_indices:
							(sentiment, sentiment_polarity, sentiment_word_count, sentiment_adjusted) = adjust_sentiment(sentiment, sentiment_polarity, sentiment_word_count, child.text, domain_config)
							if sentiment_adjusted:
								sentiment_words_and_indices.append((child.text, child.i - sent.start))

				possible_triggers.append((token.text, token.i - sent.start, token.tag_, (belief_strength / strength_word_count) * strength_polarity, strength_words_and_indices, (sentiment / sentiment_word_count) * sentiment_polarity, sentiment_words_and_indices, belief_types, is_positive_modality, is_negative_modality, root_negation_children_count))

		for trigger, trigger_index, pos, strength, strength_words_and_indices, sentiment, sentiment_words_and_indices, belief_types, is_positive_modality, is_negative_modality, root_negation_children_count in possible_triggers:

				for belief_type, belief_type_strength, belief_type_event_sentiment in belief_types:
					stance_details = process_stance(trigger, trigger_index, pos, sent.text, srl, belief_type, strength, belief_type_event_sentiment, domain_config, text_number) #TODO make sure to remove text_number from here 

					if stance_details:
						belief_type = stance_details[0]
						sentiment_strength = stance_details[1]
						content_with_indices = stance_details[2]

						content_with_indices.sort(key=lambda x:x[1])
						#(allen_prob_sentiment, sentiment_probs) = get_sentiment_score(trigger + " " + ' '.join([x[0] for x in content_with_indices]))
						(allen_prob_sentiment, sentiment_probs) = get_sentiment_score(sent.text)
						allen_sentiment = get_valuation_score(allen_prob_sentiment)


						# Check if the belief type came from the sentiment lexicon if so the strength should 
						# be 3.00
						if stance_details[3]:
							strength = 3.00
							#sentiment_strength = sentiment
							if sentiment < 0:
								sentiment_strength *= -1
						elif stance_details[4]:
							strength = float(stance_details[5])
							

						if all_predargs_version:
							if belief_type == "NA":
								sentiment_strength = 0 
						
						if strength * sentiment_strength < 0:
							if is_positive_modality or is_negative_modality:
								is_odd = root_negation_children_count % 2
								if (is_positive_modality and is_odd) or (is_negative_modality and not is_odd):
									strength *= -1
									sentiment_strength *= -1
								
						#print("trigger: ", trigger, "strength words: ", strength_words_and_indices)
						#if strength == 0:
							
							#strength = 3.00

						#NOTE targeted sentiment is the quotes. Need to be filled in or removed at some point
						sent_stances.append(build_stance_dict(belief_type, (trigger, trigger_index), content_with_indices, strength_words_and_indices, strength, sentiment_strength, '', sentiment_probs, allen_sentiment, sent.text, author, timestamp, doc_id, text_number))

		
		if sent_stances:
			exist_stance_exists = False
			exist_stance_indices = []

			for index, stance in enumerate(sent_stances):
				if stance["belief_type"] == default_belief_type:
					exist_stance_exists = True
					exist_stance_indices.append(index)

			if len(exist_stance_indices) != len(sent_stances):
				for index in sorted(exist_stance_indices, reverse=True):
					if all_predargs_version:
						sent_stances[index]["belief_trigger"] = "NA"
					else:
						del sent_stances[index]

			stances.extend(sent_stances)

	if stances:
		return stances

def adjust_belief_strength(belief_strength, polarity, word_count, word, domain_config):
	strength_adjusted = False
	lowered_word = word.lower()
	details = {}

	if lowered_word in domain_config.get("modality_lexicon"):
		details = domain_config.get("modality_lexicon").get(lowered_word)
	elif morphRootNoun(lowered_word) in domain_config.get("modality_lexicon"):
		details = domain_config.get("modality_lexicon").get(morphRootNoun(lowered_word))
	elif morphRootVerb(lowered_word) in domain_config.get("modality_lexicon"):
		details = domain_config.get("modality_lexicon").get(morphRootVerb(lowered_word))

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

def adjust_sentiment(sentiment, polarity, word_count, word, domain_config):
	sentiment_adjusted = False
	lowered_word = word.lower()
	details = {}

	if lowered_word in domain_config.get("modality_lexicon"):
		details = domain_config.get("modality_lexicon").get(lowered_word)
	elif morphRootNoun(lowered_word) in domain_config.get("modality_lexicon"):
		details = domain_config.get("modality_lexicon").get(morphRootNoun(lowered_word))
	elif morphRootVerb(lowered_word) in domain_config.get("modality_lexicon"):
		details = domain_config.get("modality_lexicon").get(morphRootVerb(lowered_word))

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

def build_stance_dict(belief_type, belief_trigger_with_index, belief_content_with_indices, strength_words_and_indices, belief_strength, sentiment_strength, target_sentiment, sentiment_probs, allen_sentiment, sentence, author, timestamp, doc_id, text_number):
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

	attitude = belief_strength * sentiment_strength
	belief_strength = f'{belief_strength:.2f}'

	#In cases where the default belief type is used, meaning belief type was gotten from sentiment words,
	#it is inaccurate to have the trigger in the representation
	if belief_type == default_belief_type:
		belief_trigger = ''
			
	stance_dict["stance_rep"] = f'<{belief_type}[{belief_trigger}[{belief_content}]],{belief_strength},{sentiment_strength}>'
	stance_dict["belief"] = f'{belief_type}[{belief_trigger}[{belief_content}]]'
	stance_dict["belief_string"] = belief_string
	stance_dict["sentiment_string"] = sentiment_string
	stance_dict["evidence"] = sentence
	stance_dict["belief_strength"] = belief_strength
	stance_dict["sentiment_strength"] = sentiment_strength
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
	stance_dict["text_number"] = text_number - 1

	return stance_dict

#TODO make sure to remove text_number as a parameter
def process_stance(word, word_index, word_pos, sentence, srl, belief_type, strength, event_sentiment, domain_config, text_number):
	is_sentiment_belief_type = False 
	is_aspect_modality = False
	modality_belief_strength = 3
	word = word.lower()
	lem_word = morphRootVerb(word)
	'''
	(belief_types, event_sentiment, strength) = getBeliefType(word, word_pos)
	(lem_belief_types, lem_event_sentiment, lem_strength) = getBeliefType(lem_word, word_pos)

	belief_types = appendListNoDuplicates(lem_belief_types, belief_types)
	'''

	(arg0_with_indices, arg1_with_indices, arg2_with_indices, arg3_with_indices) = extractStanceFromSrl(sentence, srl, word, word_index, text_number) #TODO Make sure to remove text_number from here

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
		if  ((word in domain_config.get("light_verbs") or lem_word in domain_config.get("light_verbs")) and light_verbs_version) or no_filter_version: #(catvar_dict.get(word) or catvar_dict.get(lem_word)):
			belief_type_freq = {}
			for arg1_word, word_index in arg1_with_indices:
				lowered_arg1_word = arg1_word.lower()
				lem_arg1_word = morphRootNoun(lowered_arg1_word)
				if not lem_arg1_word:
					lem_arg1_word = morphRootVerb(lowered_arg1_word)

				#(belief_types, event_sentiment) = getBeliefTypeFromContent(word)
				#(lem_belief_types, lem_event_sentiment) = getBeliefTypeFromContent(lem_word)

				#if catvar_dict.get(lowered_arg1_word) or catvar_dict.get(lem_arg1_word):
				belief_types = getBeliefTypeFromContent(lowered_arg1_word, domain_config.get("content_buckets"))
				lem_belief_types = getBeliefTypeFromContent(lem_arg1_word, domain_config.get("content_buckets"))

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
				

			#print("arg1 words: ", arg1_with_indices)
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

		elif word in domain_config.get("modality_lexicon") and exist_case_version:
			details = domain_config.get("modality_lexicon").get(word)
			modality = details.get("modality")
			if details.get("sentiment") and modality.lower() != "aspect":
				belief_type = default_belief_type
				event_sentiment = details.get("sentiment")
				is_sentiment_belief_type = True
			elif modality.lower() == "aspect":
				belief_type = default_belief_type
				modality_belief_strength = details.get("strength")
				event_sentiment = 1
				is_aspect_modality = True
		elif lem_word in domain_config.get("modality_lexicon") and exist_case_version:
			details = domain_config.get("modality_lexicon").get(lem_word)
			modality = details.get("modality")
			if details.get("sentiment") and modality.lower() != "aspect":
				belief_type = default_belief_type
				event_sentiment = details.get("sentiment")
				is_sentiment_belief_type = True
			elif modality.lower() == "aspect":
				belief_type = default_belief_type
				modality_belief_strength = details.get("strength")
				event_sentiment = 1
				is_aspect_modality = True
				


	#If there is no belief type at this point there is no point in building the content
	# so the function is cut short in order to be more efficient
	if not belief_type:
		if all_predargs_version:
			belief_type = "NA"
		else:
			return

	#NOTE This is back off to get details for a target from for an argument that are most appropriate to the specific domain
	# in the current case (12/11/2020) that is PITT/Covid. Here we are checking, from SRL, arg1, then arg0, then arg3
	content = build_content(arg1_with_indices, belief_type, is_sentiment_belief_type, is_aspect_modality, domain_config)

	if arg0_with_indices:
		content.extend(build_content(arg0_with_indices, belief_type, is_sentiment_belief_type, is_aspect_modality,  domain_config))
	if arg3_with_indices:
		content.extend(build_content(arg3_with_indices, belief_type, is_sentiment_belief_type, is_aspect_modality, domain_config))

	if all_predargs_version:
		if not content:
			content = all_predargs_build_content(arg1_with_indices)

			if arg0_with_indices:
				content.extend(all_predargs_build_content(arg0_with_indices))
			if arg3_with_indices:
				content.extend(all_predargs_build_content(arg3_with_indices))

			if content:
				content = [("NA", 0)]
	
	print(word, "content: ", content, "arg1: ", arg1_with_indices, "arg0: ", arg0_with_indices, "arg3: ", arg3_with_indices)
	return_tuple = (belief_type, event_sentiment, content, is_sentiment_belief_type, is_aspect_modality, modality_belief_strength)
	#(ask_who, ask, ask_recipient, ask_when, ask_action, confidence, descriptions, belief_type, t_ask_confidence, word_number, arg2, event_sentiment, content, is_sentiment_belief_type)
		
	#if not ask_action:
	#	return_tuple = (ask_who, ask, ask_recipient, ask_when, ask_negation_dep_based, ask_action, confidence) = extractAskInfoFromDependencies(word, dependencies, t_ask_types)

	if belief_type and content:
		return return_tuple

def all_predargs_build_content(potential_content_with_indices):
	content_words_with_indices = []

	for word, word_index in potential_content_with_indices:
		content_words_with_indices.append((word, word_index))

	return content_words_with_indices

def build_content(potential_content_with_indices, trigger_belief_type, is_sentiment_belief_type, is_aspect_modality, domain_config): 
	content_buckets = domain_config.get("content_buckets")
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

		#content_words_with_indices.append((word, word_index))

		if trigger_belief_type == "NA" and all_predargs_version:
			content_words_with_indices.append((word, word_index))
		else:
			if content_belief_types:
				for content_belief_type in content_belief_types:
					#Constrain allowed content by only adding it if the word appears in a counterpart bucket.
					#If the trigger bucket label is PROTECT the word must be in the PROTECT content bucket.
					#Also if they belief type came from the sentiment the check words from any content bucket
					if content_belief_type.get("belief_type") == trigger_belief_type or is_sentiment_belief_type or is_aspect_modality or not mutually_constrained_version: 
						content_words_with_indices.append((word, word_index))
						break
			'''
			else:
				content_words_with_indices.append((word, word_index))
			'''

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
	
