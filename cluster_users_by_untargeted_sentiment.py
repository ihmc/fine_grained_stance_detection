import json

stances_by_user = {}
clusters = {}
stance_file_path = "/home/user/pitt/stance_detection/user_provided_stance_output/user_provided_json_stances_20210802(All IEEE_geo_mask_distance_vaccine).v1.3.jsonl"
with open(stance_file_path, "r") as stance_file:
	stance_lines = stance_file.readlines()

	for stance_json in stance_lines:
		stance = json.loads(stance_json)

		user_id = stance["attribution"]["author"]
		b_type = stance["belief_type"]
		attitude = float(stance["attitude"])
		untarg_senti = float(stance["allen_sentiment"])
		

		if user_id not in stances_by_user:
			stances_by_user[user_id] = {}
			stances_by_user[user_id]["positive"] = 0
			stances_by_user[user_id]["negative"] = 0 

			if untarg_senti > 0:
				stances_by_user[user_id]["positive"] = 1 

			elif untarg_senti < 0:
				stances_by_user[user_id]["negative"] = 1 

		else:
			if untarg_senti > 0:
				stances_by_user[user_id]["positive"] += 1 

			elif untarg_senti < 0:
				stances_by_user[user_id]["negative"] += 1 

user_total_stances = {}

for user, pos_or_neg in stances_by_user.items():
	user_total_stances[user] = 0
	for attitude_polarity, count in pos_or_neg.items():
		user_total_stances[user] += count

user_percent_untarg_senti = {}
for user, pos_or_neg in stances_by_user.items():
	user_percent_untarg_senti[user] = {}
	for attitude_polarity, count in pos_or_neg.items():
		user_percent_untarg_senti[user][attitude_polarity] = count / user_total_stances[user]

print(user_percent_untarg_senti)

clusters = {"positive": [], "negative": []}

for user, percent_by_untarg_senti in user_percent_untarg_senti.items():
	cluster = max(percent_by_untarg_senti, key=percent_by_untarg_senti.get)
	if user not in clusters[cluster]:
		clusters[cluster].append(user)


for cluster, users in clusters.items():
	print("cluster: ", cluster, "count: ", len(users))
