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
		

		if user_id not in stances_by_user:
			stances_by_user[user_id] = {}
			stances_by_user[user_id]["positive"] = {} 
			stances_by_user[user_id]["negative"] = {} 

			if attitude > 0:
				if b_type not in stances_by_user[user_id]["positive"]:
					stances_by_user[user_id]["positive"][b_type] = 1
					clusters["positive_" + b_type] = []
				else:
					stances_by_user[user_id]["positive"][b_type] += 1

			elif attitude < 0:
				if b_type not in stances_by_user[user_id]["negative"]:
					stances_by_user[user_id]["negative"][b_type] = 1
					clusters["negative_" + b_type] = []
				else:
					stances_by_user[user_id]["negative"][b_type] += 1

		else:
			if attitude > 0:
				if b_type not in stances_by_user[user_id]["positive"]:
					stances_by_user[user_id]["positive"][b_type] = 1
				else:
					stances_by_user[user_id]["positive"][b_type] += 1

			elif attitude < 0:
				if b_type not in stances_by_user[user_id]["negative"]:
					stances_by_user[user_id]["negative"][b_type] = 1
				else:
					stances_by_user[user_id]["negative"][b_type] += 1


##print(stances_by_user)

#print(len(stances_by_user.keys()))

user_total_stances = {}

for user, pos_or_neg in stances_by_user.items():
	user_total_stances[user] = 0
	for attitude_polarity, b_types in pos_or_neg.items():
		#print("this is b_types", b_types)
		if len(b_types) > 0:
			for b_type, count in b_types.items():
				user_total_stances[user] += count		
				#print("user: ", user, "pos_or_neg: ", attitude_polarity, "type: ", b_type, "count: ", count)

	

#print(user_total_stances)

user_percent_belief_w_attitude = {}
for user, pos_or_neg in stances_by_user.items():
	user_percent_belief_w_attitude[user] = {}
	for attitude_polarity, b_types in pos_or_neg.items():
		if len(b_types) > 0:
			for b_type, count in b_types.items():
				user_percent_belief_w_attitude[user][attitude_polarity + "_" + b_type] = count/user_total_stances[user]	



#print(user_percent_belief_w_attitude)

for user, percent_by_type in user_percent_belief_w_attitude.items():
	cluster = max(percent_by_type, key=percent_by_type.get)

	if user not in clusters[cluster]:
		clusters[cluster].append(user)

#print(clusters)

#print(clusters.keys())

with open("./user_clusters.json", "w+") as cluster_file:
        cluster_file.write(json.dumps(clusters, indent=4))


cluster_totals = {}
for cluster, users in clusters.items():
	cluster_totals[cluster] = len(users)

with open("./user_clusters_totals.json", "w+") as totals_file:
        totals_file.write(json.dumps(cluster_totals, indent=4))

#print(cluster_totals)

