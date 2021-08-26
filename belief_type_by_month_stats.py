import os
import json
import pandas as pd
from datetime import datetime

stance_stats = {}
stance_stats_rows = []
state = "pennsylvania"
state_abbrev = "PA"
stance_folder = "/Users/brodieslab/Documents/carley_state_data/" + state + "/stances/"

for stance_file in os.listdir(stance_folder):
	stance_file_path = stance_folder + stance_file
	city = os.path.basename(stance_file_path).split("_")[5]

	print(city)

	today = datetime.today().strftime('%Y%m%d')

	with open(stance_file_path, "r") as stance_file:
		stance_lines = stance_file.readlines() 

		

		for stance_json in stance_lines:
			stance = json.loads(stance_json)

			b_type = stance["belief_type"]
			attitude = float(stance["attitude"])
			date_info = datetime.strptime(stance["attribution"]["timestamp"].strip(), "%m/%d/%Y %H:%M:%S")
			month = date_info.strftime("%b")
			year = date_info.year

			if year not in stance_stats:
				stance_stats[year] = {}
				if month not in stance_stats[year]:
					stance_stats[year][month] = {}
					stance_stats[year][month]["positive"] = {} 
					stance_stats[year][month]["negative"] = {}

					if attitude > 0:
						if b_type not in stance_stats[year][month]["positive"]:
							stance_stats[year][month]["positive"][b_type] = 1
						else:
							stance_stats[year][month]["positive"][b_type] += 1

					elif attitude < 0:
						if b_type not in stance_stats[year][month]["negative"]:
							stance_stats[year][month]["negative"][b_type] = 1
						else:
							stance_stats[year][month]["negative"][b_type] += 1

			else:
				if month not in stance_stats[year]:
					stance_stats[year][month] = {}
					stance_stats[year][month]["positive"] = {} 
					stance_stats[year][month]["negative"] = {}
				if attitude > 0:
					if b_type not in stance_stats[year][month]["positive"]:
						stance_stats[year][month]["positive"][b_type] = 1
					else:
						stance_stats[year][month]["positive"][b_type] += 1

				elif attitude < 0:
					if b_type not in stance_stats[year][month]["negative"]:
						stance_stats[year][month]["negative"][b_type] = 1
					else:
						stance_stats[year][month]["negative"][b_type] += 1	


	city_rows = []	
	for year, months in stance_stats.items():
		for month, polarities in months.items():
			for attitude_polarity, b_types in polarities.items():
				if len(b_types) > 0:
					for b_type, count in b_types.items():
						city_rows.append([b_type, attitude_polarity, count, month, year, city])

	stance_stats_rows.extend(city_rows)
					
#print(stance_stats_rows)

stance_stats_column_names = ["Belief Type", "Attitude", "Count", "Month", "Year", "City"]

stance_stats_df = pd.DataFrame(stance_stats_rows, columns=stance_stats_column_names)
stance_stats_df.to_excel("./" + today + "_" + state_abbrev + "_stance_belief_w_attitude_stats.xlsx", index=False)

