from datetime import datetime

sms_received_count = 0
linkedin_received_count = 0
email_received_count = 0
sms_processed_count = 0
linkedin_processed_count = 0
email_processed_count = 0

health_json = { 
	"name": "Ask Detection Component",
	"timestamp": "",
	"errors": [], 
	"message_counts": {},
	"ta1_or_ta2": "TA2",
	"other": {}
}
def getHealth():
	health_json["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	getMessageCounts()
	return health_json

def addError(message_id, error_message):
	health_json['errors'].append({
		"message_id": message_id, 
		"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
		"error_message": error_message,
	})

def getMessageCounts():
	message_counts = health_json["message_counts"]
	message_counts["sms_received_message_count"] = sms_received_count
	message_counts["linkedin_received_message_count"] = linkedin_received_count
	message_counts["email_received_message_count"] = email_received_count
	message_counts["sms_processed_message_count"] = sms_processed_count
	message_counts["linkedin_processed_message_count"] = linkedin_processed_count
	message_counts["email_processed_message_count"] = email_processed_count

def incrementSMSReceivedCount():
	sms_received_count += 1

def incrementLinkedinReceivedCount():
	linkedin_received_count += 1

def incrementEmailReceivedCount():
	email_received_count += 1

def incrementSMSProcessedCount():
	sms_processed_count += 1

def incrementLinkedinProcessedCount():
	linkedin_processed_count += 1

def incrementEmailProcessedCount():
	email_processed_count += 1
