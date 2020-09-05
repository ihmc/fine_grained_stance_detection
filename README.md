1. In terminal/command prompt navigate into the corenlp directory
2. Assuming you have docker(https://docs.docker.com/get-docker/) installed on your machine run the following commands:
	docker build -t corenlp .
	docker run -d -p 9000:9000 corenlp

3. Now that corenlp is running on your system we need to start ask detection
4. Still in the terminal navigate into the ask_detection folder
5. Assuming you have python installed on your system run the following commands (Note that some systems make specific differentiarion between python 2 and python 3, so if the below commands don't work try using pip3 and python3 instead)
	pip install -r requirements.txt
	python app.py
6. Now that both ask detection and corenlp are running we should be able to send http requests to it. 
	a. Personally I used Postman (https://www.postman.com/downloads/). Inside postman you will need to make a POST request to the address panacea:ask_detection@localhost:5000/srl/sms
	b. Below the url you will need to select the body to send to ask detection. 
	c. In the dropdown menu to the right for what type of body it will be, select JSON
	d. The body will need to be just like the following:
		{
		  "text": "Hello Patrick,\nHope your day is going well. I will need you to make a wire transfer for me today. What would you need to get it done?\nThanks\nBenny Czarny",
		  "links": {
		  }
		}
	e. Now simply replace the text portion with whatever text you want to send and hit the send button.
	f. Ask detection can take a fair bit of time to process so you may not get results right away but eventually they should appear in the lower part of postman with the response. 

