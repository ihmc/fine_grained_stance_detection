let pairsWithSents = {} 
let pairCount = 0 
let pairsArray = []

if (localStorage.getItem("pairCount") === null) {
	//pairsWithSents = isis_trig_content_pairs
	pairsWithSents = afghanistan_withdrawal_trig_content_pairs["trig_content_top_pairs"]

	for (let trigger in pairsWithSents){
		contentWords = pairsWithSents[trigger]["content_words"]

		for (let word in contentWords) {
			pairsArray.push([trigger, word, contentWords[word]["sentences"]])

		}
	}
} else {
	if (parseInt(JSON.parse(localStorage.getItem("pairCount"))["pairCount"]) != JSON.parse(localStorage.getItem("pairsArray")).length) {
		pairsWithSents = JSON.parse(localStorage.getItem("pairsWithSents"))

		pairCount = parseInt(JSON.parse(localStorage.getItem("pairCount"))["pairCount"])
		pairsArray = JSON.parse(localStorage.getItem("pairsArray"))
	}
}

console.log(pairsWithSents)

let container = document.createElement("div")
container.id = "container"

//body.style.width = "1000px"

let instructions = document.createElement("div")
instructions.id = "instructions"
instructions.innerText = `INSTRUCTIONS:
Please categorize the following two-word (trigger-content) pairs into:
 - LOSE (e.g., a word like forfeit, relinquish)
 - GAIN (e.g., a word like win, possess).
 - BOTH
 - NEITHER
Representative sentences for the trigger-content pair are shown below.`

container.appendChild(instructions)

let pairContainer =	document.createElement('div')
let trigContPair = document.createElement('div')
let trigger = document.createElement('span')
let content = document.createElement('span')
let exampleSentences = document.createElement('div')
let sentList = document.createElement('ul')

exampleSentences.appendChild(sentList)

pairContainer.id = "pairContainer"
trigContPair.id = "trigContPair"
trigger.id = "trigger"
content.id = "content"
exampleSentences.id = "exampleSentences"

exampleSentences.style = "border: 1px solid black; padding: 10px;"

//pairContainer.innerText = "Trigger-Content pair: "
trigContPair.innerText = "Trigger-Content pair: "

// When page loads show first trigger content pair
trigger.innerText = pairsArray[pairCount][0]
content.innerText = " " + pairsArray[pairCount][1]
let sentences = pairsArray[pairCount][2]

for (sent of sentences){
	let sentence = document.createElement('li')
	sentence.innerText = sent
	sentence.classList.add("sent")

	sentList.appendChild(sentence)
	//exampleSentences.innerHTML += sent + "<br>"
}

trigContPair.appendChild(trigger)
trigContPair.appendChild(content)
pairContainer.appendChild(trigContPair)
pairContainer.appendChild(buildCategorizeForm(0))
pairContainer.appendChild(exampleSentences)

container.appendChild(pairContainer)

body.appendChild(container)

/*
for (let trigger in pairsWithSents){
	let body = document.getElementById("body")
	//console.log(document.body)

	contentWords = pairsWithSents[trigger]["content_words"]

	
	

	for (let word in contentWords) {
		pairCount++

		let pairContainer =	document.createElement('div')
		pairContainer.id = "pairContainer" + pairCount
		pairContainer.className = "pairContainer"

		let trigContPair = document.createElement('div')
		trigContPair.id = "trigContPair" + pairCount
		trigContPair.className = "trigContPair"
		trigContPair.innerHTML = trigger + " " + word


		let exampleSentences = document.createElement('div')
		exampleSentences.id = "exampleSentences" + pairCount
		exampleSentences.className = "exampleSentences"

		for (sent of contentWords[word]["sentences"]){
			exampleSentences.innerHTML += sent + "<br>"
		}	

		//contentWords[word]["sentences"].forEach(sent => {
		//	exampleSentences.text += sent + "\n"
		//})

		pairContainer.appendChild(trigContPair)
		pairContainer.appendChild(buildCategorizeForm(pairCount))
		pairContainer.appendChild(exampleSentences)

		container.appendChild(pairContainer)
	}

	console.log(pairsWithSents[trigger])

}
*/

function buildCategorizeForm(pPairCount) {
	let form = document.createElement('form')

	let loseRadio = document.createElement('input')
	let gainRadio = document.createElement('input')
	let bothRadio = document.createElement('input')
	let neitherRadio = document.createElement('input')

	let submit = document.createElement('input')

	let loseLabel = document.createElement('label')
	let gainLabel = document.createElement('label')
	let bothLabel = document.createElement('label')
	let neitherLabel = document.createElement('label')



	form.setAttribute("onSubmit", "submitAndLoadNext(event);")

	loseLabel.innerText = "Lose"
	gainLabel.innerText = "Gain"
	bothLabel.innerText = "Both"
	neitherLabel.innerText = "Neither"

	loseLabel.classList.add("radioCategory")
	gainLabel.classList.add("radioCategory")
	bothLabel.classList.add("radioCategory")
	neitherLabel.classList.add("radioCategory")
	

	loseRadio.type = "radio"
	gainRadio.type = "radio"
	bothRadio.type = "radio"
	neitherRadio.type = "radio"

	submit.type = "submit"
	submit.value = "Submit"


	form.id = "categorize" + pPairCount
	loseRadio.name = "categorize" + pPairCount
	gainRadio.name = "categorize" + pPairCount
	bothRadio.name = "categorize" + pPairCount
	neitherRadio.name = "categorize" + pPairCount

	loseRadio.value = "lose"
	gainRadio.value = "gain"
	bothRadio.value = "both"
	neitherRadio.value = "neither"


	loseLabel.appendChild(loseRadio)
	gainLabel.appendChild(gainRadio)
	bothLabel.appendChild(bothRadio)
	neitherLabel.appendChild(neitherRadio)

	form.appendChild(loseLabel)
	form.appendChild(gainLabel)
	form.appendChild(bothLabel)
	form.appendChild(neitherLabel)

	form.appendChild(submit)

	return form
}

function submitAndLoadNext(e){
	e.preventDefault()

	//let trigger = document.getElementById('trigger')
	//let content = document.getElementById('content')

	

	let category = document.querySelector('input[name="categorize0"]:checked').value;
	
	pairsWithSents[trigger.innerText]["content_words"][content.innerText.trim()]["topLevelCat"] = category

	console.log(pairsWithSents)
	
	console.log("wee we submitted")

	//After the category has been set we need to populate the elements with the new pair's data

	document.querySelector('input[name="categorize0"]:checked').checked = false	


	pairCount++

	if (pairCount == pairsArray.length) {
		localStorage.setItem("categorizedPairs", JSON.stringify(pairsWithSents));
		localStorage.removeItem("userCategories")
		localStorage.removeItem("specificCategories")
		window.location = "user_categories.html"
		return
	}

	trigger.innerText = pairsArray[pairCount][0]
	content.innerText = " " + pairsArray[pairCount][1]
	let sentences = pairsArray[pairCount][2]

	//let exampleSentences = document.getElementById("exampleSentences")
	//exampleSentences.innerHTML = ""
	while (sentList.firstChild) {
		sentList.removeChild(sentList.firstChild)
	}

	for (sent of sentences){
		let sentence = document.createElement('li')
		sentence.innerText = sent
		sentence.classList.add("sent")

		sentList.appendChild(sentence)
		//exampleSentences.innerHTML += sent + "<br>"
	}

	return false
	
}

window.onbeforeunload = function(){
	localStorage.setItem("pairCount", JSON.stringify({"pairCount" : pairCount}));
	localStorage.setItem("pairsWithSents", JSON.stringify(pairsWithSents));
	localStorage.setItem("pairsArray", JSON.stringify(pairsArray));
}
