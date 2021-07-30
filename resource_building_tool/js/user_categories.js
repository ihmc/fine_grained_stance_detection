localStorage.removeItem("pairCount")
localStorage.removeItem("pairsArray")
localStorage.removeItem("pairsWithSents")

let categorizedPairs = JSON.parse(localStorage.getItem("categorizedPairs"));

let userCategories = []



let elementsToAppendBody = []

let container = document.createElement("div")
container.id = "container"

let instructions = document.createElement("div")
instructions.id = "instructions"
instructions.innerText = "INSTRUCTIONS: Below are the LOSE/GAIN trigger-content pairs you selected. Check the blocks for those pairs that have an underlying \"meaning\" in common. Specify this meaning as a single action word, a \"meaning category\" that you choose. Type this category into the text box to be added to the drop down list, or select it if it is already added. (Other domains have used CONTROL, PROTECT, RESTRICT, REJECT, SPREAD_ILLNESS, and others.) For your convenience, once you have typed in an action word you will see semantically related terms. Pick the preferred term and click \"Add category\" when your selection is final. Continue to inspect and categorize trigger-content pairs until all have been processed."
elementsToAppendBody.push(instructions)

let categoryInput = document.createElement("input")
categoryInput.type = "text"
elementsToAppendBody.push(categoryInput)

let categoryInputButton = document.createElement("input")
categoryInputButton.type = "button"
categoryInputButton.id = "addCategory"
categoryInputButton.value = "Add Category"
categoryInputButton.setAttribute("onclick", "addCategory()")
elementsToAppendBody.push(categoryInputButton)

let categoryDropDown = document.createElement("select")
categoryDropDown.name = "categoryList"
categoryDropDown.id = "categoryDropDown"

let defaultValue = document.createElement("option")
defaultValue.value = ""
defaultValue.innerText = "Select Category"
defaultValue.selected = "selected"
categoryDropDown.appendChild(defaultValue)
elementsToAppendBody.push(categoryDropDown)


let categoryDropDownLabel = document.createElement("label")
categoryDropDownLabel.htmlFor = "categoryList"
elementsToAppendBody.push(categoryDropDownLabel)

let categoryAssignButton = document.createElement("input")
categoryAssignButton.type = "button"
categoryAssignButton.id = "assignPairs"
categoryAssignButton.value = "Assign Pairs to Category"
categoryAssignButton.setAttribute("onclick", "assignToCategory()")
elementsToAppendBody.push(categoryAssignButton)

let finishButton = document.createElement("input")
finishButton.type = "button"
finishButton.value = "Finish Assigning"
finishButton.setAttribute("onclick", "finishAssigning()")
elementsToAppendBody.push(finishButton)


let listsContainer = document.createElement("div")
listsContainer.id = "listsContainer"

let loseListContainer = document.createElement("div")
let gainListContainer = document.createElement("div")
let bothListContainer = document.createElement("div")
let neitherListContainer = document.createElement("div")
loseListContainer.innerText = "Lose:"
gainListContainer.innerText = "Gain:"
bothListContainer.innerText = "Both:"
neitherListContainer.innerText = "Neither:"
listsContainer.appendChild(loseListContainer)
listsContainer.appendChild(gainListContainer)
listsContainer.appendChild(bothListContainer )
listsContainer.appendChild(neitherListContainer)
elementsToAppendBody.push(listsContainer)
//elementsToAppendBody.push(loseListContainer)
//elementsToAppendBody.push(gainListContainer)
//elementsToAppendBody.push(bothListContainer )
//elementsToAppendBody.push(neitherListContainer)

loseListContainer.className = "categoryList"
gainListContainer.className = "categoryList"
bothListContainer.className = "categoryList"
neitherListContainer.className = "categoryList"

let loseList = document.createElement("ul")
let gainList = document.createElement("ul")
let bothList = document.createElement("ul")
let neitherList = document.createElement("ul")

loseList.id = "loseList"
gainList.id = "gainList"
bothList.id = "bothList"
neitherList.id = "neitherList"

loseListContainer.appendChild(loseList) 
gainListContainer.appendChild(gainList) 
bothListContainer.appendChild(bothList) 
neitherListContainer.appendChild(neitherList) 


for (let ele of elementsToAppendBody) {
	container.appendChild(ele)
}

body.appendChild(container)


if (localStorage.getItem("userCategories") !== null){
	userCategories = JSON.parse(localStorage.getItem("userCategories"))

	for (let category of userCategories){
		let newCategory = document.createElement("option")
		newCategory.innerText = category
		newCategory.value = category

		categoryDropDown.appendChild(newCategory)		
	}
}


let pairCount = 0
for (let trigger in categorizedPairs){
	contentWords = categorizedPairs[trigger]["content_words"]

	for (let word in contentWords) {
		let listItem = document.createElement("li") 
		let pairCheckbox = document.createElement("input")
		let tooltipText = document.createElement("span")

		tooltipText.classList.add("tooltiptext")
		for (let sentence of contentWords[word]["sentences"]) {
			
			tooltipText.innerText += sentence + "\n"
		}

		pairCheckbox.type = "checkbox"
		pairCheckbox.value = trigger + " " + word

		listItem.classList.add("tooltip")
		listItem.innerText = trigger + " " + word
		listItem.appendChild(tooltipText)

		listItem.setAttribute("title", contentWords[word]["sentences"].join("\n"))

		listItem.append(pairCheckbox)
		document.getElementById(contentWords[word]["topLevelCat"] + "List").appendChild(listItem)

	}
}

if (loseList.innerHTML.trim() == "") {
	loseListContainer.className += " empty"
}

if (gainList.innerHTML.trim() == "") {
	gainListContainer.className += " empty"
}

if (bothList.innerHTML.trim() == "") {
	bothListContainer.className += " empty"
}

if (neitherList.innerHTML.trim() == "") {
	neitherListContainer.className += " empty"
}


function addCategory(){
	let newCategory = document.createElement("option")
	newCategory.innerText = categoryInput.value.toLowerCase()
	newCategory.value = categoryInput.value.toLowerCase()

	let duplicates = document.querySelector("#categoryDropDown option[value='" + categoryInput.value.toLowerCase() + "']")

	if (duplicates) {
		return
	}

	userCategories.push(categoryInput.value.toLowerCase())
	categoryDropDown.appendChild(newCategory)
	
}

function assignToCategory() {
	//console.log(document.querySelectorAll("input:checked"))

	for (let assigned of document.querySelectorAll("input:checked")){
		if (!assigned.classList.contains("assigned")) {
			
			assigned.parentElement.style = "color: lightgray; opacity: 0.6;"
			triggerAndContent = assigned.value.split(' ')
			assignedTrigger = triggerAndContent[0]
			assignedContent = triggerAndContent[1]

			categorizedPairs[assignedTrigger]["content_words"][assignedContent]["specificCat"] = categoryDropDown.value

			assigned.disabled = true;

			assigned.classList.add("assigned")

			//localStorage.setItem("categorizedPairs", JSON.stringify(categorizedPairs));

		}

	}

	
}

function finishAssigning() {
	specificCategories = {}
	triggerCategories = {}
	contentCategories = {}
	

	for (let trigger in categorizedPairs){
		contentWords = categorizedPairs[trigger]["content_words"]

		let category
		for (let word in contentWords) {
			if ("specificCat" in categorizedPairs[trigger]["content_words"][word]) {
				category = categorizedPairs[trigger]["content_words"][word]["specificCat"]
			}
			else {
				categorizedPairs[trigger]["content_words"][word]["specificCat"] = "unassigned"
				category = "unassigned"
			}

			if (!(category in specificCategories)) {
				specificCategories[category] = [trigger, word]
				triggerCategories[category] = [trigger]
				contentCategories[category] = [word]
			} 
			else {
				if (!specificCategories[category].includes(trigger)) {
					specificCategories[category].push(trigger)
					triggerCategories[category].push(trigger)
				}

				if (!specificCategories[category].includes(word)) {
					specificCategories[category].push(word)
					contentCategories[category].push(word)
				}
			}
		}
	}

	console.log(specificCategories)

	localStorage.setItem("specificCategories", JSON.stringify(specificCategories));
	localStorage.setItem("triggerCategories", JSON.stringify(triggerCategories));
	localStorage.setItem("contentCategories", JSON.stringify(contentCategories));
	window.location = "show_categorizations.html"
}

window.onbeforeunload = function(){
	localStorage.setItem("userCategories", JSON.stringify(userCategories));
}
