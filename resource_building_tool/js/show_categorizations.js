let specificCategories = JSON.parse(localStorage.getItem("specificCategories"));
let triggerCategories = JSON.parse(localStorage.getItem("triggerCategories"));
let contentCategories = JSON.parse(localStorage.getItem("contentCategories"));
console.log("Below is the trigger categories json")
console.log(triggerCategories)
console.log("Below is the content categories json")
console.log(contentCategories)


for (let category in specificCategories) {
	let categoryListContainer = document.createElement("div")
	categoryListContainer.innerText = category.toUpperCase()

	let categoryList = document.createElement("ul")

	for (let word of specificCategories[category]) {
		let listItem = document.createElement("li") 

		listItem.innerText = word

		categoryList.appendChild(listItem)
	}

	categoryListContainer.appendChild(categoryList)
	categoryListContainer.style = "float: left; border: 1px solid black; margin: 10px; padding: 8px;"

	body.appendChild(categoryListContainer)
	
}
