// Adds click and press event listener to button with id "send-btn" and retrieves value from input field with id "user-input"
document.getElementById("send-btn").addEventListener("click", handleSend);

document.getElementById("user-input").addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
        handleSend();
    }
});

// Function to automatically scroll the chat window to the bottom
function scrollToBottom() {
    const chatWindow = document.getElementById("chat-window"); // Target the chat window
    chatWindow.scrollTop = chatWindow.scrollHeight; // Set the scroll position to the bottom
}

// Main message handling function
function handleSend() {
    const userInput = document.getElementById("user-input").value;
    const imageFile = document.getElementById("image-upload") ? document.getElementById("image-upload").files[0] : null;

    if (!userInput && !imageFile) {
        return;
    }

    // Display user message in chat window
    const userMessageDiv = document.createElement("div"); // Create new <div> element for user message
    userMessageDiv.textContent = userInput; // Set the user's message as the content
    userMessageDiv.classList.add("message", "user-message"); // Add the classes for styling
    document.getElementById("messages").appendChild(userMessageDiv); // Add user's message to chat window

    // Automatically scroll to the bottom after appending the user's message
    scrollToBottom();

    // Create QueryData to send text and image
    const queryData = new QueryData();
    queryData.append("query", userInput);
    if (imageFile) {
        queryData.append("image", imageFile);
    } 

    // Send user's message as JSON to Backend (Flask /chat endpoint)
    fetch("/process_query", {
        method: "POST",
        body: queryData 
    })
        //headers: {
        //#    "Content-Type": "application/json"
        //},
        //body: JSON.stringify({ query: userInput })  // Ensure "query" is used
    //})
    .then(response => response.json())
    .then(data => {
        const botResponseDiv = document.createElement("div");
        botResponseDiv.textContent = data.response || "No response";  // Adjust to match Flask return
        botResponseDiv.classList.add("message", "bot-message");
        document.getElementById("messages").appendChild(botResponseDiv);
        scrollToBottom();
    })    
    .catch(error => {
        const errorDiv = document.createElement("div");
        errorDiv.textContent = "Error: Unable to connect to the server.";
        errorDiv.classList.add("message", "bot-message"); // Add styling class for bot error messages
        document.getElementById("messages").appendChild(errorDiv);

        // Automatically scroll to the bottom after appending the error message
        scrollToBottom();
    });

    // Clear input field after sending message
    document.getElementById("user-input").value = "";
    if (imageFile) {
        document.getElementById("image-upload").value = "";
    }
}
