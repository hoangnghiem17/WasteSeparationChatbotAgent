// Reset session on page load or refresh
window.onload = function() {
    fetch('/reset', { method: 'POST' });
};

// Existing event listeners for send button and enter key
document.getElementById("send-btn").addEventListener("click", handleSend);
document.getElementById("user-input").addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
        handleSend();
    }
});

// Listen for image upload (change event on file input and display file name
document.getElementById("image-upload").addEventListener("change", handleFileSelect);

function handleFileSelect(event) {
    const fileNameDisplay = document.getElementById("file-name");
    const file = event.target.files[0];
    if (file) {
        fileNameDisplay.textContent = file.name;  // Show the uploaded file name
    } else {
        fileNameDisplay.textContent = "Keine Datei ausgewählt";
    }
}

// Function to automatically scroll the chat window to the bottom
function scrollToBottom() {
    const chatWindow = document.getElementById("chat-window");
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Show typing indicator in chat (inline spinner)
function showTypingIndicator() {
    const typingIndicator = document.createElement("div");
    typingIndicator.classList.add("message", "bot-message", "typing-indicator");
    typingIndicator.setAttribute("id", "loading");
    typingIndicator.innerHTML = `
        <img src="static/loading.gif" alt="Loading...">
        <span>Assistent tippt...</span>
    `;
    document.getElementById("messages").appendChild(typingIndicator);
    scrollToBottom();
}

// Remove typing indicator from chat once response arrives
function hideTypingIndicator() {
    const typingElement = document.getElementById("loading");
    if (typingElement) {
        typingElement.remove();
    }
}

// Main message handling function
function handleSend() {
    const userInput = document.getElementById("user-input").value;
    const imageFile = document.getElementById("image-upload") ? document.getElementById("image-upload").files[0] : null;

    if (!userInput && !imageFile) {
        return;
    }

    // Display user text message in chat window
    if (userInput) {
        addMessage(userInput, "user-message");
    }

    // Display image preview in chat window
    if (imageFile) {
        displayImagePreview(imageFile);
    }

    // Bundle text and image for backend submission
    const queryData = new FormData();
    queryData.append("query", userInput);
    if (imageFile) {
        queryData.append("image", imageFile);
    }

    // Show typing indicator before sending the request
    showTypingIndicator();

    // Send user's message and image to Flask backend
    fetch("/process_query", {
        method: "POST",
        body: queryData
    })
    .then(response => response.json())
    .then(data => {
        hideTypingIndicator();
        
        // Display the bot's response
        const botResponseDiv = document.createElement("div");
        botResponseDiv.textContent = data.response || "No response";
        botResponseDiv.classList.add("message", "bot-message");
        document.getElementById("messages").appendChild(botResponseDiv);
    })
    .catch(error => {
        hideTypingIndicator();
        
        // Display error message in the chat
        const errorDiv = document.createElement("div");
        errorDiv.textContent = "Error: Unable to connect to the server.";
        errorDiv.classList.add("message", "bot-message");
        document.getElementById("messages").appendChild(errorDiv);
    })
    .finally(() => {
        scrollToBottom();
    });

    // Clear input and file fields after sending message
    document.getElementById("user-input").value = "";
    if (imageFile) {
        document.getElementById("image-upload").value = "";
        document.getElementById("file-name").textContent = "Keine Datei ausgewählt";
    }
}

// Utility function to add text messages to chat window
function addMessage(content, className) {
    const messageDiv = document.createElement("div");
    messageDiv.textContent = content;
    messageDiv.classList.add("message", className);
    document.getElementById("messages").appendChild(messageDiv);
    scrollToBottom();
}

// Display image preview in chat window
function displayImagePreview(file) {
    const reader = new FileReader();
    reader.onload = function(event) {
        const img = document.createElement("img");
        img.src = event.target.result;
        img.alt = "Uploaded Image";
        img.classList.add("uploaded-image");

        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", "user-message");
        messageDiv.appendChild(img);
        document.getElementById("messages").appendChild(messageDiv);
        scrollToBottom();
    };
    reader.readAsDataURL(file);
}
