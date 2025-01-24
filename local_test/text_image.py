import openai
import base64
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the OpenAI client with your API key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to encode an image to base64
def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your local image file
image_path = 'static/uploads/obst_test.jpg'
base64_image = encode_image_to_base64(image_path)

# Prepare the messages payload
messages = [
    {
        "role": "system",
        "content": "You are an assistant that classifies inputs into predefined categories."
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Classify the following input."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    }
]

# Send the request to the OpenAI API
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    max_tokens=100
)

# Output the assistant's response
print(response.choices[0].message.content)