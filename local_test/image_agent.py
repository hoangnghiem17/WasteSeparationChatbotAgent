import base64
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from typing import Literal

# Load environment variables
load_dotenv()

# Initialize the model
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Define the tool for image description
@tool
def describe_image(description: str) -> None:
    """Provide a description of the given image."""
    pass

# Bind the tool to the model
model_with_tools = model.bind_tools([describe_image])

# Path to your local image file
local_image_path = "static/uploads/obst_test.jpg"

# Read and encode the local image file
with open(local_image_path, "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode("utf-8")

# Create the message for image description
message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe this image in detail."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
    ],
)

# Invoke the model with the message
response = model_with_tools.invoke([message])
print(response.tool_calls)
