import base64
import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from typing import List, Optional, Union, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

# Initialize the model
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))

# General system prompt to handle mixed input types
system_prompt = SystemMessage(
    content=(
        "You are an advanced assistant capable of handling user queries with text, images, or both. "
        "For text inputs, provide a detailed, context-aware response. "
        "For image inputs, describe the image or answer specific visual questions. "
        "For combinations, integrate both to deliver a comprehensive reply."
    )
)

# Define a tool for image description
@tool
def describe_image(description: str) -> None:
    """Provide a description of the given image."""
    pass

# Bind the tool to the model
model_with_tools = model.bind_tools([describe_image])

# Function to encode images as base64
def create_message(
    text: Optional[str] = None,
    image_path: Optional[str] = None
) -> List[Dict]:
    """
    Create a structured message for the model, handling text, images, or both.
    """
    content = []
    if text:
        logging.info(f"Adding text to message: {text}")
        content.append({"type": "text", "text": text})
    if image_path:
        logging.info(f"Adding image to message from path: {image_path}")
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}})
            logging.debug("Image added successfully")
        except FileNotFoundError:
            logging.error(f"Image file not found: {image_path}")
            raise
        except Exception as e:
            logging.error(f"Error encoding image: {e}")
            raise
    return content


def process_input(
    text: Optional[str] = None,
    image_path: Optional[str] = None
) -> str:
    """
    Process the user input (text, image, or both) and invoke the model.
    """
    try:
        # Validate input
        if not text and not image_path:
            logging.warning("No input provided. Returning error message.")
            return "Please provide either text, an image, or both."

        # Create the dynamic message
        logging.info("Creating message for the model")
        message_content = create_message(text=text, image_path=image_path)
        logging.debug(f"Constructed message content: {message_content}")

        # Create a HumanMessage with the content
        message = HumanMessage(content=message_content)
        logging.debug(f"Constructed HumanMessage: {message}")

        # Invoke the model with the message
        logging.info("Invoking the model")
        response = model_with_tools.invoke([system_prompt, message])
        logging.debug(f"Raw model response: {response}")

        # Check if the response contains content
        if hasattr(response, 'content') and response.content:
            logging.info(f"Model response content: {response.content}")
            return response.content
        
        # Check for tool call output
        tool_calls = response.additional_kwargs.get('tool_calls', [])
        if tool_calls:
            logging.info(f"Tool call detected: {tool_calls}")
            for tool_call in tool_calls:
                if 'arguments' in tool_call['function']:
                    # Extract the tool's output from the arguments
                    tool_output = tool_call['function']['arguments']
                    logging.info(f"Tool output: {tool_output}")
                    return tool_output

        # If no content or tool call output is present
        logging.warning("Model response did not contain valid content or tool call output")
        return "No valid response received from the model."

    except Exception as e:
        logging.error(f"Error processing input: {e}")
        return f"An error occurred: {e}"


# Test cases
logging.info("Test case 1: Text input only")
print(process_input(text="What is the capital of France?"))

logging.info("Test case 2: Image input only")
print(process_input(image_path="static/uploads/obst_test.jpg"))

logging.info("Test case 3: Text and image combination")
print(process_input(
    text="Describe this image and explain its artistic style.",
    image_path="static/uploads/obst_test.jpg"
))
