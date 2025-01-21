from typing import Optional, Union
import base64
import os
import logging


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize the model
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Define a tool for adding two numbers
@tool
def add(a: float, b: float) -> float:
    """Takes in two numbers and adds them."""
    return a + b

# Bind tools directly to the language model
model_with_tools = llm.bind_tools([add])
logging.info("Tools successfully bound to the model.")

def encode_image_to_base64(image_path: str) -> str:
    """
    Encodes a local image file to a Base64 string for sending image data via API.
    
    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64-encoded image string.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logging.error(f"Error encoding image: {e}")
        raise e

def invoke_model(conversation_history):
    """
    Invokes the model with the given conversation history.
    """
    try:
        response = model_with_tools.invoke(conversation_history)
        logging.info(f"Model response: {response}")
        if isinstance(response, AIMessage):
            return response  
        else:
            logging.error("Unexpected response type from the model.")
            return {"error": "Unexpected response type from the model."}
    except Exception as e:
        logging.error(f"Error invoking model: {e}")
        return {"error": str(e)}

def handle_tool_call(response, conversation_history):
    """
    Processes tool calls dynamically if present in the response (requested by model).
    """
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            try:
                # Log the tool invocation
                logging.info(f"Tool '{tool_name}' invoked with arguments: {tool_args}")

                # Dynamically resolve the tool function by name
                tool_function = globals().get(tool_name)
                if not tool_function:
                    logging.error(f"Tool '{tool_name}' not found.")
                    return {"error": f"Tool '{tool_name}' is not implemented."}

                # Execute the tool and get the result
                result = tool_function.invoke(input=tool_args)
                logging.info(f"Tool '{tool_name}' result: {result}")

                # Add the result directly to conversation history as an assistant response
                conversation_history.append(HumanMessage(content=f"The result is {result}."))

                return f"The result is {result}."
            except Exception as e:
                logging.error(f"Error executing tool '{tool_name}': {e}")
                return {"error": f"Tool '{tool_name}' execution failed."}
    else:
        logging.warning("No tool calls found in the response.")
        return response.content if isinstance(response, AIMessage) else "No valid response received."


def process_input(
    text: Optional[str] = None,
    image_path: Optional[str] = None,
    a: Optional[float] = None,
    b: Optional[float] = None
) -> Union[str, dict]:
    """
    Handles text, numerical, and image inputs, and routes input to handling methods.
    """
    try:
        # Handle numerical inputs
        if a is not None and b is not None:
            user_input = f"Add {a} and {b}"
            conversation_history = [HumanMessage(content=user_input)]
            response = invoke_model(conversation_history)
            return handle_tool_call(response, conversation_history)

        # Handle text and image inputs
        if text or image_path:
            content = []

            # Add text if provided
            if text:
                content.append({"type": "text", "text": text})

            # Add image if provided
            if image_path:
                try:
                    image_data = encode_image_to_base64(image_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    })
                except FileNotFoundError as e:
                    logging.error(f"Error encoding image: {e}")
                    return {"error": str(e)}

            # Create a HumanMessage with combined content
            message = HumanMessage(content=content)
            response = invoke_model([message])
            return response.content if isinstance(response, AIMessage) else "No valid response received."

        # If no valid input provided
        return "Please provide valid input (text, image, or numbers)."

    except Exception as e:
        logging.error(f"Error processing input: {e}")
        return {"error": str(e)}


# Example test cases
if __name__ == "__main__":
    logging.info("Test case 1: Add two numbers")
    print(process_input(a=5, b=10))

    logging.info("Test case 2: Text input")
    print(process_input(text="What is the capital of France?"))

    logging.info("Test case 3: Image input")
    print(process_input(image_path="static/uploads/obst_test.jpg"))
    
    logging.info("Test case 4: Text + Image input")
    print(process_input(text="Describe in german you can see on the picture.",
                        image_path="static/uploads/obst_test.jpg")
    )
