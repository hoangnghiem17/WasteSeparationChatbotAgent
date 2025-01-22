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

from typing import TypedDict, Union, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    Represents the state of the agent during execution.
    """
    input: str  # The user's input to the agent
    chat_history: List[BaseMessage]  # All messages exchanged in the session
    agent_outcome: Union[str, None]  # The agent's final response or None if still running
    intermediate_steps: List[tuple]  # Logs of reasoning and tool actions

def initialize_state(user_input: str, image_path: Optional[str] = None) -> AgentState:
    """
    Initializes the AgentState with the user's input and optional image data.
    """
    chat_history = [HumanMessage(content=user_input)]

    if image_path:
        try:
            image_data = encode_image_to_base64(image_path)
            chat_history.append(HumanMessage(content=[
                {"type": "text", "text": user_input},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            ]))
        except FileNotFoundError as e:
            logging.error(f"Error encoding image: {e}")
            return {
                "input": user_input,
                "chat_history": [],
                "agent_outcome": {"error": f"Image not found: {e}"},
                "intermediate_steps": [],
            }

    return AgentState(
        input=user_input,
        chat_history=chat_history,
        agent_outcome=None,
        intermediate_steps=[],
    )

    
def invoke_model(state: AgentState):
    """
    Invokes the model using the conversation history stored in the state.
    Updates the state with the model's response.
    """
    try:
        response = model_with_tools.invoke(state["chat_history"])
        logging.info(f"Model response: {response}")
        
        # Append the response to the chat history
        state["chat_history"].append(response)
        
        return response
    except Exception as e:
        logging.error(f"Error invoking model: {e}")
        state["agent_outcome"] = {"error": str(e)}
        return None


def handle_tool_call(state: AgentState):
    """
    Processes tool calls from the model's response and updates the state.
    """
    response = state["chat_history"][-1]  # Get the most recent message from the model
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            # Handle specific tool calls
            if tool_name == "add":
                try:
                    logging.info(f"Executing tool '{tool_name}' with arguments: {tool_args}")
                    result = add.invoke(input=tool_args)
                    logging.info(f"Tool '{tool_name}' result: {result}")

                    # Log the intermediate step
                    state["intermediate_steps"].append((tool_call, result))

                    # Append the result as a message in chat history
                    state["chat_history"].append(HumanMessage(content=f"The result is {result}."))
                    
                    return result
                except Exception as e:
                    logging.error(f"Error executing tool '{tool_name}': {e}")
                    state["agent_outcome"] = {"error": f"Tool '{tool_name}' execution failed."}
                    return None
    else:
        logging.warning("No tool calls found in the response.")
        state["agent_outcome"] = response.content if isinstance(response, AIMessage) else "No valid response received."



def process_input(
    text: Optional[str] = None,
    image_path: Optional[str] = None,
    a: Optional[float] = None,
    b: Optional[float] = None,
) -> Union[str, dict]:
    """
    Processes the user's input and runs the workflow, including text and image handling.
    """
    try:
        # Determine input type
        if a is not None and b is not None:
            user_input = f"Add {a} and {b}"
            state = initialize_state(user_input)
        elif text and image_path:
            user_input = text
            state = initialize_state(user_input, image_path)
        elif text:
            user_input = text
            state = initialize_state(user_input)
        elif image_path:
            user_input = "Image input provided."
            state = initialize_state(user_input, image_path)
        else:
            return "Please provide valid input."

        # Invoke the model
        response = invoke_model(state)
        if not response:
            return state["agent_outcome"]

        # Handle tools or finalize response
        if hasattr(response, "tool_calls"):
            handle_tool_call(state)

        # Return the final outcome if present
        return state["agent_outcome"] or response.content
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
