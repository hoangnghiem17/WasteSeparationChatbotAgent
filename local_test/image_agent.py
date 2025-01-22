import base64
import os
import logging
from typing import Optional, Union, List, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Helper function: Encode image to Base64
def encode_image_to_base64(image_path: str) -> str:
    """
    Encodes a local image file to a Base64 string for sending image data via API.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64-encoded image string.
    """
    logging.info(f"Encoding image at path: {image_path}")
    if not os.path.isfile(image_path):
        logging.error(f"Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        logging.info("Image successfully encoded to Base64.")
        return encoded_image
    except Exception as e:
        logging.error(f"Error encoding image: {e}")
        raise e

# Helper function: Prepare image message
def prepare_image_message(user_input: str, image_path: str) -> List[dict]:
    """
    Prepares a message with text and encoded image data into message structure for structured LLM input.

    Args:
        user_input (str): The user's input text.
        image_path (str): Path to the image file.

    Returns:
        list[dict]: A message containing text and Base64-encoded image data.
    """
    logging.info("Preparing image message with user input and image data.")
    image_data = encode_image_to_base64(image_path)
    return [
        {"type": "text", "text": user_input},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
    ]

# Helper function: Update agent outcome
def update_outcome(state: 'AgentState', outcome: Union[str, dict]):
    """
    Updates the agent's outcome and logs the result.

    Args:
        state (AgentState): The current state of the agent.
        outcome (Union[str, dict]): The outcome to update in the agent state.

    Returns:
        None
    """
    state["agent_outcome"] = outcome
    logging.info(f"Updated agent outcome: {outcome}")

# Tool for adding two numbers
@tool
def add(a: float, b: float) -> float:
    """
    Takes in two numbers and adds them.

    Args:
        a (float): The first number.
        b (float): The second number.

    Returns:
        float: The sum of the two numbers.
    """
    logging.info(f"Executing add tool with arguments: a={a}, b={b}")
    return a + b

# Initialize the model with tools
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))
model_with_tools = llm.bind_tools([add])
logging.info("Tools successfully bound to the model.")

# Define AgentState
class AgentState(TypedDict):
    """
    Represents the state of the agent during execution to centralize all data-related into a single session for workflow transitions.
    """
    input: str  # User input to the agent
    chat_history: List[BaseMessage]  # Messages exchanged in the session
    agent_outcome: Union[str, dict, None]  # Final response or None if still running
    intermediate_steps: List[tuple]  # Logs of reasoning and tool actions

# Initialize AgentState
def initialize_state(user_input: str, image_path: Optional[str] = None) -> AgentState:
    """
    Initializes the AgentState with the user's input and optional image data.

    Args:
        user_input (str): The user's input text.
        image_path (Optional[str]): Path to an optional image file.

    Returns:
        AgentState: The initialized state of the agent.
    """
    logging.info(f"Initializing AgentState for input: {user_input}")
    chat_history = [HumanMessage(content=user_input)]

    if image_path:
        try:
            chat_history.append(HumanMessage(content=prepare_image_message(user_input, image_path)))
            logging.info("Image data successfully attached to chat history.")
        except FileNotFoundError as e:
            logging.error(f"Image not found: {e}")
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
    
# Invoke the model
def invoke_model(state: AgentState):
    """
    Invokes the model using the conversation history stored in the state. Centralizes model interaction and updates chat history.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AIMessage or None: The model's response or None if an error occurred.
    """
    logging.info("Invoking the model with the current conversation history.")
    try:
        response = model_with_tools.invoke(state["chat_history"])
        logging.info(f"Model response received: {response}")
        state["chat_history"].append(response)
        return response
    except Exception as e:
        logging.error(f"Error invoking model: {e}")
        update_outcome(state, {"error": str(e)})
        return None

# Workflow decision logic
def should_continue(state: AgentState) -> str:
    """
    Determines whether the agent should continue execution or terminate based on 'agent_outcome'.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        str: "CONTINUE" if the agent should execute more tools, "END" otherwise.
    """
    if state["agent_outcome"] is not None:
        logging.info("Agent finished execution.")
        return "END"
    else:
        logging.info("Agent continuing execution.")
        return "CONTINUE"

# Node: Reasoning (agent)
def run_tool_agent(state: AgentState) -> AgentState:
    """
    Handles reasoning to determine if tool needs to be invoked or if workflow is complete.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state of the agent after reasoning.
    """
    logging.info("Running the agent reasoning step.")
    response = invoke_model(state)

    if response is None:
        logging.error("No response from model. Terminating workflow.")
        update_outcome(state, {"error": "No response from model."})
        return state

    if hasattr(response, "tool_calls") and response.tool_calls:
        logging.info("Tool call detected. Transitioning to action node.")
    else:
        logging.info("No tool calls detected. Marking as complete.")
        update_outcome(state, response.content if response.content else "No valid response received.")

    return state

# Node: Tool execution (action)
def execute_tools(state: AgentState) -> AgentState:
    """
    Executes the tools requested by the agent and updates the state.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state of the agent after tool execution.
    """
    logging.info("Executing tools based on the model's request.")
    response = state["chat_history"][-1]
    if hasattr(response, "tool_calls"):
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            if tool_name == "add":
                try:
                    result = add.invoke(input=tool_args)
                    logging.info(f"Tool '{tool_name}' executed. Result: {result}")
                    tool_message = {
                        "role": "tool",
                        "name": tool_name,
                        "content": str(result),
                        "tool_call_id": tool_call_id,
                    }
                    state["chat_history"].append(tool_message)
                    update_outcome(state, f"The result is {result}.")
                except Exception as e:
                    logging.error(f"Error executing tool '{tool_name}': {e}")
                    update_outcome(state, {"error": str(e)})
    else:
        logging.warning("No tool calls found. Skipping action node.")
    return state

# Build the workflow graph - defining nodes for reasoning (agent) and tool execution (action) - edges for transitions based on should_continue
workflow = StateGraph(AgentState)
workflow.add_node("agent", run_tool_agent)
workflow.add_node("action", execute_tools)
workflow.set_entry_point("agent")
workflow.add_edge("action", "agent")
workflow.add_conditional_edges("agent", should_continue, {"CONTINUE": "action", "END": END})
app = workflow.compile()

# Process user input
def process_input(
    text: Optional[str] = None,
    image_path: Optional[str] = None,
    a: Optional[float] = None,
    b: Optional[float] = None,
) -> Union[str, dict]:
    """
    Processes user input using the graph-based workflow - main entry point for user interaction.

    Args:
        text (Optional[str]): The user's input text.
        image_path (Optional[str]): Path to an optional image file.
        a (Optional[float]): The first number for numerical operations.
        b (Optional[float]): The second number for numerical operations.

    Returns:
        Union[str, dict]: The final outcome of the agent's workflow execution,
        or an error message if the workflow fails.
    """
    logging.info(f"Processing input with parameters: text={text}, image_path={image_path}, a={a}, b={b}")
    if a is not None and b is not None:
        user_input = f"Add {a} and {b}"
    elif text and image_path:
        user_input = f"{text} (with image)"
    elif text:
        user_input = text
    elif image_path:
        user_input = "Image input provided."
    else:
        return "Please provide valid input."

    state = initialize_state(user_input, image_path)

    try:
        app.invoke(state)
        logging.info("Workflow execution completed.")
        return state["agent_outcome"]
    except Exception as e:
        logging.error(f"Error running workflow: {e}")
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
    print(process_input(
        text="Was ist auf dem Bild zu sehen?",
        image_path="static/uploads/obst_test.jpg"
    ))
