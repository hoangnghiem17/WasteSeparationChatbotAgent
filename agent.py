import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import TypedDict, Annotated, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.messages import BaseMessage
import operator
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from dotenv import load_dotenv
import io
from PIL import Image
import base64

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Tool to encode images
@tool
def encode_image_tool(image_file):
    """
    Encode an uploaded image to base64 format.

    Args:
        image_file: Uploaded image file content (binary).

    Returns:
        str: Base64 encoded string of the image or error message.
    """
    logging.info("Encoding image to base64.")
    try:
        # Open the image from binary content
        image = Image.open(io.BytesIO(image_file))
        img_buffer = io.BytesIO()
        image.save(img_buffer, format=image.format)
        base64_image = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        return base64_image
    except Exception as e:
        logging.error(f"Error encoding image: {e}", exc_info=True)
        return "Failed to process the image."

# Toolkit setup
toolkit = [encode_image_tool]

# Initializing the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=openai_api_key
)

# System prompt
system_prompt = """
You are an assistant specialized in processing images provided by the user. If the user uploads an image, encode it into base64 format using the `encode_image_tool`. Provide the result to the user.

Do not answer questions unrelated to image processing. If you cannot process the request, explain why and guide the user to rephrase or provide an image.
"""

# Customize the LLM's interaction with tools
tool_calling_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# Create an agent
tool_runnable = create_tool_calling_agent(llm, toolkit, tool_calling_prompt)
logging.info("Tool calling agent created.")

# Define the agent's reasoning process
def run_tool_agent(state):
    """
    Executes the tool-calling agent to process user input and invoke the appropriate tool.

    Args:
        state (dict): A dictionary containing the agent's state, including user input and chat history.

    Returns:
        dict: Updated state with the agent's outcome after invoking tools.
    """
    agent_outcome = tool_runnable.invoke(state)
    logging.info(f"Agent outcome: {agent_outcome}")
    return {"agent_outcome": agent_outcome}

# Create the Tool Executor
tool_executor = ToolExecutor(toolkit)
logging.info("Tool executor initialized.")

# Define tool execution logic
def execute_tools(state):
    """
    Executes tools selected by the agent and processes their outputs.

    Args:
        state (dict): A dictionary containing the agent's outcome and tool-related details.

    Returns:
        dict: Updated state with intermediate steps (tool outputs) for each executed tool.
    """
    agent_action = state['agent_outcome']
    if type(agent_action) is not list:
        agent_action = [agent_action]

    steps = []
    for action in agent_action:
        logging.info(f"Executing tool: {action.tool} with input: {action.tool_input}")
        output = tool_executor.invoke(action)
        logging.info(f"Tool result: {output}")
        steps.append((action, str(output)))
    
    return {"intermediate_steps": steps}

# Define agent state structure
class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, list, ToolAgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[Union[tuple[AgentAction, str], tuple[ToolAgentAction, str]]], operator.add]

# Define conditional edge logic
def should_continue(data):
    """
    Determines whether the agent should continue execution or terminate.

    Args:
        data (dict): State data containing the agent's outcome.

    Returns:
        str: "CONTINUE" if the agent should execute more tools, "END" otherwise.
    """
    if isinstance(data['agent_outcome'], AgentFinish):
        logging.info("Agent finished execution.")
        return "END"
    else:
        logging.info("Agent continuing execution.")
        return "CONTINUE"

# Build the workflow graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", run_tool_agent)
workflow.add_node("action", execute_tools)

workflow.set_entry_point("agent")
workflow.add_edge("action", "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "CONTINUE": "action",
        "END": END
    }
)

# Compile the workflow
app = workflow.compile()
logging.info("Workflow compiled successfully.")

# Run the workflow for a sample input
# Run the workflow for a sample input
if __name__ == "__main__":
    # Read the image file as binary
    with open("static/uploads/obst_test.jpg", "rb") as image_file:
        image_data = image_file.read()  # Ensure the data is read as bytes
        
        # Pass the binary image data correctly
        inputs = {
            "input": {"image_file": image_data},  # Provide binary image data
            "chat_history": []  # Include any chat history if necessary
        }
        config = {"configurable": {"thread_id": "1"}}

        # Process the input through the workflow
        for s in app.stream(inputs, config=config):
            output = list(s.values())[0]
            print("------------------------------------------------------")
            print(output)
