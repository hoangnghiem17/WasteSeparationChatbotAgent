import base64
import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain.prompts import PromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain_core.tools import tool
from typing import List, Union, Optional
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

# Define AgentState - representing state of agent during execution
class AgentState(BaseModel):
    input: str
    chat_history: List[BaseMessage] = Field(default_factory=list)
    agent_outcome: Optional[Union[str, dict]] = None
    intermediate_steps: List = Field(default_factory=list)

# Initialize the model
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Define the system prompt
system_prompt = PromptTemplate(
    input_variables=["input", "chat_history", "agent_scratchpad"],
    template=(
        """
        You are an advanced assistant capable of performing mathematical operations, such as adding two numbers. 
        You can also handle user queries with text and images.
        
        Input: {input}
        Chat History: {chat_history}
        Reasoning and Actions: {agent_scratchpad}
        """
    )
)

# Define a tool for adding two numbers
@tool
def add(a: float, b: float) -> float:
    """Takes in two numbers and adds them."""
    return a + b

# Create the tool-calling agent
toolkit = [add]
tool_runnable = create_tool_calling_agent(llm, toolkit, system_prompt)
logging.info("Tool calling agent created.")

# Encode images into base64-string for LLM input
def encode_image(image_path: str) -> str:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Initialize agent state instance using provided user input
def initialize_state(user_input: str) -> AgentState:
    return AgentState(input=user_input)

# Iterates through agent_outcome actions and manually invokes tools
def execute_tools(state: AgentState) -> AgentState:
    agent_actions = state.agent_outcome
    if not isinstance(agent_actions, list):
        agent_actions = [agent_actions]

    intermediate_steps = []
    for action in agent_actions:
        if hasattr(action, "tool"):
            tool_name = action.tool
            tool = next((t for t in toolkit if t.name == tool_name), None)
            if tool:
                logging.info(f"Executing tool: {tool_name} with input: {action.tool_input}")
                tool_result = tool.invoke(input=action.tool_input)
                logging.info(f"Tool result: {tool_result}")
                intermediate_steps.append((action, str(tool_result)))
            else:
                logging.warning(f"Tool '{tool_name}' not found.")
    state.intermediate_steps = intermediate_steps
    return state

# Run the tool agent by combining LangChain's tool_runnable agent with manual tool execution and updates agent state
def run_tool_agent(state: AgentState) -> AgentState:
    """
    Executes the tool-calling agent and invokes tools manually for actions.

    Args:
        state (AgentState): The agent's state containing input and actions.

    Returns:
        AgentState: Updated state with the agent's outcome and intermediate steps.
    """
    try:
        # Invoke the LangChain agent
        state_dict = state.model_dump()
        agent_outcome = tool_runnable.invoke(state_dict)
        state.agent_outcome = agent_outcome

        # Execute tools manually and capture intermediate steps
        state = execute_tools(state)
        return state
    except Exception as e:
        state.agent_outcome = {"error": str(e)}
        return state

# Processes user input based on type - initialize and execute agent state to produce a response
def process_input(
    text: Optional[str] = None, 
    image_path: Optional[str] = None, 
    a: Optional[float] = None, 
    b: Optional[float] = None
) -> Union[str, dict]:
    try:
        if text:
            user_input = text
        elif a is not None and b is not None:
            user_input = f"Add {a} and {b}"
        elif image_path:
            user_input = encode_image(image_path)
        else:
            return "Please provide valid input."

        state = initialize_state(user_input=user_input)
        return run_tool_agent(state).agent_outcome or "No valid response received."
    except Exception as e:
        logging.error(f"Error processing input: {e}")
        return {"error": str(e)}

# Test cases
logging.info("Test case 1: Add two numbers")
print(process_input(a=5, b=10))

logging.info("Test case 2: Text input")
print(process_input(text="What is the capital of France?"))

# Uncomment to test image input
# logging.info("Test case 3: Image input")
# print(process_input(image_path="static/uploads/obst_test.jpg"))
