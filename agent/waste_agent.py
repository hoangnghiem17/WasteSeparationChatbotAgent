import logging
from typing import  Union, TypedDict, Annotated, List
import operator

from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain.agents import create_tool_calling_agent
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentFinish
from langchain.agents.output_parsers.tools import ToolAgentAction
from dotenv import load_dotenv

from config.llm import llm
from config.prompt import system_prompt
from agent.query_classification import classify_query
from agent.retrieval import retrieve_chunks
from agent.geocoding import geocode_address
from agent.facility import find_closest_facility

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global variables
faiss_store_path = "faiss_store"
db_path = "database/waste_separation_frankfurt.db"

# Define Tools wrapped with LangChain decorators
@tool
def classify_query_tool(user_query: str = None, image_path: str = None) -> str:
    """
    Tool wrapper for classify_query function. Classifies the user query into predefined waste categories.

    Args:
        user_query (str, optional): The text input from the user.
        image_path (str, optional): Path to an image file for additional context.

    Returns:
        str: The classification category or 'fallback' if classification fails.
    """
    logging.info(f"Invoking classify_query with query: {user_query} and image: {image_path}")
    return classify_query(user_query, image_path)

@tool
def retrieve_chunks_tool(query: str, category: str, k: int = 2) -> List[str]:
    """
    Tool to retrieve the most relevant document chunks for a given query and category.

    Args:
        query (str): The user query.
        category (str): The waste category for filtering (e.g., "bio_waste").
        k (int): Number of top results to retrieve.

    Returns:
        List[str]: A list of relevant document chunks.
    """
    return retrieve_chunks(query, faiss_store_path, category, k)

@tool
def geocode_address_tool(address: str):
    """
    Takes in an address and geocodes it to return latitude and longitude.
    """
    lat, lon = geocode_address(address)
    if lat is not None and lon is not None:
        return {"latitude": lat, "longitude": lon}
    return "Geocoding failed."

@tool
def find_closest_facility_tool(user_coords: tuple):
    """
    Takes in the user coordinates and finds the closest recycling facility to the user's coordinates.
    """
    if not db_path:
        return {"error": "Database path is not set. Please check the configuration."}
    return find_closest_facility(user_coords, db_path)
    
# Create tool calling agent - llm with available tools and system prompt
tool_calling_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
toolkit = [classify_query_tool, retrieve_chunks_tool, geocode_address_tool, find_closest_facility_tool]
tool_executor = ToolExecutor(toolkit)

# Creates an agent that can reason and invoke tools dynamically
tool_runnable = create_tool_calling_agent(
    llm, toolkit, ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
)

# Agent Reasoning Process - determines what tools to call with what inputs
def run_tool_agent(state):
    """
    Executes the tool-calling agent to process the user's input and invoke the appropriate tools.

    Args:
        state (dict): A dictionary containing the agent state, including user input and chat history.

    Returns:
        dict: Updated state with the agent's outcome after invoking tools.
    """
    #logging.info(f"Agent reasoning started with input: {state['input']}")
    agent_outcome = tool_runnable.invoke(state)
    #logging.info(f"Agent reasoning completed. Outcome: {agent_outcome}")
    return {"agent_outcome": agent_outcome}

# Agent Execution Process - invokes selected tools, processes their outputs and updates state
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
        
        # Update tool input handling
        tool_input = action.tool_input
        if isinstance(tool_input, dict):
            tool_input["image_path"] = state["input"].get("image_path")  # Add image_path if available
        
        try:
            output = tool_executor.invoke(action)
            logging.info(f"Tool '{action.tool}' returned result: {output}")
            steps.append((action, str(output)))
        except Exception as e:
            logging.error(f"Error executing tool '{action.tool}': {e}")
            steps.append((action, "error"))

    return {"intermediate_steps": steps}

# Conditional Edge Logic - evaluates agent's output based on AgentFinish
def should_continue(data):
    """
    Determines whether the agent should continue execution or terminate.

    Args:
        data (dict): State data containing the agent's outcome.

    Returns:
        str: "CONTINUE" if the agent should execute more tools, "END" otherwise.
    """
    if isinstance(data['agent_outcome'], AgentFinish):
        logging.info("Agent has finished execution.")
        return "END"
    else:
        logging.info("Agent reasoning requires further tool execution.")
        return "CONTINUE"

# Define Agent State
class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, list, ToolAgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[Union[tuple[AgentAction, str], tuple[ToolAgentAction, str]]], operator.add]
    
# Build the Graph - represents workflow logic as directed graph (agent=reasoning; action=execution)
workflow = StateGraph(AgentState)

# Agent for reasoning, action for tool execution
workflow.add_node("agent", run_tool_agent) 
workflow.add_node("action", execute_tools) 

# Define workflow
workflow.set_entry_point("agent")
workflow.add_edge('action', 'agent')

# Determine to continue/terminate execution based on agent's output
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "CONTINUE": "action",
        "END": END
    }
)
agent_app = workflow.compile()
