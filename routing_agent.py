
import os
import sqlite3

import logging
import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langchain.agents import create_tool_calling_agent
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import TypedDict, Annotated, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.messages import BaseMessage
import operator
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global variables
openai_api_key = os.getenv("OPENAI_API_KEY")
db_path = os.getenv("DB_PATH")

# Geocoding function
def geocode_address(address):
    """
    Geocodes a user's address using the Nominatim API.

    Args:
        address (str): User's address as a string.

    Returns:
        Tuple[float, float]: A tuple containing the latitude and longitude of the address. Returns (None, None) if geocoding fails.
    """
    print(f"[INFO][geocode_address] Attempting to geocode address: {address}")
    url = "https://nominatim.openstreetmap.org/search"
    headers = {'User-Agent': 'WasteSeparationChatbot/1.0 (nghhoang@gmail.com)'}
    params = {'q': address, 'format': 'json', 'limit': 1}

    try:
        response = requests.get(url, headers=headers, params=params)
        print(f"[INFO][geocode_address] Received response with status code: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        if data:
            lat, lon = float(data[0]['lat']), float(data[0]['lon'])
            print(f"[INFO][geocode_address] Successfully geocoded to Latitude: {lat}, Longitude: {lon}")
            return lat, lon
    except requests.exceptions.RequestException as e:
        print(f"[ERROR][geocode_address] Error geocoding address '{address}': {e}")

    print("[WARNING][geocode_address] Geocoding failed, returning None, None")
    return None, None

def find_closest_facility(user_coords, db_path):
    """
    Finds the closest recycling facility to the user's coordinates using SQLite database and OSRM API.

    Args:
        user_coords (Tuple[float, float]): A tuple of the user's latitude and longitude.
        db_path (str): Path to the SQLite3 database file containing facility data.

    Returns:
        dict: Dictionary containing the closest facility's details, including:
              - name (str): Facility name.
              - address (str): Full address of the facility.
              - district (str): Facility's district.
              - distance_km (float): Distance to the facility in kilometers.
              - duration_min (float): Travel time to the facility in minutes.
              - opening_time (str): Facility's opening hours.
    """
    print(f"[INFO][find_closest_facility_with_route] Searching for the closest facility to {user_coords}")
    osrm_url_template = "http://router.project-osrm.org/route/v1/driving/{from_lon},{from_lat};{to_lon},{to_lat}"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name, street, zip, district, lat, long, opening_time FROM recyclinghof")
        facilities = cursor.fetchall()

        closest_facility = None
        min_distance = float('inf')

        for facility in facilities:
            name, street, zip_code, district, lat, lon, opening_time = facility
            to_coords = (lat, lon)

            print(f"[DEBUG][find_closest_facility_with_route] Checking facility '{name}' at {to_coords}")
            
            # Construct the OSRM URL
            osrm_url = osrm_url_template.format(
                from_lon=user_coords[1], from_lat=user_coords[0],
                to_lon=lon, to_lat=lat
            )
            
            try:
                response = requests.get(osrm_url, params={"overview": "false"})
                response.raise_for_status()
                route_data = response.json()

                if route_data.get("routes"):
                    route = route_data["routes"][0]
                    distance_km = route["distance"] / 1000  # Convert meters to kilometers
                    duration_min = route["duration"] / 60  # Convert seconds to minutes

                    print(f"[DEBUG][find_closest_facility_with_route] Facility '{name}' Distance: {distance_km} km, Duration: {duration_min} min")
                    
                    # Update closest facility if the current one is nearer
                    if distance_km < min_distance:
                        print(f"[DEBUG][find_closest_facility_with_route] Found a closer facility: {name}")
                        min_distance = distance_km
                        closest_facility = {
                            "name": name,
                            "address": f"{street}, {zip_code} Frankfurt am Main",
                            "district": district,
                            "distance_km": distance_km,
                            "duration_min": duration_min,
                            "opening_time": opening_time,
                        }

            except requests.exceptions.RequestException as e:
                print(f"[ERROR][find_closest_facility_with_route] Error calculating route to '{name}': {e}")

        conn.close()

        if closest_facility:
            print(f"[INFO][find_closest_facility_with_route] Closest facility: {closest_facility['name']} at {closest_facility['address']}")
            return closest_facility
        else:
            print("[WARNING][find_closest_facility_with_route] No facilities found")
            return {"error": "No facilities found."}

    except Exception as e:
        print(f"[ERROR][find_closest_facility_with_route] Error accessing database: {e}")
        return {"error": "Database query failed."}

# 1) Wrap functions into LangChain tools, making them callable by the agent
@tool
def geocode_address_tool(address: str):
    """
    Takes in an address and geocodes it to return latitude and longitude.
    """
    print(f"[INFO][Tool: geocode_address_tool] Tool called with address: {address}")
    lat, lon = geocode_address(address)
    if lat is not None and lon is not None:
        result = {"latitude": lat, "longitude": lon}
        print(f"[INFO][Tool: geocode_address_tool] Tool returning result: {result}")
        return result
    else:
        print("[WARNING][Tool: geocode_address_tool] Tool returning failure message.")
        return "Geocoding failed."

@tool
def find_closest_facility_tool(user_coords: tuple):
    """
    Takes in the user coordinates, result from geocode_address_tool and calculates the route, distance and duration between all facilities in the database and finds the closest recycling facility to the user's coordinates.
    """
    print(f"[INFO][Tool: find_closest_facility_tool] Tool called with user_coords: {user_coords}")
    if not db_path:
        print("[ERROR][Tool: find_closest_facility_tool] Database path is not configured.")
        return {"error": "Database path is not set. Please check the configuration."}
    return find_closest_facility(user_coords, db_path)

# Setup the toolkit
toolkit = [geocode_address_tool, find_closest_facility_tool]

# 2) Initiating LLM 
logging.info("Initializing LLM with model gpt-4o-mini")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=openai_api_key
)

# 3) Define Nodes
system_prompt = """
You are an assistant specializing in finding the closest facility for a given address. You should get the user address as input including atleast street, house number and zip code. 
The user can just mention their address or include it in a sentence by for example saying "Bergerstraße 148, 60385, Frankfurt am Main", for the latter case extract the address from the query. 

If you are not able to discern this information, ask them to clarity! After you are able to discern all information, use the following tools to answer the query:
- geocode_address_tool: Takes in an address and returns latitude and longitude.
- find_closest_facility_tool: Takes in latitude and longitude to calculate the distance to all facilities in the database and return the closest one.

Given the user address in the query, you should first geocode the address and then use it as input for the find_closest_facility_tool. In other cases, just answer the query.

Focus on understanding the user question and solving it with the available tools. If you do not have a tool or knowledge to answer the question, say so. 
Avoid discussing topics that are religious, political, harmful, violent, sexual, filthy, or in any way negative, sad, or provocative.
"""

# Customize how LLM interacts with tools, including placeholders for dynamic inputs and outputs
tool_calling_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# Creates an agent that can reason and invoke tools dynamically
tool_runnable = create_tool_calling_agent(llm, toolkit, tool_calling_prompt)
logging.info("Tool calling agent created")

# Agent reasons based on state and decides which tools to invoke and how - what tools to call with what input
def run_tool_agent(state):
    """
    Executes the tool-calling agent to process the user's input and invoke the appropriate tools.

    Args:
        state (dict): A dictionary containing the agent state, including user input and chat history.

    Returns:
        dict: Updated state with the agent's outcome after invoking tools.
    """
    agent_outcome = tool_runnable.invoke(state)
    logging.info(f"Agent outcome: {agent_outcome}")
    return {"agent_outcome": agent_outcome}

# Create Tool Executor - manages tool execution and tracks tool outcomes
tool_executor = ToolExecutor(toolkit)
logging.info("Tool executor initialized")

# Execute tools selected by agent based on reasoning process - processes tool output and update state
def execute_tools(state):
    """
    Executes tools selected by the agent and processes their outputs.

    Args:
        state (dict): A dictionary containing the agent's outcome and tool-related details.

    Returns:
        dict: Updated state with intermediate steps (tool outputs) for each executed tool.
    """
    # Retrieves agent outcome from state
    agent_action = state['agent_outcome']
    if type(agent_action) is not list:
        agent_action = [agent_action]

    # Iterate over actions
    steps = []
    for action in agent_action:
        logging.info(f"Executing tool: {action.tool} with input: {action.tool_input}")
        output = tool_executor.invoke(action) 
        logging.info(f"Tool result: {output}")
        steps.append((action, str(output)))
    
    return {"intermediate_steps": steps}

# 4) Define Graph State Structure - what is passed between workflow nodes
class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, list, ToolAgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[Union[tuple[AgentAction, str], tuple[ToolAgentAction, str]]], operator.add]
    
# 5) Add conditional Edge Logic - evaluate agent's current state and decide on further steps
def should_continue(data):
    """
    Determines whether the agent should continue execution or terminate.

    Args:
        data (dict): State data containing the agent's outcome.

    Returns:
        str: "CONTINUE" if the agent should execute more tools, "END" otherwise.
    """
    # Inspects agent_outcome in data dictionary - if it's AgentFinish then the agent has completed task
    if isinstance(data['agent_outcome'], AgentFinish):
        logging.info("Agent finished execution")
        return "END"
    else:
        logging.info("Agent continuing execution")
        return "CONTINUE"

# 6) Build the Graph - represents workflow logic as directed graph
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

app = workflow.compile()
logging.info("Workflow compiled successfully")

# Run the agent - iteratively executes workflow and result are streamed back after each node's execution
if __name__ == "__main__":
    # Run the workflow with input
    inputs = {"input": "Meine Adresse ist: Frankenallee 41, 60327, Frankfurt am Main", "chat_history": []}
    config = {"configurable": {"thread_id": "1"}}

    for s in app.stream(inputs, config=config):
        output = list(s.values())[0]
        print("------------------------------------------------------")