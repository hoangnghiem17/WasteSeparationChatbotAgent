import os
import logging
import requests
from typing import  Union, TypedDict, Annotated, List
import operator
import sqlite3
import json
import base64

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain.agents import create_tool_calling_agent
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.agents import AgentFinish
from langchain.agents.output_parsers.tools import ToolAgentAction
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global variables
openai_api_key = os.getenv("OPENAI_API_KEY")
faiss_store_path = "faiss_store"
db_path = "database/waste_separation_frankfurt.db"
   
# Prompt construction
with open("config/categories.json", "r", encoding="utf-8") as file:
    categories = json.load(file)
    
formatted_categories = "\n".join([f"    - {key}: {value}" for key, value in categories.items()])
keywords_categories = list(categories.keys())

# Agent System Prompt Formatting  
with open("config/agent_prompt.txt", "r", encoding="utf-8") as file:
    prompt_template = file.read()    

system_prompt = prompt_template.format(
    formatted_categories=formatted_categories,
    keywords_categories=keywords_categories
)

# Initialize LLM 
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

# Define tool functions
def classify_query(user_query: str = None, image_path: str = None) -> str:
    """
    Classifies the input (text query and/or image) into predefined waste categories using the LLM.

    Args:
        user_query (str, optional): The text input from the user.
        image_path (str, optional): Path to an image file for additional context.

    Returns:
        str: The classification category or 'fallback' if classification fails.
    """
    if not user_query and not image_path:
        logging.error("Both user_query and image_path are missing. Cannot classify without input.")
        return "fallback"

    logging.info(f"Starting classification for query: {user_query or '(image only)'}")

    # Define the base system prompt
    prompt = f"""
    You are an expert in classifying user queries about waste separation into the following predefined categories:
    {formatted_categories}

    Input:
    - Analyze the text query and/or the image (if provided) to determine the most appropriate category.

    Response Format:
    - Respond with one keyword from these categories: {keywords_categories}.
    - If no category applies, respond with 'fallback'.
    """
    logging.info("Constructed classification prompt for LLM.")

    # Prepare the system message
    system_message = SystemMessage(content=prompt)

    # Prepare the user message (text, image, or both)
    if image_path:
        try:
            logging.info(f"Encoding image at path: {image_path}")

            # Validate and encode the image
            if not os.path.isfile(image_path):
                logging.error(f"Image file not found: {image_path}")
                return "fallback"

            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            logging.info("Image successfully encoded to Base64.")

            # Create a HumanMessage with image and optional text
            user_message_content = []
            if user_query:
                user_message_content.append({"type": "text", "text": user_query})
            user_message_content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            )
            user_message = HumanMessage(content=user_message_content)
        except Exception as e:
            logging.error(f"Error encoding image: {e}")
            return "fallback"
    elif user_query:
        # Create a text-only HumanMessage
        user_message = HumanMessage(content=user_query)
    else:
        logging.error("Both query and image are missing.")
        return "fallback"

    # Combine system and user messages
    messages = [system_message, user_message]

    try:
        # Invoke the LLM with the prepared messages
        classification_response = llm.invoke(input=messages)
        category = classification_response.content.strip().lower()

        # Validate the returned category
        if category in categories:
            logging.info(f"Query classified as category: {category}")
            return category
        else:
            logging.warning(f"Unknown category returned by LLM: {category}. Defaulting to 'fallback'.")
            return "fallback"
    except Exception as e:
        logging.error(f"Error during classification: {e}")
        return "fallback"


# Retrieval Function
def retrieve_chunks(query: str, faiss_store_path: str, category: str, k: int = 2) -> List[str]:
    """
    Loads a FAISS vector store, filters by metadata, and performs a similarity search for the input query.

    Args:
        query (str): The user query for similarity search.
        faiss_store_path (str): Path to the saved FAISS vector store.
        category (str): The waste category for filtering (e.g., "bio_waste").
        k (int, optional): The number of most similar chunks to retrieve. Defaults to 2.

    Returns:
        List[str]: A list of the most similar text chunks.
    """
    try:
        logging.info(f"Starting similarity search for query: '{query}' in category: '{category}'")

        # Load the FAISS vector store
        vector_store = FAISS.load_local(
            faiss_store_path,
            OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
            allow_dangerous_deserialization=True
        )

        # Filter vector store documents by the specified category
        filtered_docs = [
            doc for doc_id, doc in vector_store.docstore._dict.items()
            if doc.metadata.get("source") == category
        ]
        if not filtered_docs:
            logging.warning(f"No matching documents found for category: {category}")
            return []

        logging.info(f"Filtered {len(filtered_docs)} documents for category: {category}")

        # Create a temporary vector store for the filtered documents
        embeddings = vector_store.embeddings
        temp_vector_store = FAISS.from_documents(filtered_docs, embeddings)
        logging.info("Temporary vector store successfully created.")

        # Perform similarity search
        logging.info(f"Performing similarity search with top {k} results.")
        results_with_scores = temp_vector_store.similarity_search_with_score(query, k=k)

        if results_with_scores:
            for idx, (doc, score) in enumerate(results_with_scores, 1):
                logging.info(f"Chunk {idx} with score {score}.")
            return [result[0].page_content for result in results_with_scores]
        else:
            logging.info("No similar chunks found.")
            return []
    except Exception as e:
        logging.error(f"Error during similarity search: {e}")
        return []
    
# Geocoding function
def geocode_address(address):
    """
    Geocodes a user's address using the Nominatim API.

    Args:
        address (str): User's address as a string.

    Returns:
        Tuple[float, float]: A tuple containing the latitude and longitude of the address. Returns (None, None) if geocoding fails.
    """
    url = "https://nominatim.openstreetmap.org/search"
    headers = {'User-Agent': 'WasteSeparationChatbot/1.0 (nghhoang@gmail.com)'}
    params = {'q': address, 'format': 'json', 'limit': 1}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    except requests.exceptions.RequestException:
        pass

    return None, None

def find_closest_facility(user_coords, db_path):
    """
    Finds the closest recycling facility to the user's coordinates using SQLite database and OSRM API.

    Args:
        user_coords (Tuple[float, float]): A tuple of the user's latitude and longitude.
        db_path (str): Path to the SQLite3 database file containing facility data.

    Returns:
        dict: Dictionary containing the closest facility's details or an error message if no facility is found.
    """
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

                    if distance_km < min_distance:
                        min_distance = distance_km
                        closest_facility = {
                            "name": name,
                            "address": f"{street}, {zip_code} Frankfurt am Main",
                            "district": district,
                            "distance_km": distance_km,
                            "duration_min": duration_min,
                            "opening_time": opening_time,
                        }

            except requests.exceptions.RequestException:
                continue

        conn.close()

        return closest_facility or {"error": "No facilities found."}

    except Exception:
        return {"error": "Database query failed."}

# Tools wrapped with LangChain decorators
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
    
# Create tool calling agent
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

# Agent reasons based on state and decides which tools to invoke and how - what tools to call with what input
def run_tool_agent(state):
    """
    Executes the tool-calling agent to process the user's input and invoke the appropriate tools.

    Args:
        state (dict): A dictionary containing the agent state, including user input and chat history.

    Returns:
        dict: Updated state with the agent's outcome after invoking tools.
    """
    logging.info(f"Agent reasoning started with input: {state['input']}")
    agent_outcome = tool_runnable.invoke(state)
    logging.info(f"Agent reasoning completed. Outcome: {agent_outcome}")
    return {"agent_outcome": agent_outcome}

# Execute tools selected by agent based on reasoning process - processes tool output and update state
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

# Conditional Edge Logic
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

# Example Test
if __name__ == "__main__":
    # Define test cases for text-only, image-only, and combined inputs
    test_cases = [
        #{"user_input": "Where is the next recycling facility to Frankallee 41, 60327, Franfurt am Main?", "image_path": None},  # Example 1: Address input
        #{"user_input": "Where do I dispose my wine bottle?", "image_path": None},            # Example 2: Glas waste
        #{"user_input": "In which trash do diapers belong?", "image_path": None},             # Example 3: Bulky waste
        #{"user_input": "How does the recycling system in Frankfurt works?", "image_path": None},  # Example 4: General Waste
        #{"user_input": "What is the capital of France?", "image_path": None},                # Example 5: Edge case
        #{"user_input": None, "image_path": "static/uploads/paket_test.jpg"},             # Example 6: Image-only input
        #{"user_input": "What category does this belong to?", "image_path": "static/uploads/obst_test.jpg"},  # Example 7: Text and image input
    ]

    # Testing Classify_Query
    """
    for i, test_case in enumerate(test_cases, start=1):
        print(f"\n--- Test Case {i} ---")
        user_query = test_case["user_input"]
        image_path = test_case["image_path"]

        result = classify_query(user_query=user_query, image_path=image_path)
    """
    
    # Testing Agent
    for i, test_case in enumerate(test_cases, start=1):
        user_input = test_case["user_input"]
        image_path = test_case["image_path"]
        
        print(f"\n--- Test Case {i} ---")
        
        # Initialize agent state
        agent_state = {
            "input": {"user_query": user_input, "image_path": image_path},  # Pass both user_query and image_path
            "chat_history": [],
            "agent_outcome": None,
            "intermediate_steps": []
        }

        # Run the workflow
        output_state = app.invoke(agent_state)