import requests
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

# Geocoding function
def geocode_address(address):
    """
    Geocode a user's address using the Nominatim API.

    :param address: User's address as a string.
    :return: Tuple of (latitude, longitude) or (None, None) if geocoding fails.
    """
    print(f"[INFO] Attempting to geocode address: {address}")
    url = "https://nominatim.openstreetmap.org/search"
    headers = {
        'User-Agent': 'WasteSeparationChatbot/1.0 (nghhoang@gmail.com)'
    }
    params = {
        'q': address,
        'format': 'json',
        'limit': 1
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        print(f"[INFO] Received response: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        if data:
            lat, lon = float(data[0]['lat']), float(data[0]['lon'])
            print(f"[INFO] Geocoded address to Latitude: {lat}, Longitude: {lon}")
            return lat, lon
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Error geocoding address '{address}': {e}")

    print("[WARNING] Geocoding failed, returning None, None")
    return None, None

# Define the geocoding tool
@tool
def geocode_tool(address: str):
    """
    Geocodes an address to latitude and longitude.
    """
    print(f"[INFO] Tool called with address: {address}")
    lat, lon = geocode_address(address)
    if lat is not None and lon is not None:
        result = f"Latitude: {lat}, Longitude: {lon}"
        print(f"[INFO] Tool returning result: {result}")
        return result
    else:
        print("[WARNING] Tool returning failure message.")
        return "Geocoding failed."

# Initialize the model
model = ChatOpenAI(model="gpt-4o-mini",
                   api_key=os.getenv("OPENAI_API_KEY"),
                   temperature=0).bind_tools([geocode_tool])

# Define the function that calls the model
def call_model(state: MessagesState):
    print("[INFO] Calling model with current state messages.")
    messages = state['messages']
    response = model.invoke(messages)
    print(f"[INFO] Model response received: {response}")
    return {"messages": [response]}

# Define the function that determines the next step
def should_continue(state: MessagesState):
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        print("[INFO] Model decided to call a tool.")
        return "tools"
    print("[INFO] Model has no further steps, ending conversation.")
    return END

# Initialize the graph
workflow = StateGraph(MessagesState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode([geocode_tool]))

# Define edges
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile()

# Run the agent
if __name__ == "__main__":
    # Define the input message
    input_message = HumanMessage(content="I live in Bergerstraße 148, 60385.")
    print("[INFO] Starting agent with input message.")

    # Invoke the agent
    final_state = app.invoke({"messages": [input_message]})

    # Output the response
    print("[INFO] Final response from agent:")
    print(final_state["messages"][-1].content)
