import base64
import os
import logging
import json
import requests

from typing import Optional, Union, List, TypedDict, Tuple
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global variables
openai_api_key = os.getenv("OPENAI_API_KEY")
faiss_store_path="faiss_store"
with open("categories.json", "r", encoding="utf-8") as file:
    categories = json.load(file)

# Helper function: Encode image to Base64 and structure correctly for API
def create_image_message(user_input: str, image_path: str) -> HumanMessage:
    """
    Combines image encoding and message preparation into one function to return a HumanMessage.

    Args:
        user_input (str): The textual input to include in the message.
        image_path (str): The file path to the image.

    Returns:
        HumanMessage: A message containing the user input and the base64-encoded image.
    """
    logging.info(f"Creating image message with user input and image at path: {image_path}")
    
    # Validate the file path
    if not os.path.isfile(image_path):
        logging.error(f"Image file not found: {image_path}")
    
    try:
        # Encode the image as Base64
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        logging.info("Image successfully encoded to Base64.")
        
        # Return a HumanMessage with the image and user input
        return HumanMessage(
            content=[
                {"type": "text", "text": user_input},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]
        )
    except Exception as e:
        logging.error(f"Error creating image message: {e}")
        raise e

# Helper function: Update agent outcome
def update_outcome(state: 'AgentState', outcome: Union[str, dict]):
    state["agent_outcome"] = outcome
    logging.info(f"Updated agent outcome: {outcome}")

# Tool for adding two numbers
@tool
def geocode_address(address: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Takes an address and returns its latitude and longitude.
    """
    url = "https://nominatim.openstreetmap.org/search"
    headers = {'User-Agent': 'WasteSeparationChatbot/1.0 (nghhoang@gmail.com)'}
    params = {'q': address, 'format': 'json', 'limit': 1}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        if data:
            lat, lon = float(data[0]['lat']), float(data[0]['lon'])
            return lat, lon
    except Exception as e:
        logging.error(f"Error geocoding address '{address}': {e}")
        return None, None

# Initialize the model with tools
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
model_with_tools = llm.bind_tools([geocode_address])
logging.info("Tools successfully bound to the model.")

# Define AgentState
class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    agent_outcome: Union[str, dict, None]
    intermediate_steps: List[tuple]

# Initialize AgentState
def initialize_state(user_input: str, image_path: Optional[str] = None) -> AgentState:
    logging.info(f"Initializing AgentState for input: {user_input}")
    chat_history = [HumanMessage(content=user_input)]

    if image_path:
        try:
            chat_history.append(HumanMessage(content=create_image_message(user_input, image_path)))
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

# Classify query into waste categories
def classify_waste_query_llm(user_query: Optional[str] = None, image_path: Optional[str] = None) -> str:
    try:
        # Prepare the HumanMessage
        if image_path:
            human_message = create_image_message(user_query or "Classify this image:", image_path)
        else:
            # For text-only input
            human_message = HumanMessage(
                content=[{"type": "text", "text": user_query or "Classify this input:"}]
            )
    except Exception as e:
        logging.error(f"Error creating HumanMessage: {e}. Returning fallback.")
        return "fallback"

    # Convert the dictionary into a formatted string
    formatted_categories = "\n".join([f"- {key}: {value}" for key, value in categories.items()])
    keywords_categories = list(categories.keys())
    
    # Construct the system prompt for classification
    system_prompt = SystemMessage(
        content=(
            f"""
            You are an expert in classifying user queries about waste separation that include text and/or images into the following predefined categories:
            {formatted_categories}

            Input Types:
            1. Only Text: Analyze the text query and classify it based on the provided category descriptions and your knowledge.
            2. Only Image: Analyze the objects or content in the image and classify it based on the category descriptions and your knowledge.
            3. Text and Image:
                - First, identify the objects or content in the image.
                - Then, interpret the text query and determine how it relates to the image.
                - Combine both inputs to classify the query. If there is a conflict, prioritize the image content unless the text explicitly specifies otherwise.

            Examples:
            - If the text says "Where to dispose this?" with an image of food scraps, classify as 'bio_waste'.

            Response Format:
            - Respond with **only one keyword** from the following categories: {keywords_categories}.
            - If the input cannot fit into any of the categories, respond with 'fallback'.

            Important Note:
            - Be specific in recognizing objects and interpreting intent based on both inputs.
            - For ambiguous cases, rely on the most clear and relevant input (text or image).
            """
        )
    )

    try:
        # Invoke the LLM with the system prompt and HumanMessage
        classification_response = model_with_tools.invoke([system_prompt, human_message])
        category = classification_response.content.strip().lower()
        logging.info(f"LLM Query Classification Response: {category}")

        if category in categories:
            return category
        logging.warning(f"Unmatched category returned by LLM: {category}")
        return "fallback"
    except Exception as e:
        logging.error(f"Error classifying query: {e}")
        return "fallback"



# Retrieval Function
def retrieve_similar_chunks(query: str, faiss_store_path: str, category: str, k: int = 2) -> List[str]:
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

# Invoke the model
def invoke_model(state: AgentState):
    try:
        response = model_with_tools.invoke(state["chat_history"])
        logging.info(f"Model invoked with conversation history. Response: {response.content}")
        state["chat_history"].append(response)
        return response
    except Exception as e:
        logging.error(f"Error invoking model: {e}")
        update_outcome(state, {"error": str(e)})
        return None
    
# Workflow decision logic
def should_continue(state: AgentState) -> str:
    if state["agent_outcome"] is not None:
        logging.info("Agent finished execution.")
        return "END"
    else:
        logging.info("Agent continuing execution.")
        return "CONTINUE"

# Node: Reasoning (agent)
def run_tool_agent(state: AgentState) -> AgentState:
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
    response = state["chat_history"][-1]
    if hasattr(response, "tool_calls"):
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            if tool_name == "geocode":
                try:
                    address = tool_args.get("address")
                    if address:
                        lat, lon = geocode_address(address)
                        logging.info(f"Tool '{tool_name}' executed successfully. Coordinates: {lat}, {lon}")
                        tool_message = {
                            "role": "tool",
                            "name": tool_name,
                            "content": f"Coordinates: Latitude={lat}, Longitude={lon}" if lat and lon else "Geocoding failed.",
                            "tool_call_id": tool_call_id,
                        }
                        state["chat_history"].append(tool_message)
                        update_outcome(state, f"Coordinates: {lat}, {lon}" if lat and lon else "Failed to geocode address.")
                    else:
                        logging.warning("No address provided for geocode tool.")
                        update_outcome(state, {"error": "No address provided."})
                except Exception as e:
                    logging.error(f"Error executing tool '{tool_name}': {e}")
                    update_outcome(state, {"error": str(e)})
    else:
        logging.warning("No tool calls found. Skipping action node.")
    return state


# Build the workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", run_tool_agent)
workflow.add_node("action", execute_tools)
workflow.set_entry_point("agent")
workflow.add_edge("action", "agent")
workflow.add_conditional_edges("agent", should_continue, {"CONTINUE": "action", "END": END})
app = workflow.compile()

# Process user input and route based on classification
def process_input(
    text: Optional[str] = None,
    image_path: Optional[str] = None,
    address: Optional[str] = None,
    faiss_store_path: str = "vectorstore_path"
) -> Union[str, dict]:
    logging.info(f"Processing input with parameters: text={text}, image_path={image_path}, address={address}")

    # Handle geocoding for an address
    if address:
        logging.info(f"Geocoding address: {address}")
        try:
            lat, lon = geocode_address(address)
            if lat is not None and lon is not None:
                final_response = {"response": f"Coordinates for '{address}': Latitude={lat}, Longitude={lon}"}
            else:
                final_response = {"error": f"Failed to geocode the address: {address}"}
            logging.info(f"Final agent answer: {final_response}")
            return final_response
        except Exception as e:
            logging.error(f"Error geocoding address: {e}")
            return {"error": "Failed to geocode the address."}

    # Handle text and image inputs
    if text and image_path:
        user_input = f"{text} (with image)"
    elif text:
        user_input = text
    elif image_path:
        user_input = "Image input provided."
    else:
        return {"error": "Please provide valid input (text, image, or address)."}

    # Classify query or image
    category = classify_waste_query_llm(user_query=user_input, image_path=image_path)
    logging.info(f"Query classified as '{category}'. User input: {user_input}")

    if category == "fallback":
        logging.info("Handling fallback query by generating a response directly with the LLM.")
        prompt = f"""
        The user asked a question that does not match any specific waste category.
        Please provide an accurate and helpful response.

        Query:
        {user_input}

        Answer the query concisely and informatively.
        """
        try:
            fallback_response = model_with_tools.invoke([HumanMessage(content=prompt)])
            final_answer = fallback_response.content.strip()
            final_response = {"response": final_answer}
            logging.info(f"Final agent answer for fallback: {final_response}")
            return final_response
        except Exception as e:
            logging.error(f"Error generating fallback response: {e}")
            return {"error": "Failed to generate a response for the fallback query."}

    # Perform retrieval for the classified category
    retrieved_chunks = retrieve_similar_chunks(user_input or "", faiss_store_path, category)
    if not retrieved_chunks:
        final_response = {"category": category, "retrieved_chunks": [], "final_answer": "No relevant information found."}
        logging.info(f"Final agent answer: {final_response}")
        return final_response

    logging.info("Retrieved relevant chunks successfully.")
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
    Based on the following information about {category.replace('_', ' ')}, answer the user's query:
    Context:
    {context}

    Query:
    {user_input}

    Answer the query concisely.
    """
    try:
        final_answer_response = model_with_tools.invoke([HumanMessage(content=prompt)])
        final_answer = final_answer_response.content.strip()
        final_response = {"category": category, "retrieved_chunks": retrieved_chunks, "final_answer": final_answer}
        logging.info(f"Final agent answer: {final_response}")
        return final_response
    except Exception as e:
        logging.error(f"Error generating final answer: {e}")
        final_response = {"category": category, "retrieved_chunks": retrieved_chunks, "final_answer": "Error generating final answer."}
        logging.info(f"Final agent answer: {final_response}")
        return final_response


# Example test cases
if __name__ == "__main__":   
    
    logging.info("Test case 1: Geocode Address")
    print(process_input(address="Bergerstraße 148, 60385, Frankfurt am Main"))
    """
    logging.info("Test case 2: Text input")
    print(process_input(text="What is the capital of France?"))
    
    logging.info("Test case 3: Waste Text input")
    print(process_input(text="Where do I dispose old fruits?", faiss_store_path=faiss_store_path))
   
    logging.info("Test case 4: Image input")
    print(process_input(image_path="static/uploads/paket_test.jpg", faiss_store_path=faiss_store_path))

    logging.info("Test case 5: Text + Image input")
    print(process_input(
        text="Where do I dispose the objects on the image?",
        image_path="static/uploads/obst_test.jpg",
        faiss_store_path=faiss_store_path
    ))
    """
