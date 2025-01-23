import base64
import os
import logging
from typing import Optional, Union, List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Helper function: Encode image to Base64
def encode_image_to_base64(image_path: str) -> str:
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
    logging.info("Preparing image message with user input and image data.")
    image_data = encode_image_to_base64(image_path)
    return [
        {"type": "text", "text": user_input},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
    ]

# Helper function: Update agent outcome
def update_outcome(state: 'AgentState', outcome: Union[str, dict]):
    state["agent_outcome"] = outcome
    logging.info(f"Updated agent outcome: {outcome}")

# Tool for adding two numbers
@tool
def add(a: float, b: float) -> float:
    """Takes in two numbers and returns the sum."""
    logging.info(f"Executing add tool with arguments: a={a}, b={b}")
    return a + b

# Initialize the model with tools
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))
model_with_tools = llm.bind_tools([add])
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

# Classify query into waste categories
def classify_waste_query_llm(user_query: str) -> str:
    predefined_categories = {
        "battery_waste": "If the query is about the disposal of batteries and related items.",
        "bio_waste": "If the query is about the disposal of organic waste.",
        "electronic_waste": "If the query is about the disposal of electronic waste and related items.",
        "glas_waste": "If the query is about the disposal of glass waste.",
        "package_waste": "If the query is about the disposal of packaging waste.",
        "paper_waste": "If the query is about the disposal of paper waste.",
        "residual_waste": "If the query is about the disposal of non-recyclable waste or items that are not assigned to any specific category, such as heavily soiled materials, hygiene products, or broken household items.",
        "other_waste": "If the query is about the disposal of waste types that do not fit other categories.",
        "general": "If the query is about general waste separation information in Frankfurt am Main.",
        "fallback": "If the query does not match any of the above mentioned categories."
    }

    prompt = f"""
    Classify the following query into one of the categories, considering the description of each: 
    {', '.join([f'{key} ({description})' for key, description in predefined_categories.items()])}.
    
    Answer with only the keywords, for example 'battery_waste'. If the query does not fit any category, respond with 'fallback'.

    Query: {user_query}
    """
    try:
        classification_response = model_with_tools.invoke([HumanMessage(content=prompt)])
        category = classification_response.content.strip().lower()

        # Ensure the response matches one of the predefined categories
        if category in predefined_categories:
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

        # Debug filtered documents metadata
        for doc in filtered_docs:
            logging.debug(f"Filtered Document Metadata: {doc.metadata}")

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
        logging.error(f"Error during similarity search: {e}", exc_info=True)
        return []

# Invoke the model
def invoke_model(state: AgentState):
    logging.info("Invoking the model with the current conversation history.")
    try:
        response = model_with_tools.invoke(state["chat_history"])
        logging.info(f"Model response received: {response.content}")
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
    a: Optional[float] = None,
    b: Optional[float] = None,
    faiss_store_path: str = "vectorstore_path"
) -> Union[str, dict]:
    logging.info(f"Processing input with parameters: text={text}, image_path={image_path}, a={a}, b={b}")

    if a is not None and b is not None:
        user_input = f"Add {a} and {b}"
        try:
            result = add.run({"a": a, "b": b})
            final_response = {"response": f"The sum of {a} and {b} is {result}."}
            logging.info(f"Final agent answer: {final_response}")
            return final_response
        except Exception as e:
            logging.error(f"Error running the add tool: {e}")
            return {"error": "Failed to execute the add tool."}

    elif text and image_path:
        user_input = f"{text} (with image)"
    elif text:
        user_input = text
    elif image_path:
        user_input = "Image input provided."
    else:
        return "Please provide valid input."

    # Classify query using LLM
    category = classify_waste_query_llm(user_input)
    logging.info(f"Query classified as '{category}'. User input: {user_input}")

    if category == "fallback":
        # Generate an LLM-based response for fallback cases
        logging.info("Handling fallback query by generating a response directly with the LLM.")
        prompt = f"""
        The user asked a question that does not match any specific waste category.
        Please provide an accurate and helpful response.

        Query:
        {user_input}

        Answer the query concisely and informatively.
        """
        logging.debug(f"Generated prompt for fallback case: {prompt}")

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
    faiss_store_path="faiss_store"
    
    #logging.info("Test case 1: Add two numbers")
    #print(process_input(a=5, b=10))

    #logging.info("Test case 2: Text input")
    #print(process_input(text="What is the capital of France?"))

    #logging.info("Test case 3: Waste Text input")
    #print(process_input(text="Where to dispose diapers?", faiss_store_path=faiss_store_path))
    
    logging.info("Test case 4: Image input")
    print(process_input(image_path="static/uploads/obst_test.jpg", faiss_store_path=faiss_store_path))

    #logging.info("Test case 5: Text + Image input")
    #print(process_input(
    #    text="Was ist auf dem Bild zu sehen?",
    #    image_path="static/uploads/obst_test.jpg",
    #    faiss_store_path=faiss_store_path
    #))
