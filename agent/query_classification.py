import logging
import base64
import os

from langchain_core.messages import SystemMessage, HumanMessage

from config.llm import llm
from config.prompt import categories, formatted_categories, keywords_categories

def classify_query(user_query: str = None, image_path: str = None) -> str:
    """
    Classifies the input (text query and/or image) into predefined categories using the LLM.

    Args:
        user_query (str, optional): The text input from the user.
        image_path (str, optional): Path to an image file for additional context.

    Returns:
        str: The classification category or 'fallback' if classification fails.
    """
    # Validate input that text or image is given
    if not user_query and not image_path:
        logging.error("Both user_query and image_path are missing. Cannot classify without input.")
        return "fallback"

    logging.info(f"Starting classification for query: {user_query or '(image only)'}")

    # Construct query classification prompt
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
    system_message = SystemMessage(content=prompt)

    # Prepare the user message (text, image, or both)
    if image_path:
        try:
            logging.info(f"Encoding image at path: {image_path}")

            # Validate and encode the image
            if not os.path.isfile(image_path):
                logging.error(f"Image file not found: {image_path}")
                return "fallback"

            # Encode image to base64
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            logging.info("Image successfully encoded to Base64.")

            # Construct LangChain HumanMessage object with text and base64-encoded image
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

    # Combine system and user messages and invoke LLM
    messages = [system_message, user_message]

    try:
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
