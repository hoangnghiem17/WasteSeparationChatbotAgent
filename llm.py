import os
import logging

import openai
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(prompt, base64_image=None):
    """
    Process a text query using OpenAI's language model.
    """
    try:
        logging.info("call_llm() invoked with prompt: %s", prompt)
        messages = [{"role": "system", "content": "You are an assistant that provides recycling tips for Frankfurt am Main, answer in German."}]
        
        if prompt:
            messages.append({"role": "user", "content": prompt})
            logging.debug("User prompt added to messages.")
            
        if base64_image:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                ]
            })
            logging.debug("Base64 image appended to messages.")
            
        logging.info("Sending request to OpenAI LLM...")
            
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        logging.info("OpenAI API call successful. Extracting response.")
        result = response.choices[0].message.content
        logging.debug("LLM Response: %s", result)
        
        return result
        
    except Exception as e:
        logging.error("Error in call_llm(): %s", str(e), exc_info=True)
        return f"Error: {e}"
    