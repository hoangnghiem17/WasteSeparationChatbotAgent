import os
import logging
import openai
from dotenv import load_dotenv

from image import encode_image

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(prompt, image_file=None):
    """
    Call OpenAI's multimodal LLM with optional image input.
    """
    try:
        logging.info("call_llm() invoked with prompt: %s", prompt)
        
        # Encode image if provided
        base64_image = None
        if image_file:
            base64_image = encode_image(image_file)

        # Construct message content
        message_content = [{"type": "text", "text": prompt}]
        logging.debug("User prompt added to messages.")
        
        if base64_image:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            )
            logging.debug("Base64 image appended to messages.")
        
        # Log the constructed payload
        logging.debug("Sending request Payload to OpenAI LLM: %s", message_content)
        
        system_prompt = "You are an assistant that provides recycling tips for Frankfurt am Main. Users can input their questions as text or image. In case of images, try to recognize what is on the picture and answer based on the text provided. Answer in German."
        
        # Send request to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message_content}
            ]
        )
        
        logging.info("OpenAI API call successful. Extracting response.")
        result = response.choices[0].message.content
        logging.debug("LLM Response: %s", result)
        
        return result
        
    except Exception as e:
        logging.error("Error in call_llm(): %s", str(e), exc_info=True)
        return f"Error: {e}"
